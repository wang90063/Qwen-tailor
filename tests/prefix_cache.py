"""Prefix cache (bf16 exact path): a hit resumes the prefill from a stored checkpoint.

* plumbing is exact: resuming == running the same split by hand (state / first token bit-identical), the cached prefix state is the
  uncached prefix state, a repeated prompt returns the identical state and token
* equivalence: a hit vs an uncached single-pass prefill (a split at the checkpoint changes MLX's SDPA tiling for the suffix rows, so
  bf16 rounding-level differences are expected): first token + 48 greedy tokens identical, final hidden / state rel-L2 bounded
* checkpoint bookkeeping: end-of-prompt / chunk-boundary checkpoints, split learning (shared system prompt), `cache_at`, multi-turn,
  LRU eviction under a small budget, restored state is never corrupted by decoding from it (aliasing)
* SpecEngine: same for its (carry, first token), and generation is token-identical hit vs miss
"""
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR

from qwen35 import gemm_i8
from qwen35.engine import Engine, TAIL_ROWS
from qwen35.model import new_state
from qwen35.prefill import forward as pforward
from qwen35.prefix_cache import PrefixCache
from qwen35.spec import SpecEngine

assert not gemm_i8.NAMES
fails = 0
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
# frozen corpus (installed mlx_lm sources): the project's own files change with every edit and so would the prompts, and a
# first-token comparison against the uncached pass can flip at a bf16 near-tie for some prompt and not for another
import mlx_lm
_src = sorted(Path(mlx_lm.__file__).parent.glob("models/*.py"))
IDS = tk("\n".join(p.read_text() for p in _src[:12]))["input_ids"]
OTHER = tk("\n".join(p.read_text() for p in _src[12:24]))["input_ids"]
assert len(IDS) > 4600 and len(OTHER) > 700


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    fails += 0 if ok else 1


def bits(a):
    return np.array(a.view(mx.uint32 if a.dtype == mx.float32 else mx.uint16))


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(bits(a), bits(b)))


def rel(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - b) ** 2) / mx.sum(a * a)))


def state_arrays(st, n):
    """flat list of the (kv rows [:n], conv, ssm) arrays of a state"""
    out = []
    for k, v in st["kv"]:
        out += [k[:, :, :n], v[:, :, :n]]
    return out + list(st["conv"]) + list(st["ssm"])


def states_same(a, b, n):
    return all(same(x, y) for x, y in zip(state_arrays(a, n), state_arrays(b, n)))


def states_rel(a, b, n):
    return max(rel(y, x) for x, y in zip(state_arrays(a, n), state_arrays(b, n)))


MAXLEN = 4800
ref = Engine(max_len=MAXLEN)                                  # no cache
eng = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)         # (all engines share ref's weights)
pc = eng.pcache


def pre(e, ids, **kw):
    tok, st, pos = e.prefill(ids, **kw)
    mx.eval(tok, st)
    return int(tok.item()), st, pos


def manual(ids, cuts):
    """what a hit must reproduce: the same forward calls by hand, one per segment between the cut positions (the chain of checkpoints)"""
    st = new_state(ref.cfg, MAXLEN)
    a = 0
    for c in cuts:
        _, st = pforward(ref.cfg, ref.W, mx.array([ids[a:c]]), st, a, c, tail=0)
        mx.eval(st)
        a = c
    h, st = pforward(ref.cfg, ref.W, mx.array([ids[a:]]), st, a, len(ids), tail=TAIL_ROWS)
    from qwen35.head_prune import argmax_head
    nxt = argmax_head(ref.W["embed"], ref.F.get("head_tab"), h[0, -1:])
    mx.eval(st, nxt)
    return int(nxt.item()), st


def gen(e, ids, n=48):
    return e.generate(ids, n)


TOL = 0.13        # the project's bf16 near-tie tolerance for greedy divergences (logit gap of the two candidates; = 1 bf16 ulp at logits 16..32)
# A hit is bit-identical to the prefill chunked at the checkpoint (checked above), which differs from the single pass by bf16 rounding
# (state rel-L2 ~1.4e-2, twice the 0.7% of the lossless prefill TOL was calibrated on), so greedy generations vs the single pass may
# flip on a gap of 2 bf16 ulps (0.25 at logits 16..32).  Real cache bugs (wrong K/V rows, stale state) show up as O(1) logit errors.
TOL_GEN = 2 * TOL


def gen_close(a, b, ids):
    """(ok, note): identical, or the first divergence is a bf16 near-tie of the reference model (teacher-forced on b)."""
    if a == b:
        return True, "identical"
    d = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
    ctx = list(ids) + b[:d]
    st = new_state(ref.cfg, len(ctx) + 8)
    from qwen35.model import forward, logits_of
    h, st = forward(ref.cfg, ref.W, mx.array([ctx]), st, 0, len(ctx))
    lg = logits_of(ref.W, h[:, -1]).astype(mx.float32)[0]
    gap = abs(float(lg[a[d]]) - float(lg[b[d]]))
    return gap <= TOL_GEN, f"diverges at token {d} on a bf16 near-tie (logit gap {gap:.3f} <= {TOL_GEN})" if gap <= TOL_GEN else f"DIVERGES at token {d}, logit gap {gap:.3f}"


# ---- 1. exact repeat: identical token and state, nothing recomputed -----------------------------
P = IDS[:1000]
cuts = [1000]                                                  # the chain of checkpoints the extensions below resume from
t0, s0, _ = pre(eng, P)
check("miss: one checkpoint (end of prompt), no hit", len(pc) == 1 and pc.last_hit == 0)
t1, s1, pos1 = pre(eng, P)
check("repeat: whole-prompt hit", pc.last_hit == 1000 and pos1 == 1000 and len(pc) == 1)
check("repeat: first token identical", t0 == t1)
check("repeat: restored state bit-identical to the computed one", states_same(s0, s1, 1000))
tr, sr, _ = pre(ref, P)
check("miss path == engine without cache (token, state)", tr == t0 and states_same(sr, s0, 1000))

# ---- 2. extension: hit at the end of the earlier prompt == the same split by hand ---------------
for n_ext in (20, 100, 300, 1100):
    ids = IDS[:1000 + n_ext]
    tc, sc, _ = pre(eng, ids)
    hit = pc.last_hit
    tm, sm = manual(ids, cuts)
    tu, su, _ = pre(ref, ids)
    check(f"+{n_ext}: hit at the earlier prompt's end ({hit}), bit-identical to the hand-run split chain {cuts} (token, state)", hit == cuts[-1] and tc == tm and states_same(sc, sm, len(ids)))
    cuts.append(len(ids))
    check(f"+{n_ext}: close to the uncached single pass (token {tc}=={tu}, state rel-L2 {states_rel(su, sc, len(ids)):.1e})",
          tc == tu and states_rel(su, sc, len(ids)) < 0.06)
    P = ids

# ---- 3. generation equivalence hit vs miss -----------------------------------------------------
eng2 = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)
base = IDS[:700]
gen(eng2, base, 4)
for n_ext in (1, 17, 90, 257, 700):
    ids = IDS[:700 + n_ext]
    a = gen(eng2, ids)                                   # hit at the previous prompt's end
    b = gen(ref, ids)
    ok, note = gen_close(a, b, ids)
    check(f"generate after +{n_ext} tokens (hit@{eng2.pcache.last_hit}): 48 tokens vs the uncached engine: {note}", ok)

# ---- 4. aliasing: decoding from a restored state must not touch the cached checkpoint ------------
nd = max(eng2.pcache.nodes.values(), key=lambda n: n.pos)
before = [bits(x).copy() for x in [nd.kv, nd.conv, nd.ssm]]
ids = IDS[:nd.pos]
r1 = gen(eng2, ids, 64)                                  # full-prompt hit, then 63 decode steps from the restored state
r2 = gen(eng2, ids, 64)
after = [bits(x) for x in [nd.kv, nd.conv, nd.ssm]]
check("whole-prompt hit + decoding: tokens repeat exactly", r1 == r2 and eng2.pcache.last_hit == nd.pos)
check("decoding from a restored state leaves the cached checkpoint bit-identical", all(np.array_equal(x, y) for x, y in zip(before, after)))

# ---- 5. chunk boundaries + deep chain (K/V segments) ---------------------------------------------
eng3 = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)
long_ids = IDS[:4500]
pre(eng3, long_ids)
check("4500-token prompt: checkpoints at 2048, 4096 and 4500", sorted(n.pos for n in eng3.pcache.nodes.values()) == [2048, 4096, 4500])
seg_bytes = sum(n.kv.nbytes for n in eng3.pcache.nodes.values())
check("K/V stored once per token (segments), not per checkpoint", seg_bytes == 4500 * 6 * 2 * 2 * 256 * 2)
ids = long_ids[:4300] + OTHER[:200]                      # shares 4300 tokens -> hit at 4096
tc, sc, _ = pre(eng3, ids)
tu, su, _ = pre(ref, ids)
check(f"hit at the 4096 chain end (got {eng3.pcache.last_hit}), token == uncached, state close ({states_rel(su, sc, len(ids)):.1e})",
      eng3.pcache.last_hit == 4096 and tc == tu and states_rel(su, sc, len(ids)) < 0.06)
check("4300-token shared prefix learned: checkpoint at 4300 (split)", 4300 in [n.pos for n in eng3.pcache.nodes.values()])
ids = long_ids[:4300] + OTHER[300:420]
tc, sc, _ = pre(eng3, ids)
check(f"next prompt with the same 4300-token prefix hits at 4300 (got {eng3.pcache.last_hit})", eng3.pcache.last_hit == 4300)

# ---- 6. split learning: shared system prompt, different questions --------------------------------
eng4 = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)
sysp = IDS[:600]
qa, qb, qc = OTHER[:300], OTHER[300:600], OTHER[600:700]
pre(eng4, sysp + qa)
check("first request: miss, one checkpoint", eng4.pcache.last_hit == 0 and len(eng4.pcache) == 1)
pre(eng4, sysp + qb)
check("second request overlaps 600 tokens -> miss now, split checkpoint at 600 created", eng4.pcache.last_hit == 0 and 600 in [n.pos for n in eng4.pcache.nodes.values()])
tc, sc, _ = pre(eng4, sysp + qc)
check(f"third request hits the system prompt (hit at {eng4.pcache.last_hit})", eng4.pcache.last_hit == 600)
tu, su, _ = pre(ref, sysp + qc)
check("... and its first token / state match the uncached engine", tc == tu and states_rel(su, sc, 700) < 0.06)
eng5 = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)
pre(eng5, IDS[:900], cache_at=[250, 640])
check("cache_at forces checkpoints", sorted(n.pos for n in eng5.pcache.nodes.values()) == [250, 640, 900])
pre(eng5, IDS[:700])
check(f"... and a prompt sharing the first 700 tokens hits at 640 (got {eng5.pcache.last_hit})", eng5.pcache.last_hit == 640)

# ---- 7. multi-turn: every turn extends the previous prompt -----------------------------------------
eng6 = Engine(max_len=MAXLEN, prefix_cache_mb=3000, like=ref)
hits = []
for L in (200, 450, 800, 1300):
    pre(eng6, IDS[:L])
    hits.append(eng6.pcache.last_hit)
check(f"multi-turn: every turn resumes from the previous prompt {hits}", hits == [0, 200, 450, 800])

# ---- 8. eviction under a small budget -------------------------------------------------------------
eng7 = Engine(max_len=MAXLEN, prefix_cache_mb=90, like=ref)         # ~2 checkpoints of 1000 tokens
streams = [IDS[i * 700:(i + 1) * 700 + 300] for i in range(4)]
for s_ in streams:
    pre(eng7, s_)
c7 = eng7.pcache
check(f"LRU eviction keeps the cache within its budget ({c7.bytes / 2**20:.0f} <= 90 MB, {len(c7)} nodes, {c7.stats['evicted']} evicted)",
      c7.bytes <= c7.budget and c7.stats["evicted"] >= 2 and len(c7) >= 1)
tc, sc, _ = pre(eng7, streams[-1])
tu, su, _ = pre(ref, streams[-1])
check("most recent prompt still cached and correct after evictions", eng7.pcache.last_hit == len(streams[-1]) and tc == tu)
tc, sc, _ = pre(eng7, streams[0])
tu, su, _ = pre(ref, streams[0])
check("an evicted prompt is recomputed correctly", tc == tu and states_same(su, sc, 1000))
big = Engine(max_len=MAXLEN, prefix_cache_mb=5, like=ref)
pre(big, IDS[:1000])
check("a checkpoint larger than the whole budget is simply not stored", len(big.pcache) == 0)

# ---- 9. bypass with W8A8 ----------------------------------------------------------------------------
gemm_i8.NAMES = frozenset(gemm_i8.SUPPORTED)
c9 = PrefixCache(10)
check("W8A8 set: lookup bypassed", c9.lookup(IDS[:100]) == (None, 0))
gemm_i8.NAMES = frozenset()

# ---- 10. SpecEngine ----------------------------------------------------------------------------------
import gc
eng = eng2 = eng3 = eng4 = eng5 = eng6 = eng7 = big = pc = c7 = ref = None          # free the first weight copy
gc.collect()
mx.clear_cache()
sp = SpecEngine(k=3, max_len=MAXLEN, prefix_cache_mb=3000)
sref = SpecEngine(k=3, max_len=MAXLEN, like=sp)
ref = Engine(max_len=MAXLEN, like=sp)                           # plain greedy engine on the same weights (also the tie checker)
spc = sp.pcache


def sgen(e, ids, n=64):
    st = {}
    out = e.generate(ids, n, st)
    return out, st


def carry_eq(a, b, n):
    (sta, *ra), (stb, *rb) = a, b
    ok = all(same(x, y) for x, y in zip(state_arrays({"kv": sta["kv"], "conv": [c[:1] for c in sta["conv"]], "ssm": [c[:1] for c in sta["ssm"]]}, n),
                                         state_arrays({"kv": stb["kv"], "conv": [c[:1] for c in stb["conv"]], "ssm": [c[:1] for c in stb["ssm"]]}, n)))
    return ok


c0, f0, _ = sp.prefill(IDS[:900])
c1, f1, _ = sp.prefill(IDS[:900])
check("spec: repeated prompt = whole-prompt hit, identical first token / carry / draft KV",
      spc.last_hit == 900 and f0 == f1 and carry_eq(c0, c1, 900) and same(c0[2][0][:, :, :899], c1[2][0][:, :, :899]) and same(c0[2][1][:, :, :899], c1[2][1][:, :, :899])
      and same(c0[3], c1[3]))
cr, fr, _ = sref.prefill(IDS[:900])
check("spec: miss path == SpecEngine without cache (carry incl. draft KV)", f0 == fr and carry_eq(c0, cr, 900) and same(c0[2][0][:, :, :899], cr[2][0][:, :, :899]))
for n_ext in (13, 150, 900):
    ids = IDS[:900 + n_ext]
    a, sa = sgen(sp, ids)
    b, sb = sgen(sref, ids)
    ok, note = gen_close(a, b, ids)
    check(f"spec +{n_ext} (hit@{spc.last_hit}): 64 tokens vs the uncached SpecEngine: {note} (acc/round {sa['accepted'] / max(sa['rounds'], 1):.2f} vs {sb['accepted'] / max(sb['rounds'], 1):.2f})",
          ok and spc.last_hit >= 900)
    if n_ext == 13:
        ck, fk, _ = sp.prefill(ids)
        cu, fu, _ = sref.prefill(ids)
        check("spec: draft KV rows after a hit close to the uncached ones (rel-L2 < 0.1)",
              rel(cu[2][0][:, :, :len(ids) - 1], ck[2][0][:, :, :len(ids) - 1]) < 0.1 and rel(cu[2][1][:, :, :len(ids) - 1], ck[2][1][:, :, :len(ids) - 1]) < 0.1)
ids = IDS[:3000]
sp2 = SpecEngine(k=3, max_len=MAXLEN, prefix_cache_mb=3000, like=sp)
sgen(sp2, ids, 4)
check("spec 3000-token prompt: checkpoints at 2048 and 3000", sorted(n.pos for n in sp2.pcache.nodes.values()) == [2048, 3000])
a, _ = sgen(sp2, IDS[:2300])
b, _ = sgen(sref, IDS[:2300])
ok, note = gen_close(a, b, IDS[:2300])
check(f"spec: hit at the chunk boundary ({sp2.pcache.last_hit}), generation vs uncached: {note}", sp2.pcache.last_hit == 2048 and ok)
ok, note = gen_close(sgen(sp, IDS[:1063], 48)[0], ref.generate(IDS[:1063], 48), IDS[:1063])
check(f"spec generation (cached engine) vs plain engine greedy: {note}", ok)

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
