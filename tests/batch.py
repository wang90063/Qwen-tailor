"""Continuous batching (batch.py + serve.py): everything must be bit-identical to solo decoding.

* deltanet_b / attn_b == the solo kernels (`deltanet` M=1, MLX sdpa_vector via a masked SDPA) per row, ragged positions, slots
  scattered in the pool; untouched slots and rows are never written; the new K / V row lands at `pos`
* Pools.load / read round trip
* the batched step (compiled, async-pipelined, ragged positions, padded bucket) generates the same tokens as Engine.generate
* the scheduler (continuous and static policies; more requests than slots; staggered arrivals; EOS; chunked prefill between decode
  steps): every request's tokens == solo prefill(chunk=same) + solo decode, truncated at its EOS
* prefix cache + scheduler: same, and later requests do hit the cache
"""
import sys
import time

import numpy as np
import mlx.core as mx

from qwen35 import gemm_i8
from qwen35 import batch as Bt
from qwen35.engine import Engine
from qwen35.kernels import deltanet
from qwen35.serve import Server, Request

assert not gemm_i8.NAMES
fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    fails += 0 if ok else 1


def eq(a, b):
    return bool(mx.array_equal(a, b))


rng = np.random.default_rng(11)
eng = Engine(max_len=1024)
cfg, F = eng.cfg, eng.F
S = 6

# ---- deltanet_b -----------------------------------------------------------------------------------------------------------------
L = next(l for i, l in enumerate(F["layers"]) if cfg.is_linear(i))
B = 5
slots = np.array([3, 0, 5, 1, 2], dtype=np.int32)
pos = np.array([0, 1, 7, 100, 2047], dtype=np.int32)
proj = mx.array((rng.standard_normal((B, 8224)) * 2).astype(np.float32)).astype(mx.bfloat16)
cv = mx.array(rng.standard_normal((S + 1, 3, 6144)).astype(np.float32)).astype(mx.bfloat16)
ssm0 = mx.array((rng.standard_normal((S + 1, 16, 128, 128)) * 0.3).astype(np.float32))
cp = np.zeros(((S + 1) * 2, 3, 6144), np.float32)
cvn = np.array(cv.astype(mx.float32))
for b in range(B):
    cp[slots[b] * 2 + (pos[b] & 1)] = cvn[slots[b]]
conv_pool = mx.array(cp).astype(mx.bfloat16)
ssm_pool = mx.array(ssm0)
mx.eval(proj, cv, ssm0, conv_pool, ssm_pool)
y = Bt.deltanet_b(proj, conv_pool, L["conv_w"], L["A_log"], L["dt_bias"], ssm_pool, mx.array(slots), mx.array(pos))
mx.eval(y)
ok_y = ok_s = ok_c = True
for b in range(B):
    s_ = int(slots[b])
    ry, rc, rs = deltanet(proj[b:b + 1], cv[s_:s_ + 1], L["conv_w"], L["A_log"], L["dt_bias"], ssm0[s_:s_ + 1], eng._sel0)
    ok_y &= eq(y[b:b + 1], ry)
    ok_s &= eq(ssm_pool[s_:s_ + 1], rs)
    ok_c &= eq(conv_pool[s_ * 2 + ((int(pos[b]) & 1) ^ 1)][None], rc)
check("deltanet_b: y == solo deltanet (5 slots, ragged positions)", ok_y)
check("deltanet_b: new ssm state in place == solo", ok_s)
check("deltanet_b: new conv window written to the other buffer == solo", ok_c)
check("deltanet_b: slots without a row untouched", eq(ssm_pool[4], ssm0[4]) and eq(ssm_pool[6], ssm0[6]))
check("deltanet_b: the old conv buffer is preserved (double buffering)", all(
    eq(conv_pool[int(slots[b]) * 2 + (int(pos[b]) & 1)], mx.array(cp[int(slots[b]) * 2 + (int(pos[b]) & 1)]).astype(mx.bfloat16)) for b in range(B)))

# ---- attn_b ---------------------------------------------------------------------------------------------------------------------
Lmax, H, KV, D = 1024, 8, 2, 256
Kp = mx.array(rng.standard_normal((S + 1, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
Vp = mx.array(rng.standard_normal((S + 1, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
Kref, Vref = mx.array(Kp), mx.array(Vp)
B = 6
slots = np.array([3, 0, 5, 1, 2, 4], dtype=np.int32)
pos = np.array([0, 1, 31, 32, 300, 1023], dtype=np.int32)
q = mx.array(rng.standard_normal((B, H, D)).astype(np.float32)).astype(mx.bfloat16)
kn = mx.array(rng.standard_normal((B, KV, D)).astype(np.float32)).astype(mx.bfloat16)
vn = mx.array(rng.standard_normal((B, KV, D)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(Kp, Vp, Kref, Vref, q, kn, vn)
o = Bt.attn_b(q, kn, vn, Kp, Vp, mx.array(slots), mx.array(pos))
mx.eval(o)
ok_o = ok_a = ok_p = True
for b in range(B):
    s_, p = int(slots[b]), int(pos[b])
    K1 = mx.slice_update(Kref[s_:s_ + 1], kn[b:b + 1][:, :, None, :], mx.array([p], dtype=mx.int32), (2,))
    V1 = mx.slice_update(Vref[s_:s_ + 1], vn[b:b + 1][:, :, None, :], mx.array([p], dtype=mx.int32), (2,))
    Lb = 256 * (-(-(p + 1) // 256))
    r = mx.fast.scaled_dot_product_attention(q[b:b + 1][:, :, None, :], K1[:, :, :Lb], V1[:, :, :Lb], scale=D ** -0.5,
                                             mask=mx.arange(Lb)[None, :] <= p)[:, :, 0, :]
    ok_o &= eq(o[b:b + 1], r)
    ok_a &= eq(Kp[s_, :, p], kn[b]) and eq(Vp[s_, :, p], vn[b])
    ok_p &= eq(Kp[s_, :, :p], Kref[s_, :, :p]) and eq(Kp[s_, :, p + 1:], Kref[s_, :, p + 1:])
check("attn_b: o == MLX sdpa (bool mask) bit for bit, positions 0..1023, GQA 4", ok_o)
check("attn_b: new K / V row appended at pos", ok_a)
check("attn_b: every other row of the pool untouched", ok_p and eq(Kp[6], Kref[6]))

# ---- Pools.load / read ----------------------------------------------------------------------------------------------------------
ids0 = [int(t) for t in rng.integers(1000, 30000, 77)]
tok0, st0, n0 = eng.prefill(ids0)
mx.eval(tok0)
P = Bt.Pools(cfg, 3, 1024)
P.load(2, st0, n0)
back = P.read(2, n0, 1024)
okr = all(eq(a, b) for a, b in zip(back["ssm"], st0["ssm"])) and all(eq(a, b) for a, b in zip(back["conv"], st0["conv"]))
okr &= all(eq(a[0][:, :, :n0], b[0][:, :, :n0]) and eq(a[1][:, :, :n0], b[1][:, :, :n0]) for a, b in zip(back["kv"], st0["kv"]))
check("Pools.load -> read round trip (prefill state of 77 tokens)", okr)
check("Pools.load: other slots stay zero", all(float(mx.abs(x[0]).max()) == 0 and float(mx.abs(x[1]).max()) == 0 for x in P.ssm + P.K))


# ---- batched step vs solo generate -----------------------------------------------------------------------------------------------
def solo_gen(e, ids, n, chunk=2048):
    tok, st, pos_ = e.prefill(ids, chunk=chunk)
    out = [int(tok.item())]
    for _ in range(n - 1):
        nxt, st = e.step(tok, mx.array(pos_, dtype=mx.int32), st, pos_ + 1)
        pos_ += 1
        tok = nxt
        out.append(int(tok.item()))
    return out


lens = [17, 64, 130, 301, 520]
NG = 20
prompts = [[int(t) for t in rng.integers(1000, 30000, n)] for n in lens]
solo = [solo_gen(eng, p, NG) for p in prompts]
S2 = 8
P = Bt.Pools(cfg, S2, 1024)
cur = mx.zeros((S2 + 1,), mx.int32)
posn = {}
for i, p in enumerate(prompts):
    tok, st, n = eng.prefill(p)
    mx.eval(tok)
    P.load(i + 1, st, n)                         # slots 1..5: slot 0 stays free, ragged and not contiguous
    cur[i + 1] = tok.reshape(()).astype(mx.int32)
    posn[i + 1] = n
mx.eval(cur)
tree = P.as_tree()
fn = mx.compile(lambda F_, P_, c, sl, ps: Bt.step(cfg, F_, P_, c, sl, ps))
rows = [1, 2, 3, 4, 5]
res = []
for it in range(NG - 1):
    sl = np.array(rows + [S2], dtype=np.int32)                           # + one padding row on the trash slot
    ps = np.array([posn[r] for r in rows] + [0], dtype=np.int32)
    ids, cur = fn(F, tree, cur, mx.array(sl), mx.array(ps))
    mx.async_eval(ids, cur)
    res.append(ids)
    for r in rows:
        posn[r] += 1
got = [[solo[i][0]] + [int(r[i].item()) for r in res] for i in range(len(rows))]
check("batched step (compiled, pipelined, 5 ragged rows + pad row): tokens == Engine.generate", all(got[i] == solo[i] for i in range(5)))
check("batched step: the free slot 0 was never written", float(mx.abs(P.ssm[0][0]).max()) == 0 and float(mx.abs(P.K[0][0]).max()) == 0)


# ---- B >= 2: split norm kernels (B >= SPLIT_MIN), per-B GEMV tilings, tensor-op head K1 (M >= hp.NAX_MIN), M = 13 and M = 19 (two 16-row tiles) ----
def big_batch(nreq, NG2=8):
    lens2 = [int(v) for v in rng.integers(9, 260, nreq)]
    prompts2 = [[int(t) for t in rng.integers(1000, 30000, n)] for n in lens2]
    solo2 = [solo_gen(eng, p, NG2) for p in prompts2]
    S3 = nreq + 2
    P3 = Bt.Pools(cfg, S3, 512)
    cur3 = mx.zeros((S3 + 1,), mx.int32)
    pos3 = {}
    for i, p in enumerate(prompts2):
        tok, st, n = eng.prefill(p)
        mx.eval(tok)
        P3.load(i + 1, st, n)
        cur3[i + 1] = tok.reshape(()).astype(mx.int32)
        pos3[i + 1] = n
    mx.eval(cur3)
    tree3 = P3.as_tree()
    fn3 = mx.compile(lambda F_, P_, c, sl, ps: Bt.step(cfg, F_, P_, c, sl, ps))
    rows3 = list(range(1, nreq + 1))
    res3 = []
    for it in range(NG2 - 1):
        sl = np.array(rows3 + [S3], dtype=np.int32)
        ps = np.array([pos3[r] for r in rows3] + [0], dtype=np.int32)
        ids, cur3 = fn3(F, tree3, cur3, mx.array(sl), mx.array(ps))
        mx.async_eval(ids, cur3)
        res3.append(ids)
        for r in rows3:
            pos3[r] += 1
    got3 = [[solo2[i][0]] + [int(r[i].item()) for r in res3] for i in range(nreq)]
    return [i for i in range(nreq) if got3[i] != solo2[i]]


for nreq in (1, 2, 3, 4, 6, 12, 18):                 # M = nreq + 1 = 2, 3, 4, 5, 7 (the small-B tilings / split-norm switch), 13, 19
    bad_rows = big_batch(nreq)
    check(f"batched step (compiled, {nreq} ragged rows + pad row, M={nreq + 1}): tokens == Engine.generate (bad rows {bad_rows})", not bad_rows)


# ---- scheduler ------------------------------------------------------------------------------------------------------------------
def make_requests(n, seed, max_prompt=600, spread=0.5, eos_every=3):
    r_ = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        ln = int(r_.integers(8, max_prompt))
        m = int(r_.integers(4, 40))
        specs.append(([int(t) for t in r_.integers(1000, 30000, ln)], m))
    return specs, [float(x) for x in r_.uniform(0, spread, n)]


def expect(ref, r):
    for k, t in enumerate(ref):
        if t in r.eos:
            return ref[:k + 1]
    return ref[:r.max_new]


CH = 128
specs, arr = make_requests(11, 21)
refs = [solo_gen(eng, p, m, chunk=CH) for p, m in specs]
for policy in ("continuous", "static"):
    srv = Server(eng, slots=3, max_len=1024, chunk=CH, policy=policy)
    srv.warmup()
    reqs = []
    for i, (p, m) in enumerate(specs):
        eos = frozenset([refs[i][3]]) if i % 3 == 0 and m > 5 else frozenset()
        reqs.append(Request(rid=i, prompt=p, max_new=m, arrival=arr[i], eos=eos))
    stats = srv.run(reqs)
    bad = [r.rid for r in reqs if r.out != expect(refs[r.rid], r)]
    check(f"scheduler [{policy}]: 11 requests, 3 slots, chunked prefill, EOS: every request == solo (bad: {bad})", not bad)
    check(f"scheduler [{policy}]: all slots returned, every request finished",
          all(r.t_done is not None and r.t_first is not None for r in reqs) and stats["mean_batch"] > 1.0)
    if policy == "continuous":
        cont_wall = stats["wall"]
    else:
        print(f"     (wall: continuous {cont_wall:.2f} s, static {stats['wall']:.2f} s)")

# slot-count independence: one slot is plain sequential serving
srv = Server(eng, slots=1, max_len=1024, chunk=CH)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=0.0) for i, (p, m) in enumerate(specs[:4])]
srv.run(reqs)
check("scheduler: 1 slot (sequential), simultaneous arrivals: == solo", all(r.out == refs[r.rid][:r.max_new] for r in reqs))

# ---- prefix cache + scheduler -----------------------------------------------------------------------------------------------------
ceng = Engine(max_len=1024, prefix_cache_mb=512, like=eng)
oeng = Engine(max_len=1024, prefix_cache_mb=512, like=eng)               # the oracle runs the same prompts in the same order
sysp = [int(t) for t in rng.integers(1000, 30000, 300)]
cs = [(sysp + [int(t) for t in rng.integers(1000, 30000, int(rng.integers(5, 60)))], int(rng.integers(6, 25))) for _ in range(5)]
refs_c = [solo_gen(oeng, p, m, chunk=CH) for p, m in cs]
srv = Server(ceng, slots=3, max_len=1024, chunk=CH)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=0.35 * i) for i, (p, m) in enumerate(cs)]       # spaced: FIFO admission order
srv.run(reqs)
check("prefix cache + scheduler: 5 requests sharing a 300-token prefix: == solo with the same cache history",
      all(r.out == refs_c[r.rid] for r in reqs))
check(f"prefix cache + scheduler: later requests hit the cache ({[r.hit_tokens for r in reqs]})",
      reqs[0].hit_tokens == 0 and all(r.hit_tokens >= 256 for r in reqs[2:]))


# ---- request state machine, admission control, memory plan -------------------------------------------------------------------------
from qwen35 import serve as Sv

check("slot_bytes * (slots + 1) == Pools.nbytes()", Bt.Pools(cfg, 3, 256).nbytes() == 4 * Bt.Pools.slot_bytes(cfg, 256))
try:
    Sv.Request(0, [1], 1).to(Sv.DECODE, 0.0)
    illegal = False
except AssertionError:
    illegal = True
check("state machine: QUEUED/PREFILL/DECODE skipping is refused", illegal)

ML = 200
srv = Server(eng, slots=2, max_len=ML, chunk=64, max_queue=4)
srv.warmup()
mk = lambda n: [int(t) for t in rng.integers(1000, 30000, n)]
p0, p1, p4, p5 = mk(50), mk(90), mk(30), mk(30)
ref0, ref1, ref4, ref5 = (solo_gen(eng, p, m, chunk=64) for p, m in ((p0, 6), (p1, ML - 90 + 1), (p4, 3), (p5, 3)))
reqs = [Sv.Request(0, p0, 6, 0.0), Sv.Request(1, p1, 500, 0.0), Sv.Request(2, mk(ML + 1), 4, 0.0), Sv.Request(3, [], 4, 0.0),
        Sv.Request(4, p4, 3, 0.0, eos=frozenset([ref4[0]])), Sv.Request(5, p5, 3, 0.0)] + [Sv.Request(6 + i, mk(30), 3, 0.0) for i in range(4)]
srv.run(reqs)
byid = {r.rid: r for r in reqs}
check("states: too-long and empty prompts REJECTED, queue_full beyond max_queue=4 (requests 0,1,4,5 queued; 6-9 rejected)",
      [byid[i].state for i in (2, 3)] == [Sv.REJECTED] * 2 and [byid[i].finish_reason for i in (2, 3)] == ["prompt_too_long"] * 2
      and all(byid[i].state == Sv.REJECTED and byid[i].finish_reason == "queue_full" for i in range(6, 10))
      and all(byid[i].state == Sv.DONE for i in (0, 1, 4, 5)))
check("states: rejected requests never touched a slot (events == [REJECTED], no tokens)",
      all([e[0] for e in byid[i].events] == [Sv.REJECTED] and not byid[i].out for i in (2, 3, 6, 7, 8, 9)))
check("states: finish reasons (max_new / length clip / EOS at the first token)",
      [byid[i].finish_reason for i in (0, 1, 4, 5)] == ["max_new", "length", "eos", "max_new"] and len(byid[1].out) == ML - 90 + 1
      and byid[4].out == [ref4[0]])
check("states: event sequences legal, times monotone; EOS-at-first-token skips DECODE",
      [e[0] for e in byid[0].events] == [Sv.QUEUED, Sv.PREFILL, Sv.DECODE, Sv.DONE]
      and [e[0] for e in byid[4].events] == [Sv.QUEUED, Sv.PREFILL, Sv.DONE]
      and all(all(a[1] <= b[1] for a, b in zip(byid[i].events, byid[i].events[1:])) for i in (0, 1, 4, 5)))
check("admission is FIFO", [byid[i].t_admit for i in (0, 1, 4, 5)] == sorted(byid[i].t_admit for i in (0, 1, 4, 5)))
check("admitted requests' tokens == solo (incl. the clipped one)", byid[0].out == ref0 and byid[1].out == ref1 and byid[5].out == ref5)

# memory plan
pl = Sv.plan_memory(eng, 6 << 30, 1024, 512, 1 << 30, "auto", 16)
check(f"plan_memory: auto slots fit the budget ({pl['slots']} slots, planned {pl['total'] >> 20} of {pl['budget'] >> 20} MiB)",
      1 <= pl["slots"] <= 16 and pl["total"] <= pl["budget"] and (pl["slots"] == 16 or pl["total"] + pl["per_slot"] > pl["budget"]))
pl2 = Sv.plan_memory(eng, 12 << 30, 1024, 512, 1 << 30, "auto", 3)
check("plan_memory: max_slots caps", pl2["slots"] == 3)
try:
    Sv.plan_memory(eng, 1 << 30, 1024)
    err = False
except MemoryError:
    err = True
check("plan_memory: a budget below weights + one slot raises MemoryError", err)

print(f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
