"""Continuous-batching server with the speculative mode (serve.Server(spec=...)): every request's tokens == the solo plain greedy decode,
through arrivals / EOS / max_new inside a round / plain <-> speculative switches in both directions / prefix-cache hits.

Prompts are slices of the installed mlx_lm sources (real code: the MTP drafts get accepted); the oracle is Engine.prefill(chunk) + the
solo compiled decode step, i.e. what tests/batch.py compares the plain server to.
"""
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx_lm
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR

from qwen35 import serve as Sv, batch as Bt
from qwen35.engine import Engine
from qwen35.serve import Server, Request
from qwen35.spec import SpecEngine

fails = []


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


ML = 1024
sp = SpecEngine(k=2, max_len=ML)
eng = Engine(max_len=ML, like=sp)
cfg = eng.cfg
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
text = "\n".join(p.read_text() for p in sorted(Path(mlx_lm.__file__).parent.glob("models/*.py"))[:12])
IDS = tk(text)["input_ids"]
rng = np.random.default_rng(5)


def snippet(n):
    a = int(rng.integers(0, len(IDS) - n))
    return IDS[a:a + n]


def solo_gen(e, ids, n, chunk=2048):
    tok, st, pos_ = e.prefill(ids, chunk=chunk)
    out = [int(tok.item())]
    for _ in range(n - 1):
        nxt, st = e.step(tok, mx.array(pos_, dtype=mx.int32), st, pos_ + 1)
        pos_ += 1
        tok = nxt
        out.append(int(tok.item()))
    return out


def expect(ref, r):
    for k, t in enumerate(ref):
        if t in r.eos:
            return ref[:k + 1]
    return ref[:r.max_new]


CH = 128
# ---- ragged arrivals, 4 slots, speculation only up to 2 sequences: the load crosses the threshold both ways -----------------------------
specs = [(snippet(int(n)), int(m)) for n, m in zip([40, 300, 90, 18, 220, 60, 150], [50, 30, 60, 24, 40, 36, 45])]
arr = [0.0, 0.05, 0.1, 0.15, 0.6, 1.4, 1.5]
refs = [solo_gen(eng, p, m, chunk=CH) for p, m in specs]
srv = Server(eng, slots=4, max_len=ML, chunk=CH, spec=sp, spec_max=2, spec_dwell=2)
srv.warmup()
reqs = []
for i, (p, m) in enumerate(specs):
    eos = frozenset([refs[i][7]]) if i in (1, 4) else frozenset()
    reqs.append(Request(rid=i, prompt=p, max_new=m, arrival=arr[i], eos=eos))
st = srv.run(reqs)
bad = [r.rid for r in reqs if r.out != expect(refs[r.rid], r)]
check(f"spec server: 7 ragged requests, 4 slots, spec_max 2, EOS in rounds: every request == solo (bad: {bad})", not bad)
check(f"spec server: both modes ran and the load crossed the threshold (rounds {st['rounds']}, plain steps {st['steps']}, switches {st['mode_switches']})",
      st["rounds"] > 0 and st["steps"] > 0 and st["mode_switches"] >= 2)
check(f"spec server: speculation pays on code ({st['tokens_per_round']:.2f} tokens / sequence / round)", st["tokens_per_round"] > 1.3)
check("spec server: all spec entries returned", all(r.state == Sv.DONE for r in reqs))

# ---- low load only: a single long request stays speculative the whole time ------------------------------------------------------------
p, m = snippet(120), 90
ref = solo_gen(eng, p, m, chunk=CH)
srv = Server(eng, slots=4, max_len=ML, chunk=CH, spec=sp)
srv.warmup()
r1 = Request(rid=0, prompt=p, max_new=m)
st = srv.run([r1])
check(f"spec server: one request, 90 tokens == solo ({st['rounds']} rounds, {st['tokens_per_round']:.2f} tokens/round, no plain steps {st['steps']})",
      r1.out == ref and st["steps"] == 0 and st["rounds"] < 90)

# ---- max_new cut inside a round, max_new = 1 / 2, the context limit --------------------------------------------------------------------
cases = [(snippet(30), 1), (snippet(30), 2), (snippet(30), 3), (snippet(30), 7), (snippet(30), 8)]
refs2 = [solo_gen(eng, p, m, chunk=CH) for p, m in cases]
srv = Server(eng, slots=3, max_len=ML, chunk=CH, spec=sp)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=0.05 * i) for i, (p, m) in enumerate(cases)]
srv.run(reqs)
check("spec server: max_new 1,2,3,7,8 (cut inside a round): == solo, exact lengths",
      all(r.out == refs2[r.rid] and len(r.out) == r.max_new for r in reqs))

ML2 = 160
pl = snippet(100)
ref_l = solo_gen(eng, pl, ML2 - 100 + 1, chunk=CH)
srv = Server(eng, slots=2, max_len=ML2, chunk=CH, spec=sp)
srv.warmup()
rl = Request(rid=0, prompt=pl, max_new=500)
srv.run([rl])
check(f"spec server: context limit (max_len {ML2}, rounds in flight past the end): == solo, {len(rl.out)} tokens, reason {rl.finish_reason}",
      rl.out == ref_l and rl.finish_reason == "length")

# ---- heavy churn: 10 requests, 6 slots, spec_max 3, dwell 1 ---------------------------------------------------------------------------------
specs3 = [(snippet(int(n)), int(m)) for n, m in zip(rng.integers(10, 260, 10), rng.integers(8, 60, 10))]
arr3 = sorted(float(x) for x in rng.uniform(0, 1.2, 10))
refs3 = [solo_gen(eng, p, m, chunk=CH) for p, m in specs3]
srv = Server(eng, slots=6, max_len=ML, chunk=CH, spec=sp, spec_max=3, spec_dwell=1)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=arr3[i]) for i, (p, m) in enumerate(specs3)]
st = srv.run(reqs)
bad = [r.rid for r in reqs if r.out != refs3[r.rid]]
check(f"spec server: 10 requests, 6 slots, spec_max 3, dwell 1: == solo (bad: {bad}; switches {st['mode_switches']}, mean spec batch {st['mean_spec_batch']:.2f})",
      not bad)

# ---- thrash: speculation only for a single sequence, dwell 1: every arrival / completion flips the mode ----------------------------------
specs4 = [(snippet(int(n)), int(m)) for n, m in zip(rng.integers(10, 120, 12), rng.integers(10, 40, 12))]
arr4 = [1.4 * (i // 4) + 0.03 * (i % 4) for i in range(12)]              # three waves of four
refs4 = [solo_gen(eng, p, m, chunk=CH) for p, m in specs4]
srv = Server(eng, slots=5, max_len=ML, chunk=CH, spec=sp, spec_max=1, spec_dwell=1)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=arr4[i]) for i, (p, m) in enumerate(specs4)]
st = srv.run(reqs)
bad = [r.rid for r in reqs if r.out != refs4[r.rid]]
check(f"spec server: thrash (3 waves x 4 requests, spec_max 1, dwell 1): == solo (bad: {bad}; {st['mode_switches']} mode switches, {st['rounds']} rounds, {st['steps']} plain steps)",
      not bad and st["mode_switches"] >= 5)

# ---- adaptive mode: noise prompts (drafts rejected) at a load where speculation does not pay -> the server stays / goes plain ---------------
noise = [rng.integers(1000, 200000, int(n)).tolist() for n in (60, 90, 40, 75)]
refs5 = [solo_gen(eng, p, 120, chunk=CH) for p in noise]
srv = Server(eng, slots=4, max_len=ML, chunk=CH, spec=sp, spec_max=4, spec_dwell=2)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=120, arrival=0.02 * i) for i, p in enumerate(noise)]
st = srv.run(reqs)
check(f"adaptive: 4 noise prompts == solo (acceptance {st['tokens_per_round']:.2f} tokens/round; {st['rounds']} rounds vs {st['steps']} plain steps; estimate {st['tpr']:.2f})",
      all(r.out == refs5[r.rid] for r in reqs))
check("adaptive: low acceptance at B=4 -> the server falls back to plain mode and the estimate follows", st["steps"] > 0 and st["mode_switches"] >= 1 and st["tpr"] < 1.6)
srv.warmup()
reqs = [Request(rid=i, prompt=snippet(80), max_new=330, arrival=0.0) for i in range(2)]
srv.tpr = 1.0                                                                       # same server, estimate forced low, code at B=2
st = srv.run(reqs)
check(f"adaptive: a low estimate recovers by itself on code ({st['rounds']} rounds, {st['steps']} plain steps, estimate {st['tpr']:.2f})",
      st["rounds"] > 0 and st["tpr"] > 1.5)

# ---- static policy ------------------------------------------------------------------------------------------------------------------------
srv = Server(eng, slots=3, max_len=ML, chunk=CH, spec=sp, spec_max=3, policy="static")
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=arr3[i] * 0.3) for i, (p, m) in enumerate(specs3[:6])]
srv.run(reqs)
check("spec server: static policy == solo", all(r.out == refs3[r.rid] for r in reqs))

# ---- prefix cache + speculation ---------------------------------------------------------------------------------------------------------
ceng = Engine(max_len=ML, prefix_cache_mb=512, like=sp)
oeng = Engine(max_len=ML, prefix_cache_mb=512, like=sp)
sysp = snippet(300)
cs = [(sysp + snippet(int(rng.integers(5, 60))), int(rng.integers(10, 30))) for _ in range(5)]
refs_c = [solo_gen(oeng, p, m, chunk=CH) for p, m in cs]
srv = Server(ceng, slots=3, max_len=ML, chunk=CH, spec=sp)
srv.warmup()
reqs = [Request(rid=i, prompt=p, max_new=m, arrival=0.5 * i) for i, (p, m) in enumerate(cs)]
st = srv.run(reqs)
check(f"spec server + prefix cache: 5 requests sharing a 300-token prefix == solo (hits {[r.hit_tokens for r in reqs]})",
      all(r.out == refs_c[r.rid] for r in reqs) and all(r.hit_tokens >= 256 for r in reqs[2:]))

# ---- memory plan ----------------------------------------------------------------------------------------------------------------------------
p0 = Sv.plan_memory(eng, 6 << 30, 1024, 512, 1 << 30, "auto", 16)
p1 = Sv.plan_memory(eng, 6 << 30, 1024, 512, 1 << 30, "auto", 16, spec=(4, 3))
check(f"plan_memory: speculation costs memory ({p0['slots']} -> {p1['slots']} slots at 6 GiB; spec entries {p1['spec_entries'] >> 20} MiB)",
      p1["per_slot"] > p0["per_slot"] and p1["spec_entries"] > 0 and p1["slots"] <= p0["slots"] and p1["total"] <= p1["budget"])

print(f"FAILURES: {len(fails)}")
sys.exit(1 if fails else 0)
