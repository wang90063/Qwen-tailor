"""Round time of the batched speculative decoder vs the plain batched step, B = 1..N (quiet-probe sampling, interleaved).

usage: python bench/batch_spec_bench.py [LMAX] [POS] [B,B,...] [k]
Throughput = B * tokens_per_round / round_time; tokens_per_round depends on the text (measured 2.4-2.7 with k=2 on the golden prompts),
so the table prints it for 2.0 / 2.5 next to the plain step's B / step_time.
"""
import sys, time, statistics
import numpy as np, mlx.core as mx
from quiet import Probe
from qwen35.spec import SpecEngine
from qwen35 import batch as Bt, batch_spec as BS

LM = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
POS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
Bs = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else (1, 2, 3, 4, 6, 8)
k = int(sys.argv[4]) if len(sys.argv) > 4 else 2
M = k + 1
sp = SpecEngine(k=k, max_len=LM)
cfg, F, MT = sp.cfg, sp.F, sp.MT
NSPEC = max(Bs)
S = NSPEC
Pm = Bt.Pools(cfg, S, LM)
SPl = BS.SpecPools(cfg, NSPEC, S, M, LM)
P, SPt = Pm.as_tree(), SPl.as_tree()
rng = np.random.default_rng(0)
cur = mx.array(rng.integers(1000, 30000, S + 1).astype(np.int32))
C = {n: mx.array(v) for n, v in SPl.carry.items()}
C["H"] = (mx.random.normal(C["H"].shape) * 0.5).astype(mx.bfloat16)
C["toks"] = mx.array(rng.integers(1000, 30000, C["toks"].shape).astype(np.int32))
C["ids"] = mx.array(rng.integers(1000, 30000, C["ids"].shape).astype(np.int32))
mx.eval(cur, C)
fn_p = mx.compile(lambda F_, P_, cur_, sl, ps: Bt.step(cfg, F_, P_, cur_, sl, ps))
fn_s = mx.compile(lambda F_, MT_, Ws, im, nv, P_, SP_, C_, sl, ssl, par: BS.spec_round(cfg, k, F_, MT_, Ws, im, nv, P_, SP_, C_, sl, ssl, par))
probe = Probe(tol=1.15)


def run_plain(B, n=20):
    sl = mx.array(np.arange(B, dtype=np.int32)); ps = mx.array(rng.integers(POS // 2, POS, B).astype(np.int32))
    c = cur
    t = time.perf_counter()
    for _ in range(n):
        ids, c = fn_p(F, P, c, sl, ps)
        mx.async_eval(ids, c)
    mx.eval(ids)
    return (time.perf_counter() - t) / n


def run_spec(B, n=20):
    sl = mx.array(np.arange(B, dtype=np.int32)); ssl = mx.array(np.arange(B, dtype=np.int32))
    cc = dict(C)
    cc["pos"] = mx.array(rng.integers(POS // 2, POS, C["pos"].shape).astype(np.int32))
    cc["sel"] = mx.zeros_like(C["sel"])
    mx.eval(cc)
    t = time.perf_counter()
    for r in range(n):
        ids, nn, cc = fn_s(F, MT, sp.Wsub, sp.idmap, sp.nvalid, P, SPt, cc, sl, ssl, mx.full((B,), r & 1, dtype=mx.int32))
        mx.async_eval(ids, nn)
    mx.eval(ids)
    return (time.perf_counter() - t) / n


def med(f, B):
    ts = []
    for _ in range(12):
        dt, ok = probe.sample(lambda: f(B))
        if ok:
            ts.append(dt)
    return statistics.median(ts or [f(B)]) * 1e3, len(ts)


print(f"Lmax {LM}  pos ~{POS // 2}-{POS}  k={k} (M={M})")
for B in Bs:
    for f in (run_plain, run_spec):
        f(B, 3); f(B, 3)
    tp, np_ = med(run_plain, B)
    ts, ns = med(run_spec, B)
    plain = B / tp * 1e3
    print(f"B={B:2d}  plain step {tp:6.2f} ms ({plain:6.1f} tok/s)   spec round {ts:6.2f} ms   "
          f"agg @2.0 tok/round {B * 2.0 / ts * 1e3:6.1f} ({B * 2.0 / ts * 1e3 / plain:4.2f}x)   @2.5 {B * 2.5 / ts * 1e3:6.1f} ({B * 2.5 / ts * 1e3 / plain:4.2f}x)"
          f"   ({np_}/{ns} quiet)", flush=True)
