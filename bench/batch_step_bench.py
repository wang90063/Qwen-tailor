"""Batched decode step time and aggregate tok/s vs batch size B (quiet-probe sampling).
usage: batch_step_bench.py [max_len=4096] [pos=600] [B,B,...]"""
import sys, time, statistics
import numpy as np, mlx.core as mx
from quiet import Probe
from qwen35.engine import Engine
from qwen35 import batch as Bt
LM = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
POS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
eng = Engine(max_len=LM)
cfg, F = eng.cfg, eng.F
S = 16
P = Bt.Pools(cfg, S, LM); tree = P.as_tree()
rng = np.random.default_rng(0)
cur = mx.array(rng.integers(1000, 30000, S + 1).astype(np.int32))
mx.eval(cur)
fn = mx.compile(lambda F_, P_, cur_, sl, ps: Bt.step(cfg, F_, P_, cur_, sl, ps))
probe = Probe(tol=1.15)
def run(B, n=20):
    sl = mx.array(np.arange(B, dtype=np.int32)); ps = mx.array(rng.integers(POS // 2, POS, B).astype(np.int32))
    c = cur
    t = time.perf_counter()
    for _ in range(n):
        ids, c = fn(F, tree, c, sl, ps)
        mx.async_eval(ids, c)
    mx.eval(ids)
    return (time.perf_counter() - t) / n
print(f"Lmax {LM}  pos ~{POS//2}-{POS}")
for B in [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else (1, 2, 3, 4, 6, 8, 10, 12, 14, 16):
    run(B, 3); run(B, 3)
    ts = []
    for _ in range(14):
        dt, ok = probe.sample(lambda: run(B))
        if ok: ts.append(dt)
    if not ts: ts = [run(B)]
    m = statistics.median(ts) * 1e3
    print(f"B={B:2d}  step {m:6.2f} ms  per-seq {m / B:5.2f} ms  agg {B / m * 1e3:7.1f} tok/s   ({len(ts)} quiet)", flush=True)
