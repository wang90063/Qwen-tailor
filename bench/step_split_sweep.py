"""Batched decode step: fused vs split-norm / fp32-x / R=2 / wf GEMV variants per batch size (quiet-probe sampling).
usage: step_split_sweep.py B,B,..."""
import sys, time, statistics
import numpy as np, mlx.core as mx
from quiet import Probe
from qwen35.engine import Engine
from qwen35 import batch as Bt
Bs = [int(a) for a in sys.argv[1].split(",")]
LM, POS = 4096, 600
eng = Engine(max_len=LM); cfg, F = eng.cfg, eng.F
P = Bt.Pools(cfg, 16, LM); tree = P.as_tree()
rng = np.random.default_rng(0)
cur = mx.array(rng.integers(1000, 30000, 17).astype(np.int32)); mx.eval(cur)
probe = Probe(tol=1.15)
# (split, xf32, tiling)
cands = [("fused", 99, 99, {}), ("split", 0, 99, {}), ("split+xf", 0, 0, {}), ("split+xf+R2", 0, 0, dict(R=2)), ("split+R2", 0, 99, dict(R=2)), ("split+xf+R2+wf", 0, 0, dict(R=2, wf=True))]
for B in Bs:
    sl = mx.array(np.arange(B, dtype=np.int32)); ps = mx.array(rng.integers(POS // 2, POS, B).astype(np.int32))
    fns = []
    for name, sm, fm, tl in cands:
        Bt.SPLIT_MIN = sm if sm < 99 else 99; Bt.F32X_MIN = fm; Bt.gemv_tiling = lambda b, tl=tl: dict(tl)
        if name == "fused": Bt.SPLIT_MIN = 99
        fn = mx.compile(lambda F_, P_, cc, s, p: Bt.step(cfg, F_, P_, cc, s, p))
        ids, _ = fn(F, tree, cur, sl, ps); mx.eval(ids); fns.append((name, fn))
    def run(fn, n=20):
        cc = cur; t = time.perf_counter()
        for _ in range(n):
            ids, cc = fn(F, tree, cc, sl, ps); mx.async_eval(ids, cc)
        mx.eval(ids); return (time.perf_counter() - t) / n
    for _, fn in fns: run(fn, 3); run(fn, 3)
    res = {i: [] for i in range(len(fns))}; loose = {i: [] for i in range(len(fns))}
    for rnd in range(8):
        for i, (c, fn) in enumerate(fns):
            dt, ok = probe.sample(lambda: run(fn)); loose[i].append(dt)
            if ok: res[i].append(dt)
    print(f"B={B}", flush=True)
    for i, (c, _) in enumerate(fns):
        v = res[i] if len(res[i]) >= 3 else [min(loose[i])]
        print(f"   {c:20s} {statistics.median(v)*1e3:6.2f} ms ({len(res[i])})", flush=True)
