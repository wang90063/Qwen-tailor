"""Standalone cost of the residual epilogue: EPI_PLAIN vs EPI_ADD GEMM, add_norm vs norm kernels (quiet-GPU sampling).
usage: epi_add.py [M] [reps]"""
import sys, time, statistics
import mlx.core as mx
from quiet import Probe
from qwen35 import gemm_nax as G, prefill as pf
M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N_IN = 12                                       # distinct weights/inputs so the L2/SLC does not make the chain unrealistically warm
cases = {}
for name, K, N in (("down", 3584, 1024), ("out_proj", 2048, 1024)):
    ws = [(mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16) for _ in range(N_IN)]
    x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
    r = (mx.random.normal((M, N)) * 0.5).astype(mx.bfloat16)
    mx.eval(ws, x, r)
    cases[f"{name} plain"] = (lambda ws=ws, x=x, name=name: [G.linear(x, w, name) for w in ws])
    cases[f"{name} +add "] = (lambda ws=ws, x=x, r=r, name=name: [G.linear(x, w, name, epi=G.EPI_ADD, r=r) for w in ws])
xs = [(mx.random.normal((M, 1024)) * 0.5).astype(mx.bfloat16) for _ in range(N_IN)]
rs = [(mx.random.normal((M, 1024)) * 0.5).astype(mx.bfloat16) for _ in range(N_IN)]
w = mx.random.normal((1024,)).astype(mx.bfloat16)
mx.eval(xs, rs, w)
cases["add_norm"] = lambda: [pf.add_norm(a, b, w) for a, b in zip(xs, rs)]
cases["norm    "] = lambda: [pf.norm(a, w) for a in xs]
probe = Probe()


def timed(f):
    t = time.perf_counter(); mx.eval(f()); return time.perf_counter() - t


for f in cases.values(): mx.eval(f())
res = {k: [] for k in cases}
for _ in range(REPS):
    for k, f in cases.items():
        dt, ok = probe.sample(lambda: timed(f))
        if ok: res[k].append(dt / N_IN)
for k, v in res.items():
    if v: print(f"M={M} {k:14s}: median {statistics.median(v) * 1e3:.4f} ms  best {min(v) * 1e3:.4f} ms (n={len(v)})")
