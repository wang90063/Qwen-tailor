"""Fused GEMV family (norm / swiglu / residual epilogues) and head_argmax bandwidth at M = 1, 2, 4 rows (the speculative-verify shapes)."""
import sys, time
import mlx.core as mx
from qwen35.kernels import fused_gemv, head_argmax
bf = mx.bfloat16
def tm(f, n=20):
    mx.eval(f()); t = time.perf_counter(); outs = [f() for _ in range(n)]; mx.eval(outs); return (time.perf_counter() - t) / n
nw = (mx.random.normal((1024,)) * .1 + 1).astype(bf)
shapes = {"in_proj 8224x1024": (8224, 1024, dict(nw=nw)), "gate_up 7168x1024 swiglu": (7168, 1024, dict(nw=nw, swiglu=True)),
          "down 1024x3584 +res": (1024, 3584, dict(res=True)), "out_proj 1024x2048": (1024, 2048, dict(res=True))}
for name, (N, K, kw) in shapes.items():
    Ws = [(mx.random.normal((N, K)) * .02).astype(bf) for _ in range(12)]; mx.eval(Ws)
    for M in (1, 2, 4):
        x = (mx.random.normal((M, K)) * .5).astype(bf)
        kws = dict(kw)
        if kws.get("res"): kws["res"] = (mx.random.normal((M, N))).astype(bf)
        t = tm(lambda: [fused_gemv(w, x, **kws) for w in Ws])
        print(f"{name:28s} M={M}  {12*N*K*2/t/1e9:6.1f} GB/s  {t/12*1e6:6.1f} us")
E = (mx.random.normal((248320, 1024)) * .02).astype(bf)
for M in (1, 2, 4):
    x = (mx.random.normal((M, 1024))).astype(bf)
    t = tm(lambda: head_argmax(E, x)); print(f"head M={M} {E.nbytes/t/1e9:6.1f} GB/s {t*1e3:.2f} ms")
