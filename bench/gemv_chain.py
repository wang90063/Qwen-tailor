"""Same sweep as gemv_bench.py over a chain of 24 distinct weights (cold caches, as in the real model)."""
import sys, time, statistics
import mlx.core as mx
from qwen35.kernels import gemv
def tm(f, n=30):
    for _ in range(4): mx.eval(f())
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    return statistics.median(ts), min(ts)
for (N,K,label) in [(3584*2,1024,"gate_up"),(1024,3584,"down"),(8208,1024,"dn_in_proj")]:
    Ws=[(mx.random.normal((N,K))*0.05).astype(mx.bfloat16) for _ in range(24)]; x=mx.random.normal((1,1,K)).astype(mx.bfloat16); mx.eval(Ws,x)
    nb=sum(w.nbytes for w in Ws)
    m,b=tm(lambda:[x@w.T for w in Ws]); print(f"{label:10s} mlx     {nb/m/1e9:6.1f} / {nb/b/1e9:6.1f} GB/s  ({m*1e3/24:.3f} ms/op)")
    for R in (1,2,4,8):
        for sg in (2,4,8):
            m,b=tm(lambda:[gemv(w,x,R=R,sg=sg) for w in Ws]); print(f"{label:10s} R={R} sg={sg} {nb/m/1e9:6.1f} / {nb/b/1e9:6.1f} GB/s  ({m*1e3/24:.3f} ms/op)")
