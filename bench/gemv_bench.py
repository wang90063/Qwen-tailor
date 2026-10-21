"""M1 GEMV kernel sweep (rows per threadgroup R x simdgroups sg) for lm_head / gate_up / down vs mx matmul: GB/s and max error."""
import sys, time, statistics
import mlx.core as mx
from qwen35.kernels import gemv
def tm(f, n=40):
    for _ in range(5): mx.eval(f())
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    return statistics.median(ts), min(ts)
for (N,K,label) in [(248320,1024,"lm_head"),(3584*2,1024,"gate_up"),(1024,3584,"down")]:
    W=(mx.random.normal((N,K))*0.05).astype(mx.bfloat16); x=mx.random.normal((1,1,K)).astype(mx.bfloat16); mx.eval(W,x)
    ref=(x@W.T)[0,0].astype(mx.float32)
    nb=W.nbytes
    m,b=tm(lambda:x@W.T); print(f"{label:8s} mlx      {nb/m/1e9:6.1f} / {nb/b/1e9:6.1f} GB/s")
    for R in (1,2,4,8):
        for sg in (2,4,8):
            try:
                y=gemv(W,x,R=R,sg=sg)
                err=(y[0,0].astype(mx.float32)-ref).abs().max().item()
                m,b=tm(lambda:gemv(W,x,R=R,sg=sg)); print(f"{label:8s} R={R} sg={sg} {nb/m/1e9:6.1f} / {nb/b/1e9:6.1f} GB/s  maxerr {err:.4f}")
            except Exception as e: print(label,R,sg,"ERR",str(e)[:200]); 
