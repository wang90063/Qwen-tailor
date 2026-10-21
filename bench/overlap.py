"""Do the gated-delta recurrence and a GEMM overlap when issued on two GPU streams?  (No gain, see docs/PLAN.md M3.)"""
import sys, time
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update
S=2048
q=mx.random.normal((1,S,16,128)).astype(mx.bfloat16); k=q; v=q
a=mx.random.normal((1,S,16)).astype(mx.bfloat16); b=a
A_log=mx.zeros((16,)); dtb=mx.zeros((16,),mx.bfloat16)
s0=mx.zeros((1,16,128,128),mx.float32)
x=(mx.random.normal((S,1024))*.1).astype(mx.bfloat16); w=(mx.random.normal((7168,1024))*.1).astype(mx.bfloat16)
mx.eval(q,a,x,w,s0)
s1=mx.new_stream(mx.gpu); s2=mx.new_stream(mx.gpu)
def T(name,f,n=9):
    mx.eval(f()); ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    ts.sort(); print(f"{name:30s} {ts[len(ts)//2]*1e3:8.2f} ms")
NG=6
def gdn_chain(stream, n):
    out=[]
    with mx.stream(stream):
        st=s0
        for i in range(n):
            y,st=gated_delta_update(q,k,v,a,b,A_log,dtb,st); out.append(y)
    return out
def gemm_chain(stream, n):
    out=[]
    with mx.stream(stream):
        for i in range(n): out.append(x@w.T)
    return out
T(f"{NG}x gdn alone", lambda: gdn_chain(s1,NG))
T(f"{NG*2}x gemm alone", lambda: gemm_chain(s2,NG*2))
T("both, separate streams", lambda: gdn_chain(s1,NG)+gemm_chain(s2,NG*2))
T("both, same stream", lambda: gdn_chain(mx.gpu,NG)+gemm_chain(mx.gpu,NG*2))
