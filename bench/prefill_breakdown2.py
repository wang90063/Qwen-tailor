"""Second prefill breakdown at S = 2048: linear-attention layer (in_proj, conv prep, gdn, gnorm, out_proj) and MLP pieces, using qwen35.prefill helpers."""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf
cfg = m.Cfg(); W = m.load_weights(cfg); S = 2048
def T(name, f, n=7):
    mx.eval(f()); ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    ts.sort(); print(f"{name:34s} {ts[len(ts)//2]*1e3:8.2f} ms"); return ts[len(ts)//2]
x = (mx.random.normal((1,S,1024))*0.5).astype(mx.bfloat16)
L = W["layers"][0]; A = W["layers"][3]
cs=mx.zeros((1,3,6144),mx.bfloat16); ss=mx.zeros((1,16,128,128))
proj=(x@L["in_proj"].T)[0]; mx.eval(proj)
T("in_proj", lambda: x @ L["in_proj"].T)
T("prep", lambda: pf.conv_prep(proj, cs[0], L["conv_w"]))
q,k,v=pf.conv_prep(proj, cs[0], L["conv_w"]); mx.eval(q,k,v)
from mlx_lm.models.gated_delta import gated_delta_update
T("gdn", lambda: gated_delta_update(q[None],k[None],v[None],proj[:,8208:][None],proj[:,8192:8208][None],L["A_log"],L["dt_bias"],ss)[0])
y=mx.random.normal((1,S,16,128)).astype(mx.bfloat16)
T("gnorm", lambda: pf.gated_norm(y, proj, L["gnorm"]))
yy=mx.random.normal((1,S,2048)).astype(mx.bfloat16)
T("out_proj", lambda: yy @ L["out_proj"].T)
T("linear layer total", lambda: pf._linear_attn(cfg, L, x, cs, ss)[0])
T("mlp", lambda: m._mlp(L, x))
T("gate_up", lambda: x @ L["gate_up"].T)
hh=mx.random.normal((1,S,3584)).astype(mx.bfloat16); mx.eval(hh)
T("down", lambda: hh @ L["down"].T)
g,u=mx.split(x@L["gate_up"].T,2,axis=-1); mx.eval(g,u)
import mlx.nn as nn
T("swiglu elementwise", lambda: nn.silu(g)*u)
T("rmsnorm", lambda: mx.fast.rms_norm(x, L["ln1"], 1e-6))
T("residual add", lambda: x + x)
kv=(mx.zeros((1,2,S,256),mx.bfloat16),)*2
T("full_attn layer", lambda: m._full_attn(cfg, A, x, kv, 0, S, "causal")[0])
