"""M3-era prefill breakdown at S = 2048 with stock mlx-lm ops: per-GEMM, gated-delta, conv, SDPA, norms, rope, per layer type."""
import sys, time
import mlx.core as mx, mlx.nn as nn
from qwen35 import model as m
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
cfg = m.Cfg(); W = m.load_weights(cfg)
S = 2048
def T(name, f, n=5):
    mx.eval(f()); ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    ts.sort(); print(f"{name:34s} {ts[len(ts)//2]*1e3:8.2f} ms"); return ts[len(ts)//2]
x = (mx.random.normal((1,S,1024))*0.5).astype(mx.bfloat16)
L = W["layers"][0]; A = W["layers"][3]
T("in_proj gemm (x18)", lambda: x @ L["in_proj"].T)
T("out_proj gemm", lambda: mx.random.normal((1,S,2048)).astype(mx.bfloat16) @ L["out_proj"].T)
T("mlp gate_up", lambda: x @ L["gate_up"].T)
T("mlp down", lambda: mx.random.normal((1,S,3584)).astype(mx.bfloat16) @ L["down"].T)
T("lm_head last row only", lambda: x[:, -1] @ W["embed"].T)
# delta attention pieces
q=mx.random.normal((1,S,16,128)).astype(mx.bfloat16); k=q; v=q
a=mx.random.normal((1,S,16)).astype(mx.bfloat16); b=a
s0=mx.zeros((1,16,128,128),mx.float32)
T("gated_delta_update (x18)", lambda: gated_delta_update(q,k,v,a,b,L["A_log"],L["dt_bias"],s0))
cin = mx.random.normal((1,S+3,6144)).astype(mx.bfloat16)
T("conv1d+silu", lambda: nn.silu(mx.conv1d(cin, L["conv_w"], groups=6144)))
T("linear_attn full layer", lambda: m._linear_attn(cfg, L, x, mx.zeros((1,3,6144),mx.bfloat16), s0)[0])
kv=(mx.zeros((1,2,S,256),mx.bfloat16),)*2
mask=(mx.arange(S)[None,:]<=mx.arange(S)[:,None])
T("full_attn layer (x6)", lambda: m._full_attn(cfg, A, x, kv, 0, S, mask)[0])
T("mlp layer (x24)", lambda: m._mlp(L, x))
print("---- details")
H,KV,D=8,2,256
qq=mx.random.normal((1,H,S,D)).astype(mx.bfloat16); kk=mx.random.normal((1,KV,S,D)).astype(mx.bfloat16)
T("sdpa bool mask", lambda: mx.fast.scaled_dot_product_attention(qq,kk,kk,scale=D**-0.5,mask=mask))
T("sdpa causal str", lambda: mx.fast.scaled_dot_product_attention(qq,kk,kk,scale=D**-0.5,mask="causal"))
qkv_ = mx.random.normal((1,S,6144)).astype(mx.bfloat16)
T("qk normalize", lambda: normalize_qk(q,k,inv_scale=128**-0.5,eps=1e-6))
o=mx.random.normal((1,S,16,128)).astype(mx.bfloat16); z=o
def gn():
    n = mx.fast.rms_norm(o, L["gnorm"], cfg.eps)
    return (nn.silu(z.astype(mx.float32)) * n.astype(mx.float32)).astype(mx.bfloat16)
T("gated norm", gn)
T("qkv gemm full-attn", lambda: x @ A["qkv"].T)
T("o_proj gemm full-attn", lambda: qq.reshape(1,S,2048) @ A["o_proj"].T)
rp = mx.random.normal((1,H,S,D)).astype(mx.bfloat16)
T("rope q", lambda: mx.fast.rope(rp, 64, traditional=False, base=1e7, scale=1.0, offset=0))
T("rmsnorm x", lambda: mx.fast.rms_norm(x, W["norm"], 1e-6))
