"""Prefill full-attention layer pieces at S = 512 / 2048: qkv GEMM, fused causal SDPA vs unfused matmul + softmax, o_proj, q-norm, rope, the whole `_full_attn`."""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf
cfg = m.Cfg(); W = m.load_weights(cfg)
A = W["layers"][3]
def T(name, f, n=9):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); print(f"{name:40s} {ts[len(ts)//2]*1e3:8.2f} ms"); return ts[len(ts)//2]
for S in (512, 2048):
    print(f"--- S={S}")
    x = (mx.random.normal((1, S, 1024)) * 0.5).astype(mx.bfloat16)
    H, KV, D = 8, 2, 256
    q = mx.random.normal((1, H, S, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, KV, S, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, KV, S, D)).astype(mx.bfloat16)
    mx.eval(q, k, v, x)
    T("qkv gemm", lambda: x @ A["qkv"].T)
    T("sdpa causal (fused call)", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal"))
    def unfused():
        kk = mx.repeat(k, 4, axis=1); vv = mx.repeat(v, 4, axis=1)
        s = (q @ kk.transpose(0, 1, 3, 2)) * (D ** -0.5)
        s = s + mx.where(mx.arange(S)[None, :] <= mx.arange(S)[:, None], 0.0, -mx.inf).astype(s.dtype)
        return mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype) @ vv
    T("sdpa unfused (matmul+softmax)", unfused)
    o = mx.random.normal((1, S, 2048)).astype(mx.bfloat16); mx.eval(o)
    T("o_proj gemm", lambda: o @ A["o_proj"].T)
    qq = mx.random.normal((1, S, H, 2 * D)).astype(mx.bfloat16); mx.eval(qq)
    T("q rms_norm + transpose", lambda: mx.fast.rms_norm(qq[..., :D], A["q_norm"], 1e-6).transpose(0, 2, 1, 3))
    qt = mx.random.normal((1, H, S, D)).astype(mx.bfloat16); mx.eval(qt)
    T("rope(q)", lambda: mx.fast.rope(qt, 64, traditional=False, base=1e7, scale=1.0, offset=0))
    T("full layer (_full_attn)", lambda: m._full_attn(cfg, A, x, (mx.zeros((1, 2, S, 256), mx.bfloat16),) * 2, 0, S, "causal")[0])
