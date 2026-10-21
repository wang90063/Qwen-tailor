"""gate_up GEMM with fused SwiGLU epilogue vs (mx.matmul + prefill swiglu kernel) and vs (custom plain GEMM + swiglu)."""
import sys, time
import mlx.core as mx
from qwen35 import gemm_nax as G, prefill as pf

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
K, N = 1024, 7168
R = 8


def interleave16(w):
    """[2*I, K] with gate rows first then up rows -> blocks of 16 gate rows / 16 up rows alternating."""
    I = w.shape[0] // 2
    g = w[:I].reshape(I // 16, 16, -1); u = w[I:].reshape(I // 16, 16, -1)
    return mx.concatenate([g, u], axis=1).reshape(2 * I, -1)


def timeit(f, n=7):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
ws = [(mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16) for _ in range(R)]
wil = [interleave16(w) for w in ws]
mx.eval(x, *ws, *wil)
ref = pf.swiglu(x @ ws[0].T); mx.eval(ref)
t_ref = timeit(lambda: [pf.swiglu(x @ w.T) for w in ws]) / R
print(f"M={M}: mx.matmul + swiglu kernel          {t_ref * 1e3:.3f} ms")
t_gemm = timeit(lambda: [x @ w.T for w in ws]) / R
print(f"M={M}: mx.matmul alone                    {t_gemm * 1e3:.3f} ms  (=> swiglu kernel ~{(t_ref - t_gemm) * 1e3:.3f} ms)")
CFGS = [(128, 128, 256, 4, 4), (64, 128, 256, 2, 4), (64, 128, 512, 2, 4), (64, 256, 256, 2, 4), (32, 256, 256, 1, 4),
        (128, 128, 512, 4, 4), (64, 128, 256, 2, 2), (64, 256, 512, 2, 4), (128, 256, 512, 4, 4)]
for cfg in CFGS:
    try:
        y = G.matmul_nt(x, wil[0], epi=G.EPI_SWIGLU, cfg=cfg); mx.eval(y)
    except Exception as e:
        print(f"cfg {cfg}: FAILED {str(e).splitlines()[0][:120]}"); continue
    eq = float(mx.mean((y == ref).astype(mx.float32))) * 100
    t = timeit(lambda: [G.matmul_nt(x, w, epi=G.EPI_SWIGLU, cfg=cfg) for w in wil]) / R
    tp = timeit(lambda: [pf.swiglu(G.matmul_nt(x, w, cfg=cfg)) for w in ws]) / R
    print(f"cfg {str(cfg):22s} fused {t * 1e3:.3f} ms | plain gemm + swiglu kernel {tp * 1e3:.3f} ms | exact vs ref {eq:.2f}% "
          f"| fused is {t_ref / t:.2f}x of baseline")
