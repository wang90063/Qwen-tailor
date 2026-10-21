"""Custom NAX GEMM (qwen35/gemm_nax.py) vs mx.matmul on the real prefill shapes; tile-config sweep.
usage: gemm_nax.py [M] [shape names comma separated] [cfg;cfg;...]   cfg = BM,BN,BK,WM,WN"""
import sys, time, itertools
import mlx.core as mx
from qwen35 import gemm_nax as G

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SHAPES = {                      # name: (K, N)
    "in_proj": (1024, 8224),
    "gate_up": (1024, 7168),
    "down": (3584, 1024),
    "out_proj": (2048, 1024),
    "q_proj": (1024, 4096),
}
sel = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(SHAPES)
CFGS = [(128, 128, 256, 4, 4), (128, 128, 512, 4, 4), (128, 128, 128, 4, 4), (128, 256, 256, 4, 4), (64, 128, 256, 2, 4),
        (128, 64, 256, 4, 2), (64, 64, 256, 2, 2), (128, 128, 256, 2, 4), (128, 128, 256, 4, 2), (256, 128, 256, 4, 4),
        (64, 256, 256, 2, 4)]
if len(sys.argv) > 3:
    CFGS = [tuple(int(v) for v in c.split(",")) for c in sys.argv[3].split(";")]
R = 8                           # distinct weights per timed graph (amortizes eval overhead, defeats cache reuse)


def timeit(f, n=7):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


for name in sel:
    K, N = SHAPES[name]
    x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
    ws = [(mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16) for _ in range(R)]
    mx.eval(x, *ws)
    flops = 2.0 * M * N * K * R
    t_ref = timeit(lambda: [x @ w.T for w in ws])
    print(f"== {name} M={M} K={K} N={N}: mx.matmul {t_ref / R * 1e3:.3f} ms  {flops / t_ref / 1e12:.2f} TFLOPS")
    ref = (x @ ws[0].T); mx.eval(ref)
    for cfg in CFGS:
        try:
            y = G.matmul_nt(x, ws[0], cfg=cfg); mx.eval(y)
        except Exception as e:
            print(f"   cfg {cfg}: FAILED {str(e).splitlines()[0][:100]}"); continue
        d = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))
        rel = float(mx.sqrt(mx.sum(d * d) / mx.sum(ref.astype(mx.float32) ** 2)))
        eq = float(mx.mean((y == ref).astype(mx.float32)))
        t = timeit(lambda: [G.matmul_nt(x, w, cfg=cfg) for w in ws])
        print(f"   cfg {str(cfg):22s} {t / R * 1e3:7.3f} ms  {flops / t / 1e12:6.2f} TFLOPS  ({t_ref / t:5.2f}x vs mlx)  "
              f"rel-L2 {rel:.1e} exact {eq * 100:.1f}%")
