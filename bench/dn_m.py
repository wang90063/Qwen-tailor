"""DeltaNet decode kernel (kernels.deltanet): time for 18 layers at M = 1, 2, 4 rows."""
import sys, time
import mlx.core as mx
from qwen35.kernels import deltanet
bf = mx.bfloat16
def tm(f, n=10):
    mx.eval(f()); t = time.perf_counter(); outs = [f() for _ in range(n)]; mx.eval(outs); return (time.perf_counter() - t) / n
cw = (mx.random.normal((6144, 4, 1)) * .3).astype(bf); Al = mx.zeros((16,)); dtb = mx.zeros((16,), bf)
sel = mx.zeros((1,), dtype=mx.int32)
for M in (1, 2, 4):
    sts = [mx.random.normal((M, 16, 128, 128)) * .01 for _ in range(18)]
    css = [(mx.random.normal((M, 3, 6144))).astype(bf) for _ in range(18)]
    proj = (mx.random.normal((M, 8224)) * .5).astype(bf); mx.eval(sts, css, proj)
    t = tm(lambda: [deltanet(proj, c, cw, Al, dtb, s, sel)[2] for c, s in zip(css, sts)])
    print(f"deltanet M={M}: {t*1e3:.2f} ms for 18 layers")
