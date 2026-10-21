"""conv_prep v2 (token-blocked) vs v1: must be bit-identical; then timing in a quiet-GPU chain."""
import sys, time
import mlx.core as mx
from qwen35 import prefill as pf
from quiet import Probe

cw = (mx.random.normal((6144, 4)) * 0.5).astype(mx.bfloat16)
cs = (mx.random.normal((3, 6144)) * 1.0).astype(mx.bfloat16)
ok = True
for S in (1, 2, 3, 5, 16, 17, 100, 2048, 2051):
    proj = (mx.random.normal((S, 8224)) * 1.0).astype(mx.bfloat16)
    pf.PREP_IMPL = "v1"; a = pf.conv_prep(proj, cs, cw)
    pf.PREP_IMPL = "v2"; b = pf.conv_prep(proj, cs, cw)
    mx.eval(*a, *b)
    same = [bool(mx.all(x == y).item()) for x, y in zip(a, b)]
    print(f"S={S:5d}: q/k/v identical {same}")
    ok &= all(same)
print("bit-identical:", ok)

S = 2048
projs = [(mx.random.normal((S, 8224)) * 1.0).astype(mx.bfloat16) for _ in range(8)]
mx.eval(*projs)
probe = Probe()
def chain(impl, tb):
    pf.PREP_IMPL = impl; pf.PREP_TB = tb
    def f():
        t = time.perf_counter(); mx.eval([pf.conv_prep(p, cs, cw) for p in projs]); return (time.perf_counter() - t) / 8
    f()
    return min(probe.sample(f)[0] for _ in range(5))
print(f"v1: {chain('v1', 16) * 1e3:.3f} ms/layer")
for tb in (4, 8, 16, 32, 64):
    print(f"v2 TB={tb:3d}: {chain('v2', tb) * 1e3:.3f} ms/layer")
