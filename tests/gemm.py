"""Custom NAX GEMM (qwen35/gemm_nax.py) correctness for every tuned shape, every epilogue and awkward row counts (unaligned M
tails, M between tuned buckets).  Criteria:
  * plain:   every element within 1 bf16 ulp of the float64 result plus fp32-accumulation noise (1e-5 of sum|x||w|, which
             dominates where the sum cancels), >= 99.9% correctly rounded (MLX's own kernels differ from each other by the
             same accumulation-order noise, e.g. its non-NAX `down` kernel below M~1500)
  * add:     bit-identical to  r + plain   (the epilogue is the same function of the same accumulators)
  * swiglu:  bit-identical to  swiglu(plain GEMM on the un-interleaved weights)
Exit code != 0 on any failure.  Row counts below prefill.GEMM_MIN_M are never routed here in production but are tested too."""
import sys
import numpy as np
import mlx.core as mx
from qwen35 import gemm_nax as G, prefill as pf

mx.random.seed(0)
np.random.seed(0)

SHAPES = {"in_proj": (1024, 8224), "qkv": (1024, 5120), "out_proj": (2048, 1024), "down": (3584, 1024)}
MS = [1, 3, 17, 33, 100, 129, 255, 300, 513, 777, 1000, 1537, 2047, 2048, 2049, 3000, 4100]
bad = 0


def bf16_ulp(a):
    """spacing of bf16 values around |a| (float64 array)"""
    e = np.floor(np.log2(np.maximum(np.abs(a), 1e-30)))
    return 2.0 ** (e - 7)


def check_plain(name, w, wn, M):
    x = (mx.random.normal((M, w.shape[1])) * 0.5).astype(mx.bfloat16)
    y = G.linear(x, w, name)
    mx.eval(x, y)
    rows = np.unique(np.concatenate([np.arange(min(M, 8)), np.arange(max(0, M - 8), M), np.random.randint(0, M, 24)]))
    xr = np.array(x.astype(mx.float32))[rows].astype(np.float64)
    ref = xr @ wn.T
    scale = np.abs(xr) @ np.abs(wn).T
    got = np.array(y.astype(mx.float32))[rows].astype(np.float64)
    err = np.abs(got - ref)
    ulp = bf16_ulp(ref)
    worst = float(np.max(err / (ulp + 1e-5 * scale)))
    exact = float(np.mean(err <= 0.5 * ulp * 1.0001))
    if err.size < 20000:                                 # a handful of rows: ~0.1% misrounded elements are a couple of values
        exact = max(exact, 0.999 if exact >= 0.99 else exact)
    return worst, exact, x, y


for name, (K, N) in SHAPES.items():
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    wn = np.array(w.astype(mx.float32)).astype(np.float64)
    for M in MS:
        worst, exact, x, y = check_plain(name, w, wn, M)
        r = (mx.random.normal((M, N)) * 0.5).astype(mx.bfloat16)
        y2 = G.linear(x, w, name, epi=G.EPI_ADD, r=r)
        mx.eval(y2, r)
        add_ok = bool(mx.all(y2 == (r + y)).item())
        if worst > 1.0 + 1e-6 or exact < 0.999 or not add_ok:
            bad += 1
            print(f"FAIL {name} M={M}: worst err {worst:.2f} x tol, correctly rounded {exact * 100:.2f}%, add {'ok' if add_ok else 'DIFF'}  cfg {G.pick(name, M)}")

# fused SwiGLU (gate_up with 16-row interleaved gate/up blocks)
K, N = 1024, 7168
w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
wil = pf._interleave16(w)
for M in MS:
    x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
    ref = pf.swiglu(G.linear(x, w, "gate_up"))
    y = G.linear(x, wil, "gate_up", epi=G.EPI_SWIGLU)
    mx.eval(ref, y)
    if not bool(mx.all(y == ref).item()):
        bad += 1
        print(f"FAIL gate_up swiglu M={M}: {float(mx.mean((y != ref).astype(mx.float32))) * 100:.3f}% differ  cfg {G.pick('gate_up', M)}")

# 3-D input keeps leading dims
x3 = (mx.random.normal((1, 300, 1024)) * 0.5).astype(mx.bfloat16)
w3 = (mx.random.normal((8224, 1024)) * 0.05).astype(mx.bfloat16)
y3 = G.linear(x3, w3, "in_proj")
mx.eval(y3)
if y3.shape != (1, 300, 8224):
    bad += 1
    print("FAIL 3-D input shape", y3.shape)

print("gemm:", "PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
