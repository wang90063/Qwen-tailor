"""Lossless 12-bit weight compression (qwen35/wcomp.py): codec round trip + compressed fused_gemv == bf16 fused_gemv, bit for bit."""
import itertools
import sys

import numpy as np
import mlx.core as mx

from qwen35 import wcomp
from qwen35.kernels import fused_gemv

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


def bits(a):
    return np.array(a.view(mx.uint16))


rng = np.random.default_rng(0)


def rand_bf16(N, K, kind):
    if kind == "normal":                  # typical trained weights
        a = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    elif kind == "wide":                  # many exponents out of the window -> lots of exception chunks
        a = (rng.standard_normal((N, K)) * np.exp(rng.uniform(-30, 8, (N, K)))).astype(np.float32)
    elif kind == "bits":                  # arbitrary bit patterns incl. NaN / Inf / denormal / -0
        u = rng.integers(0, 1 << 16, (N, K), dtype=np.uint32).astype(np.uint16)
        return mx.array(u).view(mx.bfloat16)
    elif kind == "special":
        a = rng.standard_normal((N, K)).astype(np.float32) * 0.02
        a[::3, ::5] = 0.0
        a[1::7, ::4] = -0.0
        a[2::7, 1::6] = np.inf
        a[3::7, 2::9] = -np.inf
        a[4::11, 3::8] = 1e-39
        a[5::13, ::7] = 3e38
    elif kind == "none":                  # no exceptions at all
        a = rng.uniform(0.01, 0.1, (N, K)).astype(np.float32) * rng.choice([-1, 1], (N, K))
    return mx.array(a).astype(mx.bfloat16)


# --- codec round trip -----------------------------------------------------------------------------------------------------
for kind, (N, K) in itertools.product(["normal", "wide", "bits", "special", "none"], [(5, 64), (33, 1024), (17, 3584)]):
    w = rand_bf16(N, K, kind)
    Wc, X = wcomp.encode(w)
    check(f"round trip {kind:8s} {N}x{K} (exception chunks {X.shape[0] - 1})",
          Wc.shape == (N, 3 * K // 8) and wcomp.shape_of((Wc, X)) == (N, K) and np.array_equal(wcomp.decode(Wc, X), bits(w)))

# --- compressed GEMV == bf16 GEMV (all prologue / epilogue combos, M = 1..4, N not a multiple of R) ---------------------------
def gemv_case(N, K, M, kind, pro, epi, T=None, R=None):
    w = rand_bf16(N, K, kind)
    x = (mx.array(rng.standard_normal((M, K)).astype(np.float32)) * 0.5).astype(mx.bfloat16)
    nw = (mx.array(rng.uniform(0.5, 1.5, K).astype(np.float32))).astype(mx.bfloat16)
    kw = {}
    if pro == "rms":
        kw["nw"] = nw
    if epi == "add":
        kw["res"] = (mx.array(rng.standard_normal((M, N)).astype(np.float32))).astype(mx.bfloat16)
    if epi == "swiglu":
        kw["swiglu"] = True
    if T:
        kw["T"] = T
    if R:
        kw["R"] = R
    ref = fused_gemv(w, x, **kw)
    out = fused_gemv(wcomp.encode(w), x, **kw)
    mx.eval(ref, out)
    return np.array_equal(bits(ref), bits(out))


for kind, (N, K), M, (pro, epi) in itertools.product(
        ["normal", "wide", "special"], [(64, 1024), (1022, 1024), (1024, 3584), (50, 2048)], [1, 2, 3, 4],
        [("none", "none"), ("rms", "none"), ("rms", "swiglu"), ("none", "add")]):
    if epi == "swiglu" and N % 2:
        continue
    r = gemv_case(N, K, M, kind, pro, epi)
    check(f"gemv {kind:7s} {N}x{K} M={M} {pro}/{epi}", bool(r))

for T, R in [(128, 4), (256, 8), (64, 2)]:
    check(f"gemv T={T} R={R} M=4", bool(gemv_case(1024, 2048, 4, "wide", "none", "add", T=T, R=R)))

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
