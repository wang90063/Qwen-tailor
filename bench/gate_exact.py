"""Find the exact bf16 rounding structure of mx.sigmoid / nn.softplus (what mlx-lm's gated_delta_update feeds the kernel)."""
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn

bits = np.arange(65536, dtype=np.uint32) << 16
f = bits.view(np.float32)
ok = np.isfinite(f) & (np.abs(f) < 60)
x = mx.array(f).astype(mx.bfloat16)
rb = lambda a: np.array(mx.array(np.asarray(a, dtype=np.float32)).astype(mx.bfloat16).astype(mx.float32))
xf = np.array(x.astype(mx.float32))

ref_sig = np.array(mx.sigmoid(x).astype(mx.float32))
ref_sp = np.array(nn.softplus(x).astype(mx.float32))
e32 = np.exp
with np.errstate(all="ignore"):
    sigA = rb(1 / (1 + np.exp(-xf)))
    e = rb(np.exp(-np.abs(xf))); d = rb(1 + e); y = rb(1 / d)
    sigB = np.where(xf < 0, rb(1 - y), y)
    ax = np.abs(xf)
    spA = rb(np.maximum(xf, 0) + np.log1p(np.exp(-ax)))
    e1 = rb(np.exp(-ax)); l1 = rb(np.log1p(e1))
    spB = rb(np.maximum(xf, 0) + l1)
    spC = rb(np.maximum(xf, 0) + rb(np.log1p(np.exp(-ax))))
    spD = rb(np.maximum(xf, 0) + rb(np.log(1 + e1)))
    eP = rb(np.exp(np.abs(xf))); dP = rb(1 + eP); yP = rb(1 / dP)
    sigC = np.where(xf < 0, yP, rb(1 - yP))
    sigD = np.where(xf < 0, rb(1 / (1 + np.exp(np.abs(xf)))), rb(1 - rb(1 / (1 + np.exp(np.abs(xf))))))
for name, cand, ref in (("sigmoid C  (MLX structure, rounded)", sigC, ref_sig), ("sigmoid D  (round y only)", sigD, ref_sig), ("sigmoid A  (BF(1/(1+exp(-x))))", sigA, ref_sig), ("sigmoid B  (rounded e,d,y,1-y)", sigB, ref_sig),
                        ("softplus A (current kernels)", spA, ref_sp), ("softplus B (round exp,log1p,add)", spB, ref_sp),
                        ("softplus C (round log1p,add)", spC, ref_sp), ("softplus D (round exp, log(1+e), add)", spD, ref_sp)):
    mism = ok & (cand != ref)
    print(f"{name:40s}: {int(mism.sum()):5d} / {int(ok.sum())} mismatches")

# ---- silu = x * sigmoid(x) as mlx_lm's nn.silu (compiled, bf16 intermediates)
ref_silu = np.array(nn.silu(x).astype(mx.float32))
with np.errstate(all="ignore"):
    siluA = rb(xf / (1 + np.exp(-xf)))
    siluB = rb(xf * sigC)
    siluC = rb(xf * sigmoid_ref) if False else None
for name, cand in (("silu A (single rounding)", siluA), ("silu B (x * bf16 sigmoid)", siluB)):
    mism = ok & (cand != ref_silu)
    print(f"{name:40s}: {int(mism.sum()):5d} / {int(ok.sum())} mismatches")
