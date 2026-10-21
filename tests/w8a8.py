"""Opt-in lossy W8A8 prefill (gemm_i8.py + prefill.norm_q): the int8 NAX GEMM is exact (int32 accumulation, dequant epilogue rounds once
to bf16), the norm kernel's rotation + quantizer matches Hadamard-rotated per-row symmetric int8 of norm(), and the whole forward with
the GEMMs on int8 stays close to the exact bf16 path (the loss itself is documented in docs/PLAN.md).  QWEN35_W8A8_ROT=0 tests the unrotated mode."""
import sys

import numpy as np
import mlx.core as mx

from qwen35 import model as m, prefill as pf, gemm_i8 as I

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


def bits(a):
    return np.array(a.view(mx.uint16))


mx.random.seed(0)
# 1. GEMM kernels vs an exact reference (|acc| <= 127 * 127 * K < 2^24 for K <= 1024: exact in fp32), ragged M / N included
for (M, N, K) in ((300, 704, 1024), (2048, 7168, 1024), (130, 8224, 1024), (37, 1024, 512), (2000, 5120, 1024)):
    x = (mx.random.normal((M, K)) * 0.8).astype(mx.bfloat16)
    w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
    xq, sx = I.quant_rows(x)
    wq, sw = I.quant_rows(w)
    acc = xq.astype(mx.float32) @ wq.astype(mx.float32).T
    ref = (acc * (sx[:, None] * sw[None, :])).astype(mx.bfloat16)
    ok = True
    for cfg in ((128, 128, 256, 4, 4), (128, 128, 512, 4, 4), (64, 128, 256, 1, 4), (64, 128, 512, 2, 4), (64, 64, 256, 2, 2)):
        got = I.matmul_i8(xq, sx, wq, sw, cfg=cfg)
        mx.eval(got, ref)
        ok &= np.array_equal(bits(got), bits(ref))
    check(f"plain  M={M} N={N} K={K}: bit-identical on 5 tile configs", ok)
    if N % 32 == 0:                                       # swiglu epilogue on 16-row interleaved weights == swiglu of the plain GEMM
        wil = pf._interleave16(w)
        wq2, sw2 = I.quant_rows(wil)
        wq1, sw1 = I.quant_rows(w)
        gu = I.matmul_i8(xq, sx, wq1, sw1)                                # [M, N] = gate | up (un-interleaved)
        ref2 = pf.swiglu(gu) if N == 7168 else None
        got2 = I.matmul_i8(xq, sx, wq2, sw2, epi=I.EPI_SWIGLU)
        if ref2 is not None:
            mx.eval(got2, ref2)
            check(f"swiglu M={M} N={N}: bit-identical to plain GEMM + swiglu kernel", np.array_equal(bits(got2), bits(ref2)))

# 2. norm_q == (rotated) per-row symmetric int8 of norm()
H = np.array(I.hadamard(), dtype=np.float64)
for S in (2048, 777):
    x = (mx.random.normal((S, 1024)) * 3).astype(mx.bfloat16)
    w = (mx.random.normal((1024,)) * 0.5 + 1).astype(mx.bfloat16)
    h = pf.norm(x, w)
    qa = pf.norm_q(x, w)
    mx.eval(h, qa.q, qa.s)
    hf = np.array(h.astype(mx.float32)).astype(np.float64)
    if I.ROT:
        hf = hf @ H
    amax = np.abs(hf).max(-1)
    q, s = np.array(qa.q).astype(np.float64), np.array(qa.s)
    check(f"norm_q S={S} rot={I.ROT}: scale = amax / 127", np.allclose(s, amax / 127, rtol=1e-5))
    ref_q = np.rint(hf * (127 / amax)[:, None])
    check(f"norm_q S={S} rot={I.ROT}: int8 values equal per-row quantization (<= 1 level, > 99.9% identical)",
          np.abs(q - ref_q).max() <= 1 and np.mean(q == ref_q) > 0.999 and np.abs(q).max() <= 127)
wr = (mx.random.normal((64, 1024)) * 0.05).astype(mx.bfloat16)
wq, sw = I.quant_weight(wr)
mx.eval(wq, sw)
tgt = np.array(wr.astype(mx.float32)).astype(np.float64) @ (H if I.ROT else np.eye(1024))             # what is quantized: W H (rotated) or W
err = np.abs(np.array(wq).astype(np.float64) * np.array(sw)[:, None] - tgt) / np.array(sw)[:, None]
check("quant_weight: int8 * scale within half a step (+ fp32 noise) of W H", err.max() <= 0.51)

# 3. whole forward: W8A8 vs exact
cfg = m.Cfg()
W = m.load_weights(cfg)
S = 700
ids = mx.array([[(i * 7919 + 13) % 150000 for i in range(S)]])


def run(names):
    I.NAMES = frozenset(names)
    st = m.new_state(cfg, S + 8)
    h, st = pf.forward(cfg, W, ids, st, 0, S, tail=128)
    mx.eval(h, st)
    I.NAMES = frozenset()
    return np.array(h.astype(mx.float32)).ravel()


ref = run([])
TOLS = (0.03, 0.025, 0.03, 0.04) if I.ROT else (0.05, 0.04, 0.06, 0.08)
for names, tol in zip((["in_proj"], ["qkv"], ["gate_up"], ["in_proj", "qkv", "gate_up"]), TOLS):
    got = run(names)
    rel = float(np.linalg.norm(ref - got) / np.linalg.norm(ref))
    check(f"forward S={S} W8A8={','.join(names)}: hidden rel-L2 {rel:.4f} < {tol}", np.isfinite(got).all() and 0 < rel < tol)
check("exact path unchanged after W8A8 runs", np.array_equal(ref, run([])))
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
