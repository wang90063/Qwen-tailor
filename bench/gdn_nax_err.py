"""Attribute the NAX kernel's error: compare phase-A outputs with float64 numpy, then emulate phase B in float64."""
import sys
import numpy as np
import mlx.core as mx
from qwen35 import model as m, prefill as pf, gdn_nax
from bench.gdn_nax import L, cmp  # noqa

S = 256; NC = S // 16
mx.random.seed(0)
x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
proj = x @ L["in_proj"].T
cs = mx.zeros((3, 6144), mx.bfloat16)
q, k, v = pf.conv_prep(proj, cs, L["conv_w"])
mx.eval(q, k, v, proj)
ka, kb = gdn_nax._kernels()
Tb, Am, gv = ka(inputs=[q, k, proj, L["A_log"], L["dt_bias"]], output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
                output_dtypes=[mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1))
mx.eval(Tb, Am, gv)
Tb, Am, gv = (np.array(t) for t in (Tb, Am, gv))
qn = np.array(q.astype(mx.float32), dtype=np.float64); kn = np.array(k.astype(mx.float32), dtype=np.float64)
pn = np.array(proj.astype(mx.float32), dtype=np.float64)
Alog = np.array(L["A_log"].astype(mx.float32), dtype=np.float64); dtb = np.array(L["dt_bias"].astype(mx.float32), dtype=np.float64)

def bf(x):  # emulate bf16 rounding
    return np.array(mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))

errs_T, errs_A, errs_g = [], [], []
for c in (0, 5, 9, 15):
    for h in (0, 3, 11):
        t0 = c * 16
        a_ = bf(pn[t0:t0 + 16, 8208 + h] + dtb[h]).astype(np.float64)
        sp = bf(np.maximum(a_, 0) + np.log1p(np.exp(-np.abs(a_)))).astype(np.float64)
        lg = -np.exp(Alog[h]) * sp
        Lc = np.cumsum(lg)
        beta = bf(1 / (1 + np.exp(-pn[t0:t0 + 16, 8192 + h]))).astype(np.float64)
        K = kn[t0:t0 + 16, h]; Q = qn[t0:t0 + 16, h]
        G = np.exp(Lc[:, None] - Lc[None, :])
        Lm = np.tril(beta[:, None] * G * (K @ K.T), -1)
        T = np.linalg.inv(np.eye(16) + Lm)
        Tbx = T * beta[None, :]
        Ax = np.tril(G * (Q @ K.T))
        rec = c * 16 + h
        gpuT = Tb[c, h].reshape(16, 16); gpuA = Am[c, h].reshape(16, 16)
        errs_T.append(np.linalg.norm(gpuT - Tbx) / np.linalg.norm(Tbx))
        errs_A.append(np.linalg.norm(gpuA - Ax) / np.linalg.norm(Ax))
        gg = gv[c, h]
        errs_g.append(max(np.abs(gg[:16] - np.exp(Lc)).max(), np.abs(gg[16:] - np.exp(Lc[-1] - Lc)).max()))
print("T  rel-L2 err (gpu vs fp64 inverse):", ["%.2e" % e for e in errs_T])
print("A  rel-L2 err                      :", ["%.2e" % e for e in errs_A])
print("gamma/dec max abs err              :", ["%.2e" % e for e in errs_g])
