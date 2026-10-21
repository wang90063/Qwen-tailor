"""Accuracy of mx.fast.gated_delta_update (bf16 / fp32 / fp16 inputs) against mlx-lm's reference kernel at S = 2048, and how much of the gap is g / beta being rounded to bf16."""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf
from mlx_lm.models.gated_delta import gated_delta_update, compute_g
cfg = m.Cfg(); W = m.load_weights(cfg); L = W["layers"][0]
def T(name, f, n=9):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); print(f"{name:44s} {ts[len(ts)//2]*1e3:8.2f} ms"); return ts[len(ts)//2]
def cmp(tag, yr, sr, yn, sn):
    yr, yn = yr.astype(mx.float32), yn.astype(mx.float32)
    d = mx.abs(yr - yn); ds = mx.abs(sr - sn)
    rel = float(mx.sqrt(mx.sum(d * d) / mx.sum(yr * yr)))
    rels = float(mx.sqrt(mx.sum(ds * ds) / mx.sum(sr * sr)))
    print(f"  {tag:26s} y: max {float(d.max()):.3g} rel-L2 {rel:.3g} | state: max {float(ds.max()):.3g} rel-L2 {rels:.3g}")

S = 2048
mx.random.seed(0)
x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
proj = x @ L["in_proj"].T
cs = mx.zeros((3, 6144), mx.bfloat16); ss = mx.zeros((1, 16, 128, 128), mx.float32)
q, k, v = pf.conv_prep(proj, cs, L["conv_w"])
a = proj[:, 8208:]; b = proj[:, 8192:8208]
g = compute_g(L["A_log"], a, L["dt_bias"]); beta = mx.sigmoid(b)
mx.eval(q, k, v, a, b, g, beta)
yr, sr = gated_delta_update(q[None], k[None], v[None], a[None], b[None], L["A_log"], L["dt_bias"], ss); mx.eval(yr, sr)
print("g quantiles:", [round(float(mx.sort(g.reshape(-1))[int(g.size*p)]), 4) for p in (0.01, 0.1, 0.5, 0.9, 0.99)])
# 1) bf16 inputs as before (g fp32 -> cast inside?)
yn, sn = mx.fast.gated_delta_update(q[None], k[None], v[None], g[None], beta[None], ss); mx.eval(yn, sn)
cmp("native bf16 in", yr, sr, yn, sn)
# 2) fp32 inputs
f = lambda t: t.astype(mx.float32)
try:
    nat32 = lambda: mx.fast.gated_delta_update(f(q)[None], f(k)[None], f(v)[None], g[None], f(beta)[None], ss)
    yn, sn = nat32(); mx.eval(yn, sn); print("fp32 path out dtype", yn.dtype)
    cmp("native fp32 in", yr, sr, yn, sn)
    qf, kf, vf, bf = f(q), f(k), f(v), f(beta); mx.eval(qf, kf, vf, bf)
    T("native fp32 in", lambda: mx.fast.gated_delta_update(qf[None], kf[None], vf[None], g[None], bf[None], ss)[0])
except Exception as e:
    print("fp32 failed:", type(e).__name__, e)
# 3) fp16 inputs
h = lambda t: t.astype(mx.float16)
try:
    yn, sn = mx.fast.gated_delta_update(h(q)[None], h(k)[None], h(v)[None], g[None], h(beta)[None], ss); mx.eval(yn, sn)
    cmp("native fp16 in", yr, sr, yn, sn)
except Exception as e:
    print("fp16 failed:", type(e).__name__, e)
# 4) how bad is rounding g,beta to bf16 in the *reference* kernel? (isolates the cause)
gq = g.astype(mx.bfloat16).astype(mx.float32)
yq, sq = gated_delta_update(q[None], k[None], v[None], a[None], b[None], L["A_log"], L["dt_bias"], ss)
from mlx_lm.models.gated_delta import gated_delta_kernel
y2, s2 = gated_delta_kernel(q[None], k[None], v[None], gq[None], beta.astype(mx.bfloat16)[None], ss); mx.eval(y2, s2)
cmp("reference w/ g rounded to bf16", yr, sr, y2, s2)
