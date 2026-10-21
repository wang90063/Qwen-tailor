"""Does MLX's native mx.fast.gated_delta_update beat mlx-lm's packed kernel on the prefill shapes (S = 512 / 2048)?  Timing and max|diff|."""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf
from mlx_lm.models.gated_delta import gated_delta_update, compute_g
cfg = m.Cfg(); W = m.load_weights(cfg); L = W["layers"][0]
def T(name, f, n=9):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); print(f"{name:40s} {ts[len(ts)//2]*1e3:8.2f} ms"); return ts[len(ts)//2]

for S in (512, 2048):
    print(f"--- S={S}")
    x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
    proj = x @ L["in_proj"].T
    cs = mx.zeros((3, 6144), mx.bfloat16); ss = mx.zeros((1, 16, 128, 128), mx.float32)
    q, k, v = pf.conv_prep(proj, cs, L["conv_w"])
    a = proj[:, 8208:]; b = proj[:, 8192:8208]
    mx.eval(q, k, v, a, b)
    g = compute_g(L["A_log"], a, L["dt_bias"]); beta = mx.sigmoid(b); mx.eval(g, beta)
    print("g", g.shape, g.dtype, "beta", beta.shape, beta.dtype, "range g", float(g.min()), float(g.max()))
    ref = lambda: gated_delta_update(q[None], k[None], v[None], a[None], b[None], L["A_log"], L["dt_bias"], ss)
    T("mlx_lm packed (current)", lambda: ref()[0])
    try:
        nat = lambda: mx.fast.gated_delta_update(q[None], k[None], v[None], g[None], beta[None], ss)
        o = nat(); mx.eval(o)
        print("native outs:", [t.shape for t in o], [t.dtype for t in o])
        T("mx.fast.gated_delta_update (native)", lambda: nat()[0])
        yr, sr = ref(); yn, sn = nat(); mx.eval(yr, sr, yn, sn)
        d = mx.abs(yr.astype(mx.float32) - yn.astype(mx.float32))
        print("y  max|diff| %.4g  (|y| max %.4g, mean|diff| %.3g)" % (float(d.max()), float(mx.abs(yr.astype(mx.float32)).max()), float(d.mean())))
        ds = mx.abs(sr - sn.reshape(sr.shape)) if sn.size == sr.size else None
        if ds is not None:
            print("state max|diff| %.4g (|s| max %.4g)" % (float(ds.max()), float(mx.abs(sr).max())))
        else:
            print("state shape differs", sn.shape, sr.shape)
    except Exception as e:
        print("native failed:", type(e).__name__, e)
