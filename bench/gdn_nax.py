"""Accuracy + speed of the two-phase NAX chunked gated-delta kernel vs mlx-lm's sequential packed kernel."""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf, gdn_nax
from mlx_lm.models.gated_delta import gated_delta_update

cfg = m.Cfg(); W = m.load_weights(cfg); L = W["layers"][0]


def T(name, f, n=9):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); print(f"{name:44s} {ts[len(ts)//2]*1e3:8.3f} ms"); return ts[len(ts)//2]


def cmp(tag, yr, sr, yn, sn):
    yr, yn = yr.astype(mx.float32).reshape(-1), yn.astype(mx.float32).reshape(-1)
    d = mx.abs(yr - yn); sr, sn = sr.reshape(-1), sn.reshape(-1); ds = mx.abs(sr - sn)
    rel = float(mx.sqrt(mx.sum(d * d) / mx.sum(yr * yr)))
    rels = float(mx.sqrt(mx.sum(ds * ds) / mx.sum(sr * sr)))
    print(f"  {tag:30s} y: max {float(d.max()):.3g} rel-L2 {rel:.3g} | state: max {float(ds.max()):.3g} rel-L2 {rels:.3g}")


def case(S, relaxed=True, seed=0, init_state=False):
    mx.random.seed(seed)
    x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
    proj = x @ L["in_proj"].T
    cs = mx.zeros((3, 6144), mx.bfloat16)
    ss = (mx.random.normal((1, 16, 128, 128)) * 0.05) if init_state else mx.zeros((1, 16, 128, 128), mx.float32)
    q, k, v = pf.conv_prep(proj, cs, L["conv_w"])
    mx.eval(q, k, v, proj, ss)
    yr, sr = gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None], proj[:, 8192:8208][None],
                                L["A_log"], L["dt_bias"], ss)
    yn, sn = gdn_nax.gdn_chunked(q, k, v, proj, L["A_log"], L["dt_bias"], ss, relaxed=relaxed)
    mx.eval(yr, sr, yn, sn)
    cmp(f"S={S} relaxed={relaxed} st0={init_state}", yr, sr, yn, sn)
    return q, k, v, proj, ss


if __name__ == "__main__":
    for S in (16, 48, 50, 512, 2048):
        case(S)
    case(2048, init_state=True)
    q, k, v, proj, ss = case(2048)
    T("packed mlx_lm (ref)", lambda: gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None],
                                                         proj[:, 8192:8208][None], L["A_log"], L["dt_bias"], ss)[0])
    ka, kb = gdn_nax._kernels()
    S = 2048; NC = S // 16
    Tb, Am, gv = ka(inputs=[q, k, proj, L["A_log"], L["dt_bias"]], output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
                    output_dtypes=[mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1))
    mx.eval(Tb, Am, gv)
    T("phase A only", lambda: ka(inputs=[q, k, proj, L["A_log"], L["dt_bias"]],
                                  output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
                                  output_dtypes=[mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1)))
    T("phase B only", lambda: kb(inputs=[q, k, v, Tb, Am, gv, ss], output_shapes=[(S, 16, 128), (1, 16, 128, 128)],
                                 output_dtypes=[mx.bfloat16, mx.float32], grid=(256, 1, 16), threadgroup=(256, 1, 1)))
    T("A + B (gdn_chunked)", lambda: gdn_nax.gdn_chunked(q, k, v, proj, L["A_log"], L["dt_bias"], ss))
