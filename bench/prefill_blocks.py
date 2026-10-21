"""Does running the prefill layer pieces in token blocks (intermediates stay in cache) beat one big pass?"""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf

cfg = m.Cfg(); W = m.load_weights(cfg); S = 2048
L = W["layers"][0]; A = W["layers"][3]


def T(name, f, n=9):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); med = ts[len(ts) // 2]
    print(f"{name:42s} {med * 1e3:8.2f} ms"); return med


x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
mx.eval(x)


def mlp_blocked(x, bs):
    outs = []
    for s in range(0, S, bs):
        h = x[s:s + bs]
        outs.append(pf.swiglu(h @ L["gate_up"].T) @ L["down"].T)
    return mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


print("== MLP (gate_up -> swiglu -> down) ==")
for bs in (2048, 1024, 512, 256, 128):
    T(f"mlp block={bs}", lambda bs=bs: mlp_blocked(x, bs))

cs = mx.zeros((3, 6144), mx.bfloat16); ss = mx.zeros((1, 16, 128, 128), mx.float32)
mx.eval(cs, ss)


def lin_blocked(x, bs):
    """DeltaNet layer body (in_proj..out_proj) in token blocks with state carry."""
    conv, st = cs, ss
    outs = []
    for s in range(0, S, bs):
        xb = x[s:s + bs]
        n = xb.shape[0]
        proj = xb @ L["in_proj"].T
        q, k, v = pf.conv_prep(proj, conv, L["conv_w"])
        out, st = pf.gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None], proj[:, 8192:8208][None],
                                        L["A_log"], L["dt_bias"], st)
        conv = proj[n - 3:, :6144] if n >= 3 else mx.concatenate([conv, proj[:, :6144]], axis=0)[-3:]
        y = pf.gated_norm(out, proj, L["gnorm"])
        outs.append(y @ L["out_proj"].T)
    return mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


print("== DeltaNet layer body ==")
for bs in (2048, 1024, 512, 256):
    T(f"linear body block={bs}", lambda bs=bs: lin_blocked(x, bs))

# correctness: blocked == unblocked
a = lin_blocked(x, 2048); b = lin_blocked(x, 256); mx.eval(a, b)
print("max|diff| blocked-vs-full:", float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max()),
      " (ref scale", float(mx.abs(a.astype(mx.float32)).max()), ")")
