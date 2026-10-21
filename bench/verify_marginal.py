"""Marginal cost of each op kind inside the real verify graph, M=1 vs M=4 (op executed one extra time, result folded into x*0)."""
import sys, time, statistics
import mlx.core as mx
from qwen35 import fast
from qwen35.fast import fused_gemv, deltanet, argmax_head, _write
from qwen35.spec import SpecEngine
from quiet import Probe
_e = None


def _engine():
    global _e
    if _e is None:
        _e = SpecEngine(k=3, max_len=1024)
    return _e


def _dims():
    cfg = _engine().cfg
    return cfg, _engine().F, cfg.n_heads, cfg.n_kv, cfg.head_dim

def verify(cfg, F, toks, st, sel, pos, kv_len, extra):
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    M = toks.shape[0]
    mask = mx.arange(kv_len)[None, :] <= (pos + mx.arange(M))[:, None]
    x = F["embed"][toks]
    new = {"kv": [], "conv": [], "ssm": []}
    ki = ci = 0
    sink = mx.zeros((), dtype=mx.float32)
    for i, L in enumerate(F["layers"]):
        if cfg.is_linear(i):
            proj = fused_gemv(L["in_proj"], x, nw=L["ln1"])
            if extra == "in_proj": sink = sink + fused_gemv(L["in_proj"], x, nw=L["ln1"]).astype(mx.float32).sum() * 0
            y, cs_, ss_ = deltanet(proj, st["conv"][ci], L["conv_w"], L["A_log"], L["dt_bias"], st["ssm"][ci], sel)
            if extra == "deltanet": sink = sink + deltanet(proj, st["conv"][ci], L["conv_w"], L["A_log"], L["dt_bias"], st["ssm"][ci], sel)[0].astype(mx.float32).sum() * 0
            new["conv"].append(cs_); new["ssm"].append(ss_)
            x0 = x
            x = fused_gemv(L["out_proj"], y, res=x, gnorm=(proj, L["gnorm"]), T=256, R=8)
            if extra == "out_proj": sink = sink + fused_gemv(L["out_proj"], y, res=x0, gnorm=(proj, L["gnorm"]), T=256, R=8).astype(mx.float32).sum() * 0
            ci += 1
        else:
            qkv = fused_gemv(L["qkv"], x, nw=L["ln1"])
            if extra == "qkv": sink = sink + fused_gemv(L["qkv"], x, nw=L["ln1"]).astype(mx.float32).sum() * 0
            def glue(qkv):
                q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
                q, gate = mx.split(q.reshape(1, M, H, 2 * D), 2, axis=-1)
                gate = gate.reshape(M, -1)
                q = mx.fast.rms_norm(q, L["q_norm"], cfg.eps).transpose(0, 2, 1, 3)
                k = mx.fast.rms_norm(k.reshape(1, M, KV, D), L["k_norm"], cfg.eps).transpose(0, 2, 1, 3)
                v = v.reshape(1, M, KV, D).transpose(0, 2, 1, 3)
                q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
                k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
                K_, V_ = st["kv"][ki]
                K_ = _write(K_, k, pos); V_ = _write(V_, v, pos)
                o = mx.fast.scaled_dot_product_attention(q, K_[:, :, :kv_len], V_[:, :, :kv_len], scale=D ** -0.5, mask=mask)
                o = o.transpose(0, 2, 1, 3).reshape(M, -1) * mx.sigmoid(gate)
                return o, K_, V_
            o, K_, V_ = glue(qkv)
            if extra == "attn_glue": sink = sink + glue(qkv)[0].astype(mx.float32).sum() * 0
            new["kv"].append((K_, V_))
            x0 = x
            x = fused_gemv(L["o_proj"], o, res=x)
            if extra == "o_proj": sink = sink + fused_gemv(L["o_proj"], o, res=x0).astype(mx.float32).sum() * 0
            ki += 1
        act = fused_gemv(L["gate_up_il"], x, nw=L["ln2"], swiglu=True)
        if extra == "gate_up": sink = sink + fused_gemv(L["gate_up_il"], x, nw=L["ln2"], swiglu=True).astype(mx.float32).sum() * 0
        x0 = x
        x = fused_gemv(L["down"], act, res=x)
        if extra == "down": sink = sink + fused_gemv(L["down"], act, res=x0).astype(mx.float32).sum() * 0
    Hn = mx.fast.rms_norm(x, F["norm"], cfg.eps)
    ids = argmax_head(F["embed"], F.get("head_tab"), Hn)
    if extra == "head": sink = sink + argmax_head(F["embed"], F.get("head_tab"), Hn).astype(mx.float32).sum() * 0
    return ids + sink.astype(mx.int32), Hn, new


EXTRAS = [None, "in_proj", "deltanet", "out_proj", "qkv", "attn_glue", "o_proj", "gate_up", "down", "head"]


def measure(M, rounds=15):
    cfg, F, _, _, _ = _dims()
    globals().update(H=cfg.n_heads, KV=cfg.n_kv, D=cfg.head_dim)
    st0 = fast.new_state(cfg, 1024)
    sel = mx.zeros((1,), dtype=mx.int32); pos = mx.array(100, dtype=mx.int32)
    toks = mx.arange(M, dtype=mx.int32) + 1000
    st = {"kv": st0["kv"], "conv": [mx.concatenate([c] * M, 0) for c in st0["conv"]], "ssm": [mx.concatenate([c] * M, 0) for c in st0["ssm"]]}
    mx.eval(st)
    fns = {x_: mx.compile(lambda F, t, s, sel, pos, x_=x_: verify(cfg, F, t, s, sel, pos, 256, x_)) for x_ in EXTRAS}
    for x_ in EXTRAS:
        for _ in range(3): mx.eval(fns[x_](F, toks, st, sel, pos))
    probe = Probe(tol=1.18)
    res = {x_: [] for x_ in EXTRAS}
    for r in range(rounds):
        for x_ in EXTRAS:
            def run():
                t = time.perf_counter(); outs = [fns[x_](F, toks, st, sel, pos) for _ in range(6)]; mx.eval(outs); return (time.perf_counter() - t) / 6
            res[x_].append(probe.sample(run)[0])
    base = statistics.median(res[None])
    return {None: base, **{x_: statistics.median(res[x_]) - base for x_ in EXTRAS[1:]}}


if __name__ == "__main__":
    for M in (1, 4):
        r = measure(M)
        print(f"M={M}: verify {r[None]*1e3:.2f} ms")
        for x_ in EXTRAS[1:]:
            print(f"   +{x_:10s} marginal {r[x_]*1e3:6.3f} ms")
