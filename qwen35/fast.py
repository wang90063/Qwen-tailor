"""Decode / verify forward built from our fused Metal kernels, for M = 1..4 tokens per call.

State layout ("snapshot state"):
  kv   : 6 x (K, V) [1, 2, Smax, 256]                    (rollback = just don't advance pos)
  conv : 18 x [S, 3, 6144]   bf16   S snapshots (state after each of the S last processed tokens)
  ssm  : 18 x [S, 16, 128, 128] fp32
  The next call starts from snapshot `sel` (int32 [1] array) -> speculative rollback without copying.
"""
import mlx.core as mx

from .head_prune import ENABLED as HEAD_PRUNE, argmax_head, build_tables
from . import wcomp
from .kernels import fused_gemv, deltanet
from .model import Cfg, _write


def prepare(W, cfg: Cfg):
    """Re-layout fused weights for the kernels: gate/up rows interleaved so one simdgroup sees both; then the lossless 12-bit
    compressed copy of every streamed layer matrix (wcomp.py)."""
    F = {"embed": W["embed"], "norm": W["norm"], "layers": []}
    if HEAD_PRUNE:
        F["head_tab"] = build_tables(W["embed"])          # int8 copy for the certified lm_head pre-filter (head_prune.py)
    for L in W["layers"]:
        M = dict(L)
        if "gate_up" in L:
            gu = L["gate_up"]
            n = gu.shape[0] // 2
            M["gate_up_il"] = mx.stack([gu[:n], gu[n:]], axis=1).reshape(2 * n, -1)
            del M["gate_up"]
        if wcomp.ENABLED:         # lossless 12 bit/weight copies for the decode GEMVs (bit-identical results, -25% bytes)
            for k in ("in_proj", "out_proj", "qkv", "o_proj", "gate_up_il", "down"):
                if k in M:
                    M[k] = wcomp.encode(M[k])
                    mx.eval(M[k])
        F["layers"].append(M)
    mx.eval(F)
    return F


def new_state(cfg: Cfg, max_len: int):
    st = {"kv": [], "conv": [], "ssm": []}
    for i in range(cfg.layers):
        if cfg.is_linear(i):
            st["conv"].append(mx.zeros((1, 3, cfg.conv_dim), mx.bfloat16))
            st["ssm"].append(mx.zeros((1, cfg.hv, cfg.dv, cfg.dk), mx.float32))
        else:
            st["kv"].append((mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16),
                             mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16)))
    mx.eval(st)
    return st


def verify(cfg: Cfg, F, toks, st, sel, pos, kv_len):
    """toks [M] int32 (M<=4) processed at positions pos..pos+M-1, starting from recurrent snapshot `sel`.
    Returns (ids [M] int32 greedy next-token at each position, Hn [M,H] final-normed hidden, new state)."""
    M = toks.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    mask = mx.arange(kv_len)[None, :] <= (pos + mx.arange(M))[:, None]
    x = F["embed"][toks]  # [M,H]
    new = {"kv": [], "conv": [], "ssm": []}
    ki = ci = 0
    for i, L in enumerate(F["layers"]):
        if cfg.is_linear(i):
            proj = fused_gemv(L["in_proj"], x, nw=L["ln1"])
            y, cs_, ss_ = deltanet(proj, st["conv"][ci], L["conv_w"], L["A_log"], L["dt_bias"], st["ssm"][ci], sel)
            new["conv"].append(cs_); new["ssm"].append(ss_)
            x = fused_gemv(L["out_proj"], y, res=x, gnorm=(proj, L["gnorm"]), T=256, R=8)
            ci += 1
        else:
            qkv = fused_gemv(L["qkv"], x, nw=L["ln1"])
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
            new["kv"].append((K_, V_))
            o = mx.fast.scaled_dot_product_attention(q, K_[:, :, :kv_len], V_[:, :, :kv_len],
                                                     scale=D ** -0.5, mask=mask)
            o = o.transpose(0, 2, 1, 3).reshape(M, -1) * mx.sigmoid(gate)
            x = fused_gemv(L["o_proj"], o, res=x)
            ki += 1
        act = fused_gemv(L["gate_up_il"], x, nw=L["ln2"], swiglu=True)
        x = fused_gemv(L["down"], act, res=x)
    Hn = mx.fast.rms_norm(x, F["norm"], cfg.eps)
    return argmax_head(F["embed"], F.get("head_tab"), Hn), Hn, new
