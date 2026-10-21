"""Own Qwen3.5-0.8B text model (functional, no nn.Module), MLX tensor backend.

Differences from mlx-lm's implementation (framework-layer choices):
  * weights are loaded straight from safetensors and *fused at load time*
    (qkv+z+b+a for DeltaNet, q/k/v for attention, gate+up for the MLP) -> fewer dispatches
  * RMSNorm "+1" folded into the weights, conv1d weight pre-transposed
  * all state is preallocated and statically shaped: KV [1,2,Smax,256] for the 6 full-attention
    layers, conv [1,3,6144] and recurrent state [1,16,128,128] fp32 for the 18 DeltaNet layers
  * one pure step function over (token, pos, state) so the whole decode step can be mx.compile'd
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"


def weights_file() -> Path:
    """The safetensors file under models/Qwen3.5-0.8B (a clear error instead of a bare StopIteration if it is missing)."""
    f = next(MODEL_DIR.glob("*.safetensors"), None)
    if f is None:
        raise FileNotFoundError(f"no *.safetensors in {MODEL_DIR}; download the weights first:\n"
                                f"  hf download Qwen/Qwen3.5-0.8B --local-dir {MODEL_DIR}")
    return f


@dataclass
class Cfg:
    hidden: int = 1024
    inter: int = 3584
    vocab: int = 248320
    layers: int = 24
    eps: float = 1e-6
    # full attention
    n_heads: int = 8
    n_kv: int = 2
    head_dim: int = 256
    rope_dim: int = 64
    rope_theta: float = 1e7
    # linear attention
    hk: int = 16
    hv: int = 16
    dk: int = 128
    dv: int = 128
    conv_k: int = 4

    @property
    def conv_dim(self):
        return 2 * self.hk * self.dk + self.hv * self.dv

    def is_linear(self, i):
        return (i + 1) % 4 != 0


def load_weights(cfg: Cfg, path=None, fuse=True):
    path = path or weights_file()
    raw = mx.load(str(path))
    P = "model.language_model."
    g = lambda k: raw[P + k]
    W = {"embed": g("embed_tokens.weight"), "norm": g("norm.weight") + 1.0, "layers": []}
    for i in range(cfg.layers):
        p = f"layers.{i}."
        L = {
            "ln1": g(p + "input_layernorm.weight") + 1.0,
            "ln2": g(p + "post_attention_layernorm.weight") + 1.0,
        }
        gate, up = g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")
        L["down"] = g(p + "mlp.down_proj.weight")
        if fuse:
            L["gate_up"] = mx.concatenate([gate, up], axis=0)
        else:
            L["gate"], L["up"] = gate, up
        if cfg.is_linear(i):
            a = p + "linear_attn."
            parts = [g(a + "in_proj_qkv.weight"), g(a + "in_proj_z.weight"),
                     g(a + "in_proj_b.weight"), g(a + "in_proj_a.weight")]
            L["in_proj"] = mx.concatenate(parts, axis=0)  # [6144 + 2048 + 16 + 16, 1024]
            L["conv_w"] = g(a + "conv1d.weight").moveaxis(2, 1)  # [conv_dim, 4, 1]
            L["A_log"] = g(a + "A_log")
            L["dt_bias"] = g(a + "dt_bias")
            L["gnorm"] = g(a + "norm.weight")
            L["out_proj"] = g(a + "out_proj.weight")
        else:
            a = p + "self_attn."
            L["qkv"] = mx.concatenate(
                [g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")], axis=0)
            L["q_norm"] = g(a + "q_norm.weight") + 1.0
            L["k_norm"] = g(a + "k_norm.weight") + 1.0
            L["o_proj"] = g(a + "o_proj.weight")
        W["layers"].append(L)
    mx.eval(W)
    return W


def new_state(cfg: Cfg, max_len: int):
    st = {"kv": [], "conv": [], "ssm": []}
    for i in range(cfg.layers):
        if cfg.is_linear(i):
            st["conv"].append(mx.zeros((1, cfg.conv_k - 1, cfg.conv_dim), mx.bfloat16))
            st["ssm"].append(mx.zeros((1, cfg.hv, cfg.dv, cfg.dk), mx.float32))
        else:
            st["kv"].append((mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16),
                             mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16)))
    mx.eval(st)
    return st


def _mlp(L, x):
    gu = x @ L["gate_up"].T
    g, u = mx.split(gu, 2, axis=-1)
    return (nn.silu(g) * u) @ L["down"].T


def _linear_attn(cfg, L, x, conv_state, ssm_state):
    B, S, _ = x.shape
    kd, vd = cfg.hk * cfg.dk, cfg.hv * cfg.dv
    proj = x @ L["in_proj"].T
    qkv, z, b, a = mx.split(proj, [2 * kd + vd, 2 * kd + 2 * vd, 2 * kd + 2 * vd + cfg.hv], axis=-1)
    z = z.reshape(B, S, cfg.hv, cfg.dv)
    conv_in = mx.concatenate([conv_state, qkv], axis=1)
    new_conv = conv_in[:, -(cfg.conv_k - 1):, :]
    conv_out = nn.silu(mx.conv1d(conv_in, L["conv_w"], groups=cfg.conv_dim))
    q, k, v = mx.split(conv_out, [kd, 2 * kd], axis=-1)
    q = q.reshape(B, S, cfg.hk, cfg.dk)
    k = k.reshape(B, S, cfg.hk, cfg.dk)
    v = v.reshape(B, S, cfg.hv, cfg.dv)
    q, k = normalize_qk(q, k, inv_scale=cfg.dk ** -0.5, eps=1e-6)
    out, new_ssm = gated_delta_update(q, k, v, a, b, L["A_log"], L["dt_bias"], ssm_state)
    # gated RMSNorm: silu(z) * rmsnorm(out) computed in fp32
    n = mx.fast.rms_norm(out, L["gnorm"], cfg.eps)
    out = (nn.silu(z.astype(mx.float32)) * n.astype(mx.float32)).astype(x.dtype)
    return out.reshape(B, S, -1) @ L["out_proj"].T, new_conv, new_ssm


def _mm(x, w, name, res=None):
    y = x @ w.T
    return y if res is None else res + y


def _full_attn(cfg, L, x, kv, pos, kv_len, mask, mm=_mm, res=None):
    """`mm(x, w, name[, res=r])` computes x @ w.T (+ r; the prefill path swaps in its tuned GEMMs); with `res` the returned
    projection already carries the residual add."""
    B, S, _ = x.shape
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    qkv = mm(x, L["qkv"], "qkv")
    q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
    q, gate = mx.split(q.reshape(B, S, H, 2 * D), 2, axis=-1)
    gate = gate.reshape(B, S, -1)
    q = mx.fast.rms_norm(q, L["q_norm"], cfg.eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(B, S, KV, D), L["k_norm"], cfg.eps).transpose(0, 2, 1, 3)
    v = v.reshape(B, S, KV, D).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    K, V = kv
    K = mx.slice_update(K, k, pos_arr(pos), (2,)) if False else _write(K, k, pos)
    V = _write(V, v, pos)
    o = mx.fast.scaled_dot_product_attention(
        q, K[:, :, :kv_len], V[:, :, :kv_len], scale=D ** -0.5, mask=mask)
    o = o.transpose(0, 2, 1, 3).reshape(B, S, -1)
    o = o * mx.sigmoid(gate)
    return (mm(o, L["o_proj"], "out_proj") if res is None else mm(o, L["o_proj"], "out_proj", res=res)), (K, V)


def pos_arr(pos):
    return pos if isinstance(pos, mx.array) else mx.array(pos)


def _write(cache, new, pos):
    """cache[:, :, pos:pos+S] = new (pos may be a traced scalar array)."""
    return mx.slice_update(cache, new, mx.reshape(pos_arr(pos), (1,)).astype(mx.int32), (2,))


def forward(cfg: Cfg, W, tokens, state, pos, kv_len):
    """tokens [1,S] int; pos: python int or int32 scalar array; kv_len: static int >= pos+S.
    Returns (hidden [1,S,H] after final norm, new_state)."""
    S = tokens.shape[1]
    x = W["embed"][tokens]
    qpos = pos_arr(pos) + mx.arange(S)
    mask = (mx.arange(kv_len)[None, :] <= qpos[:, None])  # [S, kv_len] bool
    new = {"kv": [], "conv": [], "ssm": []}
    ki = ci = 0
    for i, L in enumerate(W["layers"]):
        h = mx.fast.rms_norm(x, L["ln1"], cfg.eps)
        if cfg.is_linear(i):
            r, c, s = _linear_attn(cfg, L, h, state["conv"][ci], state["ssm"][ci])
            new["conv"].append(c)
            new["ssm"].append(s)
            ci += 1
        else:
            r, kv = _full_attn(cfg, L, h, state["kv"][ki], pos, kv_len, mask)
            new["kv"].append(kv)
            ki += 1
        x = x + r
        x = x + _mlp(L, mx.fast.rms_norm(x, L["ln2"], cfg.eps))
    return mx.fast.rms_norm(x, W["norm"], cfg.eps), new


def logits_of(W, h):
    return h @ W["embed"].T
