"""Native MTP head (mtp_num_hidden_layers=1): one full-attention decoder layer predicting token t+2 from
(main hidden at t, token t+1)."""
import os

import mlx.core as mx
import numpy as np

from . import wcomp
from .kernels import fused_gemv, subset_argmax, qsubset_argmax
from .model import Cfg, _full_attn, _mlp, _write, ROOT, weights_file


DRAFT_Q8 = os.environ.get("QWEN35_DRAFT_Q8", "1") != "0"


def load_mtp(cfg: Cfg):
    raw = mx.load(str(weights_file()))
    g = lambda k: raw["mtp." + k]
    p = "layers.0."
    gate, up = g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")
    n = gate.shape[0]
    M = {
        "fc": g("fc.weight"),                       # [1024, 2048]  input = [norm(embed), norm(hidden)]
        "ln_e": g("pre_fc_norm_embedding.weight") + 1.0, "ln_h": g("pre_fc_norm_hidden.weight") + 1.0,
        "norm": g("norm.weight") + 1.0,
        "ln1": g(p + "input_layernorm.weight") + 1.0, "ln2": g(p + "post_attention_layernorm.weight") + 1.0,
        "qkv": mx.concatenate([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"),
                               g(p + "self_attn.v_proj.weight")], axis=0),
        "q_norm": g(p + "self_attn.q_norm.weight") + 1.0, "k_norm": g(p + "self_attn.k_norm.weight") + 1.0,
        "o_proj": g(p + "self_attn.o_proj.weight"),
        "gate_up": mx.concatenate([gate, up], axis=0),
        "gate_up_il": mx.stack([gate, up], axis=1).reshape(2 * n, -1),
        "down": g(p + "mlp.down_proj.weight"),
    }
    if wcomp.ENABLED:        # lossless 12-bit copies for the draft-step GEMVs (the plain ones stay for the batched prefill)
        for k in ("fc", "qkv", "o_proj", "gate_up_il", "down"):
            M[k + "_c"] = wcomp.encode(M[k])
    mx.eval(M)
    return M


def _w(M, k):
    return M.get(k + "_c", M[k])


def draft_vocab(embed, k_static=16384, path=None):
    """Static sub-vocabulary for the draft head: most frequent tokens of a local text corpus (see
    bench/vocab_freq.py). Returns (Wsub [Vs,H], idmap [Vs] int32, nvalid [1] int32)."""
    fr = np.load(path or ROOT / "results" / "token_freq_corpus.npy")
    idx = np.sort(np.argsort(-fr)[:k_static]).astype(np.int32)
    ids = mx.array(idx)
    Wsub = embed[ids]
    nvalid = mx.array([len(idx)], dtype=mx.int32)
    if DRAFT_Q8:             # int8 per-row copy: the draft head is approximate by design, half the bytes
        e = Wsub.astype(mx.float32)
        sc = mx.maximum(mx.abs(e).max(-1) / 127.0, 1e-30)
        Wsub = (mx.clip(mx.round(e / sc[:, None]), -127, 127).astype(mx.int8), sc)
    mx.eval(Wsub, ids, nvalid)
    return Wsub, ids, nvalid


def prefill_kv(cfg: Cfg, M, embed, Hn, toks, max_len, kv=None, start=0):
    """Build the MTP layer's KV cache from a prompt in one batch (reference ops). Hn [S,H] post-norm main
    hidden at positions start..start+S-1; toks [S] tokens at positions start+1..start+S (token at p+1 for position p).
    Fills positions start..start+S-1 of `kv` (None = fresh zeros; a prefix cache resumes from the rows [0, start) of an earlier call).
    Returns (K, V)."""
    S = Hn.shape[0]
    e = mx.fast.rms_norm(embed[toks], M["ln_e"], cfg.eps)
    hh = mx.fast.rms_norm(Hn, M["ln_h"], cfg.eps)
    x = (mx.concatenate([e, hh], axis=-1) @ M["fc"].T)[None]
    mask = mx.arange(start + S)[None, :] <= (start + mx.arange(S))[:, None]
    if kv is None:
        kv = (mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16), mx.zeros((1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16))
    r, kv = _full_attn(cfg, M, mx.fast.rms_norm(x, M["ln1"], cfg.eps), kv, start, start + S, mask)
    mx.eval(kv)
    return kv


def step(cfg: Cfg, M, embed, toks, hid, kv, pos, kv_len):
    """MTP layer on m rows (m<=4): token ids toks [m] (token at position p+1), main/MTP hidden hid [m,H]
    (post-norm) for positions pos..pos+m-1. Writes the MTP KV at those positions.
    Returns (x [m,H] pre-final-norm, new kv)."""
    m = toks.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    e = mx.fast.rms_norm(embed[toks], M["ln_e"], cfg.eps)
    hh = mx.fast.rms_norm(hid, M["ln_h"], cfg.eps)
    x = fused_gemv(_w(M, "fc"), mx.concatenate([e, hh], axis=-1))
    mask = mx.arange(kv_len)[None, :] <= (pos + mx.arange(m))[:, None]
    qkv = fused_gemv(_w(M, "qkv"), x, nw=M["ln1"])
    q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
    q, gate = mx.split(q.reshape(1, m, H, 2 * D), 2, axis=-1)
    gate = gate.reshape(m, -1)
    q = mx.fast.rms_norm(q, M["q_norm"], cfg.eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, m, KV, D), M["k_norm"], cfg.eps).transpose(0, 2, 1, 3)
    v = v.reshape(1, m, KV, D).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    K_, V_ = kv
    K_ = _write(K_, k, pos); V_ = _write(V_, v, pos)
    o = mx.fast.scaled_dot_product_attention(q, K_[:, :, :kv_len], V_[:, :, :kv_len], scale=D ** -0.5, mask=mask)
    o = o.transpose(0, 2, 1, 3).reshape(m, -1) * mx.sigmoid(gate)
    x = fused_gemv(_w(M, "o_proj"), o, res=x)
    act = fused_gemv(_w(M, "gate_up_il"), x, nw=M["ln2"], swiglu=True)
    x = fused_gemv(_w(M, "down"), act, res=x)
    return x, (K_, V_)


def draft_token(cfg, M, Wsub, idmap, nvalid, x_row):
    """x_row [1,H] pre-final-norm MTP output -> (token id [1] int32, normed hidden [1,H])."""
    hn = mx.fast.rms_norm(x_row, M["norm"], cfg.eps)
    if isinstance(Wsub, tuple):
        return qsubset_argmax(Wsub, idmap, nvalid, hn), hn
    return subset_argmax(Wsub, idmap, nvalid, hn), hn
