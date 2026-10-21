"""Measure native-MTP draft acceptance (greedy) on text the main model itself generated."""
import sys, itertools
import mlx.core as mx, mlx.nn as nn
from transformers import AutoTokenizer
from qwen35.model import *
from qwen35.model import _full_attn, _mlp
from qwen35.engine import Engine

from qwen35.model import ROOT as ROOT_S
tk = AutoTokenizer.from_pretrained(f"{ROOT_S}/models/Qwen3.5-0.8B")
cfg = Cfg(); eng = Engine(max_len=1024, fused=True); W = eng.W
raw = mx.load(str(next(MODEL_DIR.glob("*.safetensors"))))
g = lambda k: raw["mtp." + k]
M = {"fc": g("fc.weight"), "ln_e": g("pre_fc_norm_embedding.weight") + 1, "ln_h": g("pre_fc_norm_hidden.weight") + 1,
     "norm": g("norm.weight") + 1, "ln1": g("layers.0.input_layernorm.weight") + 1, "ln2": g("layers.0.post_attention_layernorm.weight") + 1,
     "qkv": mx.concatenate([g("layers.0.self_attn.q_proj.weight"), g("layers.0.self_attn.k_proj.weight"), g("layers.0.self_attn.v_proj.weight")], axis=0),
     "q_norm": g("layers.0.self_attn.q_norm.weight") + 1, "k_norm": g("layers.0.self_attn.k_norm.weight") + 1,
     "o_proj": g("layers.0.self_attn.o_proj.weight"),
     "gate_up": mx.concatenate([g("layers.0.mlp.gate_proj.weight"), g("layers.0.mlp.up_proj.weight")], axis=0),
     "down": g("layers.0.mlp.down_proj.weight")}

def mtp_batch(h, toks, order, norm_first):
    """h [1,S,H] hidden at positions 0..S-1; toks [1,S] = token at position p+1. Returns logits [1,S,V]."""
    S = h.shape[1]
    e = mx.fast.rms_norm(W["embed"][toks], M["ln_e"], cfg.eps)
    hh = mx.fast.rms_norm(h, M["ln_h"], cfg.eps)
    cat = mx.concatenate([e, hh] if order == "eh" else [hh, e], axis=-1)
    x = cat @ M["fc"].T
    pos = 0
    mask = (mx.arange(S)[None, :] <= mx.arange(S)[:, None])
    r, _ = _full_attn(cfg, M, mx.fast.rms_norm(x, M["ln1"], cfg.eps), (mx.zeros((1, 2, S, 256), mx.bfloat16), mx.zeros((1, 2, S, 256), mx.bfloat16)), 0, S, mask)
    x = x + r
    x = x + _mlp(M, mx.fast.rms_norm(x, M["ln2"], cfg.eps))
    return mx.fast.rms_norm(x, M["norm"], cfg.eps) @ W["embed"].T

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Explain in two sentences why the sky is blue.", "Write a short story about a robot learning to paint.",
           "List five tips for learning a new programming language.", "Translate to French: The weather is nice today and I want to go for a walk.",
           "import numpy as np\n\ndef softmax(x):\n", "Q: What is 17 * 23?\nA:"]
import numpy as np
fr = np.load(f"{ROOT_S}/results/token_freq_corpus.npy"); order = np.argsort(-fr)
SETS = {}
for K in (8192, 16384, 32768):
    SETS[f"freq{K}"] = mx.array(np.sort(order[:K]).astype(np.int32))
    SETS[f"freq{K}+id8k"] = mx.array(np.union1d(order[:K], np.arange(8192)).astype(np.int32))
order_np = order
N = 160
res = {}
for p in PROMPTS:
    ids = tk.encode(p)
    out = eng.generate(ids, N)
    seq = ids + out
    st = new_state(cfg, 1024)
    h, _ = forward(cfg, W, mx.array([seq]), st, 0, len(seq))     # post-norm hidden at every position
    S = len(seq)
    # position p predicts seq[p+1] (main), MTP at p (given seq[p+1]) predicts seq[p+2]
    n0_ = len(ids) - 1
    hp = h[:, : S - 2]; tp = mx.array([seq[1 : S - 1]]); want = mx.array(seq[2:S])
    # dynamic: static top-K freq set + tokens already seen in the context (causal)
    lgd = None
    for order in ("eh", "he"):
        if order == "he": continue
        lg = mtp_batch(hp, tp, order, True)
        pred = mx.argmax(lg, axis=-1)[0]
        lgn = np.array(lg.astype(mx.float32))[0]
        for K in (8192, 16384, 32768):
            stat = np.zeros(248320, bool); stat[order_np[:K]] = True
            ok = 0; tot = 0
            seen = np.zeros(248320, bool); seen[seq[:n0_ + 1]] = True
            for q in range(n0_, S - 2):
                seen[seq[q + 1]] = True      # token at p+1 is known when drafting at p
                allowed = stat | seen
                row = np.where(allowed, lgn[q], -1e30)
                ok += int(row.argmax() == seq[q + 2]); tot += 1
            res.setdefault(f"static{K}+ctx", []).append(ok / tot)
        for nm, idx in SETS.items():
            pv = idx[mx.argmax(lg[..., idx], axis=-1)[0]]
            res.setdefault(nm, []).append((pv[n0_:] == want[n0_:]).astype(mx.float32).mean().item())
        n0 = len(ids) - 1  # only score generated region
        acc = (pred[n0:] == want[n0:]).astype(mx.float32).mean().item()
        res.setdefault(order, []).append(acc)
        print(f"{p[:30]!r:34s} order={order} acceptance {acc*100:5.1f}%")
for k, v in res.items(): print(k, "mean acceptance %.1f%%" % (100 * sum(v) / len(v)))
