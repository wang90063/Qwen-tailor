"""Acceptance of chained MTP drafts: d1 from (h_main, x1); d2 from (mtp_hidden, d1) ..."""
import sys
from pathlib import Path
exec((Path(__file__).resolve().parent / "mtp_accept.py").read_text().split("PROMPTS =")[0])
from qwen35.model import _full_attn, _mlp

def mtp_core(h, toks, kv, pos, kv_len, mask):
    e = mx.fast.rms_norm(W["embed"][toks], M["ln_e"], cfg.eps)
    hh = mx.fast.rms_norm(h, M["ln_h"], cfg.eps)
    x = mx.concatenate([e, hh], axis=-1) @ M["fc"].T
    r, kv = _full_attn(cfg, M, mx.fast.rms_norm(x, M["ln1"], cfg.eps), kv, pos, kv_len, mask)
    x = x + r
    x = x + _mlp(M, mx.fast.rms_norm(x, M["ln2"], cfg.eps))
    return x, kv        # pre-final-norm

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Explain in two sentences why the sky is blue.", "Write a short story about a robot learning to paint.",
           "List five tips for learning a new programming language.", "Translate to French: The weather is nice today and I want to go for a walk.",
           "import numpy as np\n\ndef softmax(x):\n", "Q: What is 17 * 23?\nA:"]
N = 120
tot = {"a1": [], "a2|a1": [], "a2": [], "a3|a2": [], "a3": []}
for variant in ("post", "pre"):
    c = {"n": 0, "d1": 0, "d2": 0, "d2and1": 0, "d3": 0, "d3and": 0}
    for p in PROMPTS:
        ids = tk.encode(p); out = eng.generate(ids, N); seq = ids + out; S = len(seq)
        st = new_state(cfg, 1024)
        h, _ = forward(cfg, W, mx.array([seq]), st, 0, len(seq))
        # true MTP batch pass: build KV with true entries
        mask = (mx.arange(S)[None, :] <= mx.arange(S)[:, None])
        kv0 = (mx.zeros((1, 2, S + 8, 256), mx.bfloat16), mx.zeros((1, 2, S + 8, 256), mx.bfloat16))
        xs, kvt = mtp_core(h[:, :S - 1], mx.array([seq[1:S]]), kv0, 0, S, mask[: S - 1, :])
        for q in range(len(ids) - 1, S - 4):
            # q: main hidden position; seq[q+1] known. draft1 predicts seq[q+2]
            x1 = xs[:, q:q + 1]
            lg1 = mx.fast.rms_norm(x1, M["norm"], cfg.eps) @ W["embed"].T
            d1 = int(mx.argmax(lg1[0, 0]).item())
            c["n"] += 1; ok1 = d1 == seq[q + 2]; c["d1"] += ok1
            # chain: input hidden = x1 (post/pre norm), token d1 at position q+1
            hin = mx.fast.rms_norm(x1, M["norm"], cfg.eps) if variant == "post" else x1
            m2 = (mx.arange(q + 2)[None, :] <= (q + 1))
            x2, kv2 = mtp_core(hin, mx.array([[d1]]), kvt, q + 1, q + 2, m2)
            d2 = int(mx.argmax((mx.fast.rms_norm(x2, M["norm"], cfg.eps) @ W["embed"].T)[0, 0]).item())
            ok2 = d2 == seq[q + 3]
            if ok1: c["d2and1"] += 1; c["d2"] += ok2
    print(variant, "n=%d  a1=%.3f  a2|a1=%.3f  (cum a1*a2|a1 = %.3f)" % (c["n"], c["d1"] / c["n"], c["d2"] / max(1, c["d2and1"]), c["d2"] / c["n"]))
