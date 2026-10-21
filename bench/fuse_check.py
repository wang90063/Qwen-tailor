"""FUSE_ADD=1 must reproduce FUSE_ADD=0 bit for bit (final hidden + every state).  usage: fuse_check.py [S ...]"""
import sys
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35 import model as m, prefill as pf
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
bad = 0
for S in [int(a) for a in sys.argv[1:]] or [2048, 777, 300, 128, 97, 64, 17, 1]:
    ids = mx.array([make_prompt(tk, max(S, 8))[:S]])
    outs = {}
    for fuse in (False, True):
        pf.FUSE_ADD = fuse
        st = m.new_state(cfg, S + 8)
        h, st = pf.forward(cfg, W, ids, st, 0, S)
        mx.eval(h, st)
        outs[fuse] = (h, st)
    (h0, s0), (h1, s1) = outs[False], outs[True]
    same = bool(mx.all(h0 == h1).item())
    leaves0 = [a for k in ("kv", "conv", "ssm") for a in (sum(([*t] if isinstance(t, tuple) else [t] for t in s0[k]), []))]
    leaves1 = [a for k in ("kv", "conv", "ssm") for a in (sum(([*t] if isinstance(t, tuple) else [t] for t in s1[k]), []))]
    st_same = all(bool(mx.all(a == b).item()) for a, b in zip(leaves0, leaves1))
    print(f"S={S:5d}: hidden identical {same}, state identical {st_same}")
    bad += (not same) + (not st_same)
print("fuse_check:", "PASS" if bad == 0 else f"{bad} DIFFS")
sys.exit(1 if bad else 0)
