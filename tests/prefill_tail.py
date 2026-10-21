"""prefill.forward(tail=R): the last layer only does the work of the last R rows.  Every K / V / state must be bit-identical to the
full computation, the returned rows too whenever SDPA tiles the same way (MLX picks its attention tiling from the query length, so a
shorter tail can differ by a bf16 ulp in some elements: those are bounded by rel-L2 instead).  R < 96 uses mx.matmul (not tested)."""
import sys
import numpy as np
import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from qwen35 import model as m, prefill as pf

cfg = m.Cfg(); W = m.load_weights(cfg)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
# frozen corpus (installed mlx_lm sources): the project's own files change with every edit and so would the prompts (and with them the
# rel-L2 of the non-bit-identical rows, which sits close to the bound)
import mlx_lm
text = "\n".join(p.read_text() for p in sorted(Path(mlx_lm.__file__).parent.glob("models/*.py"))[:12])
IDS = tk(text)["input_ids"]
fails = 0


def same(a, b):
    return np.array_equal(np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16)))


def check(label, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + label)
    fails += 0 if ok else 1


def rel(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - b) ** 2) / mx.sum(a * a)))


def states_same(s1, s2):
    return (all(same(a, b) for a, b in zip(s1["ssm"], s2["ssm"])) and all(same(a, b) for a, b in zip(s1["conv"], s2["conv"]))
            and all(same(k1, k2) and same(v1, v2) for (k1, v1), (k2, v2) in zip(s1["kv"], s2["kv"])))


for S, R in [(130, 128), (500, 128), (500, 500), (2048, 128), (2048, 512), (2048, 2048), (96, 96), (300, 0), (2048, 0)]:
    ids = mx.array([IDS[:S]])
    st = m.new_state(cfg, S + 8)
    h_full, st_full = pf.forward(cfg, W, ids, st, 0, S)
    h_t, st_t = pf.forward(cfg, W, ids, m.new_state(cfg, S + 8), 0, S, tail=R)
    mx.eval(h_full, st_full, st_t) if h_t is None else mx.eval(h_full, h_t, st_full, st_t)
    ref_rows = None if h_t is None else h_full[:, S - min(R, S):]
    e = 0.0 if h_t is None else rel(ref_rows, h_t)
    check(f"S={S} tail={R}: hidden rows {'(none)' if h_t is None else 'bit-identical' if e == 0 and same(ref_rows, h_t) else f'rel-L2 {e:.1e}'}", e < 5e-3)
    check(f"S={S} tail={R}: kv / conv / ssm state bit-identical", states_same(st_full, st_t))

# chunked prefill (second chunk uses the explicit mask, first chunk needs no output)
S1, S2 = 700, 300
st_f = m.new_state(cfg, S1 + S2 + 8); st_t = m.new_state(cfg, S1 + S2 + 8)
a, st_f = pf.forward(cfg, W, mx.array([IDS[:S1]]), st_f, 0, S1)
b, st_f = pf.forward(cfg, W, mx.array([IDS[S1:S1 + S2]]), st_f, S1, S1 + S2)
a2, st_t = pf.forward(cfg, W, mx.array([IDS[:S1]]), st_t, 0, S1, tail=0)
b2, st_t = pf.forward(cfg, W, mx.array([IDS[S1:S1 + S2]]), st_t, S1, S1 + S2, tail=128)
mx.eval(b, b2, st_f, st_t)
check("chunked: last chunk rows", rel(b[:, -128:], b2) < 5e-3)
check("chunked: state bit-identical", states_same(st_f, st_t))
print("FAILURES:", fails)
sys.exit(1 if fails else 0)
