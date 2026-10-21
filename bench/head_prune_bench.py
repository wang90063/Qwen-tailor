"""Correctness + timing of the certified int8 lm_head prefilter (qwen35/head_prune.py) against kernels.head_argmax."""
import sys, time, statistics, glob
import numpy as np, mlx.core as mx
from transformers import AutoTokenizer
from qwen35 import model as m, head_prune as hp
from qwen35.kernels import head_argmax
from quiet import Probe

from qwen35.model import ROOT
cfg = m.Cfg(); W = m.load_weights(cfg); E = W["embed"]
tk = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-0.8B")
t0 = time.time(); tab = hp.build_tables(E); print(f"tables built in {time.time() - t0:.2f}s  q {tab['q'].nbytes / 1e6:.0f} MB")

text = "\n\n".join(open(f).read() for f in sorted(glob.glob(f"{ROOT}/qwen35/*.py")))
ids = tk(text, add_special_tokens=False)["input_ids"][:2048]
st = m.new_state(cfg, 2048 + 8)
H, _ = m.forward(cfg, W, mx.array([ids]), st, 0, 2048 + 8)
H = H[0, 16:]; mx.eval(H)
P = H.shape[0]
print(f"{P} real hidden states")

# --- correctness M=1 and M=4
bad = 0
for p in range(0, P, 1):
    x = H[p:p + 1]
    a = head_argmax(E, x); b = hp.head_argmax_pruned(E, tab, x)
    mx.eval(a, b)
    if int(a.item()) != int(b.item()):
        bad += 1; print("MISMATCH M=1 at", p, int(a.item()), int(b.item()))
print(f"M=1: {P - bad}/{P} identical")
bad = 0
for p in range(0, P - 4, 4):
    x = H[p:p + 4]
    a = head_argmax(E, x); b = hp.head_argmax_pruned(E, tab, x)
    mx.eval(a, b)
    if not bool(mx.all(a == b).item()):
        bad += 1; print("MISMATCH M=4 at", p, a.tolist(), b.tolist())
print(f"M=4: {(P - 4) // 4 + 1 - bad} identical batches, {bad} bad")

# --- timing
def timed(f, n=40):
    for _ in range(5): mx.eval(f())
    ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3, min(ts) * 1e3
x1 = H[100:101]; x4 = H[100:104]
for M, x in ((1, x1), (4, x4)):
    a = timed(lambda: head_argmax(E, x)); b = timed(lambda: hp.head_argmax_pruned(E, tab, x))
    print(f"M={M}: head_argmax {a[0]:.3f} ms (best {a[1]:.3f})   pruned {b[0]:.3f} ms (best {b[1]:.3f})")
