"""Prefill throughput / TTFT on the Engine path: fast forward with the last-layer tail path, then the certified int8 head for the
first token.  usage: prefill.py [--fast] [--mlx-head] [--no-tail] [S ...]   (--fast = NAX prefill; default = reference graph)"""
import sys, time, statistics
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from quiet import Probe
from qwen35 import model as m
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
from qwen35 import prefill as pf
from qwen35.engine import TAIL_ROWS
from qwen35.head_prune import argmax_head, build_tables
# first token as Engine.prefill computes it (certified int8 pre-filter); --mlx-head: the old full bf16 GEMV + argmax
TAB = None if "--mlx-head" in sys.argv else build_tables(W["embed"])
fast = "--fast" in sys.argv
fwd = pf.forward if fast else m.forward
kw = {"tail": TAIL_ROWS} if fast and "--no-tail" not in sys.argv else {}
sizes = [int(a) for a in sys.argv[1:] if a.isdigit()] or [128, 512, 2048]
probe = Probe(tol=1.18)
for n in sizes:
    ids = mx.array([make_prompt(tk, n)])
    def run():
        st = m.new_state(cfg, n + 8)
        t = time.perf_counter()
        h, st = fwd(cfg, W, ids, st, 0, n, **kw)
        mx.eval(mx.argmax(h[:, -1] @ W["embed"].T, axis=-1) if TAB is None else argmax_head(W["embed"], TAB, h[0, -1:]), st)
        return time.perf_counter() - t
    run(); run()
    ts = [probe.sample(run)[0] for _ in range(12)]
    med = statistics.median(ts)
    print(f"prefill {n}: {n/med:8.0f} tok/s  TTFT {med*1e3:.1f} ms   (best {min(ts)*1e3:.1f} ms)")
