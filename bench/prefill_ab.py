"""A/B the prefill GEMM implementation inside one process (alternating, so machine noise cancels).
usage: prefill_ab.py [S] [rounds] [variant ...]   variant = comma separated GEMM names kept on mx.matmul ("all" = everything
on mx.matmul, "none" = everything on the NAX GEMMs); default: all vs none"""
import sys, time, statistics
import mlx.core as mx
from quiet import Probe
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35.head_prune import argmax_head, build_tables
from qwen35 import model as m, prefill as pf
S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
TAB = build_tables(W["embed"])                  # first token as Engine.prefill computes it
ids = mx.array([make_prompt(tk, S)])


VARIANTS = sys.argv[3:] or ["all", "none"]


def run(impl):
    pf.GEMM_IMPL = "mlx" if impl == "all" else "nax"
    pf.GEMM_OFF = frozenset() if impl in ("all", "none") else frozenset(impl.split(","))
    st = m.new_state(cfg, S + 8)
    t = time.perf_counter()
    h, st = pf.forward(cfg, W, ids, st, 0, S)
    mx.eval(argmax_head(W["embed"], TAB, h[0, -1:]), st)
    return time.perf_counter() - t


for impl in VARIANTS:
    run(impl)
probe = Probe()
ts = {v: [] for v in VARIANTS}
bad = {v: 0 for v in VARIANTS}
for _ in range(ROUNDS):
    for impl in VARIANTS:
        dt, ok = probe.sample(lambda: run(impl))
        ts[impl].append(dt); bad[impl] += (not ok)
for impl in VARIANTS:
    med = statistics.median(ts[impl]); best = min(ts[impl])
    print(f"S={S} off={impl:28s}: median {med * 1e3:.1f} ms ({S / med:.0f} tok/s)  best {best * 1e3:.1f} ms ({S / best:.0f} tok/s)"
          f"  [{ROUNDS - bad[impl]}/{ROUNDS} quiet]")
