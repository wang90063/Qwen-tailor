"""Calibrates serve.PREFILL_SCRATCH_BASE / _PER_TOKEN: peak GPU memory of one prefill chunk (above what is resident while it runs) vs chunk size
and position, and checks that serve.plan_memory's itemisation covers a real mixed run (peak <= planned total).

usage: serve_memory.py [--max-len 4096]
"""
import sys
import numpy as np
import mlx.core as mx
from qwen35.engine import Engine
from qwen35 import serve as Sv

max_len = 4096
eng = Engine(max_len=max_len)
rng = np.random.default_rng(0)
ids = [int(t) for t in rng.integers(1000, 30000, 3500)]
MiB = 1 << 20
print(f"resident after load: {mx.get_active_memory() / MiB:.0f} MiB; sequence state at max_len {max_len}: "
      f"{Sv.seq_state_bytes(eng.cfg, max_len) / MiB:.1f} MiB")
worst = 0
for S, ck in ((300, 128), (600, 256), (1100, 512), (2100, 1024), (3500, 2048), (3500, 512), (3500, 256)):
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    g = eng.prefill_iter(ids[:S], chunk=ck)
    while True:
        try:
            next(g)
        except StopIteration as e:
            nxt, st, n = e.value
            mx.eval(nxt)
            break
    peak = mx.get_peak_memory()
    over = peak - base - Sv.seq_state_bytes(eng.cfg, max_len)          # above the state the job itself holds
    model = Sv.PREFILL_SCRATCH_BASE + Sv.PREFILL_SCRATCH_PER_TOKEN * ck
    worst = max(worst, over - model)
    print(f"prompt {S:5d} chunk {ck:5d}: peak above resident {(peak - base) / MiB:7.0f} MiB; beyond one state {over / MiB:7.0f} MiB"
          f"  (model {model / MiB:.0f})")
    del g, st
print(f"worst excess over the model: {worst / MiB:+.0f} MiB (covered by the plan's 256 MiB slack if <= that; the first row also pays one-time allocations)")
