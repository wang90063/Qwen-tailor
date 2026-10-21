"""Decode step throughput: compiler output (fused / unfused lowering) vs hand-written fast.verify, M=1."""
import sys, time, statistics
import mlx.core as mx
from qwen35 import fast
from qwen35.model import Cfg, load_weights
from qwen35.compiler import Compiled

cfg = Cfg(); W = load_weights(cfg); F = fast.prepare(W, cfg)
sel = mx.zeros((1,), dtype=mx.int32)

def run(step, n=100):
    st = fast.new_state(cfg, 512)
    tok = mx.array([1000], dtype=mx.int32)
    for i in range(10):
        tok, _, st = step(tok, st, mx.array(10 + i, dtype=mx.int32))
    mx.eval(tok)
    ts = []
    for rep in range(5):
        t = time.perf_counter()
        for i in range(n):
            tok, _, st = step(tok, st, mx.array(20 + i, dtype=mx.int32))
            mx.async_eval(tok)
        mx.eval(tok)
        ts.append(n / (time.perf_counter() - t))
    return statistics.median(ts)

hand = mx.compile(lambda F_, t, s, sl, p: fast.verify(cfg, F_, t, s, sl, p, 256))
print(f"hand-written fast.verify          {run(lambda t, s, p: hand(F, t, s, sel, p)):6.1f} tok/s")
for name, fz in (("compiler, unfused lowering", False), ("compiler, fused", True), ("compiler, fused (again)", True)):
    c = Compiled(W, cfg, fuse_pass=fz)
    print(f"{name:34s}{run(lambda t, s, p: c.verify(t, s, sel, p, 256)):6.1f} tok/s   ({c.n_kernels()} graph nodes)")
