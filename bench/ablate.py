"""Scratch (M1 era): compiled trunk time with the gated-delta recurrence / conv1d stubbed out.  Superseded by step_ablate.py and prefill_marginal.py."""
import sys, time, statistics
import mlx.core as mx
import qwen35.model as M
cfg = M.Cfg(); W = M.load_weights(cfg); st = M.new_state(cfg, 256)
tok = mx.array([[1000]]); pos = mx.array(100, dtype=mx.int32)
def tm(f, n=60):
    for _ in range(5): mx.eval(f())
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    return statistics.median(ts)*1e3
def run(label):
    f = mx.compile(lambda W,tok,pos,st: M.forward(cfg, W, tok, st, pos, 256))
    print(f"{label:34s} {tm(lambda: f(W,tok,pos,st)):.2f} ms")
run("full trunk")
orig = M.gated_delta_update
M.gated_delta_update = lambda q,k,v,a,b,A,dt,s: (v, s)
run("no delta recurrence")
M.gated_delta_update = orig
oc = mx.conv1d
M.mx.conv1d = lambda x,w,groups: x[:, :1]  # placeholder cheap
run("no conv (+no delta? no)")
