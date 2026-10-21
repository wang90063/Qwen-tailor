"""M1-era decode breakdown: compiled trunk vs lm_head vs lm_head + argmax time, and the weight bytes of trunk and head."""
import sys, time, statistics
import mlx.core as mx
from qwen35.model import *
cfg = Cfg(); W = load_weights(cfg); st = new_state(cfg, 256)
tok = mx.array([[1000]]); 
def trunk(W, tok, pos, st): 
    h, n = forward(cfg, W, tok, st, pos, 256); return h, n
ct = mx.compile(trunk)
def tm(f, n=60):
    for _ in range(5): mx.eval(f())
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    return statistics.median(ts)*1e3, min(ts)*1e3
pos = mx.array(100, dtype=mx.int32)
print("trunk (compiled) ms med/min", tm(lambda: ct(W, tok, pos, st)))
h = mx.random.normal((1,1,1024)).astype(mx.bfloat16)
print("lm_head ms", tm(lambda: logits_of(W, h)))
print("lm_head+argmax ms", tm(lambda: mx.argmax(logits_of(W, h), axis=-1)))
# weight bytes of trunk
tb = sum(v.nbytes for L in W["layers"] for v in L.values())
print("trunk bytes GB", tb/1e9, "lm_head GB", W["embed"].nbytes/1e9)
