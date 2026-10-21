"""In-graph *marginal* cost of prefill building blocks: run each wrapped op an extra time (same inputs, result consumed by a tiny
reduction that is evaluated with the output) and compare with the unmodified forward, alternating on a quiet GPU.
usage: prefill_marginal.py S ROUNDS op [op ...]    op in: conv_prep gdn gated_norm add_norm swiglu_gemm(gate_up) lm_head"""
import sys, time, statistics
import mlx.core as mx
from quiet import Probe
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35 import model as m, prefill as pf, gemm_nax as G, gdn_nax as gdn
S = int(sys.argv[1]); ROUNDS = int(sys.argv[2]); OPS = sys.argv[3:]
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
ids = mx.array([make_prompt(tk, S)])
sink = []
orig = {"conv_prep": pf.conv_prep, "gdn": gdn.gdn_chunked, "gated_norm": pf.gated_norm, "add_norm": pf.add_norm, "swiglu_gemm": G.linear}


def tiny(a):
    a = a[0] if isinstance(a, (tuple, list)) else a
    return a.reshape(-1)[:1].astype(mx.float32)


def wrap(op):
    f = orig[op]
    if op == "swiglu_gemm":
        def g(x, w, name, epi=G.EPI_PLAIN, r=None):
            y = f(x, w, name, epi=epi, r=r)
            if epi == G.EPI_SWIGLU:
                sink.append(tiny(f(x, w, name, epi=epi, r=r)))
            return y
        return g
    def g(*a, **kw):
        y = f(*a, **kw)
        sink.append(tiny(f(*a, **kw)))
        return y
    return g


def install(op):
    pf.conv_prep, gdn.gdn_chunked, pf.gated_norm, pf.add_norm, G.linear = (orig["conv_prep"], orig["gdn"], orig["gated_norm"], orig["add_norm"], orig["swiglu_gemm"])
    if op == "conv_prep": pf.conv_prep = wrap(op)
    elif op == "gdn": gdn.gdn_chunked = wrap(op)
    elif op == "gated_norm": pf.gated_norm = wrap(op)
    elif op == "add_norm": pf.add_norm = wrap(op)
    elif op == "swiglu_gemm": G.linear = wrap(op)


def run(op):
    install(op if op != "base" else None)
    sink.clear()
    st = m.new_state(cfg, S + 8)
    t = time.perf_counter()
    h, st = pf.forward(cfg, W, ids, st, 0, S)
    mx.eval(h[:, -1] @ W["embed"].T, st, *sink)
    return time.perf_counter() - t


VARS = ["base"] + OPS
for v in VARS: run(v)
probe = Probe()
ts = {v: [] for v in VARS}
for _ in range(ROUNDS):
    for v in VARS:
        dt, ok = probe.sample(lambda: run(v))
        if ok: ts[v].append(dt)
base = statistics.median(ts["base"])
print(f"base: median {base * 1e3:.1f} ms (n={len(ts['base'])})")
for v in OPS:
    if ts[v]:
        med = statistics.median(ts[v])
        print(f"+ extra {v:12s}: median {med * 1e3:.1f} ms (n={len(ts[v])}) -> marginal {(med - base) * 1e3:+.2f} ms total")
