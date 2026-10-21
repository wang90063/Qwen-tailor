"""Marginal in-graph ablation of the batched decode step: replace one kernel family by a cheap broadcast stub (right shape / dtype) and
measure the compiled step without it.  `python bench/step_ablate.py B`.  Caveat: stubs change the downstream data (e.g. a degenerate
Hn makes the pruned head collect many candidates), so a family whose stub feeds data-dependent kernels can read too slow (out_proj,
argmax_head): trust the all-GEMV-families row and the families that only feed fixed-cost kernels."""
import sys, time, statistics
import numpy as np, mlx.core as mx
from quiet import Probe
from qwen35.engine import Engine
from qwen35 import batch as Bt
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
LM, POS = 4096, 600
eng = Engine(max_len=LM); cfg, F = eng.cfg, eng.F
P = Bt.Pools(cfg, 16, LM); tree = P.as_tree()
rng = np.random.default_rng(0)
cur = mx.array(rng.integers(1000, 30000, 17).astype(np.int32)); mx.eval(cur)
sl = mx.array(np.arange(B, dtype=np.int32)); ps = mx.array(rng.integers(POS // 2, POS, B).astype(np.int32))
def kind_fn(aa, kk):
    w0 = aa[0][0] if isinstance(aa[0], tuple) else aa[0]
    n = w0.shape[0]
    if n == 8224: return "in_proj"
    if n == 5120 or (n == 5120 + 0): return "qkv"
    if n == 7168: return "gate_up_il"
    K = aa[1].shape[-1]
    if K == 3584: return "down"
    return "out_proj" if kk.get("T") == 256 else "o_proj"
orig = {n: getattr(Bt, n) for n in ("fused_gemv", "deltanet_b", "attn_b", "rms_rows", "gnorm_rows", "argmax_head")}
shapes = {}
def first_arr(args):
    for a in args:
        if isinstance(a, mx.array): return a
def cheap(ref, shp):
    def one(s, d): return mx.broadcast_to(ref.reshape(-1)[:1].astype(d), s)
    if isinstance(shp, list): return tuple(one(s, d) for s, d in shp)
    return one(*shp)
def rec(name):
    def f(*a, **k):
        key = (name, kind_fn(a, k)) if name == "fused_gemv" else (name, "")
        o = orig[name](*a, **k)
        shapes[key] = [(t.shape, t.dtype) for t in o] if isinstance(o, tuple) else (o.shape, o.dtype)
        return o
    return f
for n in orig: setattr(Bt, n, rec(n))
Bt.step(cfg, F, tree, cur, sl, ps)
mx.eval(cur)
print({k: (v if not isinstance(v, list) else len(v)) for k, v in shapes.items()}, flush=True)
def stub(name):
    def f(*a, **k):
        key = (name, kind_fn(a, k)) if name == "fused_gemv" else (name, "")
        ref = first_arr(a) if name != "fused_gemv" else a[1]
        return cheap(ref, shapes[key])
    return f
def make(abl):
    for n, o in orig.items(): setattr(Bt, n, o)
    for a in abl:
        if a.startswith("gemv:"):
            kd = a[5:]
            prev = Bt.fused_gemv
            def f2(*aa, _kd=kd, _prev=prev, **kk):
                if kind_fn(aa, kk) == _kd: return cheap(aa[1], shapes[("fused_gemv", _kd)])
                return _prev(*aa, **kk)
            Bt.fused_gemv = f2
        else:
            setattr(Bt, a, stub(a))
    return mx.compile(lambda F_, P_, c, s, p: Bt.step(cfg, F_, P_, c, s, p))
probe = Probe(tol=1.15)
def run(fn, n=20):
    c = cur; t = time.perf_counter()
    for _ in range(n):
        ids, c = fn(F, tree, c, sl, ps); mx.async_eval(ids, c)
    mx.eval(ids); return (time.perf_counter() - t) / n
variants = [[]] + [[f"gemv:{k}"] for k in ("in_proj", "out_proj", "qkv", "o_proj", "gate_up_il", "down")] + [["gemv:in_proj", "gemv:out_proj", "gemv:qkv", "gemv:o_proj", "gemv:gate_up_il", "gemv:down"]] + [[n] for n in ("deltanet_b", "attn_b", "rms_rows", "gnorm_rows", "argmax_head")]
fns = []
for v in variants:
    fn = make(v); run(fn, 3); run(fn, 3); fns.append(fn)
res = {i: [] for i in range(len(variants))}
for rnd in range(9):
    for i, fn in enumerate(fns):
        dt, ok = probe.sample(lambda: run(fn))
        if ok: res[i].append(dt)
base = statistics.median(res[0]) * 1e3 if res[0] else float("nan")
print(f"B={B} base {base:.2f} ms ({len(res[0])} quiet)")
for i, v in enumerate(variants[1:], 1):
    if res[i]: print(f"  without {'+'.join(v):70s} {statistics.median(res[i])*1e3:6.2f} ms   -> costs {base - statistics.median(res[i])*1e3:5.2f} ms  ({len(res[i])})")
