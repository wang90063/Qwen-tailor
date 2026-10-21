"""Sweep the (R, mtmax) tiling of every GEMV kind of the batched step for M = 5..16 rows on the real (compressed) weights.
All configurations compute bit-identical results (R and mtmax only change the register tiling); this picks the fastest and prints
the table to paste into batch.GEMV_TUNE.   usage: python bench/gemv_tune_m.py [M,M,...] [R,R,...] [mt,mt,...]"""
import sys, time, itertools, statistics
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0]); sys.path.insert(0, __file__.rsplit("/", 1)[0])
import mlx.core as mx
from qwen35.engine import Engine
from qwen35.kernels import fused_gemv, _tiles, _default_T
from qwen35 import batch as Bt

eng = Engine(max_len=512)
cfg, F = eng.cfg, eng.F
Ls = F["layers"]
lin = [i for i in range(24) if cfg.is_linear(i)]
att = [i for i in range(24) if not cfg.is_linear(i)]
mx.random.seed(0)
Ms = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 6, 7, 8, 10, 12, 14, 16]
X1024 = (mx.random.normal((16, 1024)) * .5).astype(mx.bfloat16)
X2048 = (mx.random.normal((16, 2048)) * .5).astype(mx.bfloat16)
X3584 = (mx.random.normal((16, 3584)) * .5).astype(mx.bfloat16)
RES = (mx.random.normal((16, 1024)) * .5).astype(mx.bfloat16)
# kind -> (weights, input rows, kwargs)
KINDS = {
    "in_proj": ([Ls[i]["in_proj"] for i in lin[:8]], X1024, {}),
    "qkv": ([Ls[i]["qkv"] for i in att], X1024, {}),
    "gate_up": ([Ls[i]["gate_up_il"] for i in range(8)], X1024, {"swiglu": True}),
    "down": ([Ls[i]["down"] for i in range(8)], X3584, {"res": "res"}),
    "out_proj": ([Ls[i]["out_proj"] for i in lin[:8]], X2048, {"res": "res"}),
    "o_proj": ([Ls[i]["o_proj"] for i in att], X2048, {"res": "res"}),
}


def tm(f):
    """min of 3 back-to-back runs (stage 1: coarse ranking)"""
    mx.eval(f()); mx.eval(f())
    ts = []
    for _ in range(3):
        t = time.perf_counter(); outs = f(); mx.eval(outs); ts.append((time.perf_counter() - t))
    return min(ts) / len(outs)


def once(f):
    t = time.perf_counter(); outs = f(); mx.eval(outs); return (time.perf_counter() - t) / len(outs)


from quiet import Probe
sys.path.insert(0, __file__.rsplit("/", 1)[0])
P = Probe(1.08)
TABLE = {}
RS = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4, 8, 16]
MTS = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 3, 4, 5, 6, 7]
for kind, (ws, X, kw) in KINDS.items():
    for M in Ms:
        x = X[:M]
        kk = dict(kw)
        if kk.get("res") == "res":
            kk["res"] = RES[:M]
        if kind == "out_proj":
            kk["T"] = 256
        cfgs = {}
        ref = None
        K_ = X.shape[1]
        cpt = K_ // 8 // (kk.get("T") or _default_T(K_))
        seen = set()
        for R in RS:
            for mt in MTS:
                part = tuple(_tiles(M, cpt, mt))
                if (R, part) in seen or len(part) > 5:
                    continue                 # identical tiling (different mtmax) / too many tiles: no new kernel to compile
                seen.add((R, part))
                f = (lambda R=R, mt=mt: [fused_gemv(w, x, R=R, mtmax=mt, **kk) for w in ws])
                try:
                    o = fused_gemv(ws[0], x, R=R, mtmax=mt, **kk)
                    mx.eval(o)
                    if ref is None:
                        ref = o
                    assert bool(mx.array_equal(o, ref)), "NOT bit-identical"
                    cfgs[(R, mt)] = (f, tm(f))
                except AssertionError:
                    raise
                except Exception:
                    continue
        default = lambda: [fused_gemv(w, x, **kk) for w in ws]
        # stage 2: the default and the 5 best of stage 1, interleaved, medians over quiet rounds
        top = sorted(cfgs, key=lambda c: cfgs[c][1])[:4]
        cand = {"default": default, **{c: cfgs[c][0] for c in top}}
        samples = {c: [] for c in cand}
        loose = {c: [] for c in cand}
        for rnd in range(9):
            for c, f in cand.items():
                dt, ok = P.sample(lambda f=f: once(f))
                loose[c].append(dt)
                if ok:
                    samples[c].append(dt)
        if min(len(v) for v in samples.values()) < 3:        # the machine was busy: fall back to the min of all rounds
            samples = {c: [min(v)] for c, v in loose.items()}
        med = {c: statistics.median(v) for c, v in samples.items() if v}
        d = med.get("default", float("nan"))
        best = min((c for c in med if c != "default"), key=lambda c: med[c])
        TABLE[(kind, M)] = (best, med[best], d)
        print(f"{kind:9s} M={M:2d} default {d*1e6:6.1f}us   best R={best[0]:2d} mt={best[1]} {med[best]*1e6:6.1f}us   ({d/med[best]:.3f}x)   n={len(samples[best])}", flush=True)
print("\nGEMV_TUNE = {")
for (kind, M), (c, t, d) in TABLE.items():
    if d / t > 1.02:
        print(f"    ({kind!r}, {M}): ({c[0]}, {c[1]}),   # {d/t:.3f}x")
print("}")
