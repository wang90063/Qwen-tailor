"""Phase A/B timing + bit-exactness of gdn_nax kernel flag variants against the default.
usage: gdn_variants.py [S] [flags;flags;...]   (flags comma separated, e.g. "PF" or "" for the default)"""
import sys, time
import mlx.core as mx
from qwen35 import model as m, prefill as pf, gdn_nax
from bench.quiet import Probe

S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
def parse(v):
    fl, _, tg = v.partition("@")
    return frozenset(f for f in fl.split(",") if f), int(tg) if tg else 256
VARS = [parse(v) for v in (sys.argv[2].split(";") if len(sys.argv) > 2 else ["", "PF"])]
NC = (S + 15) // 16
cfg = m.Cfg(); W = m.load_weights(cfg); L = W["layers"][0]
mx.random.seed(0)
x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
proj = x @ L["in_proj"].T
ss = (mx.random.normal((1, 16, 128, 128)) * 0.05)
q, k, v = pf.conv_prep(proj, mx.zeros((3, 6144), mx.bfloat16), L["conv_w"])
mx.eval(q, k, v, proj, ss)

R = 6                                           # distinct "layers" per timed graph so the work can't be cached/fused away
def run_a(var):
    fl, _ = var
    ka, _ = gdn_nax._kernels(True, fl)
    return ka(inputs=[q, k, proj, L["A_log"], L["dt_bias"]], output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
              output_dtypes=[mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1))

base_a = run_a(VARS[0]); mx.eval(base_a)
def run_b(var, a=base_a):
    fl, tg = var
    _, kb = gdn_nax._kernels(True, fl)
    return kb(inputs=[q, k, v, *a, ss], output_shapes=[(S, 16, 128), (1, 16, 128, 128)],
              output_dtypes=[mx.bfloat16, mx.float32], grid=(256, 1, 16), threadgroup=(tg, 1, 1))

ref = run_b(VARS[0]); mx.eval(ref)
probe = Probe()
res = {}
import os
for rnd in range(int(os.environ.get("ROUNDS", "9"))):
    for vi, fl in enumerate(VARS):
        for tag, fn in (("A", lambda fl=fl: run_a(fl)), ("B", lambda fl=fl: run_b(fl)), ("AB", lambda fl=fl: gdn_nax.gdn_chunked(q, k, v, proj, L["A_log"], L["dt_bias"], ss, flags=fl[0], tg=fl[1]))):
            def go():
                t = time.perf_counter(); mx.eval([fn() for _ in range(R)]); return time.perf_counter() - t
            dt, ok = probe.sample(go)
            if ok:
                res.setdefault((vi, tag), []).append(dt / R)
for vi, fl in enumerate(VARS):
    out = run_b(fl); mx.eval(out)
    same = all(bool(mx.all(a == b).item()) for a, b in zip(out, ref))
    row = []
    for tag in ("A", "B", "AB"):
        ts = sorted(res.get((vi, tag), [0.0])); row.append(f"{tag} {ts[len(ts) // 2] * 1e3:.3f} ms (n={len(ts)})")
    print(f"{(','.join(sorted(fl[0])) or 'default') + '@' + str(fl[1]):16s} " + " | ".join(row) + f" | B output identical to default: {same}")
