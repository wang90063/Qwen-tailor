"""Alternating in-process comparison of prefill variants (quiet-GPU sampling).  A variant is a ';'-free comma separated list of
`module.attr=value` assignments (module in pf, G, gdn); '' = defaults.  Values are python literals or bare strings.
usage: prefill_cmp.py S ROUNDS "variant" "variant" ...       e.g.  prefill_cmp.py 2048 8 "gdn.TG=256" "gdn.TG=128" """
import sys, time, statistics, ast
import mlx.core as mx
from quiet import Probe
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35.head_prune import argmax_head, build_tables
from qwen35 import model as m, prefill as pf, gemm_nax as G, gdn_nax as gdn
MODS = {"pf": pf, "G": G, "gdn": gdn}
S = int(sys.argv[1]); ROUNDS = int(sys.argv[2]); VARIANTS = sys.argv[3:] or [""]
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
TAB = build_tables(W["embed"])                  # first token as Engine.prefill computes it
ids = mx.array([make_prompt(tk, S)])


def parse(v):
    out = []
    for a in filter(None, v.split(",")):
        k, _, val = a.partition("=")
        mod, _, attr = k.partition(".")
        try: val = ast.literal_eval(val)
        except Exception: pass
        out.append((MODS[mod], attr, val))
    return out


P = {v: parse(v) for v in VARIANTS}
saved = {}
for v, asg in P.items():
    for mod, attr, _ in asg:
        saved.setdefault((id(mod), attr), (mod, attr, getattr(mod, attr)))


def run(v):
    for mod, attr, val in saved.values(): setattr(mod, attr, val)
    for mod, attr, val in P[v]: setattr(mod, attr, val)
    st = m.new_state(cfg, S + 8)
    t = time.perf_counter()
    h, st = pf.forward(cfg, W, ids, st, 0, S)
    mx.eval(argmax_head(W["embed"], TAB, h[0, -1:]), st)
    return time.perf_counter() - t


outs = {}
for v in VARIANTS:
    run(v)
probe = Probe()
ts = {v: [] for v in VARIANTS}; nq = {v: 0 for v in VARIANTS}
for _ in range(ROUNDS):
    for v in VARIANTS:
        dt, ok = probe.sample(lambda: run(v))
        ts[v].append(dt); nq[v] += ok
for v in VARIANTS:
    med = statistics.median(ts[v]); best = min(ts[v])
    print(f"S={S} {v or 'defaults':34s}: median {med * 1e3:.1f} ms ({S / med:.0f} tok/s)  best {best * 1e3:.1f} ms ({S / best:.0f} tok/s)  [{nq[v]}/{ROUNDS} quiet]")
