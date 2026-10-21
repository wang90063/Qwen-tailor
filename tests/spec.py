"""Spec path vs reference greedy (token-exact up to bf16 near-ties): report match rate."""
import json, sys, time
import mlx.core as mx
from qwen35.spec import SpecEngine
from qwen35.model import ROOT

g = json.load(open(ROOT / "results" / "baseline_mlx_lm.json"))["golden"]
k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
eng = SpecEngine(k=k, max_len=1024, compile="--nocompile" not in sys.argv)
for name, d in g.items():
    st = {}
    out = eng.generate(d["prompt_ids"], 64, st)
    ref = d["greedy_ids"]
    n = 0
    while n < min(len(out), len(ref)) and out[n] == ref[n]: n += 1
    print(f"{name[:30]!r:34s} first-diff@{n}/{len(ref)}  rounds {st['rounds']} acc/round {st['accepted']/max(st['rounds'],1):.2f}")
    if n < len(ref): print("  out", out[:n+4], "\n  ref", ref[:n+4])
