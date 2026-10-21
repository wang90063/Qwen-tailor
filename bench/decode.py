"""Plain (non-speculative) decode throughput of Engine -- median of 5 runs, prefill excluded -- and free-run divergence from the golden mlx-lm greedy ids.
usage: decode.py [--eager]"""
import sys, time, statistics, json
import mlx.core as mx
from qwen35.engine import Engine
from mlx_lm import load
from qwen35.model import ROOT
compile_ = "--eager" not in sys.argv
eng = Engine(compile=compile_)
gold = json.load(open(f"{ROOT}/results/baseline_mlx_lm.json"))
tok = load(f"{ROOT}/models/Qwen3.5-0.8B")[1] if False else None
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-0.8B")
from baseline import make_prompt
p = make_prompt(tk, 64)
eng.generate(p, 16)
tps = []
N = 128
for _ in range(5):
    t0 = time.perf_counter(); out = eng.generate(p, N); dt = time.perf_counter() - t0
    # exclude prefill: measure with 1-token run
    t1 = time.perf_counter(); eng.generate(p, 1); dp = time.perf_counter() - t1
    tps.append((N - 1) / (dt - dp))
print(f"decode compile={compile_}: median {statistics.median(tps):.1f} tok/s  all {[round(x,1) for x in tps]}")
# golden free-run
for name, g in gold["golden"].items():
    o = eng.generate(g["prompt_ids"], 64)
    m = next((i for i, (a, b) in enumerate(zip(o, g["greedy_ids"])) if a != b), None)
    print(" golden", repr(name[:25]), "ALL" if m is None else f"diverge@{m}")
