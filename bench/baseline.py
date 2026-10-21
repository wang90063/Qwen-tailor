"""Baseline benchmark + golden outputs for the stock mlx-lm path.

Measures, for Qwen3.5-0.8B bf16:
  prefill tok/s at several prompt lengths (time to first token)
  decode  tok/s (greedy, fixed length, EOS ignored)
Also writes golden greedy token ids so every optimized path can be checked for token-exact output.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "Qwen3.5-0.8B"
PEAK_GBPS = 153.0
BYTES_PER_TOKEN = 1.504e9  # bf16 trunk + tied lm_head, see docs/PLAN.md

PARAGRAPH = (
    "The history of computing is a story of abstraction layers. Early programmers wrote machine code "
    "by hand, then assemblers appeared, then compilers, then interpreters and virtual machines. Each "
    "layer hid detail and made programmers more productive, but each also cost some performance, and "
    "the work of systems engineers has often been to claw that performance back without breaking the "
    "abstraction. "
)
GOLDEN_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain in two sentences why the sky is blue.",
]


def env_snapshot():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chip": sh("sysctl -n machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__ if hasattr(mx, "__version__") else "?",
        "swap": sh("sysctl -n vm.swapusage"),
        "loadavg": os.getloadavg(),
        "power": sh("pmset -g batt | head -1"),
    }


def run_tokens(model, prompt_ids, n_tokens):
    """Greedy-generate n_tokens. Returns (token_ids, t_first_token, per-token timestamps)."""
    prompt = mx.array(prompt_ids)
    ids, stamps = [], []
    t0 = time.perf_counter()
    for tok, _ in generate_step(prompt, model, max_tokens=n_tokens):
        ids.append(int(tok))
        stamps.append(time.perf_counter() - t0)
    return ids, stamps


def make_prompt(tok, n):
    ids = tok.encode(PARAGRAPH * (n // 40 + 2))
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prefill-lens", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_mlx_lm.json"))
    args = ap.parse_args()

    model, tok = load(str(MODEL))
    mx.eval(model.parameters())
    print(f"loaded; {sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 1e9:.2f} GB of parameters")

    res = {"env": env_snapshot(), "prefill": {}, "decode": {}}

    # warmup (compiles kernels, fills caches)
    run_tokens(model, make_prompt(tok, 64), 16)

    for n in args.prefill_lens:
        p = make_prompt(tok, n)
        tps = []
        for _ in range(args.runs):
            _, stamps = run_tokens(model, p, 1)
            tps.append(n / stamps[0])
        res["prefill"][str(n)] = {"median_tps": statistics.median(tps), "all": tps}
        print(f"prefill {n:5d} tok: median {statistics.median(tps):8.1f} tok/s   "
              f"(min {min(tps):.0f}, max {max(tps):.0f})")

    p = make_prompt(tok, 64)
    tps = []
    for _ in range(args.runs):
        _, stamps = run_tokens(model, p, args.decode_tokens)
        tps.append((len(stamps) - 1) / (stamps[-1] - stamps[0]))
    med = statistics.median(tps)
    res["decode"]["tps_median"] = med
    res["decode"]["all"] = tps
    res["decode"]["bandwidth_utilization"] = med * BYTES_PER_TOKEN / (PEAK_GBPS * 1e9)
    print(f"decode (ctx 64, {args.decode_tokens} tok): median {med:.1f} tok/s "
          f"(min {min(tps):.1f}, max {max(tps):.1f})  "
          f"=> {med * BYTES_PER_TOKEN / 1e9:.1f} GB/s effective = "
          f"{res['decode']['bandwidth_utilization'] * 100:.0f}% of {PEAK_GBPS:.0f} GB/s")

    golden = {}
    for g in GOLDEN_PROMPTS:
        ids, _ = run_tokens(model, tok.encode(g), 64)
        golden[g] = {"prompt_ids": tok.encode(g), "greedy_ids": ids}
        print(f"golden {g[:30]!r:34s} -> {tok.decode(ids)[:70]!r}")
    res["golden"] = golden

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
