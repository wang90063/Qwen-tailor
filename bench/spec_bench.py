"""MTP self-speculative decode on 6 prompts (192 tokens): tok/s, accepted drafts per round, round time.
usage: spec_bench.py [k=2] [draft_vocab=16384]"""
import sys, time, statistics
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from qwen35.spec import SpecEngine
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
dv = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
eng = SpecEngine(k=k, max_len=1024, draft_vocab=dv)
P = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
     "Explain in two sentences why the sky is blue.", "Write a short story about a robot learning to paint.",
     "import numpy as np\n\ndef softmax(x):\n", "Q: What is 17 * 23?\nA:"]
eng.generate(tk.encode(P[0]), 32)
tot = []
for p in P:
    st = {}
    out = eng.generate(tk.encode(p), 192, st)
    tps = (st["tokens"] - 1) / st["decode_s"]
    tot.append(tps)
    print(f"{p[:28]!r:32s} {tps:6.1f} tok/s  acc/round {st['accepted']/st['rounds']:.2f}  tok/round {(st['tokens']-1)/st['rounds']:.2f}  round {st['decode_s']/st['rounds']*1e3:.1f}ms")
print("mean", statistics.mean(tot))
