"""Server-level decode throughput with / without the speculative mode: B concurrent requests (real code prompts), the window in which
all B are decoding (last first token .. first completion), tokens counted by host observation time.

usage: python bench/serve_spec_bench.py [B,B,...] [gen] [reps] [code|noise|prose]
("noise": random-token prompts -- the drafts are rejected much more often, the adaptive mode policy must keep the server near plain speed)
"""
import sys, statistics
from pathlib import Path
import numpy as np
import mlx_lm
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from qwen35.engine import Engine
from qwen35.serve import Server, Request
from qwen35.spec import SpecEngine

Bs = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else (1, 2, 3, 4, 6, 8)
GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
KIND = sys.argv[4] if len(sys.argv) > 4 else "code"
sp = SpecEngine(k=2, max_len=2048)
eng = Engine(max_len=2048, like=sp)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
text = "\n".join(p.read_text() for p in sorted(Path(mlx_lm.__file__).parent.glob("models/*.py"))[:12])
if KIND == "prose":                      # English prose: package READMEs
    text = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path(mlx_lm.__file__).parent.parent.glob("*.dist-info/METADATA"))[:40])
IDS = tk(text)["input_ids"]
rng = np.random.default_rng(1)


def window_rate(reqs):
    lo, hi = max(r.t_first for r in reqs), min(r.t_done for r in reqs)
    n = sum(sum(1 for t in r.t_tokens if lo < t <= hi) for r in reqs)
    return n / (hi - lo), hi - lo


def run(srv, B, seed):
    r_ = np.random.default_rng(seed)
    reqs = []
    for i in range(B):
        a = int(r_.integers(0, len(IDS) - 120))
        prompt = r_.integers(1000, 200000, 100).tolist() if KIND == "noise" else IDS[a:a + 100]
        reqs.append(Request(rid=i, prompt=prompt, max_new=GEN, arrival=0.0))
    st = srv.run(reqs)
    return window_rate(reqs), st


for B in Bs:
    sp_srv = Server(eng, slots=B, max_len=2048, chunk=512, spec=sp, spec_max=8)
    pl_srv = Server(eng, slots=B, max_len=2048, chunk=512)
    sp_srv.warmup(); pl_srv.warmup()
    run(sp_srv, B, 99); run(pl_srv, B, 99)
    rs, rp, tpr = [], [], []
    for rep in range(REPS):
        (a, _), st = run(sp_srv, B, rep)
        (b, _), _ = run(pl_srv, B, rep)
        rs.append(a); rp.append(b); tpr.append(st["tokens_per_round"])
    ms, mp = statistics.median(rs), statistics.median(rp)
    print(f"B={B}  plain {mp:6.1f} tok/s   spec {ms:6.1f} tok/s  ({ms / mp:4.2f}x, {statistics.median(tpr):.2f} tokens/seq/round, {st["rounds"]} rounds / {st["steps"]} plain steps, estimate {st["tpr"]:.2f})", flush=True)
    del sp_srv, pl_srv
