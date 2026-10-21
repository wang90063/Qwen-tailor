"""Token frequency table for the vocab-pruned draft head (corpus: local text files + model generations).

The corpus is whatever text files the machine has (site-packages of the active environment, system docs), so a re-run does
not reproduce results/token_freq_*.npy bit for bit; the committed tables are the ones the draft head uses."""
import sys, glob, os, random, json, sysconfig
import numpy as np
from transformers import AutoTokenizer
from qwen35.model import ROOT
tk = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-0.8B")
V = 248320
random.seed(0)
site = sysconfig.get_paths()["purelib"]
files = []
for pat in [f"{site}/**/*.py", f"{site}/**/*.md", f"{site}/**/*.rst", f"{site}/**/*.txt",
            "/Library/Developer/CommandLineTools/usr/share/**/*.md",
            "/usr/share/**/*.txt", "/usr/share/**/*.html"]:
    files += glob.glob(pat, recursive=True)
random.shuffle(files)
print(len(files), "files")
cnt = np.zeros(V, dtype=np.int64); held = np.zeros(V, dtype=np.int64)
total = 0
for i, f in enumerate(files):
    try:
        t = open(f, errors="ignore").read()[:200000]
    except Exception: continue
    if not t.strip(): continue
    ids = tk.encode(t)
    (held if i % 10 == 0 else cnt)[:] += np.bincount(ids, minlength=V)[:V]
    total += len(ids)
    if total > 12_000_000: break
print("tokens", total, "distinct", int((cnt > 0).sum()))
np.save(f"{ROOT}/results/token_freq_corpus.npy", cnt)
np.save(f"{ROOT}/results/token_freq_heldout.npy", held)
order = np.argsort(-cnt)
for K in (8192, 16384, 32768, 65536):
    top = np.zeros(V, bool); top[order[:K]] = True
    print(K, "heldout coverage %.2f%%" % (100 * held[top].sum() / held.sum()))
