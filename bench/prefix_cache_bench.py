"""TTFT with / without the prefix cache (bf16 Engine), alternating inside one process so machine noise cancels.

usage: prefix_cache_bench.py [rounds] [prefix ...]      default prefixes 256 1024 2048, suffixes 1 16 64 128 512

For every (prefix P, suffix n):  `miss`  = prefill(P+n tokens) on an engine without a cache,
                                 `hit`   = same prompt on an engine whose cache holds the P-token prompt (different suffix every round),
                                 `repeat`= the P-token prompt again (whole prompt cached, first token included: no compute).
Also prints the restore breakdown (lookup / concatenation / eval) and the bytes of one entry."""
import sys, time, statistics
import numpy as np
import mlx.core as mx
from quiet import Probe
from qwen35.engine import Engine

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PREFIXES = [int(x) for x in sys.argv[2:]] or [256, 1024, 2048]
SUFFIXES = [1, 16, 64, 128, 512]
rng = np.random.default_rng(0)


def toks(n):
    return [int(x) for x in rng.integers(1000, 30000, n)]


def ttft(eng, ids):
    t = time.perf_counter()
    nxt, st, _ = eng.prefill(ids)
    mx.eval(nxt, st)
    return time.perf_counter() - t


base = Engine(max_len=4096)
cached = Engine(max_len=4096, prefix_cache_mb=1024, like=base)
probe = Probe(tol=1.18)

# warm up kernels / compile paths for every shape that is timed
for P in PREFIXES:
    pre = toks(P)
    for n in SUFFIXES:
        ttft(base, pre + toks(n))
        ttft(cached, pre); ttft(cached, pre + toks(n))
cached.pcache.clear()

print(f"{'prefix':>6} {'suffix':>6} | {'miss ms':>8} {'hit ms':>8} {'speedup':>7} | {'repeat ms':>9} | {'miss+ins':>8}")
for P in PREFIXES:
    pre = toks(P)
    ttft(cached, pre)                                   # seed the cache with the P-token prompt
    for n in SUFFIXES:
        if P + n > 4096:
            continue
        ms, hs, rs, ins = [], [], [], []
        for _ in range(ROUNDS):
            for acc, fn in ((ms, lambda: ttft(base, pre + toks(n))),
                            (ins, lambda: ttft(cached, toks(P + n))),     # cache miss that also stores the prompt
                            (hs, lambda: ttft(cached, pre + toks(n))),    # fresh suffix per try: a retry must not be a full-prompt hit
                            (rs, lambda: ttft(cached, pre))):
                mx.eval(mx.zeros(1))                    # drain queued cache-insert work
                dt, ok = probe.sample(fn)
                if ok:
                    acc.append(dt)
        m, h, r, i = (statistics.median(x) * 1e3 if x else float("nan") for x in (ms, hs, rs, ins))
        print(f"{P:6d} {n:6d} | {m:8.1f} {h:8.1f} {m / h:6.1f}x | {r:9.2f} | {i:8.1f}   [{len(ms)}/{len(hs)}/{len(rs)}/{len(ins)} quiet of {ROUNDS}]  mem {mx.get_active_memory() / 2**30:.2f} GiB (cache {cached.pcache.bytes / 2**30:.2f}, {len(cached.pcache)} nodes)", flush=True)
    cached.pcache.clear(); mx.clear_cache()

# restore breakdown for the largest prefix
P = PREFIXES[-1]
pre = toks(P)
ttft(cached, pre)
pc = cached.pcache
node, _ = pc.lookup(np.asarray(pre, dtype=np.int32))
print(f"\nentry at {P} tokens: {node.nbytes / 2**20:.1f} MiB  (cache {pc.bytes / 2**20:.1f} MiB, {len(pc)} nodes)")
t0 = []; t1 = []; t2 = []
for _ in range(30):
    a = time.perf_counter(); nd, _ = pc.lookup(np.asarray(pre, dtype=np.int32))
    b = time.perf_counter(); st, _ = pc.restore(nd, cached.max_len)
    c = time.perf_counter(); mx.eval(st)
    d = time.perf_counter()
    t0.append(b - a); t1.append(c - b); t2.append(d - c)
print(f"lookup {statistics.median(t0) * 1e3:.2f} ms | restore graph {statistics.median(t1) * 1e3:.2f} ms | eval {statistics.median(t2) * 1e3:.2f} ms")
