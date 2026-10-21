"""Measure achievable memory bandwidth on this GPU, in the shapes decode actually uses.

peak (spec)          : 153 GB/s for the base M5
streaming read       : sum() over a large bf16 array
lm_head GEMV (M=1)   : x[1,1024] @ W[248320,1024].T, the single biggest op in decode
"""
import json
import statistics
import sys
import time

import mlx.core as mx

PEAK_GBPS = 153.0
H, V = 1024, 248320


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return ts


def report(name, nbytes, ts):
    med, best = statistics.median(ts), min(ts)
    row = {
        "name": name,
        "bytes": nbytes,
        "median_ms": med * 1e3,
        "best_ms": best * 1e3,
        "median_gbps": nbytes / med / 1e9,
        "best_gbps": nbytes / best / 1e9,
    }
    print(f"{name:28s} median {row['median_gbps']:6.1f} GB/s ({row['median_ms']:.3f} ms)   "
          f"best {row['best_gbps']:6.1f} GB/s   = {row['best_gbps'] / PEAK_GBPS * 100:4.0f}% of peak")
    return row


def main():
    rows = []
    # 1) pure streaming read, 1 GiB
    a = mx.random.normal((512 * 1024 * 1024,)).astype(mx.bfloat16)
    mx.eval(a)
    rows.append(report("stream read 1GiB (sum)", a.nbytes, timeit(lambda: mx.sum(a))))
    del a

    # 2) lm_head-shaped GEMV, bf16
    w = (mx.random.normal((V, H)) * 0.02).astype(mx.bfloat16)
    x = mx.random.normal((1, H)).astype(mx.bfloat16)
    mx.eval(w, x)
    rows.append(report("lm_head GEMV bf16 (M=1)", w.nbytes, timeit(lambda: x @ w.T)))

    # 3) MLP-shaped GEMV (gate/up: 1024 -> 3584), many layers back to back, to see small-op efficiency
    ws = [(mx.random.normal((3584, H)) * 0.02).astype(mx.bfloat16) for _ in range(24)]
    mx.eval(ws)

    def chain():
        return [x @ wi.T for wi in ws]

    rows.append(report("24x MLP-up GEMV bf16 (M=1)", sum(wi.nbytes for wi in ws), timeit(chain)))

    # 4) same but only ONE small GEMV per eval: exposes per-dispatch latency
    rows.append(report("1x MLP-up GEMV bf16 (M=1)", ws[0].nbytes, timeit(lambda: x @ ws[0].T)))

    if "--json" in sys.argv:
        print(json.dumps(rows))


if __name__ == "__main__":
    main()
