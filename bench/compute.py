"""bf16 GEMM throughput: the compute roofline for prefill."""
import statistics, time
import mlx.core as mx

def tflops(m, k, n, iters=20):
    a = mx.random.normal((m, k)).astype(mx.bfloat16); b = mx.random.normal((k, n)).astype(mx.bfloat16)
    mx.eval(a, b)
    for _ in range(3): mx.eval(a @ b)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); mx.eval(a @ b); ts.append(time.perf_counter() - t0)
    t = statistics.median(ts); tb = min(ts)
    return 2*m*k*n/t/1e12, 2*m*k*n/tb/1e12

for (m, k, n, name) in [(4096, 4096, 4096, "square 4096^3"),
                        (2048, 1024, 3584, "prefill MLP up  (2048 tok)"),
                        (2048, 3584, 1024, "prefill MLP down(2048 tok)"),
                        (2048, 1024, 6144, "prefill DeltaNet qkv"),
                        (512, 1024, 3584, "prefill MLP up  (512 tok)")]:
    med, best = tflops(m, k, n)
    print(f"{name:28s} median {med:5.1f} TFLOPS   best {best:5.1f} TFLOPS")
