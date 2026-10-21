"""Throughput / latency of the NAX tensor-op shapes the gated-delta kernel uses (operand types matter).
Each variant runs ITERS iterations of 4 independent ops per simdgroup; reports ns per op per GPU core and cycles @ nominal clock."""
import sys, time
import mlx.core as mx
from qwen35 import nax

ITERS = 512
# (name, tile decls, op statement producing into C0..C3, flops per op)
V = {
 "bf16xbf16 16x16x32": ("NAXTile<bfloat16_t,1,2> A, B;", "MMA16x16x32(C{i}, 0, A, false, 0, B, true, 0);", 2 * 16 * 16 * 32),
 "bf16xfp32 16x16x32": ("NAXTile<bfloat16_t,1,2> A; NAXTile<float,1,2> B;", "MMA16x16x32(C{i}, 0, A, false, 0, B, true, 0);", 2 * 16 * 16 * 32),
 "fp32xfp32 16x16x32": ("NAXTile<float,1,2> A, B;", "MMA16x16x32(C{i}, 0, A, false, 0, B, true, 0);", 2 * 16 * 16 * 32),
 "bf16xbf16 16x16x16": ("NAXTile<bfloat16_t,1,1> A, B;", "MMA16x16x16(C{i}, 0, A, false, 0, B, false, 0);", 2 * 16 * 16 * 16),
 "fp32xfp32 16x16x16": ("NAXTile<float,1,1> A, B;", "MMA16x16x16(C{i}, 0, A, false, 0, B, false, 0);", 2 * 16 * 16 * 16),
 "bf16xbf16 16x32x16": ("NAXTile<bfloat16_t,1,1> A; NAXTile<bfloat16_t,1,2> B;", "MMA16x32x16(C{i}, 0, A, true, 0, B, false, 0);", 2 * 16 * 32 * 16),
 "fp32xbf16 16x32x16": ("NAXTile<float,1,1> A; NAXTile<bfloat16_t,1,2> B;", "MMA16x32x16(C{i}, 0, A, true, 0, B, false, 0);", 2 * 16 * 32 * 16),
}

def make(name, decl, op, mn):
    # C tiles: 16x32x16 writes two frags
    cn = 2 if "16x32x16" in name else 1
    ctiles = "".join(f"NAXTile<float,1,{cn}> C{i}; C{i}.clear();\n" for i in range(4))
    ops = "\n".join(op.format(i=i) for i in range(4))
    src = f"""
    const size_t base = (size_t)thread_position_in_grid.x / 32 * 0;
    {decl}
    A.load(a, 16); B.load(b, 32);
    {ctiles}
    for (int it = 0; it < {ITERS}; it++) {{
        {ops}
    }}
    float s = 0.f;
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < {cn} * 8; e++) s += AT_NAX(C0, e) + AT_NAX(C1, e) + AT_NAX(C2, e) + AT_NAX(C3, e);
    out[thread_position_in_grid.x] = s;
    """
    return mx.fast.metal_kernel(name="ops_" + "".join(c for c in name if c.isalnum()), input_names=["a", "b"], output_names=["out"],
                                source=src, header=nax.header(True))

def timeit(f, n=7):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]

NTG, TPG = 80, 256            # 640 simdgroups = 64 per core
a = (mx.random.normal((16, 64)) * 0.01).astype(mx.float32)
b = (mx.random.normal((64, 64)) * 0.01).astype(mx.float32)
for name, (decl, op, fl) in V.items():
    k = make(name, decl, op, fl)
    ain = a if "fp32x" in name else a.astype(mx.bfloat16)
    bin_ = b if "xfp32" in name else b.astype(mx.bfloat16)
    try:
        f = lambda: k(inputs=[ain, bin_], output_shapes=[(NTG * TPG,)], output_dtypes=[mx.float32], grid=(NTG * TPG, 1, 1), threadgroup=(TPG, 1, 1))[0]
        t = timeit(f)
    except Exception as e:
        print(f"{name}: FAILED {str(e).splitlines()[0][:150]}"); continue
    nsg = NTG * TPG // 32
    ops = nsg * ITERS * 4
    print(f"{name:22s} {t * 1e3:7.3f} ms | {ops / t / 1e9:7.1f} Gop/s | {t / ops * 10 * 1e9:6.2f} ns/op/core | {ops * fl / t / 1e12:6.2f} TFLOPS")
