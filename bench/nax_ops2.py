"""NAX op cost by shape x operand dtypes x transposes (ns per op per GPU core, 4 independent accumulators, 64 simdgroups/core)."""
import sys, time, itertools
import mlx.core as mx
from qwen35 import nax

ITERS = 256
SHAPES = {                      # macro, A tile cols, B tile cols, C tile cols, flops
    "16x16x16": ("MMA16x16x16", 1, 1, 1, 2 * 16 * 16 * 16),
    "16x16x32": ("MMA16x16x32", 2, 2, 1, 2 * 16 * 16 * 32),
    "16x32x16": ("MMA16x32x16", 1, 2, 2, 2 * 16 * 32 * 16),
}
TY = {"bf": "bfloat16_t", "f32": "float"}

def make(shape, ta, tb, tya, tyb):
    macro, ac, bc, cc, fl = SHAPES[shape]
    ctiles = "".join(f"NAXTile<float,1,{cc}> C{i}; C{i}.clear();\n" for i in range(4))
    ops = "\n".join(f"{macro}(C{i}, 0, A, {str(ta).lower()}, 0, B, {str(tb).lower()}, 0);" for i in range(4))
    src = f"""
    NAXTile<{TY[tya]},1,{ac}> A; NAXTile<{TY[tyb]},1,{bc}> B;
    A.load(a, 32); B.load(b, 32);
    {ctiles}
    for (int it = 0; it < {ITERS}; it++) {{
        {ops}
    }}
    float s = 0.f;
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < {cc} * 8; e++) s += AT_NAX(C0, e) + AT_NAX(C1, e) + AT_NAX(C2, e) + AT_NAX(C3, e);
    out[thread_position_in_grid.x] = s;
    """
    name = f"o_{shape}_{int(ta)}{int(tb)}_{tya}{tyb}"
    return mx.fast.metal_kernel(name=name, input_names=["a", "b"], output_names=["out"], source=src, header=nax.header(True)), fl

from bench.quiet import Probe
probe = Probe()
def timeit(f, n=5):
    mx.eval(f()); ts = []
    for _ in range(n):
        def go():
            t = time.perf_counter(); mx.eval(f()); return time.perf_counter() - t
        dt, ok = probe.sample(go)
        if ok: ts.append(dt)
    ts.sort(); return ts[len(ts) // 2] if ts else float("nan")

NTG, TPG = 80, 256
a32 = mx.random.normal((32, 32)) * 0.01
want = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SHAPES)
print(f"{'shape':9s} {'A':>4s} {'B':>4s} {'TA':>3s} {'TB':>3s} {'ns/op/core':>11s} {'TFLOPS':>7s}")
for shape in want:
    for tya, tyb in (("bf", "bf"), ("f32", "bf"), ("bf", "f32"), ("f32", "f32")):
        for ta, tb in itertools.product((False, True), repeat=2):
            try:
                k, fl = make(shape, ta, tb, tya, tyb)
                ain = a32.astype(mx.bfloat16) if tya == "bf" else a32
                bin_ = a32.astype(mx.bfloat16) if tyb == "bf" else a32
                t = timeit(lambda: k(inputs=[ain, bin_], output_shapes=[(NTG * TPG,)], output_dtypes=[mx.float32],
                                     grid=(NTG * TPG, 1, 1), threadgroup=(TPG, 1, 1))[0])
            except Exception as e:
                print(f"{shape:9s} {tya:>4s} {tyb:>4s} {int(ta):3d} {int(tb):3d}  FAILED {str(e).splitlines()[0][:90]}"); continue
            ops = NTG * TPG // 32 * ITERS * 4
            print(f"{shape:9s} {tya:>4s} {tyb:>4s} {int(ta):3d} {int(tb):3d} {t / ops * 10 * 1e9:11.2f} {ops * fl / t / 1e12:7.2f}", flush=True)
