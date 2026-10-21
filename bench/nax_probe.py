"""NAX 16x16x32 MMA precision probe: relative error vs float64 for relaxed / strict precision and fp32 / bf16 operands."""
import sys
import numpy as np
import mlx.core as mx
from qwen35 import nax

def run(relaxed, ta, tb):
    src = f"""
        NAXTile<{ta}, 1, 2> A;
        NAXTile<{tb}, 1, 2> B;
        NAXTile<float, 1, 1> C;
        A.load(a, 32); B.load(b, 32);
        C.clear();
        MMA16x16x32(C, 0, A, false, 0, B, true, 0);
        C.store(c, 16);
    """
    ia, ib = ("float" if ta == "float" else "bfloat16_t"), ("float" if tb == "float" else "bfloat16_t")
    k = mx.fast.metal_kernel(name=f"nax_probe_{relaxed}_{ta}_{tb}".replace("_t", ""), input_names=["a", "b"],
                             output_names=["c"], source=src, header=nax.header(relaxed))
    rng = np.random.default_rng(0)
    an = rng.standard_normal((16, 32)).astype(np.float32); bn = rng.standard_normal((16, 32)).astype(np.float32)
    A = mx.array(an); B = mx.array(bn)
    if ta != "float": A = A.astype(mx.bfloat16)
    if tb != "float": B = B.astype(mx.bfloat16)
    out = k(inputs=[A, B], output_shapes=[(16, 16)], output_dtypes=[mx.float32], grid=(32, 1, 1), threadgroup=(32, 1, 1))[0]
    mx.eval(out)
    # reference computed on the *given* (possibly bf16-rounded) inputs in float64
    ref = np.array(A.astype(mx.float32), dtype=np.float64) @ np.array(B.astype(mx.float32), dtype=np.float64).T
    o = np.array(out, dtype=np.float64)
    return np.linalg.norm(o - ref) / np.linalg.norm(ref)

for relaxed in (True, False):
    for ta, tb in (("float", "float"), ("bfloat16_t", "bfloat16_t"), ("bfloat16_t", "float")):
        try:
            print(f"relaxed={relaxed!s:5} A={ta:10} B={tb:10} rel-L2 err vs fp64 = {run(relaxed, ta, tb):.3e}")
        except Exception as e:
            print(f"relaxed={relaxed!s:5} A={ta:10} B={tb:10} FAILED: {str(e)[:300]}")
