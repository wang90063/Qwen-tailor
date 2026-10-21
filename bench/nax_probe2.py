"""NAX MMA precision with fp32 operands: plain / software-RNE tf32 / bf16 hi-lo split."""
import sys
import numpy as np
import mlx.core as mx
from qwen35 import nax

HDR = """
inline float tf32_rne(float x) {
    uint u = as_type<uint>(x);
    u += 0x0FFFu + ((u >> 13) & 1u);
    u &= 0xFFFFE000u;
    return as_type<float>(u);
}
"""

SRC = {
 "plain_fp32xfp32": """
    NAXTile<float,1,2> A, B; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    MMA16x16x32(C, 0, A, false, 0, B, true, 0);
    C.store(c, 16);""",
 "plain_fp32xbf16": """
    NAXTile<float,1,2> A; NAXTile<bfloat16_t,1,2> B; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    MMA16x16x32(C, 0, A, false, 0, B, true, 0);
    C.store(c, 16);""",
 "rne_fp32xfp32": """
    NAXTile<float,1,2> A, B; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    for (short e = 0; e < 16; e++) { AT_NAX(A, e) = tf32_rne(AT_NAX(A, e)); AT_NAX(B, e) = tf32_rne(AT_NAX(B, e)); }
    MMA16x16x32(C, 0, A, false, 0, B, true, 0);
    C.store(c, 16);""",
 "rne_fp32xbf16": """
    NAXTile<float,1,2> A; NAXTile<bfloat16_t,1,2> B; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    for (short e = 0; e < 16; e++) AT_NAX(A, e) = tf32_rne(AT_NAX(A, e));
    MMA16x16x32(C, 0, A, false, 0, B, true, 0);
    C.store(c, 16);""",
 "split_fp32xbf16": """
    NAXTile<float,1,2> A; NAXTile<bfloat16_t,1,2> Ah, Al, B; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    for (short e = 0; e < 16; e++) {
        float x = AT_NAX(A, e); bfloat16_t h = static_cast<bfloat16_t>(x);
        AT_NAX(Ah, e) = h; AT_NAX(Al, e) = static_cast<bfloat16_t>(x - (float)h);
    }
    MMA16x16x32(C, 0, Ah, false, 0, B, true, 0);
    MMA16x16x32(C, 0, Al, false, 0, B, true, 0);
    C.store(c, 16);""",
 "split_fp32xfp32": """
    NAXTile<float,1,2> A, B; NAXTile<bfloat16_t,1,2> Ah, Al, Bh, Bl; NAXTile<float,1,1> C;
    A.load(a, 32); B.load(b, 32); C.clear();
    for (short e = 0; e < 16; e++) {
        float x = AT_NAX(A, e); bfloat16_t h = static_cast<bfloat16_t>(x);
        AT_NAX(Ah, e) = h; AT_NAX(Al, e) = static_cast<bfloat16_t>(x - (float)h);
        float y = AT_NAX(B, e); bfloat16_t g = static_cast<bfloat16_t>(y);
        AT_NAX(Bh, e) = g; AT_NAX(Bl, e) = static_cast<bfloat16_t>(y - (float)g);
    }
    MMA16x16x32(C, 0, Ah, false, 0, Bh, true, 0);
    MMA16x16x32(C, 0, Ah, false, 0, Bl, true, 0);
    MMA16x16x32(C, 0, Al, false, 0, Bh, true, 0);
    C.store(c, 16);""",
}

rng = np.random.default_rng(0)
an = rng.standard_normal((16, 32)).astype(np.float32)
bn_f = rng.standard_normal((16, 32)).astype(np.float32)
bn_b = np.array(mx.array(bn_f).astype(mx.bfloat16).astype(mx.float32))
for name, src in SRC.items():
    for relaxed in (True,):
        bf = "bf16" in name
        k = mx.fast.metal_kernel(name="p_" + name, input_names=["a", "b"], output_names=["c"], source=src,
                                 header=HDR + nax.header(relaxed))
        A = mx.array(an)
        B = mx.array(bn_f).astype(mx.bfloat16) if bf else mx.array(bn_f)
        out = k(inputs=[A, B], output_shapes=[(16, 16)], output_dtypes=[mx.float32], grid=(32, 1, 1), threadgroup=(32, 1, 1))[0]
        mx.eval(out)
        Bn = bn_b if bf else bn_f
        ref = an.astype(np.float64) @ Bn.astype(np.float64).T
        o = np.array(out, dtype=np.float64)
        err = o - ref
        print(f"{name:18s} relaxed={relaxed}: rel-L2 {np.linalg.norm(err)/np.linalg.norm(ref):.3e}  "
              f"bias (mean err / rms ref) {err.mean()/np.sqrt((ref**2).mean()):+.2e}  max|err|/rms {np.abs(err).max()/np.sqrt((ref**2).mean()):.2e}")
