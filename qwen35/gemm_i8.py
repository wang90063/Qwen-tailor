"""OPT-IN, LOSSY prefill GEMMs on the Neural Accelerators' int8 path (W8A8): ~1.6x the bf16 NAX rate (24-25 vs 15 TOPS).

C = dequant(Aq[M,K] @ Bq[N,K]^T) with Aq per-token and Bq per-output-channel symmetric int8 (scale = amax / 127), int32 accumulation
(exact), result sx[m] * sw[n] * acc rounded to bf16.  The activations arrive already quantized from the norm kernel that produces
them (`QAct`, prefill.norm_q), so there is no extra pass over them.  This is *not* bit-compatible with the bf16 path (logit
relative error 1-5% depending on which GEMMs are on; see docs/PLAN.md) and is off unless QWEN35_W8A8 names GEMMs.

  EPI_PLAIN   D[M,N]   = BF(acc * sx * sw)
  EPI_SWIGLU  D[M,N/2] = BF(silu_bf16(BF(g)) * BF(u)), g / u the dequantized gate / up rows (16-row interleaved B, as gemm_nax)
"""
import json
import math
import os
from pathlib import Path

import mlx.core as mx

from . import gemm_nax as G

EPI_PLAIN, EPI_SWIGLU = G.EPI_PLAIN, G.EPI_SWIGLU

# GEMMs (in_proj, qkv, out_proj, gate_up, down) that run on int8; the activations of out_proj / down are produced by kernels that
# do not emit int8, so only the norm-fed ones are supported: in_proj, qkv (input of layer), gate_up (input of the MLP)
NAMES = frozenset(n for n in os.environ.get("QWEN35_W8A8", "").split(",") if n)
SUPPORTED = ("in_proj", "qkv", "gate_up")

# Hadamard rotation of the K = 1024 input dimension (QuaRot-style): the norm kernel emits H x / 32 (H = Sylvester +-1 matrix, H H = 1024 I) and
# the weights are stored as W H / 32, so the products are unchanged up to rounding while the activation outliers that dominate the
# per-token int8 error are spread over all channels (hidden-state error of the whole forward drops ~40%).  "0" = plain per-token int8.
ROT = os.environ.get("QWEN35_W8A8_ROT", "1") != "0"

_pro = G._SRC[:G._SRC.index("    NAXTile<float, TM, TN> Dtile;")]
_pro = _pro.replace("const device bfloat16_t* Ap", "const device int8_t* Ap").replace("const device bfloat16_t* Bp", "const device int8_t* Bp")

_SRC = _pro + """
    NAXTile<int, TM, TN> Dtile;
    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = gemm_loop<int8_t, SM, SN, SK, BK, false, true, kAlignedM.value, kAlignedN.value, kAlignedK.value,
                            int>(Ap, Bp, K, K, K, K / BK, sgp_sm, sgp_sn);
          if ((kAlignedM.value || sgp_sm > 0) && (kAlignedN.value || sgp_sn > 0)) {
            const short2 sc = NAXTile<int, TM, TN>::NAXFrag_t::get_coord();          // x = column, y = row of this lane's elements
            const int row0 = c_row + tm + sc.y, col0 = c_col + tn + sc.x;
#if EPI == 0
            NAXTile<bfloat16_t, TM, TN> Ot;
            const_for_loop<0, TM, 1>([&](auto mm) {
              const_for_loop<0, TN, 1>([&](auto nn) {
                constexpr int mv = decltype(mm)::value, nv = decltype(nn)::value;
                thread auto& af = Dtile.template frag_at<mv, nv>();
                thread auto& of = Ot.template frag_at<mv, nv>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 8; i++) {
                    const int r = min(row0 + mv * 16 + (i / 4) * 8, M - 1), c = min(col0 + nv * 16 + (i % 4), N - 1);
                    of[i] = static_cast<bfloat16_t>((float)af[i] * (sx[r] * sw[c]));
                }
              });
            });
            device bfloat16_t* Dp = D + (size_t)(c_row + tm) * N + (c_col + tn);
            if constexpr (kAlignedM && kAlignedN) Ot.store(Dp, N);
            else Ot.store_safe(Dp, N, short2(sgp_sn, sgp_sm));
#else
            static_assert(TN % 2 == 0, "swiglu epilogue needs an even number of 16-col fragments per simdgroup");
            NAXTile<bfloat16_t, TM, TN / 2> Ot;
            const_for_loop<0, TM, 1>([&](auto mm) {
              const_for_loop<0, TN / 2, 1>([&](auto pp) {
                constexpr int mv = decltype(mm)::value, pv = decltype(pp)::value;
                thread auto& gf = Dtile.template frag_at<mv, 2 * pv>();
                thread auto& uf = Dtile.template frag_at<mv, 2 * pv + 1>();
                thread auto& of = Ot.template frag_at<mv, pv>();
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 8; i++) {
                    const int r = min(row0 + mv * 16 + (i / 4) * 8, M - 1), c = min(col0 + 2 * pv * 16 + (i % 4), N - 17);
                    float g = (float)static_cast<bfloat16_t>((float)gf[i] * (sx[r] * sw[c]));
                    float u = (float)static_cast<bfloat16_t>((float)uf[i] * (sx[r] * sw[c + 16]));
                    of[i] = static_cast<bfloat16_t>(mx_silu_bf16(g) * u);
                }
              });
            });
            device bfloat16_t* Dp = D + (size_t)(c_row + tm) * (N / 2) + (c_col + tn) / 2;
            if constexpr (kAlignedM && kAlignedN) Ot.store(Dp, N / 2);
            else Ot.store_safe(Dp, N / 2, short2(sgp_sn / 2, sgp_sm));
#endif
          }
        });
      });
    });
"""

_k = {}


def _kernel(epi):
    if epi not in _k:
        _k[epi] = mx.fast.metal_kernel(name=f"gemm_i8_e{epi}", input_names=["A", "B", "dims", "sx", "sw"], output_names=["D"],
                                       source=f"#define EPI {epi}\n#define RPF 0\n" + _SRC, header=G._HDR)
    return _k[epi]


CFG = (128, 128, 256, 4, 4)


class QAct:
    """An int8-quantized activation matrix: q [M,K] int8, s [M] float32 (row scale); `lead` = extra leading dims of the logical shape."""
    __slots__ = ("q", "s", "lead")

    def __init__(self, q, s, lead=()):
        self.q, self.s, self.lead = q, s, lead

    @property
    def shape(self):
        return (*self.lead, *self.q.shape)

    def __getitem__(self, idx):
        assert idx is None, "QAct only supports x[None]"
        return QAct(self.q, self.s, (1, *self.lead))


_H = []


def hadamard():
    """[1024, 1024] float32 orthonormal Sylvester-Hadamard matrix H[k, j] = (-1)^popcount(k & j) / 32."""
    if not _H:
        k = mx.arange(1024, dtype=mx.uint32)
        v = k[:, None] & k[None, :]
        for sh in (16, 8, 4, 2, 1):                    # parity of popcount(v)
            v = v ^ (v >> sh)
        _H.append(mx.where((v & 1) == 1, -1.0, 1.0).astype(mx.float32) / 32.0)
    return _H[0]


def quant_weight(w):
    """bf16 [N, 1024] -> (int8, float32 scale) per output row, after the input-dimension rotation when ROT."""
    return quant_rows(w.astype(mx.float32) @ hadamard() if ROT else w)


def quant_rows(a):
    """bf16 / fp32 [N,K] -> (int8 [N,K], float32 [N]) symmetric per row (weights; activations use the norm kernel's own quantizer)."""
    a32 = a.astype(mx.float32)
    amax = mx.max(mx.abs(a32), axis=-1, keepdims=True)
    s = mx.maximum(amax, 1e-8) / 127.0
    q = mx.clip(mx.round(a32 / s), -127, 127).astype(mx.int8)
    return q, s[:, 0]


def matmul_i8(xq, sx, wq, sw, epi=EPI_PLAIN, cfg=CFG, swizzle_log=None):
    bm, bn, bk, wm, wn = cfg
    M, K = xq.shape
    N = wq.shape[0]
    tm, tn = -(-M // bm), -(-N // bn)
    if swizzle_log is None:
        swizzle_log = 0 if tm <= 3 else 1
    dims = mx.array([M, N, K, swizzle_log], dtype=mx.int32)
    tpg = 32 * wm * wn
    grid = (tn * (1 << swizzle_log) * tpg, -(-tm // (1 << swizzle_log)), 1)
    out_shape = (M, N // 2) if epi == EPI_SWIGLU else (M, N)
    return _kernel(epi)(inputs=[xq, wq, dims, sx, sw], output_shapes=[out_shape], output_dtypes=[mx.bfloat16], grid=grid,
                        threadgroup=(tpg, 1, 1),
                        template=[("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn)])[0]


# tuned tile configurations per (GEMM name, M): results/gemm_i8_cfg.json {name: {M: [BM, BN, BK, WM, WN, swizzle_log]}}, written by
# bench/gemm_i8_tune.py (the entry nearest to M in log2 applies; CFG when untuned)
_TABLE_PATH = Path(__file__).resolve().parent.parent / "results" / "gemm_i8_cfg.json"
_TABLE = {}
if _TABLE_PATH.exists():
    _TABLE = {n: {int(m): tuple(c) for m, c in d.items()} for n, d in json.loads(_TABLE_PATH.read_text()).items()}


def pick(name, M):
    t = _TABLE.get(name)
    if not t:
        return CFG, None
    c = t[min(t, key=lambda k: abs(math.log2(k) - math.log2(max(M, 1))))]
    return tuple(c[:5]), c[5]


def linear(x, wpack, name, epi=EPI_PLAIN):
    """x: QAct ([..., K]); wpack = (wq [N,K] int8, sw [N] float32); leading dims preserved."""
    wq, sw = wpack
    cfg, swz = pick(name, x.q.shape[0])
    if swz:                                              # never swizzle past the number of row tiles
        tm = -(-x.q.shape[0] // cfg[0])
        while swz and tm <= (1 << swz):
            swz -= 1
    y = matmul_i8(x.q, x.s, wq, sw, epi=epi, cfg=cfg, swizzle_log=swz)
    return y.reshape(*x.lead, *y.shape)
