"""Prefill GEMM on the Neural Accelerators: MLX's own NAX mainloop (steel `gemm_loop`, inlined at runtime) with our
epilogues fused in.  C = A[M,K] @ B[N,K]^T with bf16 operands, fp32 accumulation, bf16 result.

  EPI_PLAIN   D[M,N]   = BF(acc)
  EPI_SWIGLU  D[M,N/2] = BF(silu_bf16(BF(g)) * BF(u)) where B's rows are interleaved in blocks of 16
                       (rows 32j..32j+15 = gate rows 16j.., rows 32j+16..32j+31 = up rows 16j..)
  EPI_ADD     D[M,N]   = BF(R + BF(acc))                    (residual add, R is [M,N] bf16)
"""
import json
import math
from pathlib import Path

import mlx.core as mx

from . import nax
from .mslib import BF16_MATH

EPI_PLAIN, EPI_SWIGLU, EPI_ADD = 0, 1, 2

_HDR = nax.header(True, extra=("mlx/backend/metal/kernels/steel/gemm/gemm_nax.h",)) + BF16_MATH

_SRC = """
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;

    const int M = dims[0], N = dims[1], K = dims[2], swizzle_log = dims[3];
    const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
    uint3 tid = threadgroup_position_in_grid;
    const int tid_y = ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    const int tid_x = (tid.x) >> swizzle_log;
    if (tiles_n <= tid_x || tiles_m <= tid_y) return;

    const int c_row = tid_y * BM, c_col = tid_x * BN;
    const short sg = simdgroup_index_in_threadgroup;
    const short tm = SM * (sg / WN), tn = SN * (sg % WN);

    const bool align_M = (M % BM) == 0, align_N = (N % BN) == 0, align_K = (K % BK) == 0;
    const short sgp_sm = align_M ? SM : (short)min((int)SM, M - (c_row + tm));
    const short sgp_sn = align_N ? SN : (short)min((int)SN, N - (c_col + tn));
    const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
    const bool is_unaligned_sn = align_N ? false : (sgp_sn != SN);

    const device bfloat16_t* Ap = A + (size_t)(c_row + tm) * K;
    const device bfloat16_t* Bp = B + (size_t)(c_col + tn) * K;

    NAXTile<float, TM, TN> Dtile;
#if EPI == 2 && RPF
    // residual tile: loaded up front so its latency hides behind the main loop
    NAXTile<bfloat16_t, TM, TN> Rt;
    if ((align_M || sgp_sm > 0) && (align_N || sgp_sn > 0)) {
        const size_t roff = (size_t)(c_row + tm) * N + (c_col + tn);
        if (align_M && align_N) Rt.load(R + roff, N);
        else Rt.load_safe(R + roff, N, short2(sgp_sn, sgp_sm));
    }
#endif
    dispatch_bool(align_K, [&](auto kAlignedK) {
      dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
        dispatch_bool(align_N || !is_unaligned_sn, [&](auto kAlignedN) {
          Dtile = gemm_loop<bfloat16_t, SM, SN, SK, BK, false, true, kAlignedM.value, kAlignedN.value, kAlignedK.value,
                            float>(Ap, Bp, K, K, K, K / BK, sgp_sm, sgp_sn);
          if ((kAlignedM.value || sgp_sm > 0) && (kAlignedN.value || sgp_sn > 0)) {
#if EPI == 0
            device bfloat16_t* Dp = D + (size_t)(c_row + tm) * N + (c_col + tn);
            if constexpr (kAlignedM && kAlignedN) Dtile.store(Dp, N);
            else Dtile.store_safe(Dp, N, short2(sgp_sn, sgp_sm));
#elif EPI == 2
            const size_t off = (size_t)(c_row + tm) * N + (c_col + tn);
#if !RPF
            NAXTile<bfloat16_t, TM, TN> Rt;
            if constexpr (kAlignedM && kAlignedN) Rt.load(R + off, N);
            else Rt.load_safe(R + off, N, short2(sgp_sn, sgp_sm));
#endif
            NAXTile<bfloat16_t, TM, TN> Ot;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM * TN * 8; i++) {
                float a = (float)static_cast<bfloat16_t>(Dtile.elems()[i]);
                Ot.elems()[i] = static_cast<bfloat16_t>((float)Rt.elems()[i] + a);
            }
            device bfloat16_t* Dp = D + off;
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
                    float g = (float)static_cast<bfloat16_t>(gf[i]);
                    float u = (float)static_cast<bfloat16_t>(uf[i]);
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


# EPI_ADD: fetch the residual tile before the main loop (True) or after it (False).  Up front costs registers and ~1.7 ms over a
# whole 2048-token prefill (occupancy), so the default is the plain epilogue load
RPF = False


def _kernel(epi, rpf=False):
    key = (epi, rpf and epi == EPI_ADD)
    if key not in _k:
        ins = ["A", "B", "dims"] + (["R"] if epi == EPI_ADD else [])
        _k[key] = mx.fast.metal_kernel(name=f"gemm_nax_e{epi}{'p' if key[1] else ''}", input_names=ins, output_names=["D"],
                                       source=f"#define EPI {epi}\n#define RPF {int(key[1])}\n" + _SRC, header=_HDR)
    return _k[key]


# default tile configuration (BM, BN, BK, WM, WN)
CFG = (128, 128, 256, 4, 4)


def matmul_nt(x, w, epi=EPI_PLAIN, r=None, cfg=CFG, swizzle_log=None):
    """x [M,K] bf16, w [N,K] bf16 -> [M,N] (or [M,N/2] for EPI_SWIGLU)."""
    bm, bn, bk, wm, wn = cfg
    M, K = x.shape
    N = w.shape[0]
    tm, tn = -(-M // bm), -(-N // bn)
    if swizzle_log is None:
        swizzle_log = 0 if tm <= 3 else 1
    dims = mx.array([M, N, K, swizzle_log], dtype=mx.int32)
    tpg = 32 * wm * wn
    grid = (tn * (1 << swizzle_log) * tpg, -(-tm // (1 << swizzle_log)), 1)
    ins = [x, w, dims] + ([r] if epi == EPI_ADD else [])
    out_shape = (M, N // 2) if epi == EPI_SWIGLU else (M, N)
    return _kernel(epi, RPF)(inputs=ins, output_shapes=[out_shape], output_dtypes=[mx.bfloat16], grid=grid,
                        threadgroup=(tpg, 1, 1),
                        template=[("BM", bm), ("BN", bn), ("BK", bk), ("WM", wm), ("WN", wn)])[0]


# tuned tile configurations per (layer GEMM, M): results/gemm_cfg.json, written by bench/gemm_tune.py ("mlx" = mx.matmul is
# at least as fast at that M, keep it)
_TABLE_PATH = Path(__file__).resolve().parent.parent / "results" / "gemm_cfg.json"
_TABLE = {}
if _TABLE_PATH.exists():
    _TABLE = {n: {int(m): None if c == "mlx" else tuple(c) for m, c in d.items()}
              for n, d in json.loads(_TABLE_PATH.read_text()).items()}


# the residual-fused GEMMs (EPI_ADD) have their own entries "<name>+add" (bench/gemm_tune_add.py); without one the plain entry applies
ADD_TABLE = True


def pick(name, M, epi=EPI_PLAIN):
    """(cfg, swizzle_log) tuned for GEMM `name`, from the table entry nearest to M in log2 (CFG if untuned);
    (None, None) when mx.matmul is the better choice."""
    t = _TABLE.get(name + "+add") if epi == EPI_ADD and ADD_TABLE else None
    t = t or _TABLE.get(name)
    if not t:
        return CFG, None
    m = min(t, key=lambda k: abs(math.log2(k) - math.log2(max(M, 1))))
    c = t[m]
    return (None, None) if c is None else (tuple(c[:5]), c[5])


def linear(x, w, name, epi=EPI_PLAIN, r=None):
    """x [..., K] @ w[N, K]^T on the tuned config for (name, rows); leading dims are preserved."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    cfg, swz = pick(name, M, epi)
    if cfg is None:
        assert epi != EPI_SWIGLU
        y = x @ w.T
        return y if r is None else r + y
    if swz:                                              # never swizzle past the number of row tiles
        tm = -(-M // cfg[0])
        while swz and tm <= (1 << swz):
            swz -= 1
    y = matmul_nt(x2, w, epi=epi, r=None if r is None else r.reshape(M, -1), cfg=cfg, swizzle_log=swz)
    return y.reshape(*lead, y.shape[-1])
