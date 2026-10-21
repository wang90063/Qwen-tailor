"""Two-phase chunked gated-delta-rule for prefill, built on the Neural Accelerator (NAX) tensor ops.

Per head, chunk of C=16 tokens, S0 = state entering the chunk ([dv, dk]):
    L   = strict_lower(beta_i * G_ij * K K^T)        G_ij = exp(Lc_i - Lc_j),  Lc = cumulative log-decay
    T   = (I + L)^-1,  Tb = T diag(beta)
    D   = Tb (V - gamma * (K S0^T))                  gamma_i = exp(Lc_i)
    Y   = gamma * (Q S0^T) + A D,  A = lower_incl(G * Q K^T)
    S1  = gamma_C S0 + D^T diag(gamma_C / gamma) K
Phase A (parallel over (chunk, head)) produces Tb, A and the decay vectors; phase B is the serial state
recurrence, one simdgroup per (head, 16-wide dv slice), 8 NAX matmuls per chunk step.
"""
import mlx.core as mx

from . import nax
from .mslib import BF16_MATH

_HDR = "#define BF(v) ((float)static_cast<bfloat16_t>(v))\n"
_k = {}

_HELPERS = r"""
inline float tf32_rne(float x) {                 // round to nearest even at 10 mantissa bits (what the tensor op keeps)
    uint u = as_type<uint>(x);
    u += 0x0FFFu + ((u >> 13) & 1u);
    u &= 0xFFFFE000u;
    return as_type<float>(u);
}
template <short TM, short TN>
METAL_FUNC void rne_tile(thread NAXTile<float, TM, TN>& t) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < NAXTile<float, TM, TN>::kElemsPerTile; e++) AT_NAX(t, e) = tf32_rne(AT_NAX(t, e));
}
// x = hi + lo with both parts exactly representable in bf16 (16 mantissa bits in total)
template <short TM, short TN>
METAL_FUNC void split_tile(thread NAXTile<float, TM, TN>& x, thread NAXTile<bfloat16_t, TM, TN>& hi,
                           thread NAXTile<bfloat16_t, TM, TN>& lo) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < NAXTile<float, TM, TN>::kElemsPerTile; e++) {
        const float v = AT_NAX(x, e);
        const bfloat16_t h = static_cast<bfloat16_t>(v);
        AT_NAX(hi, e) = h;
        AT_NAX(lo, e) = static_cast<bfloat16_t>(v - (float)h);
    }
}

// everything phase B reads for one 16-token chunk of one (head, dv slice); fetch() is issued one step ahead of its use
struct GdnStep {
    NAXTile<bfloat16_t, 1, 8> Kt, Qt;
    NAXTile<bfloat16_t, 1, 1> Vt;
    NAXTile<float, 1, 1> Tb, Am;
    float gam[2], dec[2], gC;
    METAL_FUNC void fetch(const device bfloat16_t* k, const device bfloat16_t* q, const device bfloat16_t* v,
                          const device float* Tbuf, const device float* Abuf, const device float* gv,
                          int c, int S, int h, int j, short2 co) thread {
        const int t0 = c * 16;
        const short nrows = (short)min(16, S - t0);
        const size_t rec = (size_t)c * 16 + h;
        const size_t tokbase = ((size_t)t0 * 16 + h) * 128;
        Kt.load_rows(k + tokbase, 2048, nrows);
        Qt.load_rows(q + tokbase, 2048, nrows);
        Vt.load_rows(v + tokbase + j * 16, 2048, nrows);
        Tb.load(Tbuf + rec * 256, 16);
        Am.load(Abuf + rec * 256, 16);
        const device float* gp = gv + rec * 32;
        gam[0] = gp[co.y]; gam[1] = gp[co.y + 8];
        dec[0] = gp[16 + co.y]; dec[1] = gp[16 + co.y + 8];
        gC = gp[15];
    }
};
"""

_SRC_A = r"""
    constexpr int PW = 8224;
    const uint c = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const int S = q_shape[0];
    const int t0 = (int)c * 16;
    const short nrows = (short)min(16, S - t0);
    const size_t rec = (size_t)c * 16 + h;

    // per-token gate scalars (same rounding points as the reference graph)
    const int ti = t0 + (int)(lane & 15);
    const bool valid = lane < 16 && ti < S;
    float lg = 0.f, beta = 0.f;
    if (valid) {
        float av = BF((float)proj[(size_t)ti * PW + 8208 + h] + (float)dtb[h]);
        float sp = mx_softplus_bf16(av);
        lg = -metal::precise::exp((float)Alog[h]) * sp;
        float bv = (float)proj[(size_t)ti * PW + 8192 + h];
        beta = mx_sigmoid_bf16(bv);
    }
    const float Lc = simd_prefix_inclusive_sum(lg);
    const short2 co = BaseNAXFrag::get_coord();            // {fn, fm}
    float Lr[2], br[2], Lcl[4], bcl[4];
    Lr[0] = simd_shuffle(Lc, (ushort)co.y);  Lr[1] = simd_shuffle(Lc, (ushort)(co.y + 8));
    br[0] = simd_shuffle(beta, (ushort)co.y); br[1] = simd_shuffle(beta, (ushort)(co.y + 8));
    for (short i = 0; i < 4; i++) {
        Lcl[i] = simd_shuffle(Lc, (ushort)(co.x + i));
        bcl[i] = simd_shuffle(beta, (ushort)(co.x + i));
    }
    const float L15 = simd_shuffle(Lc, (ushort)15);
    if (lane < 16) {
        gv[rec * 32 + lane] = metal::precise::exp(Lc);
        gv[rec * 32 + 16 + lane] = metal::precise::exp(L15 - Lc);
    }

    const size_t tokbase = ((size_t)t0 * 16 + h) * 128;
    NAXTile<bfloat16_t, 1, 8> Kt, Qt;
    Kt.load_rows(k + tokbase, 2048, nrows);
    Qt.load_rows(q + tokbase, 2048, nrows);
    NAXTile<float, 1, 1> KK, QK;
    KK.clear(); QK.clear();
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < 8; n += 2) {
        MMA16x16x32(KK, 0, Kt, false, n, Kt, true, n);
        MMA16x16x32(QK, 0, Qt, false, n, Kt, true, n);
    }

    // L = strict_lower(beta_i G_ij K K^T) goes to threadgroup memory; T = (I+L)^-1 by exact fp32 forward
    // substitution, one column per lane (the tensor-op Neumann product loses all precision when keys correlate).
    threadgroup float Ls[256];
    NAXTile<float, 1, 1> Am;
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; e++) {
        const short ri = co.y + (e >> 2) * 8, cj = co.x + (e & 3);
        const float ex = (ri >= cj) ? metal::precise::exp(Lr[e >> 2] - Lcl[e & 3]) : 0.f;
        AT_NAX(Am, e) = ex * AT_NAX(QK, e);
        Ls[ri * 16 + cj] = (ri > cj) ? br[e >> 2] * ex * AT_NAX(KK, e) : 0.f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16) {
        float x[16];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 16; i++) {
            float acc = (i == (short)lane) ? 1.f : 0.f;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < i; jj++) acc -= Ls[i * 16 + jj] * x[jj];
            x[i] = acc;
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 16; i++) Tbuf[rec * 256 + i * 16 + lane] = x[i] * beta;       // Tb = T diag(beta)
    }
    Am.store(Abuf + rec * 256, 16);
"""


# --- conv+silu+qk-norm fused into phase A -------------------------------------------------------------------------------------
# One simdgroup per (16-token chunk, head): the token-blocked conv walk of prefill._prep2_kernel (sliding 4-tap window in
# registers) writes q/k/v for phase B and stages the normalized q/k rows in threadgroup memory, where the phase-A tensor-op
# products read them (no global round trip of q/k).  Rows past the end of the sequence are zero, as load_rows(.., nrows) gives.
_PREP_PART = r"""
    constexpr int KD = 2048, CD = 6144, DK = 128;
    threadgroup uint2 ksm[16 * 32], qsm[16 * 32];
    {
        const uint hv = h;
        const uint c0 = lane * 4;
        const int chb[3] = {(int)(hv * DK + c0), (int)(KD + hv * DK + c0), (int)(2 * KD + hv * DK + c0)};
        float w[3][4][4], win[3][4][3];
        for (int g = 0; g < 3; g++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) w[g][i][j] = (float)cw[(chb[g] + i) * 4 + j];
        for (int r = 0; r < 3; r++) {                     // rows t0-3 .. t0-1 (conv state for negative positions)
            const int row = t0 - 3 + r;
            for (int g = 0; g < 3; g++) {
                const device bfloat16_t* pp = row < 0 ? cs + (row + 3) * CD + chb[g] : proj + (size_t)row * PW + chb[g];
                float f[4]; ld4(pp, f);
                for (int i = 0; i < 4; i++) win[g][i][r] = f[i];
            }
        }
        const float inv_scale = 0.08838834764831845f;
        const float e2 = 1e-6f * inv_scale * inv_scale;
        const float sc_q = BF(inv_scale * inv_scale), sc_k = BF(inv_scale);
        for (int t = t0; t < t0 + 16; t++) {
            const int tl = t - t0;
            if (t >= S) { ksm[tl * 32 + lane] = uint2(0, 0); qsm[tl * 32 + lane] = uint2(0, 0); continue; }
            float cur[3][4], o[3][4];
            for (int g = 0; g < 3; g++) ld4(proj + (size_t)t * PW + chb[g], cur[g]);
            float qs = 0.f, ks = 0.f;
            for (int g = 0; g < 3; g++)
                for (int i = 0; i < 4; i++) {
                    float a = BF(win[g][i][0] * w[g][i][0] + win[g][i][1] * w[g][i][1] + win[g][i][2] * w[g][i][2] + cur[g][i] * w[g][i][3]);
                    o[g][i] = mx_silu_bf16(a);
                    win[g][i][0] = win[g][i][1]; win[g][i][1] = win[g][i][2]; win[g][i][2] = cur[g][i];
                }
            for (int i = 0; i < 4; i++) { qs += o[0][i] * o[0][i]; ks += o[1][i] * o[1][i]; }
            qs = simd_sum(qs); ks = simd_sum(ks);
            const float qr = metal::precise::rsqrt(qs / (float)DK + e2);
            const float kr = metal::precise::rsqrt(ks / (float)DK + e2);
            const size_t ob = ((size_t)t * 16 + hv) * DK + c0;
            const uint2 qv = uint2(bf2(BF(o[0][0] * qr) * sc_q, BF(o[0][1] * qr) * sc_q), bf2(BF(o[0][2] * qr) * sc_q, BF(o[0][3] * qr) * sc_q));
            const uint2 kv = uint2(bf2(BF(o[1][0] * kr) * sc_k, BF(o[1][1] * kr) * sc_k), bf2(BF(o[1][2] * kr) * sc_k, BF(o[1][3] * kr) * sc_k));
            *((device uint2*)(q + ob)) = qv;
            *((device uint2*)(k + ob)) = kv;
            *((device uint2*)(v + ob)) = uint2(bf2(o[2][0], o[2][1]), bf2(o[2][2], o[2][3]));
            qsm[tl * 32 + lane] = qv; ksm[tl * 32 + lane] = kv;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
"""
_PREP_HELPERS = r"""
inline void ld4(const device bfloat16_t* p, thread float* o) {      // 4 consecutive bf16 -> float
    uint2 u = *((const device uint2*)p);
    o[0] = as_type<float>(u.x << 16); o[1] = as_type<float>(u.x & 0xffff0000u);
    o[2] = as_type<float>(u.y << 16); o[3] = as_type<float>(u.y & 0xffff0000u);
}
inline uint bf2(float a, float b) {                                  // two floats -> packed bf16 pair (round to nearest even)
    return (uint)as_type<ushort>(static_cast<bfloat16_t>(a)) | ((uint)as_type<ushort>(static_cast<bfloat16_t>(b)) << 16);
}
"""
_SRC_PA = _SRC_A.replace("""    const int S = q_shape[0];
""", """    const int S = proj_shape[0];
""").replace(a_old_loads_marker := """    const size_t tokbase = ((size_t)t0 * 16 + h) * 128;
    NAXTile<bfloat16_t, 1, 8> Kt, Qt;
    Kt.load_rows(k + tokbase, 2048, nrows);
    Qt.load_rows(q + tokbase, 2048, nrows);
""", """    NAXTile<bfloat16_t, 1, 8> Kt, Qt;
    Kt.load<bfloat16_t, 128, 1>((const threadgroup bfloat16_t*)ksm);
    Qt.load<bfloat16_t, 128, 1>((const threadgroup bfloat16_t*)qsm);
""")
assert a_old_loads_marker in _SRC_A
# the prep part goes right before the gate scalars (it needs S, t0, c, h, lane, which phase A defines first)
_SRC_PA = _SRC_PA.replace("""    // per-token gate scalars""", _PREP_PART + """
    // per-token gate scalars""", 1)

_SRC_B = r"""
    const uint h = threadgroup_position_in_grid.z;
    const uint j = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const int S = q_shape[0];
    const int NC = (S + 15) / 16;
    const short2 co = BaseNAXFrag::get_coord();            // {fn, fm}

    NAXTile<float, 1, 8> St;                                // [16 dv x 128 dk]
    St.load(st_in + ((size_t)h * 128 + j * 16) * 128, 128);

    GdnStep cur, nxt;
#if PF
    cur.fetch(k, q, v, Tbuf, Abuf, gv, 0, S, h, j, co);
#endif
    for (int c = 0; c < NC; c++) {
        const int t0 = c * 16;
        const short nrows = (short)min(16, S - t0);
        const size_t tokbase = ((size_t)t0 * 16 + h) * 128;
#if PF
        nxt.fetch(k, q, v, Tbuf, Abuf, gv, min(c + 1, NC - 1), S, h, j, co);     // next step's operands load while this one computes
#else
        cur.fetch(k, q, v, Tbuf, Abuf, gv, c, S, h, j, co);
#endif
#define Kt cur.Kt          // (a macro, not a reference: the MMA helpers take decltype() of their operands)
#define Qt cur.Qt
#define Vt cur.Vt
#define Tb cur.Tb
#define Am cur.Am
        const float gam[2] = {cur.gam[0], cur.gam[1]};
        const float dec[2] = {cur.dec[0], cur.dec[1]};
        const float gC = cur.gC;

        NAXTile<float, 1, 1> P1, P2, Z, D, Dp;
        P1.clear(); P2.clear();
#if SPLIT_P1 || SPLIT_P2
        NAXTile<bfloat16_t, 1, 8> Sh, Sl;
        split_tile(St, Sh, Sl);
#endif
#if RNE
        NAXTile<float, 1, 8> Sr = St;
        rne_tile(Sr);
#define STR Sr
#else
#define STR St
#endif
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < 8; n += 2) {
#if SPLIT_P1
            MMA16x16x32(P1, 0, Kt, false, n, Sh, true, n);
            MMA16x16x32(P1, 0, Kt, false, n, Sl, true, n);
#else
            MMA16x16x32(P1, 0, Kt, false, n, STR, true, n);
#endif
#if SPLIT_P2
            MMA16x16x32(P2, 0, Qt, false, n, Sh, true, n);
            MMA16x16x32(P2, 0, Qt, false, n, Sl, true, n);
#else
            MMA16x16x32(P2, 0, Qt, false, n, STR, true, n);
#endif
        }
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; e++) {
            AT_NAX(Z, e) = (float)AT_NAX(Vt, e) - gam[e >> 2] * AT_NAX(P1, e);
            AT_NAX(P2, e) *= gam[e >> 2];
        }
        D.clear();
#if SPLIT_D
        {
            NAXTile<bfloat16_t, 1, 1> Tbh, Tbl, Zh, Zl;
            split_tile(Tb, Tbh, Tbl); split_tile(Z, Zh, Zl);
            MMA16x16x16(D, 0, Tbh, false, 0, Zh, false, 0);
            MMA16x16x16(D, 0, Tbh, false, 0, Zl, false, 0);
            MMA16x16x16(D, 0, Tbl, false, 0, Zh, false, 0);
        }
#else
  #if RNE
        rne_tile(Tb); rne_tile(Z);
  #endif
        MMA16x16x16(D, 0, Tb, false, 0, Z, false, 0);
#endif
#if SPLIT_Y
        {
            NAXTile<bfloat16_t, 1, 1> Amh, Aml, Dh, Dl;
            split_tile(Am, Amh, Aml); split_tile(D, Dh, Dl);
            MMA16x16x16(P2, 0, Amh, false, 0, Dh, false, 0);
            MMA16x16x16(P2, 0, Amh, false, 0, Dl, false, 0);
            MMA16x16x16(P2, 0, Aml, false, 0, Dh, false, 0);
        }
#else
  #if RNE
        {
            NAXTile<float, 1, 1> Dr = D;
            rne_tile(Am); rne_tile(Dr);
            MMA16x16x16(P2, 0, Am, false, 0, Dr, false, 0);
        }
  #else
        MMA16x16x16(P2, 0, Am, false, 0, D, false, 0);         // P2 now holds Y
  #endif
#endif
        P2.store_rows(y + tokbase + j * 16, 2048, nrows);

        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; e++) AT_NAX(Dp, e) = AT_NAX(D, e) * dec[e >> 2];
        SCALE_NAX(St, gC);
#if SPLIT_DP
        {
            NAXTile<bfloat16_t, 1, 1> Dph, Dpl;
            split_tile(Dp, Dph, Dpl);
            STEEL_PRAGMA_UNROLL
            for (short n = 0; n < 8; n += 2) {
                MMA16x32x16(St, n, Dph, true, 0, Kt, false, n);
                MMA16x32x16(St, n, Dpl, true, 0, Kt, false, n);
            }
        }
#else
  #if RNE
        rne_tile(Dp);
  #endif
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < 8; n += 2) {
            MMA16x32x16(St, n, Dp, true, 0, Kt, false, n);
        }
#endif
#undef STR
#undef Kt
#undef Qt
#undef Vt
#undef Tb
#undef Am
#if PF
        cur = nxt;
#endif
    }
    St.store(st_out + ((size_t)h * 128 + j * 16) * 128, 128);
"""

# phase B threads per threadgroup (= 32 x dv slices of one head sharing a threadgroup).  The recurrence is bound by the NAX unit
# (one 16-wide dv slice per simdgroup, 128 independent chains); 4 simdgroups per threadgroup spreads the 16 heads x 8 slices
# over the GPU cores evenly enough (8 per threadgroup leaves some cores with 2 threadgroups and others with 1).
TG = 128

FLAGS = ("RNE", "SPLIT_P1", "SPLIT_P2", "SPLIT_D", "SPLIT_Y", "SPLIT_DP", "PF")
DEFAULT_FLAGS = frozenset()


def _defines(flags):
    return "".join(f"#define {f} {1 if f in flags else 0}\n" for f in FLAGS)


def _kernels(relaxed=True, flags=DEFAULT_FLAGS):
    key = ("gdn", relaxed, frozenset(flags))
    if key not in _k:
        hdr = _HDR + BF16_MATH + nax.header(relaxed) + _HELPERS
        ka = mx.fast.metal_kernel(name="gdn_nax_a", input_names=["q", "k", "proj", "Alog", "dtb"],
                                  output_names=["Tbuf", "Abuf", "gv"], source=_SRC_A, header=hdr)
        kb = mx.fast.metal_kernel(name="gdn_nax_b", input_names=["q", "k", "v", "Tbuf", "Abuf", "gv", "st_in"],
                                  output_names=["y", "st_out"], source=_defines(flags) + _SRC_B, header=hdr)
        _k[key] = (ka, kb)
    return _k[key]


def _kernel_pa(relaxed=True):
    key = ("gdn_pa", relaxed)
    if key not in _k:
        hdr = _HDR + BF16_MATH + nax.header(relaxed) + _HELPERS + _PREP_HELPERS
        _k[key] = mx.fast.metal_kernel(name="gdn_nax_prep_a", input_names=["proj", "cs", "cw", "Alog", "dtb"],
                                       output_names=["q", "k", "v", "Tbuf", "Abuf", "gv"], source=_SRC_PA, header=hdr)
    return _k[key]


def prep_a(proj, conv_state, conv_w, A_log, dt_bias, relaxed=True):
    """conv+silu+qk-norm and GDN phase A in one kernel (bit-identical to conv_prep followed by phase A).
    proj [S,8224], conv_state [3,6144], conv_w [6144,1,4] -> q,k,v [S,16,128] bf16, Tbuf, Abuf [NC,16,256] f32, gv [NC,16,32] f32."""
    S = proj.shape[0]
    NC = (S + 15) // 16
    return _kernel_pa(relaxed)(inputs=[proj, conv_state, conv_w.reshape(-1), A_log, dt_bias],
                               output_shapes=[(S, 16, 128)] * 3 + [(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
                               output_dtypes=[mx.bfloat16] * 3 + [mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1))


def phase_b(q, k, v, Tb, Am, gv, state, relaxed=True, flags=DEFAULT_FLAGS, tg=None):
    """the serial state recurrence (phase B) on the outputs of phase A -> y [S,16,128] bf16, new state [1,16,128,128] f32"""
    S = q.shape[0]
    tg = TG if tg is None else tg
    _, kb = _kernels(relaxed, flags)
    return kb(inputs=[q, k, v, Tb, Am, gv, state], output_shapes=[(S, 16, 128), (1, 16, 128, 128)],
              output_dtypes=[mx.bfloat16, mx.float32], grid=(256, 1, 16), threadgroup=(tg, 1, 1))


def gdn_chunked(q, k, v, proj, A_log, dt_bias, state, relaxed=True, flags=DEFAULT_FLAGS, tg=None):
    """q,k,v [S,16,128] bf16 (q/k already normalized + scaled), proj [S,8224] bf16 (b at 8192:8208, a at 8208:8224),
    state [1,16,128,128] fp32.  Returns y [S,16,128] bf16, new_state [1,16,128,128] fp32."""
    S = q.shape[0]
    NC = (S + 15) // 16
    tg = TG if tg is None else tg
    ka, kb = _kernels(relaxed, flags)
    Tb, Am, gv = ka(inputs=[q, k, proj, A_log, dt_bias],
                    output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)], output_dtypes=[mx.float32] * 3,
                    grid=(32, NC, 16), threadgroup=(32, 1, 1))
    y, st = kb(inputs=[q, k, v, Tb, Am, gv, state],
               output_shapes=[(S, 16, 128), (1, 16, 128, 128)], output_dtypes=[mx.bfloat16, mx.float32],
               grid=(256, 1, 16), threadgroup=(tg, 1, 1))
    return y, st
