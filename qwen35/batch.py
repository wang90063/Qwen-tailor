"""Batched decode for B independent sequences at different positions (continuous batching: the building blocks).

State lives in *slot pools* that the decode kernels update in place (MLX scatter / slice_update would copy the whole pool every
step, ~0.5 GB at 16 slots; a decode step must only touch the rows it reads and writes):

  ssm  : 18 x [S+1, 16, 128, 128] fp32       slot s = one sequence; the DeltaNet kernel updates its state where it reads it
  conv : 18 x [(S+1)*2, 3, 6144]  bf16        double buffered: step n reads buffer pos&1 and writes the other one, because the
                                              4-tap window of a channel is read by several threadgroups (q / k channels are shared by
                                              all 16 simdgroups of a head) -- an in-place shift would race
  K, V : 6 x [S+1, 2, Lmax, 256]  bf16        the attention kernel appends the new row of each sequence at its position
  slot S is a trash slot: padding rows of a bucketed batch point at it.

Every kernel here mirrors the arithmetic of its single-sequence counterpart (`deltanet` M=1, MLX `sdpa_vector` 1-pass) operation for
operation, and `kernels.fused_gemv` rows are independent of M, so row b of a batched step is bit-identical to a solo decode step of
sequence b (tests/batch.py).  Ordering of the in-place writes relative to later readers rides on data dependencies: step n+1 consumes
the token produced by step n, whose chain contains every kernel of step n, and MLX puts a buffer barrier in front of every kernel
that consumes the output of an earlier one.
"""
from __future__ import annotations
import os

import mlx.core as mx

from .head_prune import argmax_head
from .kernels import _cache, _HDR, fused_gemv
from .mslib import BF16_MATH
from .model import Cfg

HV, DK, DV, CD, PW = 16, 128, 128, 6144, 8224
_FMIN = "-3.4028234663852886e+38f"           # mlx Limits<float>::finite_min


# ---------------------------------------------------------------------------------------------
# DeltaNet, one token for each of B slots, state updated in place
# ---------------------------------------------------------------------------------------------
def _deltanet_b_kernel(eps=1e-6):
    key = ("deltanet_b", eps)
    if key in _cache:
        return _cache[key]
    src = f"""
    constexpr int HV = 16, DK = 128, DV = 128, KD = 2048, CD = 6144, RPS = 8, NSL = 4, PW = 8224, NTG = HV * NSL;
    {_HDR}
    uint gtg = threadgroup_position_in_grid.x;
    uint b = gtg / NTG, tgid = gtg % NTG;
    uint hv = tgid / NSL, slice = tgid % NSL;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row_base = slice * (DV / NSL) + sg * RPS;
    uint c0 = lane * 4;
    size_t slot = (size_t)slots[b];
    uint par = (uint)pos[b] & 1u;
    const device bfloat16_t* cs_in = cs + ((slot * 2 + par) * 3) * CD;
    device bfloat16_t* cs_out = (device bfloat16_t*)cs + ((slot * 2 + (par ^ 1u)) * 3) * CD;
    const device bfloat16_t* pj = proj + (size_t)b * PW;
    const device float* st_base = st_in + slot * HV * DV * DK;

    // input u[ch], t = -3..0  (t<0 reads the stored conv window)
    auto inp = [&](int t, uint ch) -> float {{
        return t < 0 ? (float)cs_in[(t + 3) * CD + ch] : (float)pj[ch];
    }};
    auto conv = [&](uint ch) -> float {{
        float o = BF(inp(-3, ch) * (float)cw[ch * 4] + inp(-2, ch) * (float)cw[ch * 4 + 1] +
                     inp(-1, ch) * (float)cw[ch * 4 + 2] + inp(0, ch) * (float)cw[ch * 4 + 3]);
        return mx_silu_bf16(o);
    }};

    float4 s[RPS];
    const device float4* sin4 = (const device float4*)st_base;
    for (int r = 0; r < RPS; r++) s[r] = sin4[((hv * DV + row_base + r) * DK + c0) / 4];
    device float4* sout4 = (device float4*)st_base;       // in place: every element is read and written by the same thread
    const float inv_scale = 0.08838834764831845f;      // 128^-0.5
    const float e2 = {eps}f * inv_scale * inv_scale;
    const float sc_q = BF(inv_scale * inv_scale), sc_k = BF(inv_scale);
    float A = metal::precise::exp(Alog[hv]);

    float q4[4], k4[4];
    float qs = 0.f, ks = 0.f;
    for (int i = 0; i < 4; i++) {{
        q4[i] = conv(hv * DK + c0 + i);
        k4[i] = conv(KD + hv * DK + c0 + i);
        qs += q4[i] * q4[i];
        ks += k4[i] * k4[i];
    }}
    qs = simd_sum(qs); ks = simd_sum(ks);
    float qr = metal::precise::rsqrt(qs / (float)DK + e2);
    float kr = metal::precise::rsqrt(ks / (float)DK + e2);
    for (int i = 0; i < 4; i++) {{
        q4[i] = BF(BF(q4[i] * qr) * sc_q);
        k4[i] = BF(BF(k4[i] * kr) * sc_k);
    }}
    float av = BF((float)pj[CD + KD + HV + hv] + (float)dtb[hv]);
    float sp = mx_softplus_bf16(av);
    float g = metal::precise::exp(-A * sp);
    float bv = (float)pj[CD + KD + hv];
    float beta = mx_sigmoid_bf16(bv);
    float4 kk = float4(k4[0], k4[1], k4[2], k4[3]);
    float4 qq = float4(q4[0], q4[1], q4[2], q4[3]);
    for (int r = 0; r < RPS; r++) {{
        uint row = row_base + r;
        float v = conv(2 * KD + hv * DV + row);
        float4 sv = s[r] * g;
        float kvm = simd_sum(dot(sv, kk));
        float delta = (v - kvm) * beta;
        sv += kk * delta;
        float yv = simd_sum(dot(sv, qq));
        sout4[((hv * DV + row) * DK + c0) / 4] = sv;
        if (lane == 0) {{
            y[b * 2048 + hv * DV + row] = static_cast<bfloat16_t>(yv);
            uint ch = 2 * KD + hv * DV + row;
            for (int t = 0; t < 3; t++) cs_out[t * CD + ch] = static_cast<bfloat16_t>(inp(t - 2, ch));
        }}
    }}
    if (slice == 0 && sg == 0) {{
        for (int i = 0; i < 4; i++) {{
            uint chs[2] = {{hv * DK + c0 + i, KD + hv * DK + c0 + i}};
            for (int u = 0; u < 2; u++)
                for (int t = 0; t < 3; t++)
                    cs_out[t * CD + chs[u]] = static_cast<bfloat16_t>(inp(t - 2, chs[u]));
        }}
    }}
    """
    k = mx.fast.metal_kernel(
        name="deltanet_b", input_names=["proj", "cs", "cw", "Alog", "dtb", "st_in", "slots", "pos"],
        output_names=["y"], source=src, header=BF16_MATH)
    _cache[key] = k
    return k


def deltanet_b(proj, conv_pool, conv_w, A_log, dt_bias, ssm_pool, slots, pos):
    """proj [B, 8224] bf16; slots / pos int32 [B] (pos = tokens already in the slot's state).  Returns y [B, 2048] bf16 and updates
    ssm_pool[slot] and the conv buffer (pos&1)^1 of the slot IN PLACE (the arrays are not copied or returned)."""
    B = proj.shape[0]
    return _deltanet_b_kernel()(
        inputs=[proj.reshape(-1), conv_pool, conv_w.reshape(-1), A_log, dt_bias, ssm_pool, slots, pos],
        output_shapes=[(B * 2048,)], output_dtypes=[mx.bfloat16],
        grid=(B * 64 * 128, 1, 1), threadgroup=(128, 1, 1))[0].reshape(B, 2048)


# ---------------------------------------------------------------------------------------------
# full attention, one query token for each of B slots, K / V rows appended in place
# ---------------------------------------------------------------------------------------------
def _attn_b_kernel(Lmax, H=8, KV=2, D=256, M=1):
    """The arithmetic of MLX `sdpa_vector` (1-pass, one threadgroup of 32 simdgroups per query head; simdgroup g takes the keys
    g, g+32, ...) without a mask: each sequence reads exactly its own pos+1 keys.  The key / value of the current position come
    straight from the inputs (the same bf16 values the pool row will hold), so the in-place append needs no ordering against the
    reads.
    M > 1 (speculative verify / MTP draft): each sequence b has M consecutive query rows b*M+j at positions pos[b]+j; row j attends
    the pool keys below pos[b] and the new keys of rows 0..j (all taken from the inputs, never from the pool), i.e. what the masked
    MLX SDPA of the solo verify computes."""
    key = ("attn_b", Lmax, H, KV, D, M)
    if key in _cache:
        return _cache[key]
    src = f"""
    constexpr int H_ = {H}, KV_ = {KV}, D_ = {D}, GQA = {H // KV}, LM = {Lmax}, M_ = {M}, BN = 32, BD = 32, QPT = D_ / BD, VPT = D_ / BD;
    uint bh = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
    int r = bh / H_, h = bh % H_, kvh = h / GQA;
    int b = r / M_;
    size_t slot = (size_t)slots[b];
    int p0 = pos[b];
    int cur = p0 + (r % M_);
    int N = cur + 1;
    size_t pbase = ((slot * KV_ + kvh) * LM) * D_;
    const device bfloat16_t* kb = Kp + pbase;
    const device bfloat16_t* vb = Vp + pbase;
    const device bfloat16_t* kn0 = k_new + ((size_t)b * M_ * KV_ + kvh) * D_ + simd_lid * QPT;     // sequence b's first new row
    const device bfloat16_t* vn0 = v_new + ((size_t)b * M_ * KV_ + kvh) * D_ + simd_lid * VPT;
    const device bfloat16_t* kn = k_new + ((size_t)r * KV_ + kvh) * D_ + simd_lid * QPT;         // this row's own
    const device bfloat16_t* vn = v_new + ((size_t)r * KV_ + kvh) * D_ + simd_lid * VPT;

    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    float q[QPT], k[QPT], o[VPT];

    const device bfloat16_t* qp = q_in + (size_t)bh * D_ + simd_lid * QPT;
    for (int i = 0; i < QPT; i++) q[i] = 0.0625f * (float)qp[i];
    for (int i = 0; i < VPT; i++) o[i] = 0;

    float max_score = {_FMIN};
    float sum_exp_score = 0;
    for (int i = simd_gid; i < N; i += BN) {{
        const device bfloat16_t* kp = (i >= p0) ? kn0 + (size_t)(i - p0) * KV_ * D_ : kb + (size_t)i * D_ + simd_lid * QPT;
        for (int j = 0; j < QPT; j++) k[j] = kp[j];
        float score = 0;
        for (int j = 0; j < QPT; j++) score += q[j] * k[j];
        score = simd_sum(score);
        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        const device bfloat16_t* vp = (i >= p0) ? vn0 + (size_t)(i - p0) * KV_ * D_ : vb + (size_t)i * D_ + simd_lid * VPT;
        for (int j = 0; j < VPT; j++) o[j] = o[j] * factor + exp_score * vp[j];
    }}

    if (h % GQA == 0 && simd_gid == 0) {{               // append this position (the first query head of each kv group)
        device bfloat16_t* kw = (device bfloat16_t*)Kp + pbase + (size_t)cur * D_ + simd_lid * QPT;
        device bfloat16_t* vw = (device bfloat16_t*)Vp + pbase + (size_t)cur * D_ + simd_lid * VPT;
        for (int j = 0; j < QPT; j++) kw[j] = kn[j];
        for (int j = 0; j < VPT; j++) vw[j] = vn[j];
    }}

    if (simd_lid == 0) {{
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[simd_lid];
    float new_max = simd_max(max_score);
    float factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
    for (int i = 0; i < VPT; i++) {{
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (simd_lid == 0) {{
        for (int i = 0; i < VPT; i++) out[(size_t)bh * D_ + simd_gid * VPT + i] = static_cast<bfloat16_t>(o[i]);
    }}
    """
    k = mx.fast.metal_kernel(
        name=f"attn_b_{Lmax}_m{M}", input_names=["q_in", "k_new", "v_new", "Kp", "Vp", "slots", "pos"],
        output_names=["out"], source=src)
    _cache[key] = k
    return k


def attn_b(q, k, v, Kp, Vp, slots, pos, M=1):
    """q [R,H,D], k / v [R,KV,D] (normed + roped), pools [S+1,KV,Lmax,D]; R = B*M rows (row b*M+j = position pos[b]+j of slot
    slots[b]; slots / pos int32 [B]).  Returns o [R,H,D] bf16 and appends k / v at pos[b]+j (in place)."""
    R, H, D = q.shape
    Lmax = Kp.shape[2]
    return _attn_b_kernel(Lmax, H, k.shape[1], D, M)(
        inputs=[q, k, v, Kp, Vp, slots, pos], output_shapes=[(R, H, D)], output_dtypes=[mx.bfloat16],
        grid=(R * H * 1024, 1, 1), threadgroup=(1024, 1, 1))[0]


# ---------------------------------------------------------------------------------------------
# the two GEMV prologues as stand-alone row kernels
#
# With M rows a fused prologue is redundant work: every threadgroup of the GEMV (one per R output rows) re-normalises all M input rows
# (loads, sum of squares, simd_sum, barrier, scaling: about as many ALU ops as the R * M dot products it feeds at R = 8) -- 17-25% of
# the large-M GEMV time (bench/batch_step_bench.py).  These kernels do it once per row with the *same* arithmetic in the same order
# (thread tid owns elements 8*tid..8*tid+7, sequential fp32 sum, simd_sum, sequential sum over the simdgroups) and store the result as
# bf16, which loses nothing because the prologue rounds it to bf16 anyway: GEMV(pre-normalised x) == fused GEMV, bit for bit.
# ---------------------------------------------------------------------------------------------
def _rms_rows_kernel(K, T, eps, f32=False):
    key = ("rms_rows", K, T, eps, f32)
    if key in _cache:
        return _cache[key]
    assert K % 8 == 0 and (K // 8) % T == 0 and T % 32 == 0
    OT, CAST = ("float", "(float)") if f32 else ("bfloat16_t", "static_cast<bfloat16_t>")
    src = f"""
    constexpr int K_ = {K}, T_ = {T}, CPT = {K // 8 // T}, NSG = {T // 32};
    {_HDR}
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint m = threadgroup_position_in_grid.x;
    float xr[CPT][8];
    for (int c = 0; c < CPT; c++) {{
        uint4 v = ((const device uint4*)(x + (size_t)m * K_))[c * T_ + tid];
        uint w[4] = {{v.x, v.y, v.z, v.w}};
        for (int j = 0; j < 4; j++) {{
            xr[c][2*j]   = as_type<float>(w[j] << 16);
            xr[c][2*j+1] = as_type<float>(w[j] & 0xffff0000u);
        }}
    }}
    threadgroup float red[NSG];
    float ss = 0.f;
    for (int c = 0; c < CPT; c++) for (int j = 0; j < 8; j++) ss += xr[c][j] * xr[c][j];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < NSG; s++) tot += red[s];
    float inv = metal::precise::rsqrt(tot / (float)K_ + {eps}f);
    const device uint4* nv = (const device uint4*)nw;
    for (int c = 0; c < CPT; c++) {{
        uint4 v = nv[c * T_ + tid];
        uint w[4] = {{v.x, v.y, v.z, v.w}};
        device {OT}* o = y + (size_t)m * K_ + (c * T_ + tid) * 8;
        for (int j = 0; j < 4; j++) {{
            o[2*j]   = {CAST}(BF(as_type<float>(w[j] << 16) * (xr[c][2*j] * inv)));
            o[2*j+1] = {CAST}(BF(as_type<float>(w[j] & 0xffff0000u) * (xr[c][2*j+1] * inv)));
        }}
    }}
    """
    k = mx.fast.metal_kernel(name=f"rms_rows_k{K}_t{T}{'_f32' if f32 else ''}", input_names=["x", "nw"], output_names=["y"], source=src)
    _cache[key] = k
    return k


def rms_rows(x, nw, eps=1e-6, T=128, f32=False):
    """== the `rms` prologue of fused_gemv (K = x.shape[-1] with the default T), as its own kernel.  x [B, K] bf16, nw [K] bf16.
    f32: the (bf16-rounded) result is stored widened to fp32, which spares the GEMV that consumes it the per-thread unpack."""
    B, K = x.shape
    return _rms_rows_kernel(K, T, eps, f32)(inputs=[x.reshape(-1), nw], output_shapes=[(B * K,)],
                                            output_dtypes=[mx.float32 if f32 else mx.bfloat16],
                                            grid=(B * T, 1, 1), threadgroup=(T, 1, 1))[0].reshape(B, K)


def _gnorm_rows_kernel(eps):
    key = ("gnorm_rows", eps)
    if key in _cache:
        return _cache[key]
    src = f"""
    {_HDR}
    uint tid = thread_position_in_threadgroup.x;
    uint m = threadgroup_position_in_grid.x;
    float xr[8];
    {{
        uint4 v = ((const device uint4*)(x + (size_t)m * 2048))[tid];
        uint w[4] = {{v.x, v.y, v.z, v.w}};
        for (int j = 0; j < 4; j++) {{
            xr[2*j]   = as_type<float>(w[j] << 16);
            xr[2*j+1] = as_type<float>(w[j] & 0xffff0000u);
        }}
    }}
    float ss = 0.f;
    for (int j = 0; j < 8; j++) ss += xr[j] * xr[j];
    ss += simd_shuffle_xor(ss, 8); ss += simd_shuffle_xor(ss, 4);
    ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 1);
    float inv = metal::precise::rsqrt(ss / 128.f + {eps}f);
    device bfloat16_t* o = y + (size_t)m * 2048 + tid * 8;
    for (int j = 0; j < 8; j++) {{
        float zz = (float)proj[(size_t)m * 8224 + 6144 + tid * 8 + j];
        float n = gw[(tid % 16) * 8 + j] * (xr[j] * inv);
        o[j] = static_cast<bfloat16_t>(BF(zz / (1.f + metal::precise::exp(-zz)) * n));
    }}
    """
    k = mx.fast.metal_kernel(name="gnorm_rows", input_names=["x", "proj", "gw"], output_names=["y"], source=src)
    _cache[key] = k
    return k


def gnorm_rows(y, proj, gw, eps=1e-6):
    """== the `gnorm` prologue of fused_gemv (DeltaNet gated output norm, K = 2048, T = 256) as its own kernel.
    y [B, 2048] bf16 (delta-rule output), proj [B, 8224] bf16 (its z slice is at 6144:8192), gw fp32 [128]."""
    B = y.shape[0]
    return _gnorm_rows_kernel(eps)(inputs=[y.reshape(-1), proj.reshape(-1), gw], output_shapes=[(B * 2048,)],
                                   output_dtypes=[mx.bfloat16], grid=(B * 256, 1, 1), threadgroup=(256, 1, 1))[0].reshape(B, 2048)


# ---------------------------------------------------------------------------------------------
# in-place copies (admission: a prefilled state goes into its slot)
# ---------------------------------------------------------------------------------------------
def _blit_kernel(ctype):
    key = ("blit", ctype)
    if key in _cache:
        return _cache[key]
    src = f"""
    uint i = thread_position_in_grid.x, y = thread_position_in_grid.y;
    int n = meta[0];
    if ((int)i < n) {{
        device {ctype}* d = (device {ctype}*)dst;
        d[(size_t)meta[1] + (size_t)y * meta[3] + i] = srcv[(size_t)meta[2] + (size_t)y * meta[4] + i];
    }}
    if (i == 0 && y == 0) done[0] = 1;
    """
    k = mx.fast.metal_kernel(name=f"blit_{ctype}", input_names=["dst", "srcv", "meta"], output_names=["done"], source=src)
    _cache[key] = k
    return k


def blit(dst, src, n, d0, s0, ny=1, dstride=0, sstride=0):
    """dst.flat[d0 + y*dstride + i] = src.flat[s0 + y*sstride + i]  for i < n, y < ny   (in place).  Returns a token to evaluate."""
    ctype = {mx.bfloat16: "bfloat16_t", mx.float32: "float"}[dst.dtype]
    meta = mx.array([n, d0, s0, dstride, sstride], dtype=mx.int32)
    return _blit_kernel(ctype)(inputs=[dst, src, meta], output_shapes=[(1,)], output_dtypes=[mx.int32],
                               grid=(max(n, 1), ny, 1), threadgroup=(min(256, max(n, 1)), 1, 1))[0]


def _hrec_kernel():
    key = "hrec"
    if key in _cache:
        return _cache[key]
    src = """
    uint i = thread_position_in_grid.x, b = thread_position_in_grid.y;
    int Lh = meta[0], H = meta[1];
    if ((int)i < H) {
        device bfloat16_t* h = (device bfloat16_t*)hist;
        h[((size_t)slots[b] * Lh + pos[b]) * H + i] = Hn[(size_t)b * H + i];
    }
    if (i == 0 && b == 0) done[0] = 1;
    """
    k = mx.fast.metal_kernel(name="hrec", input_names=["hist", "Hn", "slots", "pos", "meta"], output_names=["done"], source=src)
    _cache[key] = k
    return k


def hrec(hist, Hn, slots, pos):
    """hist[slots[b], pos[b], :] = Hn[b]  (in place; hist [S+1, Lh, H] bf16).  The hidden-state history the speculative mode builds its MTP K / V
    from (batch_spec.py).  Returns a token that must be evaluated with the step."""
    B, H = Hn.shape
    meta = mx.array([hist.shape[1], H], dtype=mx.int32)
    return _hrec_kernel()(inputs=[hist, Hn, slots, pos, meta], output_shapes=[(1,)], output_dtypes=[mx.int32],
                          grid=(H, B, 1), threadgroup=(min(256, H), 1, 1))[0]


# ---------------------------------------------------------------------------------------------
# pools + the batched step
# ---------------------------------------------------------------------------------------------
class Pools:
    def __init__(self, cfg: Cfg, slots: int, max_len: int):
        self.cfg, self.S, self.Lmax = cfg, slots, max_len
        nl = sum(cfg.is_linear(i) for i in range(cfg.layers))
        na = cfg.layers - nl
        self.conv = [mx.zeros(((slots + 1) * 2, 3, CD), mx.bfloat16) for _ in range(nl)]
        self.ssm = [mx.zeros((slots + 1, HV, DV, DK), mx.float32) for _ in range(nl)]
        self.K = [mx.zeros((slots + 1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16) for _ in range(na)]
        self.V = [mx.zeros((slots + 1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16) for _ in range(na)]
        mx.eval(self.conv, self.ssm, self.K, self.V)

    def nbytes(self):
        return sum(a.nbytes for g in (self.conv, self.ssm, self.K, self.V) for a in g)

    @staticmethod
    def slot_bytes(cfg: Cfg, max_len: int, double_conv=True):
        """Bytes one sequence occupies in the pools (the pools hold slots + 1 of these: the extra one is the padding rows' trash slot).
        double_conv=False: a stand-alone single-sequence state (Engine.prefill's)."""
        nl = sum(cfg.is_linear(i) for i in range(cfg.layers))
        na = cfg.layers - nl
        return (nl * (HV * DV * DK * 4 + (2 if double_conv else 1) * 3 * CD * 2)
                + na * 2 * cfg.n_kv * max_len * cfg.head_dim * 2)

    def as_tree(self):
        return {"conv": self.conv, "ssm": self.ssm, "K": self.K, "V": self.V}

    def load(self, slot: int, state: dict, pos: int):
        """Copy a prefilled single-sequence state (Engine.prefill's: kv [1,2,max_len,256], conv [1,3,6144], ssm [1,16,128,128],
        valid after `pos` tokens) into `slot`, in place and synchronously."""
        assert 0 <= slot < self.S and 0 < pos <= self.Lmax, (slot, pos)
        toks = []
        par = pos & 1
        for i, (c, s) in enumerate(zip(state["conv"], state["ssm"])):
            toks.append(blit(self.conv[i], c, CD * 3, (slot * 2 + par) * 3 * CD, 0))
            toks.append(blit(self.ssm[i], s, HV * DV * DK, slot * HV * DV * DK, 0))
        nkv = self.cfg.n_kv
        D = self.cfg.head_dim
        for i, (K, V) in enumerate(state["kv"]):
            smax = K.shape[2]
            for pool, src in ((self.K[i], K), (self.V[i], V)):
                toks.append(blit(pool, src, pos * D, slot * nkv * self.Lmax * D, 0, ny=nkv,
                                 dstride=self.Lmax * D, sstride=smax * D))
        mx.eval(toks)

    def read(self, slot: int, pos: int, max_len: int | None = None):
        """The slot's state as a single-sequence state dict (fresh arrays; for tests and for handing a sequence back)."""
        nkv, D = self.cfg.n_kv, self.cfg.head_dim
        ml = max_len or self.Lmax
        par = pos & 1
        st = {"kv": [], "conv": [mx.array(self.conv[i][slot * 2 + par][None]) for i in range(len(self.conv))],
              "ssm": [mx.array(self.ssm[i][slot][None]) for i in range(len(self.ssm))]}
        for K, V in zip(self.K, self.V):
            def one(P):
                a = P[slot][None, :, :min(pos, ml)]
                return mx.pad(a, [(0, 0), (0, 0), (0, ml - a.shape[2]), (0, 0)]) if a.shape[2] < ml else mx.array(a)
            st["kv"].append((one(K), one(V)))
        mx.eval(st)
        return st


SPLIT_MIN = 2            # B >= SPLIT_MIN: the GEMV prologues run as separate row kernels (see rms_rows); at B=1 fused is as fast


F32X_MIN = int(os.environ.get("QWEN35_F32X_MIN", "4"))             # B >= F32X_MIN: the K=1024 GEMVs (in_proj, qkv, gate_up) read their normed input as fp32 (-7..11%; not for K>1024: slower)


def gemv_tiling(B):
    """Tiling of the batched GEMVs (bench/gemv_tune_m.py; none of R / mtmax / wf changes the arithmetic, results are bit-identical): R=2 rows
    per threadgroup (less weight state per thread => more resident threads) is 3-6% faster than the solo defaults (R=4, 7//CPT row
    tiles) from 2 rows on; from 6 rows the weights are decoded once (wf), from 8 the row tiles are balanced (4 + 4 + ...)."""
    if B >= 8:
        return dict(R=2, mtmax=4 if B % 4 == 0 else 5, wf=True)       # wf: weights decoded to fp32 once, not once per row tile (-3..20%)
    if B >= 6:
        return dict(R=2, wf=True)
    return dict(R=2) if B >= 2 else {}


def step(cfg: Cfg, F, P, cur, slots, pos, hist=None):
    """One decode token for each of B rows.  P: Pools.as_tree(); cur int32 [S+1] = the token each slot feeds next;
    slots / pos int32 [B].  Returns (ids int32 [B] = greedy next token of every row, cur updated for these slots).
    hist: optional [S+1, Lh, hidden] bf16 pool that records every row's final hidden state at its position (in place; the step then
    returns a third value, a token to evaluate together with ids)."""
    B = slots.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    split = B >= SPLIT_MIN
    tl = gemv_tiling(B)
    xf = B >= F32X_MIN
    x = F["embed"][cur[slots]]                                           # [B, hidden]
    ci = ki = 0
    for i, L in enumerate(F["layers"]):
        if cfg.is_linear(i):
            proj = fused_gemv(L["in_proj"], rms_rows(x, L["ln1"], cfg.eps, f32=xf), **tl) if split else fused_gemv(L["in_proj"], x, nw=L["ln1"])
            y = deltanet_b(proj, P["conv"][ci], L["conv_w"], L["A_log"], L["dt_bias"], P["ssm"][ci], slots, pos)
            if split:
                x = fused_gemv(L["out_proj"], gnorm_rows(y, proj, L["gnorm"]), res=x, T=256, **{"R": 8, **tl})
            else:
                x = fused_gemv(L["out_proj"], y, res=x, gnorm=(proj, L["gnorm"]), T=256, R=8)
            ci += 1
        else:
            qkv = fused_gemv(L["qkv"], rms_rows(x, L["ln1"], cfg.eps, f32=xf), **tl) if split else fused_gemv(L["qkv"], x, nw=L["ln1"])
            q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
            q, gate = mx.split(q.reshape(B, H, 2 * D), 2, axis=-1)
            gate = gate.reshape(B, -1)
            q = mx.fast.rms_norm(q, L["q_norm"], cfg.eps).reshape(B, H, 1, D)
            k = mx.fast.rms_norm(k.reshape(B, KV, D), L["k_norm"], cfg.eps).reshape(B, KV, 1, D)
            q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos).reshape(B, H, D)
            k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos).reshape(B, KV, D)
            o = attn_b(q, k, v.reshape(B, KV, D), P["K"][ki], P["V"][ki], slots, pos)
            o = o.reshape(B, -1) * mx.sigmoid(gate)
            x = fused_gemv(L["o_proj"], o, res=x, **tl)
            ki += 1
        act = (fused_gemv(L["gate_up_il"], rms_rows(x, L["ln2"], cfg.eps, f32=xf), swiglu=True, **tl) if split
               else fused_gemv(L["gate_up_il"], x, nw=L["ln2"], swiglu=True))
        x = fused_gemv(L["down"], act, res=x, **tl)
    Hn = mx.fast.rms_norm(x, F["norm"], cfg.eps)
    ids = argmax_head(F["embed"], F.get("head_tab"), Hn)
    cur[slots] = ids                                          # duplicate (trash-slot) rows: any value is fine
    if hist is not None:
        return ids, cur, hrec(hist, Hn, slots, pos)
    return ids, cur
