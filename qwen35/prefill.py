"""Prefill path: reference graph structure with the DeltaNet glue (conv+silu+qk-norm, gated norm) fused into
two elementwise Metal kernels and SDPA in causal-string mode."""
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update

from . import gdn_nax
from . import gemm_nax as G
from . import gemm_i8 as I8
from .mslib import BF16_MATH
from .model import Cfg, _full_attn, _mlp, _write, pos_arr

# DeltaNet prefill recurrence: "nax" = chunked two-phase tensor-op kernel (gdn_nax.py), "ref" = mlx-lm sequential kernel
GDN_IMPL = os.environ.get("QWEN35_GDN", "nax")
# precision flags of the nax kernel (comma separated subset of gdn_nax.FLAGS; empty = fastest "plain" tensor-op MMAs)
GDN_FLAGS = frozenset(f for f in os.environ.get("QWEN35_GDN_FLAGS", "").split(",") if f)

# prefill projections: "nax" = tuned Neural-Accelerator GEMMs (gemm_nax.py, fused SwiGLU epilogue), "mlx" = mx.matmul
GEMM_IMPL = os.environ.get("QWEN35_GEMM", "nax")
# below this many rows MLX's own (gemv-like) kernels are as fast, keep them
GEMM_MIN_M = 96
# GEMMs (in_proj, qkv, out_proj, gate_up, down) that stay on mx.matmul, for A/B experiments
GEMM_OFF = frozenset(f for f in os.environ.get("QWEN35_GEMM_OFF", "").split(",") if f)

# the residual adds ride on the out_proj / down GEMM epilogues (EPI_ADD: D = BF(R + BF(acc)), exactly the rounding of the separate
# add) and a norm-only kernel finishes the job; "0" = separate add+norm kernel (A/B baseline, bit-identical result)
FUSE_ADD = os.environ.get("QWEN35_FUSE_ADD", "1") != "0"

# conv+silu+qk-norm and the GDN phase-A products in one kernel (q/k never re-read from memory); "0" = separate kernels (bit-identical)
FUSE_PREP_A = os.environ.get("QWEN35_PREPA", "1") != "0"

_HDR = "#define BF(v) ((float)static_cast<bfloat16_t>(v))\n"
_k = {}


def _fast_gemm(M, name=None):
    return GEMM_IMPL == "nax" and M >= GEMM_MIN_M and name not in GEMM_OFF


# int8 copies of the weights for the opt-in W8A8 prefill (gemm_i8.py): id(bf16 weight) -> (that weight, int8 [N,K], scales [N]); the entry
# keeps the bf16 array alive, so its id cannot be reused by another array
_WQ = {}


def _lin(x, w, name, res=None):
    """x [..., K] @ w[N, K]^T (+ res); x may be a QAct (int8 activations of the opt-in W8A8 mode)"""
    if isinstance(x, I8.QAct):
        assert res is None
        return I8.linear(x, _WQ[id(w)][1:], name)
    if _fast_gemm(x.size // x.shape[-1], name):
        return G.linear(x, w, name) if res is None else G.linear(x, w, name, epi=G.EPI_ADD, r=res)
    y = x @ w.T
    return y if res is None else res + y


def prepare_w8a8(W):
    """Quantize the weights of the GEMMs named in gemm_i8.NAMES (idempotent, except the last layer which stays exact)."""
    for L in W["layers"][:-1]:
        for name, key in (("in_proj", "in_proj"), ("qkv", "qkv"), ("gate_up", "gate_up_il")):
            if name in I8.NAMES and key in L and id(L[key]) not in _WQ:
                _WQ[id(L[key])] = (L[key], *I8.quant_weight(L[key]))
                mx.eval(_WQ[id(L[key])][1:])


def _w8a8(cfg, i, S, n_layers):
    """Is layer i's input / MLP-input GEMM on int8: (attn_in, mlp_in)"""
    if not I8.NAMES or i >= n_layers - 1 or not _fast_gemm(S):
        return False, False
    return ("in_proj" if cfg.is_linear(i) else "qkv") in I8.NAMES, "gate_up" in I8.NAMES


def _interleave16(w):
    """[2I, K] rows = gate | up  ->  alternating 16-row blocks of gate and up (the fused SwiGLU epilogue pairs them)."""
    I = w.shape[0] // 2
    return mx.concatenate([w[:I].reshape(I // 16, 16, -1), w[I:].reshape(I // 16, 16, -1)], axis=1).reshape(2 * I, -1)


def prepare(W):
    """Prefill-only weight copies (idempotent, +14.7 MB/layer): interleaved gate_up for the fused SwiGLU GEMM; the last layer's
    qkv split into its q|gate rows and k|v rows (the tail path needs q only for the rows whose output is used)."""
    todo = [L for L in W["layers"] if "gate_up_il" not in L]
    for L in todo:
        L["gate_up_il"] = _interleave16(L["gate_up"])
    last = W["layers"][-1]
    if "qkv" in last and "qkv_kv" not in last:
        c = Cfg()
        n_q = c.n_heads * c.head_dim * 2                                     # rows of q|gate (k, v follow: n_kv * head_dim each)
        last["qkv_q"], last["qkv_kv"] = mx.contiguous(last["qkv"][:n_q]), mx.contiguous(last["qkv"][n_q:])
        todo.append(last)
    if todo:
        mx.eval([L[k] for L in todo for k in ("gate_up_il", "qkv_q", "qkv_kv") if k in L])


def _prep_kernel():
    if "prep" in _k:
        return _k["prep"]
    src = _HDR + """
    constexpr int KD = 2048, CD = 6144, PW = 8224, DK = 128;
    uint hv = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup, c0 = lane * 4;
    auto inp = [&](int i, uint ch) -> float {
        return i < 0 ? (float)cs[(i + 3) * CD + ch] : (float)proj[(size_t)i * PW + ch];
    };
    auto conv = [&](uint ch) -> float {
        float o = BF(inp(t - 3, ch) * (float)cw[ch * 4] + inp(t - 2, ch) * (float)cw[ch * 4 + 1] +
                     inp(t - 1, ch) * (float)cw[ch * 4 + 2] + inp(t, ch) * (float)cw[ch * 4 + 3]);
        return mx_silu_bf16(o);
    };
    const float inv_scale = 0.08838834764831845f;
    const float e2 = 1e-6f * inv_scale * inv_scale;
    const float sc_q = BF(inv_scale * inv_scale), sc_k = BF(inv_scale);
    float q4[4], k4[4], v4[4];
    float qs = 0.f, ks = 0.f;
    for (int i = 0; i < 4; i++) {
        q4[i] = conv(hv * DK + c0 + i);
        k4[i] = conv(KD + hv * DK + c0 + i);
        v4[i] = conv(2 * KD + hv * DK + c0 + i);
        qs += q4[i] * q4[i]; ks += k4[i] * k4[i];
    }
    qs = simd_sum(qs); ks = simd_sum(ks);
    float qr = metal::precise::rsqrt(qs / (float)DK + e2);
    float kr = metal::precise::rsqrt(ks / (float)DK + e2);
    size_t o = ((size_t)t * 16 + hv) * DK + c0;
    for (int i = 0; i < 4; i++) {
        q[o + i] = static_cast<bfloat16_t>(BF(q4[i] * qr) * sc_q);
        k[o + i] = static_cast<bfloat16_t>(BF(k4[i] * kr) * sc_k);
        v[o + i] = static_cast<bfloat16_t>(v4[i]);
    }
    """
    _k["prep"] = mx.fast.metal_kernel(name="prefill_prep", input_names=["proj", "cs", "cw"],
                                      output_names=["q", "k", "v"], source=src, header=BF16_MATH)
    return _k["prep"]


def _gnorm_kernel():
    if "gn" in _k:
        return _k["gn"]
    src = _HDR + """
    constexpr int PW = 8224;
    uint hv = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup, c0 = lane * 4;
    size_t o = (size_t)t * 2048 + hv * 128 + c0;
    float yy[4]; float ss = 0.f;
    for (int i = 0; i < 4; i++) { yy[i] = (float)y[o + i]; ss += yy[i] * yy[i]; }
    ss = simd_sum(ss);
    float inv = metal::precise::rsqrt(ss / 128.f + 1e-6f);
    for (int i = 0; i < 4; i++) {
        float zz = (float)proj[(size_t)t * PW + 6144 + hv * 128 + c0 + i];
        float n = gw[c0 + i] * (yy[i] * inv);
        out[o + i] = static_cast<bfloat16_t>(zz / (1.f + metal::precise::exp(-zz)) * n);
    }
    """
    _k["gn"] = mx.fast.metal_kernel(name="prefill_gnorm", input_names=["y", "proj", "gw"],
                                    output_names=["out"], source=src)
    return _k["gn"]


def _prep2_kernel(tb):
    """Token-blocked conv+silu+qk-norm: one simdgroup walks `tb` consecutive tokens of one head with a sliding 4-tap window
    in registers (every proj element is read once, not 4x), vectorized 8-byte loads/stores, conv weights loaded once."""
    key = ("prep2", tb)
    if key in _k:
        return _k[key]
    src = _HDR + """
    constexpr int KD = 2048, CD = 6144, PW = 8224, DK = 128;
    const int S = (int)proj_shape[0];
    const uint hv = threadgroup_position_in_grid.y;
    const int t0 = (int)threadgroup_position_in_grid.z * TB;
    const int t1 = min(t0 + TB, S);
    const uint c0 = thread_index_in_simdgroup * 4;
    const int chb[3] = {(int)(hv * DK + c0), (int)(KD + hv * DK + c0), (int)(2 * KD + hv * DK + c0)};
    float w[3][4][4], win[3][4][3];
    for (int g = 0; g < 3; g++)
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) w[g][i][j] = (float)cw[(chb[g] + i) * 4 + j];
    for (int r = 0; r < 3; r++) {                     // rows t0-3 .. t0-1 (conv state for negative positions)
        const int row = t0 - 3 + r;
        for (int g = 0; g < 3; g++) {
            const device bfloat16_t* p = row < 0 ? cs + (row + 3) * CD + chb[g] : proj + (size_t)row * PW + chb[g];
            float f[4]; ld4(p, f);
            for (int i = 0; i < 4; i++) win[g][i][r] = f[i];
        }
    }
    const float inv_scale = 0.08838834764831845f;
    const float e2 = 1e-6f * inv_scale * inv_scale;
    const float sc_q = BF(inv_scale * inv_scale), sc_k = BF(inv_scale);
    for (int t = t0; t < t1; t++) {
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
        *((device uint2*)(q + ob)) = uint2(bf2(BF(o[0][0] * qr) * sc_q, BF(o[0][1] * qr) * sc_q), bf2(BF(o[0][2] * qr) * sc_q, BF(o[0][3] * qr) * sc_q));
        *((device uint2*)(k + ob)) = uint2(bf2(BF(o[1][0] * kr) * sc_k, BF(o[1][1] * kr) * sc_k), bf2(BF(o[1][2] * kr) * sc_k, BF(o[1][3] * kr) * sc_k));
        *((device uint2*)(v + ob)) = uint2(bf2(o[2][0], o[2][1]), bf2(o[2][2], o[2][3]));
    }
    """
    hdr = BF16_MATH + """
inline void ld4(const device bfloat16_t* p, thread float* o) {      // 4 consecutive bf16 -> float
    uint2 u = *((const device uint2*)p);
    o[0] = as_type<float>(u.x << 16); o[1] = as_type<float>(u.x & 0xffff0000u);
    o[2] = as_type<float>(u.y << 16); o[3] = as_type<float>(u.y & 0xffff0000u);
}
inline uint bf2(float a, float b) {                                  // two floats -> packed bf16 pair (round to nearest even)
    return (uint)as_type<ushort>(static_cast<bfloat16_t>(a)) | ((uint)as_type<ushort>(static_cast<bfloat16_t>(b)) << 16);
}
"""
    _k[key] = mx.fast.metal_kernel(name="prefill_prep2", input_names=["proj", "cs", "cw"], output_names=["q", "k", "v"],
                                   source=src, header=hdr)
    return _k[key]


PREP_IMPL = os.environ.get("QWEN35_PREP", "v2")
PREP_TB = int(os.environ.get("QWEN35_PREP_TB", "16"))


def conv_prep(proj, conv_state, conv_w):
    """proj [S,8224], conv_state [3,6144] -> q,k [S,16,128] (normalized, scaled), v [S,16,128]."""
    S = proj.shape[0]
    if PREP_IMPL == "v2":
        return _prep2_kernel(PREP_TB)(inputs=[proj, conv_state, conv_w.reshape(-1)],
                                      output_shapes=[(S, 16, 128)] * 3, output_dtypes=[mx.bfloat16] * 3,
                                      grid=(32, 16, -(-S // PREP_TB)), threadgroup=(32, 1, 1),
                                      template=[("TB", PREP_TB)])
    return _prep_kernel()(inputs=[proj, conv_state, conv_w.reshape(-1)],
                          output_shapes=[(S, 16, 128)] * 3, output_dtypes=[mx.bfloat16] * 3,
                          grid=(32, 16, S), threadgroup=(32, 1, 1))


def gated_norm(y, proj, gw):
    S = proj.shape[0]
    return _gnorm_kernel()(inputs=[y.reshape(S, 2048), proj, gw], output_shapes=[(S, 2048)],
                           output_dtypes=[mx.bfloat16], grid=(32, 16, S), threadgroup=(32, 1, 1))[0]


def _linear_attn(cfg, L, x, conv_state, ssm_state, res=None):
    """x [1,S,H] normed. conv_state [1,3,6144]; ssm_state [1,16,128,128]. With `res` [S,H] the projection carries the residual."""
    S = x.shape[1]
    proj = _lin(x, L["in_proj"], "in_proj")[0]           # [S, 8224]
    if GDN_IMPL == "nax" and FUSE_PREP_A:
        q, k, v, Tb, Am, gv = gdn_nax.prep_a(proj, conv_state[0], L["conv_w"], L["A_log"], L["dt_bias"])
        out, new_ssm = gdn_nax.phase_b(q, k, v, Tb, Am, gv, ssm_state, flags=GDN_FLAGS)
    elif GDN_IMPL == "nax":
        q, k, v = conv_prep(proj, conv_state[0], L["conv_w"])
        out, new_ssm = gdn_nax.gdn_chunked(q, k, v, proj, L["A_log"], L["dt_bias"], ssm_state, flags=GDN_FLAGS)
    else:
        q, k, v = conv_prep(proj, conv_state[0], L["conv_w"])
        # mlx-lm's packed delta-rule kernel (4 lanes/row, 32 state elems/lane) is already ALU-bound
        out, new_ssm = gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None], proj[:, 8192:8208][None],
                                          L["A_log"], L["dt_bias"], ssm_state)
    tail = proj[:, :6144]
    new_conv = (tail[S - 3:] if S >= 3 else mx.concatenate([conv_state[0], tail], axis=0)[-3:])[None]
    y = gated_norm(out, proj, L["gnorm"])
    return _lin(y[None], L["out_proj"], "out_proj", res=res), new_conv, new_ssm


def _swiglu_kernel():
    if "sw" in _k:
        return _k["sw"]
    src = _HDR + """
    uint t = threadgroup_position_in_grid.y, c = thread_position_in_grid.x * 4;
    const device bfloat16_t* g = gu + (size_t)t * 7168;
    for (int i = 0; i < 4; i++) {
        float a = (float)g[c + i], b = (float)g[3584 + c + i];
        float sg = mx_silu_bf16(a);
        out[(size_t)t * 3584 + c + i] = static_cast<bfloat16_t>(sg * b);
    }
    """
    _k["sw"] = mx.fast.metal_kernel(name="prefill_swiglu", input_names=["gu"], output_names=["out"], source=src,
                                    header=BF16_MATH)
    return _k["sw"]


def swiglu(gu):
    """gu [S,7168] (gate | up) -> silu(gate) * up, rounded like the unfused bf16 graph. [S,3584]"""
    S = gu.shape[0]
    return _swiglu_kernel()(inputs=[gu], output_shapes=[(S, 3584)], output_dtypes=[mx.bfloat16],
                            grid=(3584 // 4, S, 1), threadgroup=(256, 1, 1))[0]


def _addnorm_kernel():
    if "an" in _k:
        return _k["an"]
    src = _HDR + """
    uint t = threadgroup_position_in_grid.y, tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    threadgroup float red[8];
    size_t o = (size_t)t * 1024 + tid * 4;
    float v[4]; float ss = 0.f;
    for (int i = 0; i < 4; i++) {
        v[i] = BF((float)x[o + i] + (float)r[o + i]);
        xs[o + i] = static_cast<bfloat16_t>(v[i]);
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; s++) tot += red[s];
    float inv = metal::precise::rsqrt(tot / 1024.f + 1e-6f);
    for (int i = 0; i < 4; i++) h[o + i] = static_cast<bfloat16_t>((float)w[tid * 4 + i] * (v[i] * inv));
    """
    _k["an"] = mx.fast.metal_kernel(name="prefill_addnorm", input_names=["x", "r", "w"],
                                    output_names=["xs", "h"], source=src)
    return _k["an"]


def add_norm(x, r, w):
    """x, r [S,1024] bf16 -> (x + r, rmsnorm(x + r) * w)."""
    S = x.shape[0]
    xs, h = _addnorm_kernel()(inputs=[x, r, w], output_shapes=[(S, 1024)] * 2, output_dtypes=[mx.bfloat16] * 2,
                              grid=(256, S, 1), threadgroup=(256, 1, 1))
    return xs, h


def _norm_kernel():
    if "nm" in _k:
        return _k["nm"]
    src = _HDR + """
    uint t = threadgroup_position_in_grid.y, tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    threadgroup float red[8];
    size_t o = (size_t)t * 1024 + tid * 4;
    float v[4]; float ss = 0.f;
    for (int i = 0; i < 4; i++) {
        v[i] = (float)x[o + i];
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; s++) tot += red[s];
    float inv = metal::precise::rsqrt(tot / 1024.f + 1e-6f);
    for (int i = 0; i < 4; i++) h[o + i] = static_cast<bfloat16_t>((float)w[tid * 4 + i] * (v[i] * inv));
    """
    _k["nm"] = mx.fast.metal_kernel(name="prefill_norm", input_names=["x", "w"], output_names=["h"], source=src)
    return _k["nm"]


def norm(x, w):
    """x [S,1024] bf16 -> rmsnorm(x) * w, same arithmetic as add_norm's second output."""
    S = x.shape[0]
    return _norm_kernel()(inputs=[x, w], output_shapes=[(S, 1024)], output_dtypes=[mx.bfloat16], grid=(256, S, 1),
                          threadgroup=(256, 1, 1))[0]


def _normq_kernel():
    if "nq" in _k:
        return _k["nq"]
    src = _HDR + f"#define ROT {int(I8.ROT)}\n" + """
    uint t = threadgroup_position_in_grid.y, tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    threadgroup float red[8], redm[8], xch[1024];
    size_t o = (size_t)t * 1024 + tid * 4;
    float v[4]; float ss = 0.f;
    for (int i = 0; i < 4; i++) {
        v[i] = (float)x[o + i];
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int s = 0; s < 8; s++) tot += red[s];
    float inv = metal::precise::rsqrt(tot / 1024.f + 1e-6f);
    float hv[4];
    for (int i = 0; i < 4; i++) hv[i] = BF((float)w[tid * 4 + i] * (v[i] * inv));       // the bf16 norm output, as prefill.norm
#if ROT
    // fast Walsh-Hadamard transform over index k = tid*4 + i: bits 0-1 in the thread, bits 2-6 across the lanes, bits 7-9 across simdgroups
    {
        float s0 = hv[0] + hv[1], s1 = hv[0] - hv[1], s2 = hv[2] + hv[3], s3 = hv[2] - hv[3];
        hv[0] = s0 + s2; hv[1] = s1 + s3; hv[2] = s0 - s2; hv[3] = s1 - s3;
    }
    for (ushort b = 1; b < 32; b <<= 1)
        for (int i = 0; i < 4; i++) {
            float q = simd_shuffle_xor(hv[i], b);
            hv[i] = (lane & b) ? (q - hv[i]) : (hv[i] + q);
        }
    for (int i = 0; i < 4; i++) xch[sg * 128 + lane * 4 + i] = hv[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; i++) hv[i] = 0.f;
    for (uint r = 0; r < 8; r++) {
        float sgn = (popcount(sg & r) & 1) ? -1.f : 1.f;
        for (int i = 0; i < 4; i++) hv[i] += sgn * xch[r * 128 + lane * 4 + i];
    }
    for (int i = 0; i < 4; i++) hv[i] *= 0.03125f;
#endif
    float am = 0.f;
    for (int i = 0; i < 4; i++) am = max(am, fabs(hv[i]));
    am = simd_max(am);
    if (lane == 0) redm[sg] = am;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0.f;
    for (int s = 0; s < 8; s++) amax = max(amax, redm[s]);
    float is = amax > 0.f ? 127.f / amax : 0.f;
    for (int i = 0; i < 4; i++) hq[o + i] = (int8_t)clamp(rint(hv[i] * is), -127.f, 127.f);
    if (tid == 0) hs[t] = amax / 127.f;
    """
    _k["nq"] = mx.fast.metal_kernel(name="prefill_normq", input_names=["x", "w"], output_names=["hq", "hs"], source=src)
    return _k["nq"]


def norm_q(x, w):
    """x [S,1024] bf16 -> QAct(int8 rmsnorm(x) * w, per-token scale): norm() followed by (the Hadamard rotation, if gemm_i8.ROT, and)
    symmetric per-row int8 quantization."""
    S = x.shape[0]
    hq, hs = _normq_kernel()(inputs=[x, w], output_shapes=[(S, 1024), (S,)], output_dtypes=[mx.int8, mx.float32], grid=(256, S, 1),
                             threadgroup=(256, 1, 1))
    return I8.QAct(hq, hs)


def _mlp_block(L, h, res=None):
    """h [S,H] normed -> down(silu(gate(h)) * up(h)) [S,H] (+ res)"""
    if isinstance(h, I8.QAct):
        a = I8.linear(h, _WQ[id(L["gate_up_il"])][1:], "gate_up", epi=I8.EPI_SWIGLU)
        return _lin(a, L["down"], "down", res=res)
    if _fast_gemm(h.shape[0], "gate_up"):
        a = G.linear(h, L["gate_up_il"], "gate_up", epi=G.EPI_SWIGLU)
    else:
        a = swiglu(h @ L["gate_up"].T)
    return _lin(a, L["down"], "down", res=res)


def _nax_rows(x, w, epi=G.EPI_PLAIN, r=None):
    """NAX GEMM on a short row block (the tail path): always the default tile config, bit-identical per row to the same rows inside a
    larger NAX GEMM (the k loop order does not depend on the tile shape); falls back to mx.matmul below GEMM_MIN_M rows."""
    if x.shape[0] < GEMM_MIN_M or GEMM_IMPL != "nax":
        y = x @ w.T
        if epi == G.EPI_SWIGLU:
            return swiglu(y)
        return y if r is None else r + y
    return G.matmul_nt(x, w, epi=epi, r=r)


def _last_layer_tail(cfg, L, x, h, kv, pos, kv_len, mask, R, nxt):
    """Last (full-attention) layer when only the last R rows of its output are needed (R = 0: none).  K / V need every row, but q,
    the output projection, the MLP and the final norm only the tail rows; each of these ops is row-independent so the rows are
    bit-identical to the full computation.  x [S,H] residual stream, h [S,H] normed input; returns (norm_out [R,H], (K, V))."""
    S = h.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    kvp = _lin(h, L["qkv_kv"], "qkv")                                     # [S, 2 * KV * D]
    k = mx.fast.rms_norm(kvp[:, :KV * D].reshape(1, S, KV, D), L["k_norm"], cfg.eps).transpose(0, 2, 1, 3)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos)
    v = kvp[:, KV * D:].reshape(1, S, KV, D).transpose(0, 2, 1, 3)
    K, V = kv
    K, V = _write(K, k, pos), _write(V, v, pos)
    if R == 0:
        return None, (K, V)
    xt = x[S - R:]
    qg = _nax_rows(h[S - R:], L["qkv_q"])                                  # [R, H * 2D]
    q, gate = mx.split(qg.reshape(1, R, H, 2 * D), 2, axis=-1)
    gate = gate.reshape(1, R, -1)
    q = mx.fast.rms_norm(q, L["q_norm"], cfg.eps).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=pos + S - R)
    tmask = mask if isinstance(mask, str) else mask[S - R:]
    o = mx.fast.scaled_dot_product_attention(q, K[:, :, :kv_len], V[:, :, :kv_len], scale=D ** -0.5, mask=tmask)
    o = o.transpose(0, 2, 1, 3).reshape(1, R, -1) * mx.sigmoid(gate)
    r = _nax_rows(o[0], L["o_proj"], epi=G.EPI_ADD, r=xt)                  # residual rides on the GEMM, as in forward()
    a = _nax_rows(norm(r, L["ln2"]), L["gate_up_il"], epi=G.EPI_SWIGLU)
    return norm(_nax_rows(a, L["down"], epi=G.EPI_ADD, r=r), nxt), (K, V)


def forward(cfg: Cfg, W, tokens, state, pos, kv_len, tail=None):
    """Same contract as model.forward; fast for pos == 0 (single-chunk prefill).
    `tail` = number of trailing rows of the returned hidden state that the caller uses (None = all): the last layer then skips the
    work of the other rows (their K / V are still written), and h comes back as [1, R, H] with R = min(tail, S); for a non-final
    chunk of a chunked prefill pass tail=0 (h is None)."""
    S = tokens.shape[1]
    if _fast_gemm(S):
        prepare(W)
        if I8.NAMES:
            prepare_w8a8(W)
    x = W["embed"][tokens][0]                            # [S,H]
    if isinstance(pos, int) and pos == 0 and kv_len == S:
        mask = "causal"
    else:
        qpos = pos_arr(pos) + mx.arange(S)
        mask = mx.arange(kv_len)[None, :] <= qpos[:, None]
    new = {"kv": [], "conv": [], "ssm": []}
    ki = ci = 0
    layers = W["layers"]
    q_in, _ = _w8a8(cfg, 0, S, len(layers)) if FUSE_ADD else (False, False)
    h = norm_q(x, layers[0]["ln1"]) if q_in else mx.fast.rms_norm(x, layers[0]["ln1"], cfg.eps)
    use_tail = tail is not None and FUSE_ADD and _fast_gemm(S) and not cfg.is_linear(len(layers) - 1)
    for i, L in enumerate(layers):
        if use_tail and i == len(layers) - 1:
            R = min(tail, S)
            h, kv = _last_layer_tail(cfg, L, x, h, state["kv"][ki], pos, kv_len, mask, R, W["norm"])
            new["kv"].append(kv)
            return (None if h is None else h[None]), new
        res = x if FUSE_ADD else None
        if cfg.is_linear(i):
            r, c, s = _linear_attn(cfg, L, h[None], state["conv"][ci], state["ssm"][ci], res=res)
            new["conv"].append(c); new["ssm"].append(s); ci += 1
        else:
            r, kv = _full_attn(cfg, L, h[None], state["kv"][ki], pos, kv_len, mask, mm=_lin, res=res)
            new["kv"].append(kv); ki += 1
        nxt = layers[i + 1]["ln1"] if i + 1 < len(layers) else W["norm"]
        if FUSE_ADD:
            x = r[0]
            _, q_mlp = _w8a8(cfg, i, S, len(layers))
            h = (norm_q if q_mlp else norm)(x, L["ln2"])
            x = _mlp_block(L, h, res=x)
            q_next = i + 1 < len(layers) and _w8a8(cfg, i + 1, S, len(layers))[0]
            h = (norm_q if q_next else norm)(x, nxt)
        else:
            x, h = add_norm(x, r[0], L["ln2"])
            r = _mlp_block(L, h)
            x, h = add_norm(x, r, nxt)
    return h[None], new
