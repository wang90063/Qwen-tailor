"""Certified int8 pre-filter for the lm_head argmax (decode is 34% embedding-matrix traffic: 508MB bf16 per token).

The result is bit-identical to `kernels.head_argmax` (argmax over bf16-rounded fp32-accumulated logits, first index on ties):

  K1  approx   one streaming pass over an int8 copy of the embedding (254MB).  For every row i it computes the approximate logit
               l~_i = s_i * <q_i, h> and a RIGOROUS interval [lb_i, ub_i] around the exact bf16 logit v_i of the same row:
                   |<e_i,h> - <s_i q_i,h>|  <=  ||e_i - s_i q_i||_2 * ||h||_2          (Cauchy-Schwarz)
               plus fp32 accumulation error (any summation order) and the bf16 rounding of v_i.  Per threadgroup it also emits
               max lb and max ub.
  K2  collect  one threadgroup per output row: T = max_j lb_j; every row with ub_i >= T is a candidate (the argmax, and every
               tie for it, must satisfy v_i >= v_j >= lb_j for the j that attains T, and v_i <= ub_i).
  K3  exact    the candidates are evaluated on the bf16 table with the SAME arithmetic as head_argmax (fused-gemv inner loop);
               grid-stride over the candidate list, so any candidate count is handled (a flat distribution just runs slower).
  K4  reduce   (value, then lowest index) over the K3 partials.

On real hidden states the per-row bound leaves a median of 3 / p99 of 22 / max of 48 candidates out of 248320 rows
(bench/head_prune_stats.py), so K3/K4 are negligible and the cost is the 254MB pass instead of 508MB.
"""
import os

import mlx.core as mx

from .kernels import _cache, head_argmax

ENABLED = os.environ.get("QWEN35_HEAD_PRUNE", "1") != "0"

# the safety factors of the interval (see the module doc); FP32_ERR is gamma_1023 for u=2^-24 with a factor ~2 of margin because
# both the exact and the approximate dot product contribute and the quantised row norm is slightly larger than ||e_i||
FP32_ERR = 1.3e-4
RND_REL = 0.004          # bf16 rounding of the exact logit: |BF(x)-x| <= 2^-8 |x| = 0.0039 |x|
BOUND_REL = 1.004        # = 1 + 2^-8 (rounds the widened error radius), rounded up


def build_tables(E):
    """E [N,K] bf16 -> dict(q=int8 [N,K], scale=f32 [N], cp=f32 [N]) with cp_i = ||e_i - s_i q_i||_2 + FP32_ERR*||e_i||_2."""
    N, K = E.shape
    qs, ss, cs = [], [], []
    CH = 16384
    for r0 in range(0, N, CH):
        e = E[r0:r0 + CH].astype(mx.float32)
        s = mx.maximum(mx.abs(e).max(-1, keepdims=True) / 127.0, 1e-30)
        q = mx.clip(mx.round(e / s), -127, 127)
        res = e - q * s
        c = mx.sqrt((res * res).sum(-1)) * (1 + 1e-5) + FP32_ERR * mx.sqrt((e * e).sum(-1))
        q = q.astype(mx.int8)
        mx.eval(q, s, c)
        qs.append(q); ss.append(s[:, 0]); cs.append(c)
    t = {"q": mx.concatenate(qs, 0), "scale": mx.concatenate(ss, 0), "cp": mx.concatenate(cs, 0)}
    mx.eval(t)
    return t


def _approx_kernel(N, K, M, T, R, dbg=False, mt=4):
    """M query rows are processed in register tiles of <= mt rows (the int8 weights stay in registers across the tiles): the interval
    does not depend on the summation order, so unlike the exact kernels this one may be tiled freely.  mt=4 keeps M=16 at ~1.2x the
    M=1 time instead of the 40x of one tile of 16 rows (register spills)."""
    key = ("hp_approx", N, K, M, T, R, dbg, mt)
    if key in _cache:
        return _cache[key]
    assert K % 16 == 0 and (K // 16) % T == 0 and T % 32 == 0
    CPT = K // 16 // T
    NSG = T // 32
    NTG = -(-N // R)
    nt = -(-M // mt)
    base, extra = divmod(M, nt)
    tiles, m0 = [], 0
    for i in range(nt):
        n = base + (1 if i < extra else 0)
        tiles.append((m0, n))
        m0 += n
    dbg_store = "lbrow[m * N_ + row] = lo;" if dbg else ""        # test hook: the per-row interval
    blocks = ""
    for m0, n in tiles:
        blocks += f"""
    {{
        constexpr int MT = {n};
        float xr[MT][CPT][16];
        float ss[MT];
        for (int m = 0; m < MT; m++) {{
            ss[m] = 0.f;
            const device uint4* xv = (const device uint4*)(x + ({m0} + m) * K_);
            for (int c = 0; c < CPT; c++) {{
                for (int h2 = 0; h2 < 2; h2++) {{
                    uint4 v = xv[(c * T_ + tid) * 2 + h2];
                    uint w[4] = {{v.x, v.y, v.z, v.w}};
                    for (int j = 0; j < 4; j++) {{
                        float a0 = as_type<float>(w[j] << 16), a1 = as_type<float>(w[j] & 0xffff0000u);
                        xr[m][c][h2 * 8 + 2 * j] = a0;
                        xr[m][c][h2 * 8 + 2 * j + 1] = a1;
                        ss[m] += a0 * a0 + a1 * a1;
                    }}
                }}
            }}
        }}
        for (int r = 0; r < R_; r++) {{
            float a[MT];
            for (int m = 0; m < MT; m++) a[m] = 0.f;
            for (int c = 0; c < CPT; c++) {{
                uint w[4] = {{buf[r][c].x, buf[r][c].y, buf[r][c].z, buf[r][c].w}};
                for (int j = 0; j < 4; j++) {{
                    float4 f = float4(as_type<char4>(w[j]));
                    for (int m = 0; m < MT; m++)
                        a[m] += (f.x * xr[m][c][4 * j] + f.y * xr[m][c][4 * j + 1]) + (f.z * xr[m][c][4 * j + 2] + f.w * xr[m][c][4 * j + 3]);
                }}
            }}
            for (int m = 0; m < MT; m++) {{
                float v = simd_sum(a[m]);
                if (lane == 0) part[(({m0} + m) * (R_ + 1) + r) * NSG + sg] = v;
            }}
        }}
        for (int m = 0; m < MT; m++) {{
            float v = simd_sum(ss[m]);
            if (lane == 0) part[(({m0} + m) * (R_ + 1) + R_) * NSG + sg] = v;
        }}
    }}
"""
    src = f"""
    constexpr int K_ = {K}, N_ = {N}, M_ = {M}, T_ = {T}, R_ = {R}, CPT = {CPT}, NSG = {NSG}, NTG = {NTG};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint tg = threadgroup_position_in_grid.x;
    uint row0 = tg * R_;
    threadgroup float part[M_ * (R_ + 1) * NSG];
    threadgroup float lbs[M_ * R_], ubs[M_ * R_];

    uint4 buf[R_][CPT];
    for (int r = 0; r < R_; r++) {{
        uint row = min(row0 + r, (uint)N_ - 1);
        const device uint4* wv = (const device uint4*)(q + (size_t)row * K_);
        for (int c = 0; c < CPT; c++) buf[r][c] = wv[c * T_ + tid];
    }}
    {blocks}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tt = tid; tt < M_ * R_; tt += T_) {{
        uint m = tt / R_, r = tt % R_;
        uint row = row0 + r;
        float dq = 0.f, h2 = 0.f;
        for (int g = 0; g < NSG; g++) {{ dq += part[(m * (R_ + 1) + r) * NSG + g]; h2 += part[(m * (R_ + 1) + R_) * NSG + g]; }}
        float lo = -INFINITY, hi = -INFINITY;
        if (row < (uint)N_) {{
            float ap = scale[row] * dq;
            float rad = {BOUND_REL}f * (cp[row] * metal::precise::sqrt(h2)) + {RND_REL}f * metal::abs(ap) + 1e-30f;
            lo = ap - rad; hi = ap + rad;
            ub[m * N_ + row] = hi;
            {dbg_store}
        }}
        lbs[tt] = lo; ubs[tt] = hi;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < M_) {{
        float l = lbs[tid * R_], u = ubs[tid * R_];
        for (int r = 1; r < R_; r++) {{ l = max(l, lbs[tid * R_ + r]); u = max(u, ubs[tid * R_ + r]); }}
        lbmax[tid * NTG + tg] = l; ubmax[tid * NTG + tg] = u;
    }}
    """
    k = mx.fast.metal_kernel(name=f"hp_approx_n{N}_m{M}_t{T}_r{R}_mt{mt}{'_dbg' if dbg else ''}", input_names=["q", "x", "scale", "cp"],
                             output_names=["ub", "lbmax", "ubmax"] + (["lbrow"] if dbg else []), source=src)
    _cache[key] = k
    return k


# ---------------------------------------------------------------------------------------------- tensor-op (NAX) K1 for M >= NAX_MIN
# The scalar K1 is ALU-bound at large M (int8 -> fp32 convert + 16 FMAs per weight byte per tile).  The Neural-Accelerator MMA does the
# same x[M,K] . q[N,K]^T on bf16 fragments (int8 -> bf16 is exact, the products are exact in fp32), so the 254MB pass runs at ~the DRAM
# floor.  The interval does not depend on the summation order (FP32_ERR covers any order), so using a different accumulation order here
# changes nothing about the certificate.  K2-K4 and the final argmax are untouched => result still == head_argmax, bit for bit.
NAX_MIN = int(os.environ.get("QWEN35_HEAD_NAX_MIN", "6"))     # smallest M that takes the tensor-op K1 (0 = never)


def _nax_dq_kernel(N, K, M, SN=32, NSG=2):
    """dq[m, n] = sum_k x[m,k] q[n,k] (fp32) for M <= 32 rows.  One simdgroup owns SN weight rows, the K loop streams SK=32 columns."""
    key = ("hp_nax_dq", N, K, M, SN, NSG)
    if key in _cache:
        return _cache[key]
    from . import nax
    assert K % 32 == 0 and SN % 16 == 0 and 1 <= M <= 32
    src = f"""
    constexpr short SK = 32, TM = {-(-M // 16)}, TN = {SN // 16}, TK = SK / 16;
    constexpr int SN_ = {SN}, NSG_ = {NSG}, N_ = {N}, K_ = {K}, M_ = {M};
    uint tg = threadgroup_position_in_grid.x;
    short sg = simdgroup_index_in_threadgroup;
    int n0 = ((int)tg * NSG_ + sg) * SN_;
    if (n0 >= N_) return;
    const device bfloat16_t* A = x;
    const device int8_t* B = q + (size_t)n0 * K_;
    NAXTile<float, TM, TN> D;
    D.clear();
    if (n0 + SN_ <= N_) {{
        STEEL_PRAGMA_NO_UNROLL
        for (int k = 0; k < K_; k += SK) {{
            NAXTile<bfloat16_t, TM, TK> At;
            NAXTile<bfloat16_t, TN, TK> Bt;
            At.load_rows(A + k, K_, (short)M_);
            Bt.load(B + k, K_);
            tile_matmad_nax(D, At, metal::bool_constant<false>{{}}, Bt, metal::bool_constant<true>{{}});
        }}
        D.store_rows(dq + n0, N_, (short)M_);
    }} else {{
        STEEL_PRAGMA_NO_UNROLL
        for (int k = 0; k < K_; k += SK) {{
            NAXTile<bfloat16_t, TM, TK> At;
            NAXTile<bfloat16_t, TN, TK> Bt;
            At.load_rows(A + k, K_, (short)M_);
            Bt.load_safe(B + k, K_, short2(SK, (short)(N_ - n0)));
            tile_matmad_nax(D, At, metal::bool_constant<false>{{}}, Bt, metal::bool_constant<true>{{}});
        }}
        D.store_safe(dq + n0, N_, short2((short)(N_ - n0), (short)M_));
    }}
    """
    k = mx.fast.metal_kernel(name=f"hp_nax_dq_n{N}_m{M}_s{SN}_g{NSG}", input_names=["x", "q"], output_names=["dq"], source=src,
                             header=nax.header(True))
    _cache[key] = k
    return k


def _nax_epi_kernel(N, K, M, R, dbg=False, T=256):
    """dq -> interval: ub[m,row], per-group (R rows) max lb / max ub; the arithmetic of the scalar K1 epilogue.  One thread per group."""
    key = ("hp_nax_epi", N, K, M, R, T, dbg)
    if key in _cache:
        return _cache[key]
    assert K % 32 == 0
    NTG = -(-N // R)
    NSG = T // 32
    dbg_store = "lbrow[m * N_ + row] = lo;" if dbg else ""
    src = f"""
    constexpr int N_ = {N}, K_ = {K}, M_ = {M}, R_ = {R}, NTG = {NTG}, NSG = {NSG}, T_ = {T};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    threadgroup float h2s[M_];
    for (int m = sg; m < M_; m += NSG) {{
        float ss = 0.f;
        for (int k = lane; k < K_; k += 32) {{ float a = (float)x[m * K_ + k]; ss += a * a; }}
        ss = simd_sum(ss);
        if (lane == 0) h2s[m] = ss;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint g = threadgroup_position_in_grid.x * T_ + tid;
    if (g >= (uint)NTG) return;
    uint row0 = g * R_;
    float sc[R_], cc[R_];
    for (int r = 0; r < R_; r++) {{
        uint row = min(row0 + r, (uint)N_ - 1);
        sc[r] = scale[row]; cc[r] = cp[row];
    }}
    for (int m = 0; m < M_; m++) {{
        float sh = metal::precise::sqrt(h2s[m]);
        float lmax = -INFINITY, umax = -INFINITY;
        for (int r = 0; r < R_; r++) {{
            uint row = row0 + r;
            if (row < (uint)N_) {{
                float ap = sc[r] * dq[m * N_ + row];
                float rad = {BOUND_REL}f * (cc[r] * sh) + {RND_REL}f * metal::abs(ap) + 1e-30f;
                float lo = ap - rad, hi = ap + rad;
                ub[m * N_ + row] = hi;
                {dbg_store}
                lmax = max(lmax, lo); umax = max(umax, hi);
            }}
        }}
        lbmax[m * NTG + g] = lmax; ubmax[m * NTG + g] = umax;
    }}
    """
    k = mx.fast.metal_kernel(name=f"hp_nax_epi_n{N}_m{M}_r{R}{'_dbg' if dbg else ''}", input_names=["dq", "x", "scale", "cp"],
                             output_names=["ub", "lbmax", "ubmax"] + (["lbrow"] if dbg else []), source=src)
    _cache[key] = k
    return k


def approx_nax(tab, x, R=8, dbg=False):
    """tensor-op K1: (ub [M*N], lbmax [M*NTG], ubmax [M*NTG]) (+ lbrow [M*N] when dbg) for x [M,K] bf16, M <= 32."""
    q = tab["q"]
    N, K = q.shape
    M = x.shape[0]
    ntg = -(-N // R)
    xf = x.reshape(-1)
    nsn, nsg = 32, 2
    dq, = _nax_dq_kernel(N, K, M, nsn, nsg)(inputs=[xf, q], output_shapes=[(M * N,)], output_dtypes=[mx.float32],
                                            grid=(-(-N // (nsn * nsg)) * 32 * nsg, 1, 1), threadgroup=(32 * nsg, 1, 1))
    T = 256
    return _nax_epi_kernel(N, K, M, R, dbg, T)(
        inputs=[dq, xf, tab["scale"], tab["cp"]], output_shapes=[(M * N,), (M * ntg,), (M * ntg,)] + ([(M * N,)] if dbg else []),
        output_dtypes=[mx.float32] * (4 if dbg else 3), grid=(-(-ntg // T) * T, 1, 1), threadgroup=(T, 1, 1))


def _collect_kernel(N, M, R):
    key = ("hp_collect", N, M, R)
    if key in _cache:
        return _cache[key]
    NTG = -(-N // R)
    src = f"""
    constexpr int N_ = {N}, R_ = {R}, NTG = {NTG};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint m = threadgroup_position_in_grid.x;
    float t = -INFINITY;
    for (int i = tid; i < NTG; i += 1024) t = max(t, lbmax[m * NTG + i]);
    t = simd_max(t);
    threadgroup float tv[32];
    threadgroup atomic_int cnt;
    if (lane == 0) tv[sg] = t;
    if (tid == 0) atomic_store_explicit(&cnt, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    t = tv[0];
    for (int s = 1; s < 32; s++) t = max(t, tv[s]);
    for (int i = tid; i < NTG; i += 1024) {{
        if (ubmax[m * NTG + i] >= t) {{
            for (int r = 0; r < R_; r++) {{
                int row = i * R_ + r;
                if (row < N_ && ub[m * N_ + row] >= t) {{
                    int slot = atomic_fetch_add_explicit(&cnt, 1, memory_order_relaxed);
                    cand[m * N_ + slot] = row;
                }}
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) ncand[m] = atomic_load_explicit(&cnt, memory_order_relaxed);
    """
    k = mx.fast.metal_kernel(name=f"hp_collect_n{N}_m{M}_r{R}", input_names=["ub", "lbmax", "ubmax"],
                             output_names=["cand", "ncand"], source=src)
    _cache[key] = k
    return k


def _exact_kernel(N, K, M, G, T=128, RC=4, dbg=False):
    """candidate rows evaluated exactly (bf16 table), same inner loop as kernels._fused_gemv.  grid (G*T, M)."""
    key = ("hp_exact", N, K, M, G, T, RC, dbg)
    if key in _cache:
        return _cache[key]
    assert K % 8 == 0 and (K // 8) % T == 0 and T % 32 == 0
    CPT = K // 8 // T
    NSG = T // 32
    dbg_store = "vals[m * N_ + base + r] = s;" if dbg else ""          # test hook: the exact value of every candidate, in list order
    src = f"""
    constexpr int K_ = {K}, N_ = {N}, T_ = {T}, R_ = {RC}, CPT = {CPT}, NSG = {NSG}, G_ = {G};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint g = threadgroup_position_in_grid.x;
    uint m = threadgroup_position_in_grid.y;
    threadgroup float part[R_ * NSG];
    int nc = ncand[m];

    float xr[CPT][8];
    {{
        const device uint4* xv = (const device uint4*)(x + m * K_);
        for (int c = 0; c < CPT; c++) {{
            uint4 v = xv[c * T_ + tid];
            uint w[4] = {{v.x, v.y, v.z, v.w}};
            for (int j = 0; j < 4; j++) {{
                xr[c][2*j]   = as_type<float>(w[j] << 16);
                xr[c][2*j+1] = as_type<float>(w[j] & 0xffff0000u);
            }}
        }}
    }}
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int base = g * R_; base < nc; base += G_ * R_) {{
        uint4 buf[R_][CPT];
        for (int r = 0; r < R_; r++) {{
            int j = min(base + r, nc - 1);
            uint row = (uint)cand[m * N_ + j];
            const device uint4* wv = (const device uint4*)(W + (size_t)row * K_);
            for (int c = 0; c < CPT; c++) buf[r][c] = wv[c * T_ + tid];
        }}
        for (int r = 0; r < R_; r++) {{
            float a = 0.f;
            for (int c = 0; c < CPT; c++) {{
                uint w[4] = {{buf[r][c].x, buf[r][c].y, buf[r][c].z, buf[r][c].w}};
                for (int j = 0; j < 4; j++) {{
                    float lo = as_type<float>(w[j] << 16), hi = as_type<float>(w[j] & 0xffff0000u);
                    a += lo * xr[c][2*j] + hi * xr[c][2*j+1];
                }}
            }}
            float v = simd_sum(a);
            if (lane == 0) part[r * NSG + sg] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {{
            for (int r = 0; r < R_; r++) {{
                if (base + r < nc) {{
                    float s = 0.f;
                    for (int q2 = 0; q2 < NSG; q2++) s += part[r * NSG + q2];
                    s = BF(s);
                    int row = cand[m * N_ + base + r];
                    {dbg_store}
                    if (s > best || (s == best && row < bi)) {{ best = s; bi = row; }}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0) {{ pval[m * G_ + g] = best; pidx[m * G_ + g] = bi; }}
    """
    k = mx.fast.metal_kernel(name=f"hp_exact_n{N}_m{M}_g{G}_t{T}{'_dbg' if dbg else ''}", input_names=["W", "x", "cand", "ncand"],
                             output_names=["pval", "pidx"] + (["vals"] if dbg else []), source=src,
                             header="#define BF(v) ((float)static_cast<bfloat16_t>(v))\n")
    _cache[key] = k
    return k


def _final_kernel(n_part):
    key = ("hp_final", n_part)
    if key in _cache:
        return _cache[key]
    src = f"""
    constexpr int NP = {n_part};
    uint tid = thread_position_in_threadgroup.x;
    uint m = threadgroup_position_in_grid.x;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = tid; i < NP; i += 32) {{
        float v = pval[m * NP + i]; int ix = pidx[m * NP + i];
        if (v > best || (v == best && ix < bi)) {{ best = v; bi = ix; }}
    }}
    for (uint off = 16; off > 0; off >>= 1) {{
        float ov = simd_shuffle_down(best, off); int oi = simd_shuffle_down(bi, off);
        if (ov > best || (ov == best && oi < bi)) {{ best = ov; bi = oi; }}
    }}
    if (tid == 0) out[m] = bi == 0x7fffffff ? 0 : bi;
    """
    k = mx.fast.metal_kernel(name=f"hp_final_{n_part}", input_names=["pval", "pidx"], output_names=["out"], source=src)
    _cache[key] = k
    return k


def head_argmax_pruned(Wemb, tab, x, T=64, R=8, G=64):
    """== kernels.head_argmax(Wemb, x) (bit-exact, first-index ties) reading the int8 copy `tab` for the bulk of the work.
    x [M,K] bf16 already final-normed, M <= 4."""
    N, K = Wemb.shape
    M = x.shape[0]
    ntg = -(-N // R)
    xf = x.reshape(-1)
    if NAX_MIN and M >= NAX_MIN and M <= 32:
        ub, lbmax, ubmax = approx_nax(tab, x, R)
    else:
        ub, lbmax, ubmax = _approx_kernel(N, K, M, T, R)(
            inputs=[tab["q"], xf, tab["scale"], tab["cp"]], output_shapes=[(M * N,), (M * ntg,), (M * ntg,)],
            output_dtypes=[mx.float32] * 3, grid=(ntg * T, 1, 1), threadgroup=(T, 1, 1))
    cand, ncand = _collect_kernel(N, M, R)(
        inputs=[ub, lbmax, ubmax], output_shapes=[(M * N,), (M,)], output_dtypes=[mx.int32, mx.int32],
        grid=(M * 1024, 1, 1), threadgroup=(1024, 1, 1))
    pval, pidx = _exact_kernel(N, K, M, G)(
        inputs=[Wemb, xf, cand, ncand], output_shapes=[(M * G,), (M * G,)], output_dtypes=[mx.float32, mx.int32],
        grid=(G * 128, M, 1), threadgroup=(128, 1, 1))
    return _final_kernel(G)(inputs=[pval, pidx], output_shapes=[(M,)], output_dtypes=[mx.int32],
                            grid=(M * 32, 1, 1), threadgroup=(32, 1, 1))[0]


def argmax_head(Wemb, tab, x):
    """The decode/verify/prefill-final vocabulary argmax: the certified int8 pre-filter when its table was built, else the plain
    full-vocabulary kernel (identical results)."""
    return head_argmax_pruned(Wemb, tab, x) if tab is not None else head_argmax(Wemb, x)
