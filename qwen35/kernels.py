"""Hand-written Metal kernels (runtime-compiled through mx.fast.metal_kernel).

fused_gemv: one kernel family covering every dense projection in the model, for M = 1..4 rows
(M > 1 is used to verify speculative drafts: weights are read once, M dot products per weight chunk).
    y[m] = epilogue( W[N,K] @ prologue(x[m]) )
  prologue : none | rmsnorm(x, nw) | gated-rmsnorm (DeltaNet output norm, K=2048)
  epilogue : none | add(res) | swiglu (rows interleaved gate,up) | argmax (per-threadgroup partial)
Each threadgroup has T threads (K/8/T 16-byte chunks per thread) and computes R weight rows; partial sums
are combined with simd_sum + threadgroup memory. All accumulation in fp32, intermediates rounded to bf16
exactly where the unfused MLX graph rounds them.

deltanet: conv + silu + q/k L2 norm + gates + delta-rule recurrence for M tokens in one dispatch,
emitting a state snapshot after every token (speculative rollback = choose a snapshot index).
"""
import mlx.core as mx

from . import wcomp
from .mslib import BF16_MATH

_cache = {}

_HDR = """
    #define BF(v) ((float)static_cast<bfloat16_t>(v))
"""


def _tiles(M, cpt, mtmax=None):
    """Split M rows into balanced tiles whose activations fit in registers (M * CPT * 8 floats per thread; the compiler spills
    badly beyond ~56-100 of them: M=8 at CPT=1 ran 2.5x slower than M=7).  Every tile re-decodes the weights held in registers."""
    mtmax = mtmax or max(1, 7 // cpt)
    n = -(-M // mtmax)
    base, extra = divmod(M, n)
    out, m0 = [], 0
    for i in range(n):
        mt = base + (1 if i < extra else 0)
        out.append((m0, mt)); m0 += mt
    return out


def _reduce_scatter(R, mt, m0):
    """Metal source reducing a[R][mt] over the 32 lanes of the simdgroup and storing part[((m0+m)*R_+r)*NSG+sg].

    Bit-identical to `simd_sum(a[r][m])` for every output: the hardware simd_sum is the xor butterfly with offsets 1,2,4,8,16
    (probed in bench/simdsum_probe.py: equal to it in 4000/4000 random vectors), and this is the same tree evaluated transposed
    (reduce-scatter): with V = R*mt outputs per lane, level s halves the number of values each lane carries and hands the other half
    to the partner lane, so all V outputs cost V-1 shuffles instead of 5*V, and each ends up in a lane of its own."""
    V = R * mt
    Gs = 1
    while Gs < min(V, 32):
        Gs *= 2
    k = Gs.bit_length() - 1
    src, uid = "", 0
    for g in range(-(-V // Gs)):
        cur = []
        for i in range(Gs):
            o = g * Gs + i
            cur.append(f"a[{o // mt}][{o % mt}]" if o < V else "0.f")
        for lv in range(k):
            lb = f"((lane >> {lv}) & 1u) != 0"
            new = []
            for t in range(len(cur) // 2):
                lo, hi = cur[2 * t], cur[2 * t + 1]
                n = f"rs{g}_{lv}_{t}"
                src += f"        float {n} = select({lo}, {hi}, {lb}) + simd_shuffle_xor(select({hi}, {lo}, {lb}), {1 << lv});\n"
                new.append(n)
            cur = new
        v = f"rv{g}"
        src += f"        float {v} = {cur[0]};\n"
        for lv in range(k, 5):
            src += f"        {v} += simd_shuffle_xor({v}, {1 << lv});\n"
        src += (f"        {{ uint o = {g * Gs} + (lane & {Gs - 1}); if (lane < {Gs} && o < {V}) "
                f"part[(({m0} + o % {mt}) * R_ + o / {mt}) * NSG + sg] = {v}; }}\n")
    return src


def _fused_gemv(K, N, M, T, R, prologue, epilogue, eps, dyn_n=False, comp=False, mtmax=None, xf32=False, wf=False):
    key = (K, N, M, T, R, prologue, epilogue, eps, dyn_n, comp, mtmax) + (("xf32",) if xf32 else ()) + (("wf",) if wf else ())
    if key in _cache:
        return _cache[key]
    assert K % 8 == 0 and (K // 8) % T == 0 and T % 32 == 0
    CPT = K // 8 // T
    NSG = T // 32
    ins = ["Wc", "X", "x"] if comp else ["W", "x"]
    if prologue == "rms":
        ins.append("nw")
    if prologue == "gnorm":
        ins += ["proj", "gw"]
        assert K == 2048 and CPT == 1
    if epilogue == "add":
        ins.append("res")
    if dyn_n:
        ins.append("nvalid")
    outs = ["y"] if epilogue != "argmax" else ["pval", "pidx"]
    NV = "(uint)nvalid[0]" if dyn_n else "(uint)N_"

    def norm_src(m0, mt):
        if prologue == "rms":
            return f"""
    {{
        threadgroup float red[{mt}][NSG];
        for (int m = 0; m < {mt}; m++) {{
            float ss = 0.f;
            for (int c = 0; c < CPT; c++) for (int j = 0; j < 8; j++) ss += xr[m][c][j] * xr[m][c][j];
            ss = simd_sum(ss);
            if (lane == 0) red[m][sg] = ss;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int m = 0; m < {mt}; m++) {{
            float tot = 0.f;
            for (int s = 0; s < NSG; s++) tot += red[m][s];
            float inv = metal::precise::rsqrt(tot / (float)K_ + {eps}f);
            const device uint4* nv = (const device uint4*)nw;
            for (int c = 0; c < CPT; c++) {{
                uint4 v = nv[c * T_ + tid];
                uint w[4] = {{v.x, v.y, v.z, v.w}};
                for (int j = 0; j < 4; j++) {{
                    xr[m][c][2*j]   = BF(as_type<float>(w[j] << 16) * (xr[m][c][2*j] * inv));
                    xr[m][c][2*j+1] = BF(as_type<float>(w[j] & 0xffff0000u) * (xr[m][c][2*j+1] * inv));
                }}
            }}
        }}
    }}
"""
        if prologue == "gnorm":
            # x = delta-rule output y (bf16, 16 heads x 128); z lives in proj[m][6144:8192]; gw = fp32 [128]
            return f"""
    for (int m = 0; m < {mt}; m++) {{
        float ss = 0.f;
        for (int j = 0; j < 8; j++) ss += xr[m][0][j] * xr[m][0][j];
        ss += simd_shuffle_xor(ss, 8); ss += simd_shuffle_xor(ss, 4);
        ss += simd_shuffle_xor(ss, 2); ss += simd_shuffle_xor(ss, 1);
        float inv = metal::precise::rsqrt(ss / 128.f + {eps}f);
        for (int j = 0; j < 8; j++) {{
            float zz = (float)proj[({m0} + m) * 8224 + 6144 + tid * 8 + j];
            float n = gw[(tid % 16) * 8 + j] * (xr[m][0][j] * inv);
            xr[m][0][j] = BF(zz / (1.f + metal::precise::exp(-zz)) * n);
        }}
    }}
"""
        return ""

    if epilogue == "none":
        epi = """
    for (uint i = tid; i < M_ * R_; i += T_) {
        uint m = i / R_, r = i % R_;
        float s = 0.f;
        for (int g = 0; g < NSG; g++) s += part[(m * R_ + r) * NSG + g];
        if (row0 + r < (uint)N_) y[m * N_ + row0 + r] = static_cast<bfloat16_t>(s);
    }
"""
    elif epilogue == "add":
        epi = """
    for (uint i = tid; i < M_ * R_; i += T_) {
        uint m = i / R_, r = i % R_;
        float s = 0.f;
        for (int g = 0; g < NSG; g++) s += part[(m * R_ + r) * NSG + g];
        if (row0 + r < (uint)N_) y[m * N_ + row0 + r] = static_cast<bfloat16_t>(BF(s) + (float)res[m * N_ + row0 + r]);
    }
"""
    elif epilogue == "swiglu":
        epi = """
    for (uint i = tid; i < M_ * (R_ / 2); i += T_) {
        uint m = i / (R_ / 2), p = i % (R_ / 2);
        float g = 0.f, u = 0.f;
        for (int s = 0; s < NSG; s++) { g += part[(m * R_ + 2*p) * NSG + s]; u += part[(m * R_ + 2*p+1) * NSG + s]; }
        g = BF(g); u = BF(u);
        float sg_ = mx_silu_bf16(g);
        uint o = row0 / 2 + p;
        if (o < (uint)N_ / 2) y[m * (N_ / 2) + o] = static_cast<bfloat16_t>(sg_ * u);
    }
"""
    elif epilogue == "argmax":
        epi = f"""
    for (uint m = tid; m < M_; m += T_) {{
        float best = -INFINITY; int bi = 0;
        for (int r = 0; r < R_; r++) {{
            float s = 0.f;
            for (int g = 0; g < NSG; g++) s += part[(m * R_ + r) * NSG + g];
            s = BF(s);
            if (row0 + r < {NV} && s > best) {{ best = s; bi = (int)(row0 + r); }}
        }}
        pval[m * NTG + threadgroup_position_in_grid.x] = best;
        pidx[m * NTG + threadgroup_position_in_grid.x] = bi;
    }}
"""
    else:
        raise ValueError(epilogue)

    if comp:
        wload = """
    uint2 ab[R_][CPT]; uint eb[R_][CPT];
    for (int r = 0; r < R_; r++) {
        uint row = min(row0 + r, (uint)N_ - 1);
        const device uint* rw = Wc + (size_t)row * (K_ * 3 / 8);
        for (int c = 0; c < CPT; c++) {
            ab[r][c] = ((const device uint2*)rw)[c * T_ + tid];
            eb[r][c] = rw[K_ / 4 + c * T_ + tid];
        }
    }
"""
        wget = "uint4 d = dec_chunk(ab[r][c], eb[r][c], (const device uint4*)X); uint w[4] = {d.x, d.y, d.z, d.w};"
    else:
        wload = """
    uint4 buf[R_][CPT];
    for (int r = 0; r < R_; r++) {
        uint row = min(row0 + r, (uint)N_ - 1);
        const device uint4* wv = (const device uint4*)(W + (size_t)row * K_);
        for (int c = 0; c < CPT; c++) buf[r][c] = wv[c * T_ + tid];
    }
"""
        wget = "uint w[4] = {buf[r][c].x, buf[r][c].y, buf[r][c].z, buf[r][c].w};"

    def xload(m0, mt):
        if xf32:           # x already widened to fp32 by the producer (exactly the values the bf16 path computes): no per-thread unpack
            return f"""
    for (int m = 0; m < {mt}; m++) {{
        const device float4* xv = (const device float4*)(x + ({m0} + m) * K_);
        for (int c = 0; c < CPT; c++) {{
            float4 a = xv[(c * T_ + tid) * 2], b = xv[(c * T_ + tid) * 2 + 1];
            xr[m][c][0] = a.x; xr[m][c][1] = a.y; xr[m][c][2] = a.z; xr[m][c][3] = a.w;
            xr[m][c][4] = b.x; xr[m][c][5] = b.y; xr[m][c][6] = b.z; xr[m][c][7] = b.w;
        }}
    }}
"""
        return f"""
    for (int m = 0; m < {mt}; m++) {{
        const device uint4* xv = (const device uint4*)(x + ({m0} + m) * K_);
        for (int c = 0; c < CPT; c++) {{
            uint4 v = xv[c * T_ + tid];
            uint w[4] = {{v.x, v.y, v.z, v.w}};
            for (int j = 0; j < 4; j++) {{
                xr[m][c][2*j]   = as_type<float>(w[j] << 16);
                xr[m][c][2*j+1] = as_type<float>(w[j] & 0xffff0000u);
            }}
        }}
    }}
"""

    if wf:
        wload += f"""
    float wfl[R_][CPT][8];
    for (int r = 0; r < R_; r++) for (int c = 0; c < CPT; c++) {{
        {wget}
        for (int j = 0; j < 4; j++) {{ wfl[r][c][2*j] = as_type<float>(w[j] << 16); wfl[r][c][2*j+1] = as_type<float>(w[j] & 0xffff0000u); }}
    }}
"""
    tiles = ""
    for m0, mt in _tiles(M, CPT, mtmax):
        tiles += f"""
    {{
    float xr[{mt}][CPT][8];
    {xload(m0, mt)}
    {norm_src(m0, mt)}
    float a[R_][{mt}];
    for (int r = 0; r < R_; r++) {{
        for (int m = 0; m < {mt}; m++) a[r][m] = 0.f;
        for (int c = 0; c < CPT; c++) {{
            {"" if wf else wget}
            for (int j = 0; j < 4; j++) {{
                {"float lo = wfl[r][c][2*j], hi = wfl[r][c][2*j+1];" if wf else "float lo = as_type<float>(w[j] << 16), hi = as_type<float>(w[j] & 0xffff0000u);"}
                for (int m = 0; m < {mt}; m++) a[r][m] += lo * xr[m][c][2*j] + hi * xr[m][c][2*j+1];
            }}
        }}
    }}
{_reduce_scatter(R, mt, m0)}
    }}
"""
    NTG = -(-N // R)
    src = f"""
    constexpr int K_ = {K}, N_ = {N}, M_ = {M}, T_ = {T}, R_ = {R}, CPT = {CPT}, NSG = {NSG}, NTG = {NTG};
    {_HDR}
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint row0 = threadgroup_position_in_grid.x * R_;
    threadgroup float part[M_ * R_ * NSG];
    {wload}
    {tiles}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {epi}
    """
    name = f"fgemv_{prologue}_{epilogue}_k{K}_n{N}_m{M}_t{T}_r{R}{'_dyn' if dyn_n else ''}{'_wc' if comp else ''}{f'_mt{mtmax}' if mtmax else ''}{'_xf' if xf32 else ''}{'_wf' if wf else ''}"
    k = mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=src,
                             header=BF16_MATH + (wcomp.DEC if comp else ""))
    _cache[key] = k
    return k


def _default_T(K):
    return K // 8 if K // 8 <= 256 else K // 16


def fused_gemv(W, x, *, nw=None, eps=1e-6, res=None, swiglu=False, gnorm=None, T=None, R=None, mtmax=None, wf=False):
    """y = W @ f(x). x: [M, K] (any leading dims flattened to M; M > 7 / CPT is processed in register tiles, each row's result is
    independent of M). Returns [M, N] (N/2 for swiglu)."""
    comp = wcomp.is_comp(W)         # lossless 12-bit (Wc, X) pair (wcomp.py): bit-identical result, 25% fewer bytes
    N, K = wcomp.shape_of(W)
    lead = x.shape[:-1]
    M = 1
    for d in lead:
        M *= d
    T = T or _default_T(K)
    R = R or 4
    pro = "rms" if nw is not None else ("gnorm" if gnorm is not None else "none")
    epi = "swiglu" if swiglu else ("add" if res is not None else "none")
    xf32 = x.dtype == mx.float32
    assert not xf32 or pro == "none"
    k = _fused_gemv(K, N, M, T, R, pro, epi, eps, comp=comp, mtmax=mtmax, xf32=xf32, wf=wf)
    ins = [*W, x.reshape(-1)] if comp else [W, x.reshape(-1)]
    if nw is not None:
        ins.append(nw)
    if gnorm is not None:
        ins += [gnorm[0].reshape(-1), gnorm[1]]
    if res is not None:
        ins.append(res.reshape(-1))
    n_tg = -(-N // R)
    n_out = N // 2 if swiglu else N
    y = k(inputs=ins, output_shapes=[(M * n_out,)], output_dtypes=[mx.bfloat16],
          grid=(n_tg * T, 1, 1), threadgroup=(T, 1, 1))[0]
    return y.reshape(*lead, n_out)


def _argmax_reduce_kernel(n_part, dyn_map=False):
    key = ("amax", n_part, dyn_map)
    if key in _cache:
        return _cache[key]
    ins = ["pval", "pidx"] + (["idmap"] if dyn_map else [])
    ret = "idmap[bj]" if dyn_map else "bj"
    src = f"""
    constexpr int NP = {n_part};
    uint tid = thread_position_in_threadgroup.x;
    uint m = threadgroup_position_in_grid.x;
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int i = tid; i < NP; i += 1024) {{
        float v = pval[m * NP + i]; int ix = pidx[m * NP + i];
        if (v > best || (v == best && ix < bi)) {{ best = v; bi = ix; }}
    }}
    threadgroup float tv[32]; threadgroup int ti[32];
    for (uint off = 16; off > 0; off >>= 1) {{
        float ov = simd_shuffle_down(best, off); int oi = simd_shuffle_down(bi, off);
        if (ov > best || (ov == best && oi < bi)) {{ best = ov; bi = oi; }}
    }}
    if (tid % 32 == 0) {{ tv[tid / 32] = best; ti[tid / 32] = bi; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float b = tv[0]; int bj = ti[0];
        for (int s = 1; s < 32; s++) if (tv[s] > b || (tv[s] == b && ti[s] < bj)) {{ b = tv[s]; bj = ti[s]; }}
        out[m] = {ret};
    }}
    """
    k = mx.fast.metal_kernel(name=f"amax_{n_part}_{int(dyn_map)}", input_names=ins, output_names=["out"], source=src)
    _cache[key] = k
    return k


def head_argmax(Wemb, x, T=128, R=4):
    """Per-row argmax_i (Wemb @ x[m])[i] over the full vocab (x already final-normed). x: [M,K]. -> int32 [M]."""
    N, K = Wemb.shape
    M = x.shape[0]
    k = _fused_gemv(K, N, M, T, R, "none", "argmax", 1e-6)
    n_tg = -(-N // R)
    pval, pidx = k(inputs=[Wemb, x.reshape(-1)], output_shapes=[(M * n_tg,), (M * n_tg,)],
                   output_dtypes=[mx.float32, mx.int32], grid=(n_tg * T, 1, 1), threadgroup=(T, 1, 1))
    r = _argmax_reduce_kernel(n_tg)
    return r(inputs=[pval, pidx], output_shapes=[(M,)], output_dtypes=[mx.int32],
             grid=(M * 1024, 1, 1), threadgroup=(1024, 1, 1))[0]


def subset_argmax(Wsub, idmap, nvalid, x, T=128, R=4):
    """argmax over the first nvalid rows of Wsub (a gathered sub-vocabulary); returns the original token id.
    x: [1,K] (already normed). nvalid: int32 [1] array. idmap: int32 [rows]."""
    N, K = Wsub.shape
    k = _fused_gemv(K, N, 1, T, R, "none", "argmax", 1e-6, dyn_n=True)
    n_tg = -(-N // R)
    pval, pidx = k(inputs=[Wsub, x.reshape(-1), nvalid], output_shapes=[(n_tg,), (n_tg,)],
                   output_dtypes=[mx.float32, mx.int32], grid=(n_tg * T, 1, 1), threadgroup=(T, 1, 1))
    r = _argmax_reduce_kernel(n_tg, dyn_map=True)
    return r(inputs=[pval, pidx, idmap], output_shapes=[(1,)], output_dtypes=[mx.int32],
             grid=(1024, 1, 1), threadgroup=(1024, 1, 1))[0]


def _qsubset_kernel(N, K, T, R):
    key = ("qsub", N, K, T, R)
    if key in _cache:
        return _cache[key]
    assert K % 16 == 0 and (K // 16) % T == 0 and T % 32 == 0
    CPT = K // 16 // T
    NSG = T // 32
    src = f"""
    constexpr int K_ = {K}, N_ = {N}, T_ = {T}, R_ = {R}, CPT = {CPT}, NSG = {NSG};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint tg = threadgroup_position_in_grid.x;
    uint row0 = tg * R_;
    threadgroup float part[R_ * NSG];
    float xr[CPT][16];
    for (int c = 0; c < CPT; c++) {{
        for (int h2 = 0; h2 < 2; h2++) {{
            uint4 v = ((const device uint4*)x)[(c * T_ + tid) * 2 + h2];
            uint w[4] = {{v.x, v.y, v.z, v.w}};
            for (int j = 0; j < 4; j++) {{
                xr[c][h2 * 8 + 2 * j] = as_type<float>(w[j] << 16);
                xr[c][h2 * 8 + 2 * j + 1] = as_type<float>(w[j] & 0xffff0000u);
            }}
        }}
    }}
    uint4 buf[R_][CPT];
    for (int r = 0; r < R_; r++) {{
        uint row = min(row0 + r, (uint)N_ - 1);
        const device uint4* wv = (const device uint4*)(q + (size_t)row * K_);
        for (int c = 0; c < CPT; c++) buf[r][c] = wv[c * T_ + tid];
    }}
    for (int r = 0; r < R_; r++) {{
        float a = 0.f;
        for (int c = 0; c < CPT; c++) {{
            uint w[4] = {{buf[r][c].x, buf[r][c].y, buf[r][c].z, buf[r][c].w}};
            for (int j = 0; j < 4; j++) {{
                float4 f = float4(as_type<char4>(w[j]));
                a += (f.x * xr[c][4 * j] + f.y * xr[c][4 * j + 1]) + (f.z * xr[c][4 * j + 2] + f.w * xr[c][4 * j + 3]);
            }}
        }}
        float v = simd_sum(a);
        if (lane == 0) part[r * NSG + sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float best = -INFINITY; int bi = 0;
        uint nv = nvalid[0];
        for (int r = 0; r < R_; r++) {{
            uint row = row0 + r;
            float s = 0.f;
            for (int g = 0; g < NSG; g++) s += part[r * NSG + g];
            if (row < nv && row < (uint)N_) {{ s *= scale[row]; if (s > best) {{ best = s; bi = (int)row; }} }}
        }}
        pval[tg] = best; pidx[tg] = bi;
    }}
    """
    k = mx.fast.metal_kernel(name=f"qsub_n{N}_k{K}_t{T}_r{R}", input_names=["q", "x", "scale", "nvalid"],
                             output_names=["pval", "pidx"], source=src)
    _cache[key] = k
    return k


def qsubset_argmax(Q, idmap, nvalid, x, T=64, R=8):
    """Approximate draft head: argmax over the first nvalid rows of an int8 per-row-quantised sub-vocabulary Q=(q [N,K] int8,
    scale [N] f32) -> original token id.  Not exact (draft tokens only change the acceptance rate, never the output)."""
    q, scale = Q
    N, K = q.shape
    k = _qsubset_kernel(N, K, T, R)
    n_tg = -(-N // R)
    pval, pidx = k(inputs=[q, x.reshape(-1), scale, nvalid], output_shapes=[(n_tg,), (n_tg,)],
                   output_dtypes=[mx.float32, mx.int32], grid=(n_tg * T, 1, 1), threadgroup=(T, 1, 1))
    r = _argmax_reduce_kernel(n_tg, dyn_map=True)
    return r(inputs=[pval, pidx, idmap], output_shapes=[(1,)], output_dtypes=[mx.int32],
             grid=(1024, 1, 1), threadgroup=(1024, 1, 1))[0]


# ---------------------------------------------------------------------------------------------
# DeltaNet decode / verify
# grid: 16 heads x 4 dv-slices = 64 threadgroups of 4 simdgroups; each simdgroup owns 8 state rows.
# ---------------------------------------------------------------------------------------------
def _deltanet_kernel(M, eps=1e-6):
    key = ("deltanet", M, eps)
    if key in _cache:
        return _cache[key]
    src = f"""
    constexpr int HV = 16, DK = 128, DV = 128, KD = 2048, CD = 6144, RPS = 8, NSL = 4, M_ = {M}, PW = 8224;
    {_HDR}
    uint tgid = threadgroup_position_in_grid.x;
    uint hv = tgid / NSL, slice = tgid % NSL;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row_base = slice * (DV / NSL) + sg * RPS;
    uint c0 = lane * 4;
    const device bfloat16_t* cs_in = cs + (size_t)sel[0] * 3 * CD;
    const device float* st_base = st_in + (size_t)sel[0] * HV * DV * DK;

    // input u_t[ch], t = -3..M-1  (t<0 reads the stored conv window)
    auto inp = [&](int t, uint ch) -> float {{
        return t < 0 ? (float)cs_in[(t + 3) * CD + ch] : (float)proj[t * PW + ch];
    }};
    auto conv = [&](int j, uint ch) -> float {{
        float o = BF(inp(j - 3, ch) * (float)cw[ch * 4] + inp(j - 2, ch) * (float)cw[ch * 4 + 1] +
                     inp(j - 1, ch) * (float)cw[ch * 4 + 2] + inp(j, ch) * (float)cw[ch * 4 + 3]);
        return mx_silu_bf16(o);
    }};

    float4 s[RPS];
    const device float4* sin4 = (const device float4*)st_base;
    for (int r = 0; r < RPS; r++) s[r] = sin4[((hv * DV + row_base + r) * DK + c0) / 4];
    device float4* sout4 = (device float4*)st_out;
    const float inv_scale = 0.08838834764831845f;      // 128^-0.5
    const float e2 = {eps}f * inv_scale * inv_scale;
    const float sc_q = BF(inv_scale * inv_scale), sc_k = BF(inv_scale);
    float A = metal::precise::exp(Alog[hv]);

    for (int j = 0; j < M_; j++) {{
        float q4[4], k4[4];
        float qs = 0.f, ks = 0.f;
        for (int i = 0; i < 4; i++) {{
            q4[i] = conv(j, hv * DK + c0 + i);
            k4[i] = conv(j, KD + hv * DK + c0 + i);
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
        float av = BF((float)proj[j * PW + CD + KD + HV + hv] + (float)dtb[hv]);
        float sp = mx_softplus_bf16(av);
        float g = metal::precise::exp(-A * sp);
        float bv = (float)proj[j * PW + CD + KD + hv];
        float beta = mx_sigmoid_bf16(bv);
        float4 kk = float4(k4[0], k4[1], k4[2], k4[3]);
        float4 qq = float4(q4[0], q4[1], q4[2], q4[3]);
        for (int r = 0; r < RPS; r++) {{
            uint row = row_base + r;
            float v = conv(j, 2 * KD + hv * DV + row);
            float4 sv = s[r] * g;
            float kvm = simd_sum(dot(sv, kk));
            float delta = (v - kvm) * beta;
            sv += kk * delta;
            float yv = simd_sum(dot(sv, qq));
            s[r] = sv;
            sout4[(((size_t)j * HV + hv) * DV * DK + row * DK + c0) / 4] = sv;
            if (lane == 0) {{
                y[j * 2048 + hv * DV + row] = static_cast<bfloat16_t>(yv);
                uint ch = 2 * KD + hv * DV + row;
                for (int t = 0; t < 3; t++) cs_out[(j * 3 + t) * CD + ch] = static_cast<bfloat16_t>(inp(j - 2 + t, ch));
            }}
        }}
        if (slice == 0 && sg == 0) {{
            for (int i = 0; i < 4; i++) {{
                uint chs[2] = {{hv * DK + c0 + i, KD + hv * DK + c0 + i}};
                for (int u = 0; u < 2; u++)
                    for (int t = 0; t < 3; t++)
                        cs_out[(j * 3 + t) * CD + chs[u]] = static_cast<bfloat16_t>(inp(j - 2 + t, chs[u]));
            }}
        }}
    }}
    """
    k = mx.fast.metal_kernel(
        name=f"deltanet_m{M}",
        input_names=["proj", "cs", "cw", "Alog", "dtb", "st_in", "sel"],
        output_names=["y", "cs_out", "st_out"],
        source=src,
        header=BF16_MATH,
    )
    _cache[key] = k
    return k


def deltanet(proj, conv_snap, conv_w, A_log, dt_bias, ssm_snap, sel):
    """proj [M, 8224] bf16 (qkv 6144 | z 2048 | b 16 | a 16 per token).
    conv_snap [S,3,6144] / ssm_snap [S,16,128,128]: state snapshots; the kernel starts from snapshot sel[0].
    Returns y [M,2048], conv snapshots [M,3,6144] (after each token), ssm snapshots [M,16,128,128]."""
    M = proj.shape[0]
    k = _deltanet_kernel(M)
    y, cs, st = k(
        inputs=[proj.reshape(-1), conv_snap.reshape(-1), conv_w.reshape(-1), A_log, dt_bias, ssm_snap.reshape(-1), sel],
        output_shapes=[(M * 2048,), (M * 3 * 6144,), (M * 16 * 128 * 128,)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32],
        grid=(64 * 128, 1, 1), threadgroup=(128, 1, 1),
    )
    return y.reshape(M, 2048), cs.reshape(M, 3, 6144), st.reshape(M, 16, 128, 128)
