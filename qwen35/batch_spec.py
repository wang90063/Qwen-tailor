"""Batched speculative decoding: one round = (MTP draft chain + verify of M = k+1 tokens) for each of B sequences, ragged positions.

The solo round (spec.py) re-expressed over B independent sequences with the same arithmetic, row by row:

  verify  : the B*M rows go through the same fused GEMV family (rows are independent of M) and `deltanet_bm` / `attn_b(M=M)`, which
            mirror `kernels.deltanet(M)` / the masked MLX SDPA of fast.verify per row, so the verified tokens -- and therefore the
            accepted output -- are bit-identical to plain greedy decoding (tests/batch_spec.py);
  draft   : the MTP layer on B*M rows (the repair batch over the previous verify's positions), then k-1 single-row chain steps, and
            the int8 sub-vocabulary head for B rows at once.  Drafts only decide the acceptance rate, never the output.

State: `SpecPools` -- the recurrent snapshots of the sequences that are in speculative mode (a separate, small pool: the plain pools
of batch.Pools keep their layout), the MTP layer's K / V pool (indexed by the MAIN slot, so it survives leaving speculative mode),
the per-slot history of the final hidden states `hist [NSLOTS+1, max_len, hidden]` (what MTP drafting needs to repair its K / V when a sequence
enters speculative mode; written by prefill, by plain steps (`batch.hrec`) and by the spec -> plain move), and the per-sequence round carry
(previous verify's hidden rows / tokens / ids, position, accepted count `sel`).

  ssm  : 18 x [NS+1, M, 16,128,128] fp32   snapshot j = state after the j-th token of the sequence's last verify (round r starts from
                                            snapshot sel); the kernel reads snapshot sel into registers before it writes any snapshot
                                            (every state element has exactly one owning thread), so the update is in place
  conv : 18 x [(NS+1)*2, M, 3, 6144] bf16   double buffered like batch.Pools.conv (the buffer a round reads is `par`, it writes the
                                            other one); the host counts the rounds it launched, so `par` needs no device round trip
  slot NS is the trash entry of padded rows.
"""
from __future__ import annotations

import mlx.core as mx

from . import mtp as Mtp
from .batch import CD, DK, DV, HV, PW, F32X_MIN, SPLIT_MIN, attn_b, gemv_tiling, gnorm_rows, rms_rows
from .head_prune import argmax_head
from .kernels import _HDR, _argmax_reduce_kernel, _cache, fused_gemv
from .mslib import BF16_MATH
from .model import Cfg


# Time of a speculative round (k=2) over the time of a plain step, B = 1..8 sequences at positions 300-600 (bench/batch_spec_bench.py, M5).
# A round yields `tpr` tokens per sequence (2.35 on short prompts, 2.4-2.9 on code, ~1 if the drafts keep missing), so speculation wins iff tpr > SPEC_COST[B-1]:
# the server (serve.py) switches on its running estimate of tpr.
SPEC_COST = (1.17, 1.41, 1.47, 1.61, 1.86, 1.90, 2.09, 2.29)


# ---------------------------------------------------------------------------------------------
# DeltaNet over M chained tokens per sequence, snapshot pools updated in place
# ---------------------------------------------------------------------------------------------
def _deltanet_bm_kernel(M, eps=1e-6):
    key = ("deltanet_bm", M, eps)
    if key in _cache:
        return _cache[key]
    src = f"""
    constexpr int HV = 16, DK = 128, DV = 128, KD = 2048, CD = 6144, RPS = 8, NSL = 4, M_ = {M}, PW = 8224, NTG = HV * NSL;
    {_HDR}
    uint gtg = threadgroup_position_in_grid.x;
    uint b = gtg / NTG, tgid = gtg % NTG;
    uint hv = tgid / NSL, slice = tgid % NSL;
    uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row_base = slice * (DV / NSL) + sg * RPS;
    uint c0 = lane * 4;
    size_t ss = (size_t)sslots[b];
    uint sl = (uint)sels[b];
    uint par = (uint)pars[b] & 1u;
    const device bfloat16_t* cs_in = cs + (((ss * 2 + par) * M_ + sl) * 3) * CD;
    device bfloat16_t* cs_out = (device bfloat16_t*)cs + (((ss * 2 + (par ^ 1u)) * M_) * 3) * CD;
    const device bfloat16_t* pj = proj + (size_t)b * M_ * PW;
    device float* st_pool = (device float*)st + ss * M_ * HV * DV * DK;
    const device float* st_base = st_pool + (size_t)sl * HV * DV * DK;

    // input u_t[ch], t = -3..M-1  (t<0 reads the stored conv window)
    auto inp = [&](int t, uint ch) -> float {{
        return t < 0 ? (float)cs_in[(t + 3) * CD + ch] : (float)pj[t * PW + ch];
    }};
    auto conv = [&](int j, uint ch) -> float {{
        float o = BF(inp(j - 3, ch) * (float)cw[ch * 4] + inp(j - 2, ch) * (float)cw[ch * 4 + 1] +
                     inp(j - 1, ch) * (float)cw[ch * 4 + 2] + inp(j, ch) * (float)cw[ch * 4 + 3]);
        return mx_silu_bf16(o);
    }};

    float4 s[RPS];
    const device float4* sin4 = (const device float4*)st_base;
    for (int r = 0; r < RPS; r++) s[r] = sin4[((hv * DV + row_base + r) * DK + c0) / 4];
    device float4* sout4 = (device float4*)st_pool;      // every element is read and written by the same thread: in place is safe
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
        float av = BF((float)pj[j * PW + CD + KD + HV + hv] + (float)dtb[hv]);
        float sp = mx_softplus_bf16(av);
        float g = metal::precise::exp(-A * sp);
        float bv = (float)pj[j * PW + CD + KD + hv];
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
                y[((size_t)b * M_ + j) * 2048 + hv * DV + row] = static_cast<bfloat16_t>(yv);
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
        name=f"deltanet_bm{M}", input_names=["proj", "cs", "cw", "Alog", "dtb", "st", "sslots", "sels", "pars"],
        output_names=["y"], source=src, header=BF16_MATH)
    _cache[key] = k
    return k


def deltanet_bm(proj, conv_pool, conv_w, A_log, dt_bias, ssm_pool, sslots, sels, pars, M):
    """proj [B*M, 8224] bf16 (row b*M+j = token j of sequence b); sslots / sels / pars int32 [B]: the sequence's entry in the snapshot
    pools, the snapshot it starts from, and the conv buffer it reads.  Returns y [B*M, 2048] bf16; writes snapshots 0..M-1 of
    ssm_pool[sslot] and conv buffer par^1 IN PLACE."""
    B = sslots.shape[0]
    return _deltanet_bm_kernel(M)(
        inputs=[proj.reshape(-1), conv_pool, conv_w.reshape(-1), A_log, dt_bias, ssm_pool, sslots, sels, pars],
        output_shapes=[(B * M * 2048,)], output_dtypes=[mx.bfloat16],
        grid=(B * 64 * 128, 1, 1), threadgroup=(128, 1, 1))[0].reshape(B * M, 2048)


# ---------------------------------------------------------------------------------------------
# draft head for B rows at once (int8 sub-vocabulary, kernels._qsubset_kernel with NB x-rows per weight pass)
# ---------------------------------------------------------------------------------------------
def _qsubset_b_kernel(N, K, T, R, NB):
    key = ("qsub_b", N, K, T, R, NB)
    if key in _cache:
        return _cache[key]
    assert K % 16 == 0 and (K // 16) % T == 0 and T % 32 == 0
    CPT = K // 16 // T
    NSG = T // 32
    src = f"""
    constexpr int K_ = {K}, N_ = {N}, T_ = {T}, R_ = {R}, CPT = {CPT}, NSG = {NSG}, NB_ = {NB};
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    uint tg = threadgroup_position_in_grid.x;
    uint n_tg = (N_ + R_ - 1) / R_;
    uint row0 = tg * R_;
    threadgroup float part[NB_ * R_ * NSG];
    float xr[NB_][CPT][16];
    for (int b = 0; b < NB_; b++)
        for (int c = 0; c < CPT; c++) {{
            for (int h2 = 0; h2 < 2; h2++) {{
                uint4 v = ((const device uint4*)(x + (size_t)b * K_))[(c * T_ + tid) * 2 + h2];
                uint w[4] = {{v.x, v.y, v.z, v.w}};
                for (int j = 0; j < 4; j++) {{
                    xr[b][c][h2 * 8 + 2 * j] = as_type<float>(w[j] << 16);
                    xr[b][c][h2 * 8 + 2 * j + 1] = as_type<float>(w[j] & 0xffff0000u);
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
        float a[NB_];
        for (int b = 0; b < NB_; b++) a[b] = 0.f;
        for (int c = 0; c < CPT; c++) {{
            uint w[4] = {{buf[r][c].x, buf[r][c].y, buf[r][c].z, buf[r][c].w}};
            for (int j = 0; j < 4; j++) {{
                float4 f = float4(as_type<char4>(w[j]));
                for (int b = 0; b < NB_; b++)
                    a[b] += (f.x * xr[b][c][4 * j] + f.y * xr[b][c][4 * j + 1]) + (f.z * xr[b][c][4 * j + 2] + f.w * xr[b][c][4 * j + 3]);
            }}
        }}
        for (int b = 0; b < NB_; b++) {{
            float v = simd_sum(a[b]);
            if (lane == 0) part[(b * R_ + r) * NSG + sg] = v;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NB_) {{
        uint b = tid;
        float best = -INFINITY; int bi = 0;
        uint nv = nvalid[0];
        for (int r = 0; r < R_; r++) {{
            uint row = row0 + r;
            float s = 0.f;
            for (int g = 0; g < NSG; g++) s += part[(b * R_ + r) * NSG + g];
            if (row < nv && row < (uint)N_) {{ s *= scale[row]; if (s > best) {{ best = s; bi = (int)row; }} }}
        }}
        pval[b * n_tg + tg] = best; pidx[b * n_tg + tg] = bi;
    }}
    """
    k = mx.fast.metal_kernel(name=f"qsub_b_n{N}_k{K}_t{T}_r{R}_b{NB}", input_names=["q", "x", "scale", "nvalid"],
                             output_names=["pval", "pidx"], source=src)
    _cache[key] = k
    return k


def qsubset_argmax_b(Q, idmap, nvalid, x, T=64, R=8):
    """kernels.qsubset_argmax for the B rows of x [B, K] (bf16) in one pass over the table -> int32 [B]; row b == the solo call on x[b]."""
    q, scale = Q
    N, K = q.shape
    B = x.shape[0]
    n_tg = -(-N // R)
    pval, pidx = _qsubset_b_kernel(N, K, T, R, B)(
        inputs=[q, x.reshape(-1), scale, nvalid], output_shapes=[(B * n_tg,), (B * n_tg,)],
        output_dtypes=[mx.float32, mx.int32], grid=(n_tg * T, 1, 1), threadgroup=(T, 1, 1))
    return _argmax_reduce_kernel(n_tg, dyn_map=True)(
        inputs=[pval, pidx, idmap], output_shapes=[(B,)], output_dtypes=[mx.int32],
        grid=(B * 1024, 1, 1), threadgroup=(1024, 1, 1))[0]


# ---------------------------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------------------------
class SpecPools:
    def __init__(self, cfg: Cfg, nspec: int, nslots: int, M: int, max_len: int):
        """nspec: sequences that can be in speculative mode at once; nslots / max_len: the main pools' slots and K / V length
        (the MTP K / V pool is indexed by the main slot)."""
        self.cfg, self.NS, self.M, self.Lmax = cfg, nspec, M, max_len
        nl = sum(cfg.is_linear(i) for i in range(cfg.layers))
        self.conv = [mx.zeros(((nspec + 1) * 2, M, 3, CD), mx.bfloat16) for _ in range(nl)]
        self.ssm = [mx.zeros((nspec + 1, M, HV, DV, DK), mx.float32) for _ in range(nl)]
        self.mK = mx.zeros((nslots + 1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16)
        self.mV = mx.zeros((nslots + 1, cfg.n_kv, max_len, cfg.head_dim), mx.bfloat16)
        self.hist = mx.zeros((nslots + 1, max_len, cfg.hidden), mx.bfloat16)      # final hidden state of every position (indexed by the main slot)
        # round carry, indexed by the spec entry
        self.carry = dict(H=mx.zeros((nspec + 1, M, cfg.hidden), mx.bfloat16), toks=mx.zeros((nspec + 1, M), mx.int32),
                          ids=mx.zeros((nspec + 1, M), mx.int32), pos=mx.zeros((nspec + 1,), mx.int32),
                          sel=mx.zeros((nspec + 1,), mx.int32))
        mx.eval(self.conv, self.ssm, self.mK, self.mV, self.hist, self.carry)

    def as_tree(self):
        return {"conv": self.conv, "ssm": self.ssm, "mK": self.mK, "mV": self.mV}

    @staticmethod
    def entry_bytes(cfg: Cfg, M: int):
        nl = sum(cfg.is_linear(i) for i in range(cfg.layers))
        return nl * M * (HV * DV * DK * 4 + 2 * 3 * CD * 2)

    @staticmethod
    def mtp_bytes(cfg: Cfg, max_len: int):
        """Per main slot: the MTP layer's K / V rows + the hidden-state history."""
        return 2 * cfg.n_kv * max_len * cfg.head_dim * 2 + max_len * cfg.hidden * 2


# ---------------------------------------------------------------------------------------------
# the MTP layer over B sequences x m rows
# ---------------------------------------------------------------------------------------------
def mtp_step_b(cfg: Cfg, MT, embed, toks, hid, mK, mV, slots, pos0, m):
    """== mtp.step per row.  toks [B*m] int32 (the token at position p+1), hid [B*m, H] (post-norm main / MTP hidden at position p) for
    positions pos0[b] .. pos0[b]+m-1 of sequence b (row b*m+j); writes the MTP K / V rows of those positions (in place, slot slots[b]).
    Returns x [B*m, H] (pre-final-norm)."""
    R = toks.shape[0]
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    tl = gemv_tiling(R)
    e = mx.fast.rms_norm(embed[toks], MT["ln_e"], cfg.eps)
    hh = mx.fast.rms_norm(hid, MT["ln_h"], cfg.eps)
    x = fused_gemv(Mtp._w(MT, "fc"), mx.concatenate([e, hh], axis=-1), **tl)
    qkv = fused_gemv(Mtp._w(MT, "qkv"), x, nw=MT["ln1"], **tl)
    q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
    q, gate = mx.split(q.reshape(R, H, 2 * D), 2, axis=-1)
    gate = gate.reshape(R, -1)
    rp = (pos0[:, None] + mx.arange(m, dtype=mx.int32)[None, :]).reshape(-1)
    q = mx.fast.rms_norm(q, MT["q_norm"], cfg.eps).reshape(R, H, 1, D)
    k = mx.fast.rms_norm(k.reshape(R, KV, D), MT["k_norm"], cfg.eps).reshape(R, KV, 1, D)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=rp).reshape(R, H, D)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=rp).reshape(R, KV, D)
    o = attn_b(q, k, v.reshape(R, KV, D), mK, mV, slots, pos0, m)
    o = o.reshape(R, -1) * mx.sigmoid(gate)
    x = fused_gemv(Mtp._w(MT, "o_proj"), o, res=x, **tl)
    act = fused_gemv(Mtp._w(MT, "gate_up_il"), x, nw=MT["ln2"], swiglu=True, **tl)
    return fused_gemv(Mtp._w(MT, "down"), act, res=x, **tl)


def draft_b(cfg, MT, Wsub, idmap, nvalid, x_row):
    """x_row [B,H] pre-final-norm MTP output -> (draft token ids [B] int32, normed hidden [B,H])."""
    hn = mx.fast.rms_norm(x_row, MT["norm"], cfg.eps)
    return qsubset_argmax_b(Wsub, idmap, nvalid, hn), hn


# ---------------------------------------------------------------------------------------------
# verify + round
# ---------------------------------------------------------------------------------------------
def verify_b(cfg: Cfg, F, P, SP, toks, slots, sslots, sel, par, pos, M):
    """== fast.verify per sequence.  toks [B, M] int32 at positions pos[b]..pos[b]+M-1 (slot slots[b]; main K / V pools P, snapshot
    pools SP), starting from recurrent snapshot sel[b].  Returns (ids [B, M] greedy next token after each row, Hn [B*M, H])."""
    B = toks.shape[0]
    R = B * M
    H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
    split = R >= SPLIT_MIN
    tl = gemv_tiling(R)
    xf = R >= F32X_MIN
    rp = (pos[:, None] + mx.arange(M, dtype=mx.int32)[None, :]).reshape(-1)
    x = F["embed"][toks.reshape(-1)]
    ci = ki = 0
    for i, L in enumerate(F["layers"]):
        if cfg.is_linear(i):
            proj = fused_gemv(L["in_proj"], rms_rows(x, L["ln1"], cfg.eps, f32=xf), **tl) if split else fused_gemv(L["in_proj"], x, nw=L["ln1"])
            y = deltanet_bm(proj, SP["conv"][ci], L["conv_w"], L["A_log"], L["dt_bias"], SP["ssm"][ci], sslots, sel, par, M)
            if split:
                x = fused_gemv(L["out_proj"], gnorm_rows(y, proj, L["gnorm"]), res=x, T=256, **{"R": 8, **tl})
            else:
                x = fused_gemv(L["out_proj"], y, res=x, gnorm=(proj, L["gnorm"]), T=256, R=8)
            ci += 1
        else:
            qkv = fused_gemv(L["qkv"], rms_rows(x, L["ln1"], cfg.eps, f32=xf), **tl) if split else fused_gemv(L["qkv"], x, nw=L["ln1"])
            q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
            q, gate = mx.split(q.reshape(R, H, 2 * D), 2, axis=-1)
            gate = gate.reshape(R, -1)
            q = mx.fast.rms_norm(q, L["q_norm"], cfg.eps).reshape(R, H, 1, D)
            k = mx.fast.rms_norm(k.reshape(R, KV, D), L["k_norm"], cfg.eps).reshape(R, KV, 1, D)
            q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=rp).reshape(R, H, D)
            k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=rp).reshape(R, KV, D)
            o = attn_b(q, k, v.reshape(R, KV, D), P["K"][ki], P["V"][ki], slots, pos, M)
            o = o.reshape(R, -1) * mx.sigmoid(gate)
            x = fused_gemv(L["o_proj"], o, res=x, **tl)
            ki += 1
        act = (fused_gemv(L["gate_up_il"], rms_rows(x, L["ln2"], cfg.eps, f32=xf), swiglu=True, **tl) if split
               else fused_gemv(L["gate_up_il"], x, nw=L["ln2"], swiglu=True))
        x = fused_gemv(L["down"], act, res=x, **tl)
    Hn = mx.fast.rms_norm(x, F["norm"], cfg.eps)
    ids = argmax_head(F["embed"], F.get("head_tab"), Hn)
    return ids.reshape(B, M), Hn


def spec_round(cfg: Cfg, k: int, F, MT, Wsub, idmap, nvalid, P, SP, C, slots, sslots, par):
    """One speculative round for B sequences (the batched spec.SpecEngine._round_fn).  P: batch.Pools.as_tree() (main K / V),
    SP: SpecPools.as_tree(), C: SpecPools.carry (small arrays indexed by sslots); slots / sslots / par int32 [B] (padded rows: the trash
    slot / trash entry).  Returns (ids [B, M] -- the first n_new+1 of each row are this round's output --, n_new [B], new carry)."""
    M = k + 1
    B = slots.shape[0]
    Hd = cfg.hidden
    trash = sslots == C["sel"].shape[0] - 1                                        # padded rows: a fixed position, whatever the entry holds
    sel = mx.where(trash, 0, C["sel"][sslots])
    ids_prev, toks_prev, Hprev = C["ids"][sslots], C["toks"][sslots], C["H"][sslots]
    pos_prev = mx.where(trash, 0, C["pos"][sslots])
    x0 = mx.take_along_axis(ids_prev, sel[:, None], axis=1)[:, 0]                  # [B]
    pos = pos_prev + sel + 1                                                       # position of x0
    # ---- draft: MTP batch over the previous verify's positions (repairs its K / V), then the chain ----
    tb = mx.concatenate([toks_prev[:, 1:], ids_prev[:, M - 1:M]], axis=1)
    tb = mx.where(mx.arange(M, dtype=mx.int32)[None, :] == sel[:, None], x0[:, None], tb)
    xb = mtp_step_b(cfg, MT, F["embed"], tb.reshape(-1), Hprev.reshape(B * M, Hd), SP["mK"], SP["mV"], slots, pos_prev, M)
    x_row = mx.take_along_axis(xb.reshape(B, M, Hd), sel[:, None, None], axis=1)[:, 0]
    d, hn = draft_b(cfg, MT, Wsub, idmap, nvalid, x_row)
    drafts = [d]
    for j in range(1, k):
        xj = mtp_step_b(cfg, MT, F["embed"], d, hn, SP["mK"], SP["mV"], slots, pos + (j - 1), 1)
        d, hn = draft_b(cfg, MT, Wsub, idmap, nvalid, xj)
        drafts.append(d)
    toks = mx.concatenate([x0[:, None], mx.stack(drafts, axis=1)], axis=1)         # [B, M]
    # ---- verify ----
    ids, Hn = verify_b(cfg, F, P, SP, toks, slots, sslots, sel, par, pos, M)
    eq = (toks[:, 1:] == ids[:, :k]).astype(mx.int32)
    n_new = mx.sum(mx.cumprod(eq, axis=1), axis=1).astype(mx.int32)
    C2 = dict(C)
    for name, val in (("H", Hn.reshape(B, M, Hd)), ("toks", toks), ("ids", ids), ("pos", pos), ("sel", n_new)):
        a = C[name]
        a[sslots] = val                                    # duplicate (trash entry) rows: any value is fine
        C2[name] = a
    return ids, n_new, C2
