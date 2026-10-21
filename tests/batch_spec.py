"""Batched speculative decoding (batch_spec.py): kernels row-for-row == the solo ones, and whole rounds generate Engine.generate's tokens.

* deltanet_bm == solo deltanet (M=3) per sequence: ragged entries / start snapshots / conv buffers, y + every snapshot, in place
* attn_b(M=3) == the masked MLX SDPA of fast.verify per row (ragged positions), pool rows below pos untouched, rows pos..pos+2 appended
* qsubset_argmax_b == solo qsubset_argmax per row
* mtp_step_b == mtp.step per sequence (ragged positions)
* batched rounds (compiled, B = 1..4 ragged sequences + a pad row) == Engine.generate tokens; draft acceptance == the solo SpecEngine's
"""
import sys, time
import numpy as np
import mlx.core as mx

from qwen35 import batch as Bt, batch_spec as BS, mtp as Mtp
from qwen35.engine import Engine
from qwen35.kernels import deltanet, qsubset_argmax
from qwen35.model import ROOT
from qwen35.spec import SpecEngine
import json

fails = []


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


def eq(a, b):
    return bool(mx.array_equal(a, b))


rng = np.random.default_rng(0)
K_DRAFT = 2
M = K_DRAFT + 1
sp = SpecEngine(k=K_DRAFT, max_len=1024)
cfg, F, MT = sp.cfg, sp.F, sp.MT
L0 = next(L for i, L in enumerate(F["layers"]) if cfg.is_linear(i))

# ---- deltanet_bm ----------------------------------------------------------------------------------------------------------------
NS, B = 6, 4
sslots = np.array([4, 1, 5, 0], dtype=np.int32)
sels = np.array([2, 0, 1, 2], dtype=np.int32)
pars = np.array([1, 0, 0, 1], dtype=np.int32)
conv_pool = mx.array(rng.standard_normal(((NS + 1) * 2, M, 3, 6144)).astype(np.float32)).astype(mx.bfloat16)
ssm_pool = mx.array((rng.standard_normal((NS + 1, M, 16, 128, 128)) * 0.1).astype(np.float32))
proj = mx.array((rng.standard_normal((B * M, 8224)) * 1.5).astype(np.float32)).astype(mx.bfloat16)
mx.eval(conv_pool, ssm_pool, proj)
cp0, sp0 = mx.array(conv_pool), mx.array(ssm_pool)
mx.eval(cp0, sp0)
y = BS.deltanet_bm(proj, conv_pool, L0["conv_w"], L0["A_log"], L0["dt_bias"], ssm_pool, mx.array(sslots), mx.array(sels), mx.array(pars), M)
mx.eval(y, conv_pool, ssm_pool)
ok_y = ok_c = ok_s = ok_keep = True
for b in range(B):
    s_, sl, pr = int(sslots[b]), int(sels[b]), int(pars[b])
    yr, cr, sr = deltanet(proj[b * M:(b + 1) * M], cp0[s_ * 2 + pr], L0["conv_w"], L0["A_log"], L0["dt_bias"], sp0[s_],
                          mx.array([sl], dtype=mx.int32))
    ok_y &= eq(y[b * M:(b + 1) * M], yr)
    ok_c &= eq(conv_pool[s_ * 2 + (pr ^ 1)], cr)
    ok_s &= eq(ssm_pool[s_], sr)
    ok_keep &= eq(conv_pool[s_ * 2 + pr], cp0[s_ * 2 + pr])
untouched = [e for e in range(NS + 1) if e not in set(sslots.tolist())]
ok_keep &= all(eq(ssm_pool[e], sp0[e]) and eq(conv_pool[e * 2], cp0[e * 2]) and eq(conv_pool[e * 2 + 1], cp0[e * 2 + 1]) for e in untouched)
check("deltanet_bm: y == solo deltanet per sequence (4 entries, ragged start snapshots / conv buffers)", ok_y)
check("deltanet_bm: all M conv snapshots (other buffer) == solo", ok_c)
check("deltanet_bm: all M ssm snapshots (in place) == solo", ok_s)
check("deltanet_bm: the buffer it read and the untouched entries are preserved", ok_keep)

# ---- attn_b with M query rows per sequence --------------------------------------------------------------------------------------
Lmax, H, KV, D = 1024, 8, 2, 256
S = 6
Kp = mx.array(rng.standard_normal((S + 1, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
Vp = mx.array(rng.standard_normal((S + 1, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
Kref, Vref = mx.array(Kp), mx.array(Vp)
B = 5
slots = np.array([3, 0, 5, 1, 2], dtype=np.int32)
pos = np.array([0, 1, 30, 300, 1000], dtype=np.int32)
q = mx.array(rng.standard_normal((B * M, H, D)).astype(np.float32)).astype(mx.bfloat16)
kn = mx.array(rng.standard_normal((B * M, KV, D)).astype(np.float32)).astype(mx.bfloat16)
vn = mx.array(rng.standard_normal((B * M, KV, D)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(Kp, Vp, Kref, Vref, q, kn, vn)
o = Bt.attn_b(q, kn, vn, Kp, Vp, mx.array(slots), mx.array(pos), M)
mx.eval(o)
ok_o = ok_a = ok_p = True
for b in range(B):
    s_, p = int(slots[b]), int(pos[b])
    K1, V1 = Kref[s_:s_ + 1], Vref[s_:s_ + 1]
    kk = kn[b * M:(b + 1) * M].transpose(1, 0, 2)[None]            # [1, KV, M, D]
    vv = vn[b * M:(b + 1) * M].transpose(1, 0, 2)[None]
    K1 = mx.slice_update(K1, kk, mx.array([p], dtype=mx.int32), (2,))
    V1 = mx.slice_update(V1, vv, mx.array([p], dtype=mx.int32), (2,))
    Lb = min(Lmax, 256 * (-(-(p + M) // 256)))
    qq = q[b * M:(b + 1) * M].transpose(1, 0, 2)[None]              # [1, H, M, D]
    mask = mx.arange(Lb)[None, :] <= (p + mx.arange(M))[:, None]
    r = mx.fast.scaled_dot_product_attention(qq, K1[:, :, :Lb], V1[:, :, :Lb], scale=D ** -0.5, mask=mask)       # [1, H, M, D]
    ok_o &= eq(o[b * M:(b + 1) * M], r[0].transpose(1, 0, 2))
    ok_a &= eq(Kp[s_, :, p:p + M], kk[0]) and eq(Vp[s_, :, p:p + M], vv[0])
    ok_p &= eq(Kp[s_, :, :p], Kref[s_, :, :p]) and eq(Kp[s_, :, p + M:], Kref[s_, :, p + M:])
check("attn_b(M=3): o == masked MLX sdpa (verify layout) bit for bit, positions 0..1000", ok_o)
check("attn_b(M=3): new K / V rows appended at pos..pos+2", ok_a)
check("attn_b(M=3): every other row of the pool untouched", ok_p and eq(Kp[6], Kref[6]))

# ---- qsubset_argmax_b -----------------------------------------------------------------------------------------------------------
ok_q = True
for Bq in (1, 2, 3, 4):
    xq = (mx.random.normal((Bq, 1024)) * 2).astype(mx.bfloat16)
    mx.eval(xq)
    got = BS.qsubset_argmax_b(sp.Wsub, sp.idmap, sp.nvalid, xq)
    ref = mx.concatenate([qsubset_argmax(sp.Wsub, sp.idmap, sp.nvalid, xq[b:b + 1]) for b in range(Bq)])
    ok_q &= eq(got, ref)
check("qsubset_argmax_b == solo qsubset_argmax per row (B = 1..4)", ok_q)

# ---- mtp_step_b -----------------------------------------------------------------------------------------------------------------
B = 3
mslots = np.array([2, 0, 3], dtype=np.int32)
mpos = np.array([5, 77, 400], dtype=np.int32)
mK = mx.array(rng.standard_normal((4, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
mV = mx.array(rng.standard_normal((4, KV, Lmax, D)).astype(np.float32)).astype(mx.bfloat16)
mK0, mV0 = mx.array(mK), mx.array(mV)
toksm = mx.array(rng.integers(0, 100000, B * M).astype(np.int32))
hid = mx.array(rng.standard_normal((B * M, 1024)).astype(np.float32)).astype(mx.bfloat16)
mx.eval(mK, mV, mK0, mV0, toksm, hid)
xb = BS.mtp_step_b(cfg, MT, F["embed"], toksm, hid, mK, mV, mx.array(mslots), mx.array(mpos), M)
mx.eval(xb, mK, mV)
ok_x = ok_kv = True
for b in range(B):
    s_, p = int(mslots[b]), int(mpos[b])
    xr, (K2, V2) = Mtp.step(cfg, MT, F["embed"], toksm[b * M:(b + 1) * M], hid[b * M:(b + 1) * M], (mK0[s_:s_ + 1], mV0[s_:s_ + 1]),
                            mx.array(p, dtype=mx.int32), 512 if p + M <= 512 else 1024)
    ok_x &= eq(xb[b * M:(b + 1) * M], xr)
    ok_kv &= eq(mK[s_], K2[0]) and eq(mV[s_], V2[0])
check("mtp_step_b == mtp.step per sequence (ragged positions, M=3 rows)", ok_x)
check("mtp_step_b: MTP K / V rows == solo", ok_kv)

# ---- whole rounds ---------------------------------------------------------------------------------------------------------------
g = json.load(open(ROOT / "results" / "baseline_mlx_lm.json"))["golden"]
prompts = [d["prompt_ids"] for d in g.values()]
text = [int(t) for t in rng.integers(2000, 50000, 90)]
prompts = (prompts + [text])[:5]
NEW = 48
eng = Engine(max_len=1024, like=sp)
solo = [eng.generate(p, NEW) for p in prompts]
solo_spec = []
for p in prompts:
    st = {}
    solo_spec.append((sp.generate(p, NEW, st), st))

nslots = 6
Pm = Bt.Pools(cfg, nslots, 1024)
SPl = BS.SpecPools(cfg, 4, nslots, M, 1024)
P = Pm.as_tree()
SPt = SPl.as_tree()
C = SPl.carry


def load_seq(slot, ss, carry, n):
    st, sel_, mkv, Hprev, tprev, iprev, pprev = carry
    single = {"kv": st["kv"], "conv": [c[:1] for c in st["conv"]], "ssm": [s_[:1] for s_ in st["ssm"]]}
    Pm.load(slot, single, n)
    toks = []
    for i in range(len(SPt["conv"])):
        toks.append(Bt.blit(SPt["conv"][i], st["conv"][i][0], 3 * 6144, (ss * 2) * M * 3 * 6144, 0))
        toks.append(Bt.blit(SPt["ssm"][i], st["ssm"][i][0], 16 * 128 * 128, ss * M * 16 * 128 * 128, 0))
    for pool, src in ((SPt["mK"], mkv[0]), (SPt["mV"], mkv[1])):
        toks.append(Bt.blit(pool, src, (n - 1) * D, slot * KV * 1024 * D, 0, ny=KV, dstride=1024 * D, sstride=1024 * D))
    mx.eval(toks)
    for name, val in (("H", Hprev), ("toks", tprev), ("ids", iprev), ("pos", pprev), ("sel", sel_)):
        C[name][ss] = val.reshape(C[name][ss].shape)
    mx.eval(C)


def run(sel_prompts, ragged_start=0):
    """Batched rounds for the given prompts (slots / entries scattered); returns per-sequence token lists + rounds + accepted."""
    n = len(sel_prompts)
    seqs = []
    for i, p in enumerate(sel_prompts):
        carry, first, S_ = sp.prefill(p)
        load_seq(i + 1, (n - 1 - i) % 4 if n <= 4 else i % 4, carry, S_)
        seqs.append(dict(slot=i + 1, ss=(n - 1 - i) % 4, out=[first], rounds=0, acc=0, extra=0))
    Bpad = n + (1 if n in (2, 3) else 0)
    fn = mx.compile(lambda F_, MT_, Ws, im, nv, P_, SP_, C_, sl, ssl, par: BS.spec_round(cfg, K_DRAFT, F_, MT_, Ws, im, nv, P_, SP_, C_, sl, ssl, par))
    rnd = 0
    while any(len(s_["out"]) < NEW or s_["extra"] < 1 for s_ in seqs):
        sl = np.full((Bpad,), nslots, dtype=np.int32)
        ssl = np.full((Bpad,), 4, dtype=np.int32)
        par = np.zeros((Bpad,), dtype=np.int32)
        act = [s_ for s_ in seqs if len(s_["out"]) < NEW or s_["extra"] < 1]       # + the one round the solo pipeline launches before it drains
        for i, s_ in enumerate(act):
            sl[i], ssl[i], par[i] = s_["slot"], s_["ss"], s_["rounds"] & 1
        ids, nn, C2 = fn(F, MT, sp.Wsub, sp.idmap, sp.nvalid, P, SPt, C, mx.array(sl), mx.array(ssl), mx.array(par))
        for kname in C:
            C[kname] = C2[kname]
        ids_, nn_ = np.array(ids), np.array(nn)
        for i, s_ in enumerate(act):
            s_["extra"] += len(s_["out"]) >= NEW
            s_["out"].extend(ids_[i, :nn_[i] + 1].tolist())
            s_["rounds"] += 1
            s_["acc"] += int(nn_[i])
    return seqs


for nreq in (1, 2, 3, 4):
    seqs = run(prompts[:nreq])
    bad = [i for i, s_ in enumerate(seqs) if s_["out"][:NEW] != solo[i][:NEW]]
    check(f"batched spec rounds (B={nreq}, ragged, compiled): tokens == Engine.generate (bad {bad})", not bad)
    # the draft path is bit-identical to the solo one too: same acceptance
    diff = [i for i, s_ in enumerate(seqs) if (s_["rounds"], s_["acc"]) != (solo_spec[i][1]["rounds"], solo_spec[i][1]["accepted"])]
    check(f"batched spec rounds (B={nreq}): rounds / accepted == solo SpecEngine (differ {diff}; "
          f"{[(s_['rounds'], s_['acc'], solo_spec[i][1]['rounds'], solo_spec[i][1]['accepted']) for i, s_ in enumerate(seqs)]})", not diff)

print("FAILURES:", len(fails))
sys.exit(1 if fails else 0)
