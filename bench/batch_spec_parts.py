"""Where the time of a batched spec round goes: draft part (MTP batch + chain + sub-vocab heads) vs verify (R = B*M rows + head).

usage: python bench/batch_spec_parts.py [LMAX] [POS] [B,B,...] [k]
"""
import sys, time, statistics
import numpy as np, mlx.core as mx
from quiet import Probe
from qwen35.spec import SpecEngine
from qwen35 import batch as Bt, batch_spec as BS

LM = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
POS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
Bs = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else (1, 2, 4, 8)
k = int(sys.argv[4]) if len(sys.argv) > 4 else 2
M = k + 1
sp = SpecEngine(k=k, max_len=LM)
cfg, F, MT = sp.cfg, sp.F, sp.MT
S = max(Bs)
Pm = Bt.Pools(cfg, S, LM)
SPl = BS.SpecPools(cfg, S, S, M, LM)
P, SPt = Pm.as_tree(), SPl.as_tree()
rng = np.random.default_rng(0)
C = {n: mx.array(v) for n, v in SPl.carry.items()}
C["H"] = (mx.random.normal(C["H"].shape) * 0.5).astype(mx.bfloat16)
C["toks"] = mx.array(rng.integers(1000, 30000, C["toks"].shape).astype(np.int32))
C["ids"] = mx.array(rng.integers(1000, 30000, C["ids"].shape).astype(np.int32))
C["pos"] = mx.array(rng.integers(POS // 2, POS, C["pos"].shape).astype(np.int32))
mx.eval(C)
Hd = cfg.hidden


def draft_part(MT_, Ws, im, nv, SP_, C_, sl, ssl):
    sel = C_["sel"][ssl]
    ids_prev, toks_prev, pos_prev, Hprev = C_["ids"][ssl], C_["toks"][ssl], C_["pos"][ssl], C_["H"][ssl]
    B = sl.shape[0]
    x0 = mx.take_along_axis(ids_prev, sel[:, None], axis=1)[:, 0]
    pos = pos_prev + sel + 1
    tb = mx.concatenate([toks_prev[:, 1:], ids_prev[:, M - 1:M]], axis=1)
    tb = mx.where(mx.arange(M, dtype=mx.int32)[None, :] == sel[:, None], x0[:, None], tb)
    xb = BS.mtp_step_b(cfg, MT_, F["embed"], tb.reshape(-1), Hprev.reshape(B * M, Hd), SP_["mK"], SP_["mV"], sl, pos_prev, M)
    x_row = mx.take_along_axis(xb.reshape(B, M, Hd), sel[:, None, None], axis=1)[:, 0]
    d, hn = BS.draft_b(cfg, MT_, Ws, im, nv, x_row)
    drafts = [d]
    for j in range(1, k):
        xj = BS.mtp_step_b(cfg, MT_, F["embed"], d, hn, SP_["mK"], SP_["mV"], sl, pos + (j - 1), 1)
        d, hn = BS.draft_b(cfg, MT_, Ws, im, nv, xj)
        drafts.append(d)
    return mx.stack(drafts, axis=1)


def verify_part(F_, P_, SP_, toks, sl, ssl, sel, par, pos):
    return BS.verify_b(cfg, F_, P_, SP_, toks, sl, ssl, sel, par, pos, M)


fd = mx.compile(draft_part)
fv = mx.compile(verify_part)
probe = Probe(tol=1.15)


def run(B, which, n=20):
    sl = mx.array(np.arange(B, dtype=np.int32))
    toks = mx.array(rng.integers(1000, 30000, (B, M)).astype(np.int32))
    sel = mx.zeros((B,), dtype=mx.int32)
    pos = mx.array(rng.integers(POS // 2, POS, B).astype(np.int32))
    mx.eval(toks, pos)
    t = time.perf_counter()
    for r in range(n):
        par = mx.full((B,), r & 1, dtype=mx.int32)
        if which == "draft":
            o = fd(MT, sp.Wsub, sp.idmap, sp.nvalid, SPt, C, sl, sl)
            mx.async_eval(o)
        else:
            o, h = fv(F, P, SPt, toks, sl, sl, sel, par, pos)
            mx.async_eval(o, h)
    mx.eval(o)
    return (time.perf_counter() - t) / n


def med(w, B):
    ts = []
    for _ in range(12):
        dt, ok = probe.sample(lambda: run(B, w))
        if ok:
            ts.append(dt)
    return statistics.median(ts or [run(B, w)]) * 1e3


print(f"Lmax {LM}  pos ~{POS // 2}-{POS}  k={k} (M={M})")
for B in Bs:
    for w in ("draft", "verify"):
        run(B, w, 3); run(B, w, 3)
    d, v = med("draft", B), med("verify", B)
    print(f"B={B:2d}  draft {d:6.2f} ms   verify (R={B * M}) {v:6.2f} ms   sum {d + v:6.2f}", flush=True)
