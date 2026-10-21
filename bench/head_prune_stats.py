"""Offline analysis for the certified-int8 lm_head prefilter: on real final-normed hidden states, how many vocabulary rows can
not be excluded from the argmax by a rigorous error bound on an int8-quantised embedding?
  bound (per row i):  |<e_i, h> - <e~_i, h>|  <=  ||e_i - e~_i||_2 * ||h||_2 (+ fp32 accumulation + bf16 rounding slack)
Variants: per-row scale / per-group scales (groups of g elements, bound sum_g c_ig * ||h_g||)."""
import sys, glob, time
import numpy as np, mlx.core as mx
from transformers import AutoTokenizer
from qwen35 import model as m

from qwen35.model import ROOT
cfg = m.Cfg(); W = m.load_weights(cfg)
tk = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-0.8B")
E = W["embed"]; N, K = E.shape

# ---- real hidden states: teacher-forced forward over project text (code + prose) in chunks
text = "\n\n".join(open(f).read() for f in sorted(glob.glob(f"{ROOT}/qwen35/*.py") + glob.glob(f"{ROOT}/bench/*.py")[:20] + [f"{ROOT}/docs/PLAN.md"]))
ids = tk(text, add_special_tokens=False)["input_ids"]
CH = 512
H = []
for s in range(0, min(len(ids), 40 * CH), CH):
    chunk = ids[s:s + CH]
    if len(chunk) < 64: break
    st = m.new_state(cfg, CH + 8)
    h, _ = m.forward(cfg, W, mx.array([chunk]), st, 0, CH + 8)
    mx.eval(h)
    H.append(h[0, 8:])          # skip the first few positions (attention-sink-ish start)
H = mx.concatenate(H, axis=0)
P = H.shape[0]
print(f"{P} hidden states from {len(ids)} tokens of project text")

Ef = E.astype(mx.float32)
Hf = H.astype(mx.float32)
hn = mx.sqrt((Hf * Hf).sum(-1))                                  # [P]
mx.eval(Ef, Hf, hn)


def quant(group):
    """returns (E~ fp32 [N,K], c [N,K/g] residual norms per group)"""
    g = K if group == 0 else group
    Eg = Ef.reshape(N, K // g, g)
    s = mx.abs(Eg).max(-1, keepdims=True) / 127.0
    s = mx.maximum(s, 1e-30)
    q = mx.clip(mx.round(Eg / s), -127, 127)
    Et = q * s
    c = mx.sqrt(((Eg - Et) ** 2).sum(-1))                         # [N, K/g]
    mx.eval(Et, c)
    return Et.reshape(N, K), c


def run(group, label, nb=16):
    Et, c = quant(group)
    g = K if group == 0 else group
    cs = []
    rel = []
    for p0 in range(0, P, nb):
        h = Hf[p0:p0 + nb]                                        # [nb, K]
        L = h @ Ef.T                                              # exact-ish logits fp32 [nb, N]
        Lt = h @ Et.T
        if group == 0:
            B = c[:, 0][None, :] * hn[p0:p0 + nb][:, None]
        else:
            hg = mx.sqrt((h.reshape(h.shape[0], K // g, g) ** 2).sum(-1))   # [nb, K/g]
            B = hg @ c.T                                                     # sum_g c_ig * ||h_g||
        absL = mx.abs(Lt) + B
        slack = absL * (2.0 ** -8) * 2 + 6.2e-5 * 2 * (mx.abs(Lt) + B) * 0 + 1e-3     # bf16 rounding (both ends) ; fp32 accumulation handled below
        # fp32 accumulation bound ~ 6.2e-5 * ||e|| * ||h||  (||e||~0.63 -> use actual ||e_i|| via norms)
        lo = Lt - B - slack
        hi = Lt + B + slack
        T = lo.max(-1, keepdims=True)
        cnt = (hi >= T).sum(-1)
        mx.eval(cnt, L, Lt, B)
        cs.append(np.array(cnt))
        err = mx.abs(L - Lt)
        rel.append((float(mx.max(err / B).item()), float(mx.mean(err / B).item()), float(mx.mean(B).item())))
    cs = np.concatenate(cs)
    r = np.array(rel)
    print(f"{label:22s} candidates: median {np.median(cs):7.0f}  mean {cs.mean():8.0f}  p90 {np.percentile(cs, 90):7.0f}  p99 {np.percentile(cs, 99):7.0f}  max {cs.max():7d}"
          f"   | max(err/B) {r[:, 0].max():.3f}  mean(err/B) {r[:, 1].mean():.4f}  mean B {r[:, 2].mean():.3f}")
    return cs


for grp, lab in [(0, "per-row int8"), (256, "g=256"), (128, "g=128"), (64, "g=64"), (32, "g=32")]:
    run(grp, lab)
