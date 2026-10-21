"""Fused conv_prep + GDN phase-A kernel (gdn_nax.prep_a) is bit-identical to the separate conv_prep -> phase-A kernels, and the
whole prefill forward is bit-identical with the fusion on / off (hidden state, K/V caches, conv / ssm states)."""
import sys

import numpy as np
import mlx.core as mx

from qwen35 import model as m, prefill as pf, gdn_nax as g

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


def same(a, b):
    u = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(np.array(a.view(u)), np.array(b.view(u))))


cfg = m.Cfg()
W = m.load_weights(cfg)
L = W["layers"][0]
mx.random.seed(1)
ka, kb = g._kernels()
for S in (2048, 2000, 37, 16, 5):                       # full chunks, a partial last chunk, a single partial chunk
    x = (mx.random.normal((S, 1024)) * 0.5).astype(mx.bfloat16)
    proj = x @ L["in_proj"].T
    cs = (mx.random.normal((3, 6144)) * 0.5).astype(mx.bfloat16)
    q, k, v = pf.conv_prep(proj, cs, L["conv_w"])
    NC = (S + 15) // 16
    Tb, Am, gv = ka(inputs=[q, k, proj, L["A_log"], L["dt_bias"]], output_shapes=[(NC, 16, 256), (NC, 16, 256), (NC, 16, 32)],
                    output_dtypes=[mx.float32] * 3, grid=(32, NC, 16), threadgroup=(32, 1, 1))
    got = g.prep_a(proj, cs, L["conv_w"], L["A_log"], L["dt_bias"])
    mx.eval(q, k, v, Tb, Am, gv, *got)
    check(f"prep_a S={S}: q,k,v,T,A,gate-sums bit-identical",
          all(same(a, b) for a, b in zip((q, k, v, Tb, Am, gv), got)))

# whole forward, fusion on vs off
ids = mx.array([[(i * 7919 + 13) % 150000 for i in range(700)]])


def run(flag):
    pf.FUSE_PREP_A = flag
    st = m.new_state(cfg, 708)
    h, st = pf.forward(cfg, W, ids, st, 0, 700, tail=128)
    mx.eval(h, st)
    return h, st


ha, sa = run(True)
hb, sb = run(False)
pf.FUSE_PREP_A = True
check("forward S=700 hidden bit-identical", same(ha, hb))
check("forward S=700 states bit-identical",
      all(same(a, b) for key in sa for xa, xb in zip(sa[key], sb[key])
          for a, b in (zip(xa, xb) if isinstance(xa, tuple) else [(xa, xb)])))

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
