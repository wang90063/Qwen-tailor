"""In-graph cost attribution of the prefill forward: each variant replaces one component's output by a constant array
(no kernel, no data dependence), so  T(full) - T(variant)  is what that component really costs inside the big graph
(standalone op timings are distorted by eval overhead and cache state).  Quiet-GPU sampling (bench/quiet.py).
usage: prefill_ablate.py [S] [rounds] [variant ...]   variant = [nax:|mlx:]<component>   components:
  full in_proj qkv out_proj gate_up down gdn conv_prep gnorm add_norm sdpa"""
import sys, time, statistics
import mlx.core as mx
from quiet import Probe
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35 import model as m, prefill as pf

S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
VARIANTS = sys.argv[3:] or ["nax:full", "nax:in_proj", "nax:qkv", "nax:out_proj", "nax:gate_up", "nax:down", "nax:gdn", "nax:conv_prep",
                            "nax:gnorm", "nax:add_norm", "nax:sdpa", "mlx:full", "mlx:in_proj", "mlx:gate_up", "mlx:down"]
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
cfg = m.Cfg(); W = m.load_weights(cfg)
ids = mx.array([make_prompt(tk, S)])

_fake = {}


def fake(shape, dtype=mx.bfloat16):
    key = (tuple(shape), dtype)
    if key not in _fake:
        _fake[key] = (mx.random.normal(shape) * 0.5).astype(dtype); mx.eval(_fake[key])
    return _fake[key]


orig = dict(lin=pf._lin, mlp=pf._mlp_block, conv_prep=pf.conv_prep, gdn=pf.gdn_nax.gdn_chunked, gated_norm=pf.gated_norm,
            add_norm=pf.add_norm, sdpa=mx.fast.scaled_dot_product_attention)


def install(cut):
    pf._lin, pf._mlp_block, pf.conv_prep, pf.gated_norm, pf.add_norm = (orig[k] for k in ("lin", "mlp", "conv_prep", "gated_norm", "add_norm"))
    pf.gdn_nax.gdn_chunked = orig["gdn"]; mx.fast.scaled_dot_product_attention = orig["sdpa"]
    if cut in ("in_proj", "qkv", "out_proj", "down"):
        def lin(x, w, name, **kw):
            if name == cut:
                return fake((*x.shape[:-1], w.shape[0]))
            return orig["lin"](x, w, name, **kw)
        pf._lin = lin
    elif cut == "gate_up":
        def mlp(L, h, **kw):
            return orig["lin"](fake((h.shape[0], 3584)), L["down"], "down", **kw)
        pf._mlp_block = mlp
    elif cut == "gdn":
        pf.gdn_nax.gdn_chunked = lambda q, k, v, proj, A, dtb, st, **kw: (fake(q.shape), st)
    elif cut == "conv_prep":
        pf.conv_prep = lambda proj, cs, cw: tuple(fake((proj.shape[0], 16, 128)) for _ in range(3))
    elif cut == "gnorm":
        pf.gated_norm = lambda y, proj, gw: fake((proj.shape[0], 2048))
    elif cut == "add_norm":
        pf.add_norm = lambda x, r, w: (fake(x.shape), fake(x.shape))
    elif cut == "sdpa":
        mx.fast.scaled_dot_product_attention = lambda q, k, v, scale, mask: fake(q.shape)
    elif cut != "full":
        raise SystemExit(f"unknown component {cut}")


def run(variant):
    impl, cut = variant.split(":")
    pf.GEMM_IMPL = impl
    install(cut)
    st = m.new_state(cfg, S + 8)
    t = time.perf_counter()
    h, st = pf.forward(cfg, W, ids, st, 0, S)
    mx.eval(h[:, -1] @ W["embed"].T, st)
    return time.perf_counter() - t


for v in VARIANTS:
    run(v)
probe = Probe()
ts = {v: [] for v in VARIANTS}
quiet = {v: 0 for v in VARIANTS}
for _ in range(ROUNDS):
    for v in VARIANTS:
        dt, ok = probe.sample(lambda: run(v))
        ts[v].append(dt); quiet[v] += ok
base = {impl: statistics.median(ts[f"{impl}:full"]) for impl in ("nax", "mlx") if f"{impl}:full" in ts}
for v in VARIANTS:
    med = statistics.median(ts[v]); impl = v.split(":")[0]
    d = f"  cost {1e3 * (base[impl] - med):6.1f} ms" if impl in base and not v.endswith(":full") else ""
    print(f"S={S} {v:16s} median {med * 1e3:7.1f} ms  best {min(ts[v]) * 1e3:7.1f} ms{d}   [{quiet[v]}/{ROUNDS} quiet]")
