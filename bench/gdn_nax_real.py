"""Kernel-level accuracy of gdn_chunked precision variants on *real* layer inputs captured during a prefill.

Captures (q,k,v,proj,A_log,dt_bias,state) of every DeltaNet layer from a real code-text prefill (second segment starts
from the first segment's state), then compares each variant with mlx-lm's exact fp32 sequential kernel."""
import sys, time
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR, ROOT
from mlx_lm.models.gated_delta import gated_delta_update
from qwen35 import model as m, prefill as pf, gdn_nax

cfg = m.Cfg(); W = m.load_weights(cfg)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
text = "\n".join(Path(p).read_text() for p in sorted((ROOT / "qwen35").glob("*.py")))
ids = tk(text)["input_ids"]
N1, N2 = int(sys.argv[1]) if len(sys.argv) > 1 else 1024, int(sys.argv[2]) if len(sys.argv) > 2 else 1024

cap = []
orig = gdn_nax.gdn_chunked


def spy(q, k, v, proj, A_log, dt_bias, state, **kw):
    cap.append((q, k, v, proj, A_log, dt_bias, state))
    return orig(q, k, v, proj, A_log, dt_bias, state, **kw)


gdn_nax.gdn_chunked = spy
pf.GDN_IMPL = "ref"          # propagate exact (reference) state through the layers; kernels see real data
max_len = N1 + N2 + 8
st = m.new_state(cfg, max_len)
pf.GDN_IMPL = "nax"
h, st = pf.forward(cfg, W, mx.array([ids[:N1]]), st, 0, N1)
mx.eval(h, st)
n1 = len(cap)
h, st = pf.forward(cfg, W, mx.array([ids[N1:N1 + N2]]), st, N1, max_len)
mx.eval(h, st)
gdn_nax.gdn_chunked = orig
print(f"captured {n1} layers (zero state) + {len(cap) - n1} layers (carried state), S={N1}/{N2}")
for c in cap:
    mx.eval(*c)


def rel(a, b):
    a = a.astype(mx.float32).reshape(-1); b = b.astype(mx.float32).reshape(-1)
    d = a - b
    return float(mx.sqrt(mx.sum(d * d) / mx.sum(b * b)))


refs = []
for (q, k, v, proj, A_log, dt_bias, state) in cap:
    y, s = gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None], proj[:, 8192:8208][None], A_log, dt_bias, state)
    mx.eval(y, s); refs.append((y[0], s))

VARIANTS = [
    ("plain", ()),
    ("rne", ("RNE",)),
    ("split P1", ("SPLIT_P1",)),
    ("split P1+P2", ("SPLIT_P1", "SPLIT_P2")),
    ("split Dp", ("SPLIT_DP",)),
    ("split D", ("SPLIT_D",)),
    ("split P1+Dp", ("SPLIT_P1", "SPLIT_DP")),
    ("split P1+D+Dp", ("SPLIT_P1", "SPLIT_D", "SPLIT_DP")),
    ("split P1+Dp+rne rest", ("SPLIT_P1", "SPLIT_DP", "RNE")),
    ("split all", ("SPLIT_P1", "SPLIT_P2", "SPLIT_D", "SPLIT_Y", "SPLIT_DP")),
]
sel = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for name, fl in VARIANTS:
    if sel and name not in sel:
        continue
    ye, se = [], []
    for (q, k, v, proj, A_log, dt_bias, state), (yr, sr) in zip(cap, refs):
        y, s = gdn_nax.gdn_chunked(q, k, v, proj, A_log, dt_bias, state, flags=frozenset(fl))
        mx.eval(y, s)
        ye.append(rel(y, yr)); se.append(rel(s, sr))
    # standalone timing of the full A+B on the first captured layer
    q, k, v, proj, A_log, dt_bias, state = cap[0]
    ts = []
    for _ in range(7):
        t = time.perf_counter(); mx.eval(gdn_nax.gdn_chunked(q, k, v, proj, A_log, dt_bias, state, flags=frozenset(fl))); ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"{name:24s} y rel-L2 mean {sum(ye)/len(ye):.2e} max {max(ye):.2e} | state rel-L2 mean {sum(se)/len(se):.2e} "
          f"max {max(se):.2e} | A+B {ts[3]*1e3:.3f} ms")
