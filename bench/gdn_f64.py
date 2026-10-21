"""Who is closer to exact: mlx-lm's sequential kernel or our chunked NAX kernel? float64 recurrence on real layer inputs."""
import sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR, ROOT
from mlx_lm.models.gated_delta import gated_delta_update
from qwen35 import model as m, prefill as pf, gdn_nax

cfg = m.Cfg(); W = m.load_weights(cfg)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
text = "\n".join(Path(p).read_text() for p in sorted((ROOT / "qwen35").glob("*.py")))
ids = tk(text)["input_ids"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
LAYERS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 4, 8, 16]   # DeltaNet layer ordinals

cap = []
orig = gdn_nax.gdn_chunked


def spy(q, k, v, proj, A_log, dt_bias, state, **kw):
    cap.append((q, k, v, proj, A_log, dt_bias, state))
    return orig(q, k, v, proj, A_log, dt_bias, state, **kw)


gdn_nax.gdn_chunked = spy
pf.GDN_IMPL = "nax"
st = m.new_state(cfg, N + 8)
h, st = pf.forward(cfg, W, mx.array([ids[:N]]), st, 0, N)
mx.eval(h, st)
gdn_nax.gdn_chunked = orig


def bf(x):   # round float64 array through bf16
    return np.array(mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32), dtype=np.float64)


def gates(proj, A_log, dt_bias):
    a = np.array(proj[:, 8208:].astype(mx.float32), dtype=np.float64); b = np.array(proj[:, 8192:8208].astype(mx.float32), dtype=np.float64)
    av = bf(a + np.array(dt_bias.astype(mx.float32), dtype=np.float64))
    sp = bf(np.maximum(av, 0) + np.log1p(np.exp(-np.abs(av))))
    g = np.exp(-np.exp(np.array(A_log.astype(mx.float32), dtype=np.float64)) * sp)
    beta = bf(1 / (1 + np.exp(-b)))
    return g, beta


def rec64(q, k, v, g, beta, S0):
    T = q.shape[0]; S = S0.copy(); ys = np.zeros((T, 16, 128))
    for t in range(T):
        S = S * g[t][:, None, None]
        kv = np.einsum("hvk,hk->hv", S, k[t])
        d = (v[t] - kv) * beta[t][:, None]
        S = S + d[:, :, None] * k[t][:, None, :]
        ys[t] = np.einsum("hvk,hk->hv", S, q[t])
    return ys, S


def r(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


print(f"{'layer':>5} | {'ref state':>10} {'nax state':>10} | {'ref y':>9} {'nax y':>9} {'nax y(pre-bf16 n/a)':>0}")
for li in LAYERS:
    q, k, v, proj, A_log, dt_bias, state = cap[li]
    mx.eval(q, k, v, proj)
    qn, kn, vn = (np.array(x.astype(mx.float32), dtype=np.float64) for x in (q, k, v))
    g, beta = gates(proj, A_log, dt_bias)
    y64, s64 = rec64(qn, kn, vn, g, beta, np.array(state[0], dtype=np.float64))
    yr, sr = gated_delta_update(q[None], k[None], v[None], proj[:, 8208:][None], proj[:, 8192:8208][None], A_log, dt_bias, state)
    yn, sn = gdn_nax.gdn_chunked(q, k, v, proj, A_log, dt_bias, state)
    mx.eval(yr, sr, yn, sn)
    print(f"{li:5d} | {r(np.array(sr[0]), s64):10.2e} {r(np.array(sn[0]), s64):10.2e} | "
          f"{r(np.array(yr[0].astype(mx.float32)), y64):9.2e} {r(np.array(yn.astype(mx.float32)), y64):9.2e}")
