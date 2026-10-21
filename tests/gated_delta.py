"""qwen35 does not need mlx-lm at runtime, and its vendored gated-delta code (qwen35/gated_delta.py) is bit-identical to mlx-lm's.

First every qwen35 module is imported and mlx_lm must not have been pulled in; then compute_g / normalize_qk / gated_delta_update
are compared bit for bit with the installed mlx-lm on the model's real shapes (Hk = Hv = 16, Dk = Dv = 128, B = 1..4, T = 1..2048).
mlx-lm is only needed for this comparison (pip install -e ".[dev]")."""
import importlib
import pkgutil
import sys

import numpy as np
import mlx.core as mx

import qwen35
from qwen35 import model as m
from qwen35.gated_delta import compute_g, gated_delta_update, normalize_qk

for info in pkgutil.iter_modules(qwen35.__path__):
    importlib.import_module(f"qwen35.{info.name}")
assert "mlx_lm" not in sys.modules, "a qwen35 module imports mlx_lm"
print(f"ok   imported all {len(list(pkgutil.iter_modules(qwen35.__path__)))} qwen35 modules without mlx_lm")

from mlx_lm.models import gated_delta as ref

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


def same(a, b):
    u = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(np.array(a.view(u)), np.array(b.view(u))))


cfg = m.Cfg()
L = m.load_weights(cfg)["layers"][0]
A_log, dt_bias = L["A_log"], L["dt_bias"]
mx.random.seed(0)
bf = mx.bfloat16
H, D = cfg.hv, cfg.dk
for B, T in ((1, 1), (1, 3), (2, 5), (4, 1), (1, 257), (1, 2048)):
    q = (mx.random.normal((B, T, H, D)) * 2).astype(bf)
    k = (mx.random.normal((B, T, H, D)) * 2).astype(bf)
    v = (mx.random.normal((B, T, H, D))).astype(bf)
    a = mx.random.normal((B, T, H)).astype(bf)
    b = mx.random.normal((B, T, H)).astype(bf)
    st = mx.random.normal((B, H, D, D)).astype(mx.float32) * 0.1
    qn, kn = normalize_qk(q, k, inv_scale=D ** -0.5, eps=1e-6)
    qr, kr = ref.normalize_qk(q, k, inv_scale=D ** -0.5, eps=1e-6)
    y, s = gated_delta_update(qn, kn, v, a, b, A_log, dt_bias, st)
    yr, sr = ref.gated_delta_update(qr, kr, v, a, b, A_log, dt_bias, st)
    ok = same(qn, qr) and same(kn, kr) and same(compute_g(A_log, a, dt_bias), ref.compute_g(A_log, a, dt_bias))
    ok = ok and same(y, yr) and same(s, sr)
    check(f"B={B} T={T}: normalize_qk / compute_g / y / state bit-identical to mlx-lm", ok)
    y0, s0 = gated_delta_update(qn, kn, v, a, b, A_log, dt_bias)           # state=None -> zeros
    yr0, sr0 = ref.gated_delta_update(qr, kr, v, a, b, A_log, dt_bias)
    check(f"B={B} T={T}: zero initial state", same(y0, yr0) and same(s0, sr0))

try:
    gated_delta_update(q[..., :64], k[..., :64], v[..., :64], a, b, A_log, dt_bias)
    check("unsupported head dim raises", False)
except ValueError:
    check("unsupported head dim raises", True)

print("PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
