"""Unit tests: fused GEMV family and argmax head against plain MLX ops (M = 1 and 3 rows)."""
import mlx.core as mx, mlx.nn as nn
from qwen35.kernels import fused_gemv, head_argmax, subset_argmax

mx.random.seed(0)
bf = mx.bfloat16
def rnd(*s, sc=1.0): return (mx.random.normal(s) * sc).astype(bf)
def chk(name, got, ref, tol=0.05):
    d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    sc = mx.abs(ref.astype(mx.float32)).max().item()
    print(f"{name:34s} maxdiff {d:.4f} (scale {sc:.2f}) {'ok' if d <= tol * max(sc, 1) else 'FAIL'}")
    assert d <= tol * max(sc, 1)

for M in (1, 3):
    x = rnd(M, 1024); nw = rnd(1024, sc=0.2) + 1
    xn = mx.fast.rms_norm(x, nw, 1e-6)
    W = rnd(2048, 1024, sc=0.03); Wd = rnd(1024, 3584, sc=0.03); a = rnd(M, 3584); res = rnd(M, 1024)
    chk(f"M={M} plain", fused_gemv(W, x), x @ W.T)
    chk(f"M={M} rms prologue", fused_gemv(W, x, nw=nw), xn @ W.T)
    chk(f"M={M} down+residual", fused_gemv(Wd, a, res=res), a @ Wd.T + res)
    Wg, Wu = rnd(3584, 1024, sc=0.03), rnd(3584, 1024, sc=0.03)
    Wgu = mx.stack([Wg, Wu], axis=1).reshape(7168, 1024)
    chk(f"M={M} rms+swiglu (interleaved)", fused_gemv(Wgu, x, nw=nw, swiglu=True), nn.silu(xn @ Wg.T) * (xn @ Wu.T))
    E = rnd(248320, 1024, sc=0.03)
    ref = mx.argmax((x @ E.T).astype(bf), axis=-1)
    got = head_argmax(E, x)
    n_eq = (ref == got).sum().item()
    print(f"M={M} head_argmax equal {n_eq}/{M}")
    assert n_eq == M
    ids = mx.arange(0, 248320, 7, dtype=mx.int32)[:16000]
    Ws = E[ids]
    x1 = x[:1]
    ref = ids[mx.argmax((x1 @ Ws.T).astype(bf), axis=-1)]
    got = subset_argmax(Ws, ids, mx.array([16000], dtype=mx.int32), x1)   # single-row by design (draft head)
    print(f"M={M} subset_argmax(row 0) equal {(ref == got).sum().item()}/1")
    assert (ref == got).all().item()
print("PASS")
