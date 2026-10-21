"""Certified int8 lm_head pre-filter (qwen35/head_prune.py) == kernels.head_argmax, bit for bit.

  1. interval   per-row [lb, ub] contains the exact bf16 logit v_i for EVERY row of the real 248320-row table, on real hidden
                states, random vectors of many scales, sparse/outlier vectors and adversarial ones that align h with the sign of
                one row's quantisation residual (maximal error for that row)
  2. argmax     pruned == plain on all of those, M = 1..4
  3. ties       exact duplicate rows (first index must win), all-zero h (every row ties: 248320 candidates through the grid-stride
                path), all rows equal (flat), tail rows (N not a multiple of R)
  4. tensor-op  the NAX K1 used for M >= hp.NAX_MIN (M = 6..32): the same interval / argmax checks, on the real table and on tail
                shapes (N % 32 != 0, M not a multiple of 16)
Exit code != 0 on any failure."""
import sys, glob
import numpy as np
import mlx.core as mx
from qwen35 import model as m, head_prune as hp
from qwen35.kernels import head_argmax, fused_gemv

from qwen35.model import ROOT
mx.random.seed(1)
rng = np.random.default_rng(1)
bad = 0


def fail(msg):
    global bad
    bad += 1
    print("FAIL", msg)


def bf(a):
    return mx.array(np.asarray(a, dtype=np.float32)).astype(mx.bfloat16)


def check_interval(E, tab, x, label, T=64, R=8, nax=False):
    """x [M,K] bf16: ub >= v >= lb on every row (scalar K1, or the tensor-op K1 when nax)"""
    N, K = E.shape
    M = x.shape[0]
    if nax:
        ub, lbmax, ubmax, lb = hp.approx_nax(tab, x, R, dbg=True)
    else:
        ub, lbmax, ubmax, lb = hp._approx_kernel(N, K, M, T, R, dbg=True)(
            inputs=[tab["q"], x.reshape(-1), tab["scale"], tab["cp"]],
            output_shapes=[(M * N,), (M * -(-N // R),), (M * -(-N // R),), (M * N,)], output_dtypes=[mx.float32] * 4,
            grid=(-(-N // R) * T, 1, 1), threadgroup=(T, 1, 1))
    v = fused_gemv(E, x).astype(mx.float32).reshape(-1)             # BF(fp32 dot) for every row, same arithmetic as the argmax kernel
    mx.eval(ub, lb, v, lbmax, ubmax)
    below = int(mx.sum(v < lb).item()); above = int(mx.sum(v > ub).item())
    if below or above:
        fail(f"interval {label}: {below} rows below lb, {above} rows above ub")
    # the per-threadgroup maxima must agree with the rows
    R_ = R
    lbm = np.array(lb).reshape(M, N); ubm = np.array(ub).reshape(M, N)
    pad = (-N) % R_
    lbp = np.pad(lbm, ((0, 0), (0, pad)), constant_values=-np.inf).reshape(M, -1, R_).max(-1)
    ubp = np.pad(ubm, ((0, 0), (0, pad)), constant_values=-np.inf).reshape(M, -1, R_).max(-1)
    if not (np.array_equal(lbp, np.array(lbmax).reshape(M, -1)) and np.array_equal(ubp, np.array(ubmax).reshape(M, -1))):
        fail(f"interval {label}: threadgroup maxima disagree with the rows")
    rad = (ubm - lbm) / 2
    gap = np.minimum(np.array(v).reshape(M, N) - lbm, ubm - np.array(v).reshape(M, N)) / np.maximum(rad, 1e-30)
    return float(gap.min())


def check_argmax(E, tab, x, label):
    M = x.shape[0]
    a = head_argmax(E, x); b = hp.head_argmax_pruned(E, tab, x)
    mx.eval(a, b)
    if not bool(mx.all(a == b).item()):
        fail(f"argmax {label}: plain {a.tolist()} pruned {b.tolist()}")


# ---------------------------------------------------------------------------------------------- real table
cfg = m.Cfg(); W = m.load_weights(cfg); E = W["embed"]
N, K = E.shape
tab = hp.build_tables(E)
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3.5-0.8B")
text = "\n\n".join(open(f).read() for f in sorted(glob.glob(f"{ROOT}/qwen35/*.py")))
ids = tk(text, add_special_tokens=False)["input_ids"][:768]
st = m.new_state(cfg, 776)
H, _ = m.forward(cfg, W, mx.array([ids]), st, 0, 776); H = H[0, 16:]; mx.eval(H)

# 0: the exact stage is bit-identical, row by row, to the plain kernel's logits (candidate list = the whole vocabulary)
cand_all = mx.broadcast_to(mx.arange(N, dtype=mx.int32)[None, :], (4, N)).reshape(-1)
nc_all = mx.full((4,), N, dtype=mx.int32)
for p in (0, 40, 80):
    x = H[p:p + 4]
    pv, pi, vals = hp._exact_kernel(N, K, 4, 64, dbg=True)(
        inputs=[E, x.reshape(-1), cand_all, nc_all], output_shapes=[(4 * 64,), (4 * 64,), (4 * N,)],
        output_dtypes=[mx.float32, mx.int32, mx.float32], grid=(64 * 128, 4, 1), threadgroup=(128, 1, 1))
    ref = fused_gemv(E, x).astype(mx.float32).reshape(-1)
    mx.eval(vals, ref)
    nd = int(mx.sum(vals != ref).item())
    if nd:
        fail(f"exact stage differs from fused_gemv logits on {nd} of {4 * N} values")
print("exact stage == fused_gemv logits, bit for bit, on all rows")

# 1/2: real hidden states
tight = 1e9
for p in range(0, 240, 4):
    x = H[p:p + 4]
    tight = min(tight, check_interval(E, tab, x, f"real[{p}:{p + 4}]"))
    check_argmax(E, tab, x, f"real[{p}:{p + 4}]")
for p in range(240, 400):
    check_argmax(E, tab, H[p:p + 1], f"real[{p}]")
print(f"real hidden states: min distance of v to the interval ends = {tight:.3f} x radius")

# random vectors of many scales (including very small / very large) and sparse outlier vectors
for scale in (1e-3, 0.05, 1.0, 20.0, 144.0 / 32, 500.0):
    x = bf(rng.normal(0, scale, (4, K)))
    check_interval(E, tab, x, f"normal*{scale}"); check_argmax(E, tab, x, f"normal*{scale}")
for k in (1, 3, 40):
    x = np.zeros((4, K));
    for r in range(4): x[r, rng.choice(K, k, replace=False)] = rng.normal(0, 30, k)
    x = bf(x); check_interval(E, tab, x, f"sparse{k}"); check_argmax(E, tab, x, f"sparse{k}")
# massive-activation style: a few huge dims on top of a real hidden state
x = np.array(H[10:14].astype(mx.float32)); x[:, [5, 700]] += 400.0
x = bf(x); check_interval(E, tab, x, "real+outliers"); check_argmax(E, tab, x, "real+outliers")

# adversarial: h = sign of one row's residual (error of that row at ~87% of the Cauchy-Schwarz bound), and of a row that is
# also near the top of the logits
q = np.array(tab["q"][:1000]).astype(np.float64); sc = np.array(tab["scale"][:1000]).astype(np.float64)
res = np.array(E[:1000].astype(mx.float32)).astype(np.float64) - q * sc[:, None]
worst = 0.0
for i in range(0, 1000, 125):
    for amp in (0.25, 3.0, 40.0):
        x = bf(np.sign(res[i])[None, :] * amp)
        worst = max(worst, 1 - check_interval(E, tab, x, f"adversarial row {i} amp {amp}"))
        x2 = bf(np.sign(res[i])[None, :] * amp + np.array(E[i:i + 1].astype(mx.float32)) * 60)       # near-top row under max error
        check_interval(E, tab, x2, f"adversarial+row {i}"); check_argmax(E, tab, x2, f"adversarial+row {i}")
print("adversarial residual-aligned vectors: all intervals hold")

# ---------------------------------------------------------------------------------------------- ties / tails / flat
Es = (mx.random.normal((1003, K)) * 0.03).astype(mx.bfloat16)                     # N % 8 = 3, % 4 = 3
base = np.array(Es.astype(mx.float32))
for rows in ([5, 700, 1001], [1000, 1002], [0, 1002], [333, 334]):
    En = base.copy()
    xv = rng.normal(0, 1, (1, K))
    En[rows[0]] = En[rows[0]] + 0.2 * xv[0] / np.linalg.norm(xv)             # make the first one clearly the best
    for r in rows[1:]: En[r] = En[rows[0]]                                   # exact duplicates -> exact ties
    Ec = bf(En); ts = hp.build_tables(Ec); x = bf(xv)
    a = head_argmax(Ec, x); b = hp.head_argmax_pruned(Ec, ts, x); mx.eval(a, b)
    if int(a.item()) != int(b.item()) or int(b.item()) != min(rows):
        fail(f"duplicate rows {rows}: plain {a.item()} pruned {b.item()} expected {min(rows)}")
    check_interval(Ec, ts, x, f"synthetic {rows}")
# all-zero h: every logit is exactly 0 -> every row is a candidate -> index 0 (grid-stride over 248320 candidates)
z = mx.zeros((2, K), dtype=mx.bfloat16)
a = head_argmax(E, z); b = hp.head_argmax_pruned(E, tab, z); mx.eval(a, b)
if a.tolist() != b.tolist() or b.tolist() != [0, 0]:
    fail(f"all-zero h: plain {a.tolist()} pruned {b.tolist()}")
# flat distribution on the real table: 248320 candidates, the winner is one of the tied rows
Ef = mx.broadcast_to(E[123:124], E.shape) * 1
Ef = mx.contiguous(Ef); mx.eval(Ef); tf = hp.build_tables(Ef)
x = H[20:21]
a = head_argmax(Ef, x); b = hp.head_argmax_pruned(Ef, tf, x); mx.eval(a, b)
if int(a.item()) != int(b.item()) or int(b.item()) != 0:
    fail(f"flat table: plain {a.item()} pruned {b.item()}")
# negative-only logits and a single huge negative / positive row
En = np.array(Es.astype(mx.float32)); xv = np.abs(rng.normal(0, 1, (1, K)))
En = -np.abs(En); Ec = bf(En); ts = hp.build_tables(Ec); x = bf(xv)
check_argmax(Ec, ts, x, "all-negative logits"); check_interval(Ec, ts, x, "all-negative logits")

# ---------------------------------------------------------------------------------------------- tensor-op K1 (M >= NAX_MIN)
assert hp.NAX_MIN and hp.NAX_MIN <= 8, "the NAX path must be enabled for this test"
tightn = 1e9
for M in (6, 7, 8, 12, 16, 17, 24, 32):
    for p in range(0, 200, 40):
        x = H[p:p + M]
        tightn = min(tightn, check_interval(E, tab, x, f"nax real M={M} [{p}]", nax=True))
        check_argmax(E, tab, x, f"nax real M={M} [{p}]")
print(f"tensor-op K1 on real hidden states (M = 6..32): min distance of v to the interval ends = {tightn:.3f} x radius")
for scale in (1e-3, 1.0, 144.0 / 32, 500.0):
    x = bf(rng.normal(0, scale, (16, K)))
    check_interval(E, tab, x, f"nax normal*{scale}", nax=True); check_argmax(E, tab, x, f"nax normal*{scale}")
x = np.zeros((9, K)); x[np.arange(9), rng.choice(K, 9, replace=False)] = rng.normal(0, 30, 9)
x = bf(x); check_interval(E, tab, x, "nax sparse", nax=True); check_argmax(E, tab, x, "nax sparse")
x = np.array(H[10:20].astype(mx.float32)); x[:, [5, 700]] += 400.0
x = bf(x); check_interval(E, tab, x, "nax real+outliers", nax=True); check_argmax(E, tab, x, "nax real+outliers")
for i in range(0, 1000, 250):
    x = np.concatenate([np.sign(res[i])[None, :] * amp for amp in (0.25, 3.0, 40.0)] * 3, 0)
    x = bf(x); check_interval(E, tab, x, f"nax adversarial row {i}", nax=True)
    x2 = bf(np.array(x.astype(mx.float32)) + np.array(E[i:i + 1].astype(mx.float32)) * 60)
    check_interval(E, tab, x2, f"nax adversarial+row {i}", nax=True); check_argmax(E, tab, x2, f"nax adversarial+row {i}")
# tails: N % 32 != 0 (weight rows past the end are never read), M not a multiple of 16, exact duplicate rows (first index wins)
for Nt in (1003, 1024 + 17, 64 * 5 + 1):
    Et = (mx.random.normal((Nt, K)) * 0.03).astype(mx.bfloat16)
    for rows in ([5, 700 % Nt, Nt - 2], [Nt - 1, Nt - 3], [0, Nt - 1]):
        En = np.array(Et.astype(mx.float32))
        xv = rng.normal(0, 1, (1, K))
        En[rows[0]] = En[rows[0]] + 0.2 * xv[0] / np.linalg.norm(xv)
        for r in rows[1:]: En[r] = En[rows[0]]
        Ec = bf(En); ts = hp.build_tables(Ec)
        for M in (6, 9, 16, 20):
            x = bf(np.concatenate([xv] + [xv * rng.uniform(0.5, 2) for _ in range(M - 1)], 0))
            a = head_argmax(Ec, x); b = hp.head_argmax_pruned(Ec, ts, x); mx.eval(a, b)
            if a.tolist() != b.tolist() or b.tolist() != [min(rows)] * M:
                fail(f"nax duplicate rows N={Nt} {rows} M={M}: plain {a.tolist()} pruned {b.tolist()} expected {min(rows)}")
            check_interval(Ec, ts, x, f"nax synthetic N={Nt} {rows} M={M}", nax=True)
# all-zero h: every row ties through the NAX path too
z = mx.zeros((8, K), dtype=mx.bfloat16)
a = head_argmax(E, z); b = hp.head_argmax_pruned(E, tab, z); mx.eval(a, b)
if a.tolist() != b.tolist() or b.tolist() != [0] * 8:
    fail(f"nax all-zero h: plain {a.tolist()} pruned {b.tolist()}")
print("head_prune:", "PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)
