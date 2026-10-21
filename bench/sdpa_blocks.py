"""Causal prefill attention for the 6 full-attention layers: one mx.fast.scaled_dot_product_attention call (head_dim 256 takes the
composite matmul+softmax path, which also computes the fully masked upper triangle) vs query blocks that only see keys[0:end).
usage: sdpa_blocks.py [S] [blocks comma separated]"""
import sys, time
import mlx.core as mx
from bench.quiet import Probe

S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
NB = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8, 16]
H, KV, D = 8, 2, 256
R = 6                                               # distinct layers per timed graph

qs = [(mx.random.normal((1, H, S, D)) * 0.5).astype(mx.bfloat16) for _ in range(R)]
ks = [(mx.random.normal((1, KV, S, D)) * 0.5).astype(mx.bfloat16) for _ in range(R)]
vs = [(mx.random.normal((1, KV, S, D)) * 0.5).astype(mx.bfloat16) for _ in range(R)]
mx.eval(*qs, *ks, *vs)


def sdpa(q, k, v, nb):
    if nb == 1:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
    bs = -(-S // nb)
    outs = []
    for a in range(0, S, bs):
        b = min(S, a + bs)
        outs.append(mx.fast.scaled_dot_product_attention(q[:, :, a:b], k[:, :, :b], v[:, :, :b], scale=D ** -0.5, mask="causal"))
    return mx.concatenate(outs, axis=2)


ref = [sdpa(q, k, v, 1) for q, k, v in zip(qs, ks, vs)]
mx.eval(ref)
probe = Probe()
res = {nb: [] for nb in NB}
for rnd in range(9):
    for nb in NB:
        def run():
            t = time.perf_counter(); mx.eval([sdpa(q, k, v, nb) for q, k, v in zip(qs, ks, vs)]); return time.perf_counter() - t
        dt, ok = probe.sample(run)
        if ok:
            res[nb].append(dt)
for nb in NB:
    out = [sdpa(q, k, v, nb) for q, k, v in zip(qs, ks, vs)]
    mx.eval(out)
    same = all(bool(mx.all(a == b).item()) for a, b in zip(out, ref))
    diff = max(float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) for a, b in zip(out, ref))
    ts = sorted(res[nb])
    med = ts[len(ts) // 2] if ts else float("nan")
    print(f"blocks {nb:2d}: {med / R * 1e3:7.3f} ms/layer (n={len(ts)}, best {ts[0] / R * 1e3 if ts else float('nan'):.3f})  "
          f"bit-identical {same}  max|diff| {diff:.2e}")
