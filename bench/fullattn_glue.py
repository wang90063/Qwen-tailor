"""Cost of the full-attention layer's glue (norms, rope, transposes, cache writes, output gating) vs its SDPA, S=2048.
Chains of N independent copies of each stage (distinct inputs), quiet-GPU sampling."""
import sys, time, statistics
import mlx.core as mx
from quiet import Probe
from qwen35 import model as m
S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
N = 6
cfg = m.Cfg()
H, KV, D = cfg.n_heads, cfg.n_kv, cfg.head_dim
qkvs = [(mx.random.normal((1, S, H * D * 2 + 2 * KV * D)) * 0.5).astype(mx.bfloat16) for _ in range(N)]
qn_w = mx.random.normal((D,)).astype(mx.bfloat16); kn_w = mx.random.normal((D,)).astype(mx.bfloat16)
Kc = [mx.zeros((1, KV, S + 8, D), mx.bfloat16) for _ in range(N)]; Vc = [mx.zeros((1, KV, S + 8, D), mx.bfloat16) for _ in range(N)]
mx.eval(qkvs, qn_w, kn_w, Kc, Vc)


def split(qkv):
    q, k, v = mx.split(qkv, [H * D * 2, H * D * 2 + KV * D], axis=-1)
    q, gate = mx.split(q.reshape(1, S, H, 2 * D), 2, axis=-1)
    return q, gate.reshape(1, S, -1), k, v


def stage_qk(i):
    q, gate, k, v = split(qkvs[i])
    q = mx.fast.rms_norm(q, qn_w, cfg.eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, S, KV, D), kn_w, cfg.eps).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=0)
    k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=0)
    return q, k


def stage_write(i):
    q, gate, k, v = split(qkvs[i])
    v = v.reshape(1, S, KV, D).transpose(0, 2, 1, 3)
    _, k = stage_qk(i)
    return m._write(Kc[i], k, 0), m._write(Vc[i], v, 0)


def stage_sdpa(i):
    q, k = stage_qk(i)
    _, _, _, v = split(qkvs[i])
    v = v.reshape(1, S, KV, D).transpose(0, 2, 1, 3)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")


def stage_gate(i):
    q, gate, k, v = split(qkvs[i])
    o = (qkvs[i][..., :H * D]).reshape(1, S, H, D)                 # stand-in for the attention output
    o = o.transpose(0, 2, 1, 3).transpose(0, 2, 1, 3).reshape(1, S, -1)
    return o * mx.sigmoid(gate)


def stage_base(i):                                                  # the split views alone (should be ~free)
    q, gate, k, v = split(qkvs[i])
    return q + 0, k + 0


stages = {"base(q+0,k+0)": stage_base, "qk norm+rope": lambda i: stage_qk(i), "qk + cache write": stage_write,
          "qk + sdpa": stage_sdpa, "out gate (mul sigmoid)": stage_gate}
probe = Probe(tol=1.15)


def run(st):
    t = time.perf_counter()
    outs = [st(i) for i in range(N)]
    mx.eval([o for o in outs])
    return (time.perf_counter() - t) / N


for st in stages.values(): run(st)
res = {k: [] for k in stages}
for _ in range(8):
    for k, st in stages.items():
        dt, ok = probe.sample(lambda: run(st))
        if ok: res[k].append(dt)
for k, v in res.items():
    if v: print(f"{k:26s}: median {statistics.median(v) * 1e3:.3f} ms  best {min(v) * 1e3:.3f} ms (n={len(v)}) per layer")
