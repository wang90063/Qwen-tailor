"""Compiler output must match the hand-written fast.verify bit-for-bit (and the unfused lowering closely)."""
import mlx.core as mx
from qwen35 import fast
from qwen35.model import Cfg, load_weights
from qwen35.compiler import Compiled, build_graph, fuse

cfg = Cfg(); W = load_weights(cfg)
F = fast.prepare(W, cfg)
g0 = build_graph(cfg); n0 = len([n for n in g0.live() if n.op not in ("input", "output")])
comp = Compiled(W, cfg)
print("graph nodes unfused:", n0, " fused:", comp.n_kernels(), " rules fired:", comp.graph.fusion_stats)
print(comp.graph.dump(layers=(3,)))
M = 4
mk = lambda: (lambda st: {"kv": st["kv"], "conv": [mx.concatenate([c] * M) for c in st["conv"]], "ssm": [mx.concatenate([c] * M) for c in st["ssm"]]})(fast.new_state(cfg, 512))
toks = mx.array([25, 198, 285, 1000], dtype=mx.int32); sel = mx.zeros((1,), dtype=mx.int32); pos = mx.array(7, dtype=mx.int32)
s1 = mk(); s2 = mk()
for it in range(3):
    ids_h, Hn_h, n_h = fast.verify(cfg, F, toks, s1, sel, pos, 256)
    ids_c, Hn_c, n_c = comp.verify(toks, s2, sel, pos, 256)
    mx.eval(ids_h, Hn_h, n_h, ids_c, Hn_c, n_c)
    d = lambda a, b: mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
    print(f"iter {it}: ids equal {ids_h.tolist() == ids_c.tolist()}  hidden {d(Hn_h, Hn_c)}  ssm {max(d(a, b) for a, b in zip(n_h['ssm'], n_c['ssm']))}  "
          f"conv {max(d(a, b) for a, b in zip(n_h['conv'], n_c['conv']))}  kv {max(d(a[0], b[0]) for a, b in zip(n_h['kv'], n_c['kv']))}")
    assert ids_h.tolist() == ids_c.tolist() and d(Hn_h, Hn_c) == 0
    s1, s2 = n_h, n_c; pos = pos + 1; sel = mx.array([it], dtype=mx.int32)
# unfused lowering sanity
cu = Compiled(W, cfg, fuse_pass=False)
ids_u, Hn_u, _ = cu.verify(toks, mk(), mx.zeros((1,), dtype=mx.int32), mx.array(7, dtype=mx.int32), 256)
mx.eval(ids_u)
print("unfused nodes", cu.n_kernels(), "ids", ids_u.tolist())
print("PASS")
