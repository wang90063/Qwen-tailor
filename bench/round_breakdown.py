"""Time of the parts of one MTP speculative round: compiled verify at M = 1, 2, 4, mtp.step, the 16K-subset draft head.
usage: round_breakdown.py [k=3]"""
import sys, time
import mlx.core as mx
from qwen35 import fast, mtp
from qwen35.spec import SpecEngine
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
e = SpecEngine(k=k, max_len=1024)
cfg, F, MT = e.cfg, e.F, e.MT
st = fast.new_state(cfg, 1024)
M = k + 1
def rep(a): return mx.concatenate([a] * M, axis=0)
st = {"kv": st["kv"], "conv": [rep(c) for c in st["conv"]], "ssm": [rep(c) for c in st["ssm"]]}
sel = mx.zeros((1,), dtype=mx.int32); pos = mx.array(100, dtype=mx.int32)
toks = mx.arange(M, dtype=mx.int32) + 1000
def bench(name, f, n=30):
    for _ in range(3): mx.eval(f())
    t = time.perf_counter()
    outs = [f() for _ in range(n)]
    mx.eval(outs)
    print(f"{name:34s} {(time.perf_counter()-t)/n*1e3:7.2f} ms")
for m_ in (1, 2, 4):
    s_ = {"kv": st["kv"], "conv": [c[:m_] if c.shape[0] >= m_ else c for c in st["conv"]], "ssm": [c[:m_] for c in st["ssm"]]}
    t_ = toks[:m_]
    fn = mx.compile(lambda F, t, s, sel, pos: fast.verify(cfg, F, t, s, sel, pos, 256))
    bench(f"verify M={m_} (compiled)", lambda: fn(F, t_, s_, sel, pos))
mkv = (mx.zeros((1, 2, 1024, 256), mx.bfloat16),) * 2
H = mx.random.normal((M, 1024)).astype(mx.bfloat16)
fn = mx.compile(lambda MT, t, h, kv, pos: mtp.step(cfg, MT, F["embed"], t, h, kv, pos, 256))
bench(f"mtp.step rows={M}", lambda: fn(MT, toks, H, mkv, pos))
bench("mtp.step rows=1", lambda: fn(MT, toks[:1], H[:1], mkv, pos))
fh = mx.compile(lambda MT, W, i, n, x: mtp.draft_token(cfg, MT, W, i, n, x))
bench("draft head (16K subset)", lambda: fh(MT, e.Wsub, e.idmap, e.nvalid, H[:1]))
