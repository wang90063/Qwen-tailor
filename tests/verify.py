"""verify(M=3) must equal 3 sequential verify(M=1) calls: ids, hidden, and the state snapshots."""
import json, sys
import mlx.core as mx
from qwen35.model import Cfg, load_weights, ROOT
from qwen35 import fast
cfg = Cfg(); W = load_weights(cfg); F = fast.prepare(W, cfg)
gold = list(json.load(open(ROOT / "results" / "baseline_mlx_lm.json"))["golden"].values())[1]
ids = gold["prompt_ids"] + gold["greedy_ids"]
# prefill the first P tokens one at a time with M=1 (reference), then compare
P = 20
st = fast.new_state(cfg, 256); sel = mx.array([0], dtype=mx.int32)
def run(toks, st, sel, pos):
    return fast.verify(cfg, F, mx.array(toks, dtype=mx.int32), st, sel, mx.array(pos, dtype=mx.int32), 256)
pos = 0
for t in ids[:P]:
    _, _, st = run([t], st, sel, pos); pos += 1
# path A: three single-token steps
stA = st; outsA = []
for j, t in enumerate(ids[P:P + 3]):
    i_, h_, stA = run([t], stA, sel, P + j); outsA.append((i_.item(), h_))
# path B: one M=3 call
iB, hB, stB = run(ids[P:P + 3], st, sel, P)
print("ids A", [o[0] for o in outsA], "ids B", iB.tolist())
for j in range(3):
    d = (outsA[j][1][0].astype(mx.float32) - hB[j].astype(mx.float32)).abs().max().item()
    print(f" hidden[{j}] maxdiff {d:.4f}")
# state snapshot after token 2 (index 2) vs sequential final state
dd = max((stA["ssm"][i] - stB["ssm"][i][2:3]).abs().max().item() for i in range(18))
dc = max((stA["conv"][i].astype(mx.float32) - stB["conv"][i][2:3].astype(mx.float32)).abs().max().item() for i in range(18))
dk = max((stA["kv"][i][0][:, :, :P + 3].astype(mx.float32) - stB["kv"][i][0][:, :, :P + 3].astype(mx.float32)).abs().max().item() for i in range(6))
print("final ssm diff", dd, "conv diff", dc, "K cache diff", dk)
ok = [o[0] for o in outsA] == iB.tolist() and dd == dc == dk == 0 and all(
    (outsA[j][1][0].astype(mx.float32) - hB[j].astype(mx.float32)).abs().max().item() == 0 for j in range(3))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
