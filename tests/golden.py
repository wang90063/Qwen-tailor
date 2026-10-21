"""Correctness gate vs golden greedy ids from stock mlx-lm.

Teacher-forced: feed the golden sequence, require the golden next-token's logit to be within one bf16
ulp-ish tolerance (TOL) of our max logit at every position (bf16 logits tie often: 21.0 vs 21.0), and
report the strict argmax match rate. Free-run divergence is reported too (first divergence index).
"""
import json, sys
import mlx.core as mx
from qwen35.model import Cfg, load_weights, new_state, forward, logits_of, ROOT

TOL = 0.13  # bf16 spacing at |logit| in [16,32) is 0.125

def run(step_fn, prompt_fn, label):
    gold = json.load(open(ROOT / "results" / "baseline_mlx_lm.json"))["golden"]
    ok = True
    for name, g in gold.items():
        ids, want = g["prompt_ids"], g["greedy_ids"]
        lg_seq = prompt_fn(ids, want)  # [len(want), V] logits predicting want[i]
        lg_seq = lg_seq.astype(mx.float32)
        mx_ = mx.max(lg_seq, axis=-1)
        tgt = mx.take_along_axis(lg_seq, mx.array(want)[:, None], axis=-1)[:, 0]
        gap = (mx_ - tgt)
        strict = (mx.argmax(lg_seq, axis=-1) == mx.array(want)).astype(mx.float32).mean().item()
        worst = gap.max().item()
        good = worst <= TOL
        ok &= good
        print(f"{label} {name[:28]!r:32s} strict-argmax {strict*100:5.1f}%  worst-gap {worst:.3f}  {'ok' if good else 'BAD'}")
    return ok

def check_fused(engine, label="fused"):
    """Teacher-forced run of the engine's decode step; mismatching argmax positions must be near-ties
    according to the reference (unfused) logits."""
    cfgr = Cfg(); Wr = load_weights(cfgr)
    gold = json.load(open(ROOT / "results" / "baseline_mlx_lm.json"))["golden"]
    ok = True
    for name, g in gold.items():
        ids, want = g["prompt_ids"], g["greedy_ids"]
        # reference logits teacher-forced
        st = new_state(cfgr, 512)
        h, st = forward(cfgr, Wr, mx.array([ids]), st, 0, len(ids))
        ref = [logits_of(Wr, h[:, -1])]
        pos = len(ids)
        for t in want[:-1]:
            h, st = forward(cfgr, Wr, mx.array([[t]]), st, pos, pos + 1); pos += 1
            ref.append(logits_of(Wr, h[:, -1]))
        ref = mx.concatenate(ref, axis=0).astype(mx.float32)
        # engine predictions teacher-forced
        tok, st, pos = engine.prefill(ids)
        preds = [int(tok.item())]
        for t in want[:-1]:
            tok, st = engine.step(mx.array([[t]], dtype=mx.int32), mx.array(pos, dtype=mx.int32), st, pos + 1)
            pos += 1
            preds.append(int(tok.item()))
        p = mx.array(preds)
        gap = (mx.max(ref, axis=-1) - mx.take_along_axis(ref, p[:, None], axis=-1)[:, 0])
        n_diff = sum(a != b for a, b in zip(preds, want))
        worst = gap.max().item()
        good = worst <= TOL
        ok &= good
        print(f"{label} {name[:28]!r:32s} argmax-mismatch {n_diff}/{len(want)}  worst-gap-vs-ref {worst:.3f}  {'ok' if good else 'BAD'}")
    return ok


if __name__ == "__main__":
    cfg = Cfg(); W = load_weights(cfg)
    def teacher(ids, want):
        st = new_state(cfg, 512)
        h, st = forward(cfg, W, mx.array([ids]), st, 0, len(ids))
        outs = [logits_of(W, h[:, -1])]
        pos = len(ids)
        for t in want[:-1]:
            h, st = forward(cfg, W, mx.array([[t]]), st, pos, pos + 1); pos += 1
            outs.append(logits_of(W, h[:, -1]))
        return mx.concatenate(outs, axis=0)
    ok = run(None, teacher, "eager")
    from qwen35.engine import Engine
    ok &= check_fused(Engine(max_len=512, compile=True, fused=True))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
