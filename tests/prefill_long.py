"""Long-prompt accuracy of the prefill DeltaNet implementation (QWEN35_GDN=nax vs ref).

For each prompt: prefill with 'ref' (mlx-lm sequential kernel) and with 'nax' (chunked tensor-op kernel); then decode
the same tokens with the same decode kernels (reference free-runs, nax is teacher-forced on the reference tokens),
so the only difference is the state produced by prefill.  Reports prefill-logit error, state error and, for every
decoded position, how far the nax logits' argmax is from the reference token (must be <= bf16 tie tolerance)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import mlx.core as mx
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from baseline import make_prompt
from qwen35 import model as m, prefill as pf, fast

TOL = 0.13
N_DEC = 48
cfg = m.Cfg(); W = m.load_weights(cfg); F = fast.prepare(W, cfg)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
SEL0 = mx.zeros((1,), dtype=mx.int32)


def code_prompt(n):
    # frozen corpus (installed mlx_lm sources): the project's own files change with every edit and so would the prompts
    import mlx_lm
    text = "\n".join(p.read_text() for p in sorted(Path(mlx_lm.__file__).parent.glob("models/*.py"))[:12])
    ids = tk(text)["input_ids"]
    return ids[:n]


def prefill(ids, impl, max_len):
    pf.GDN_IMPL = impl
    st = m.new_state(cfg, max_len)
    h, st = pf.forward(cfg, W, mx.array([ids]), st, 0, len(ids))
    lg = m.logits_of(W, h[:, -1]).astype(mx.float32)[0]
    mx.eval(lg, st)
    return lg, st


def decode(st, pos, tok_fn, n, max_len):
    outs, toks = [], []
    tok = tok_fn(None)
    for i in range(n):
        kv_len = min(max_len, -(-(pos + 1) // 256) * 256)
        ids, Hn, st = fast.verify(cfg, F, mx.array([tok], dtype=mx.int32), st, SEL0, mx.array(pos, dtype=mx.int32), kv_len)
        lg = (Hn @ F["embed"].T).astype(mx.float32)[0]
        mx.eval(lg)
        outs.append(lg); toks.append(tok)
        tok = tok_fn(lg); pos += 1
    return mx.stack(outs), toks


def check(label, ids):
    max_len = len(ids) + N_DEC + 8
    lg_r, st_r = prefill(ids, "ref", max_len)
    lg_n, st_n = prefill(ids, "nax", max_len)
    d = mx.abs(lg_r - lg_n)
    rel = float(mx.sqrt(mx.sum(d * d) / mx.sum(lg_r * lg_r)))
    # DeltaNet state error after prefill, per layer
    serr = []
    for a, b in zip(st_r["ssm"], st_n["ssm"]):
        dd = (a - b).reshape(-1); serr.append(float(mx.sqrt(mx.sum(dd * dd) / mx.sum(a.reshape(-1) ** 2))))
    t0 = int(mx.argmax(lg_r)); t0n = int(mx.argmax(lg_n))
    # reference free-run
    ref_toks = []
    def pick_ref(lg):
        t = t0 if lg is None else int(mx.argmax(lg)); ref_toks.append(t); return t
    ref_lg, _ = decode(st_r, len(ids), pick_ref, N_DEC, max_len)
    seq = list(ref_toks)                     # N_DEC + 1 tokens: seq[0] from prefill, seq[i+1] predicted at step i
    it = iter(seq)
    nax_lg, _ = decode(st_n, len(ids), lambda lg: next(it), N_DEC, max_len)
    mxn = mx.max(nax_lg, axis=-1); tg = mx.take_along_axis(nax_lg, mx.array(seq[1:])[:, None], axis=-1)[:, 0]
    gap = mxn - tg
    mism = int(mx.sum(mx.argmax(nax_lg, axis=-1) != mx.array(seq[1:])))
    ld = mx.max(mx.abs(nax_lg - ref_lg), axis=-1)
    print(f"{label:22s} n={len(ids):5d} | prefill logits rel-L2 {rel:.2e} top1 {'same' if t0 == t0n else 'DIFF'} | "
          f"state err mean {sum(serr)/len(serr):.2e} max {max(serr):.2e} | decode max|dlogit| {float(ld.max()):.3f} "
          f"argmax-mismatch {mism}/{N_DEC} worst-gap {float(gap.max()):.3f} {'ok' if float(gap.max()) <= TOL else 'BAD'}")
    return float(gap.max()) <= TOL


if __name__ == "__main__":
    ok = True
    for n in (130, 500, 2000):
        ok &= check("code text", code_prompt(n))
    for n in (512, 2000):
        ok &= check("repeated paragraph", make_prompt(tk, n))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
