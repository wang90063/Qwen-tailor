"""Generate text with the MTP-speculative engine:  python run.py "prompt" [-n 256] [--k 3] [--chat] [--w8a8]
--w8a8 = opt-in lossy int8 prefill GEMMs (in_proj, qkv, gate_up; ~1.25x faster prefill, logits rel-L2 ~2-3%); same as QWEN35_W8A8=in_proj,qkv,gate_up
--cache-mb N = bf16 prefix cache (N MB): a later prompt that starts with an earlier one only prefills the new tokens
--repl = multi-turn chat on stdin (one user message per line, empty line / EOF quits); uses the prefix cache (default 1024 MB), so every turn
         only prefills the new messages; prints TTFT and how many prompt tokens came from the cache"""
import argparse, sys, time
from transformers import AutoTokenizer
from qwen35 import gemm_i8
from qwen35.spec import SpecEngine
from qwen35.model import MODEL_DIR

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="")
ap.add_argument("-n", type=int, default=256)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--w8a8", action="store_true", help="int8 (W8A8, lossy) prefill GEMMs for in_proj / qkv / gate_up")
ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template")
ap.add_argument("--cache-mb", type=int, default=0, help="prefix cache budget in MB (bf16 path only; 0 = off)")
ap.add_argument("--repl", action="store_true", help="multi-turn chat on stdin")
ap.add_argument("--max-len", type=int, default=4096, help="context length for --repl")
a = ap.parse_args()
if not a.repl and not a.prompt:
    ap.error("prompt required (or --repl)")
if not MODEL_DIR.is_dir():
    sys.exit(f"model not found: {MODEL_DIR}\n  hf download Qwen/Qwen3.5-0.8B --local-dir {MODEL_DIR}")
if a.w8a8:
    gemm_i8.NAMES = frozenset(gemm_i8.SUPPORTED)
tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
eos = {tk.eos_token_id, tk.convert_tokens_to_ids("<|im_end|>"), tk.convert_tokens_to_ids("<|endoftext|>")} - {None}

if a.repl:
    eng = SpecEngine(k=a.k, max_len=a.max_len, prefix_cache_mb=a.cache_mb or 1024)
    msgs = []
    while True:
        print("> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline().strip()
        if not line:
            break
        msgs.append({"role": "user", "content": line})
        ids = tk.encode(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        if len(ids) + 64 > a.max_len:
            print(f"[context full: {len(ids)} tokens, max {a.max_len}]", file=sys.stderr)
            break
        # the next turn's prompt repeats this one only up to the generation prompt (the template drops "<think>" from past answers), so
        # checkpoint where a dummy next turn stops matching
        nxt = tk.encode(tk.apply_chat_template(msgs + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}],
                                               tokenize=False, add_generation_prompt=True))
        m = next((i for i, (p, q) in enumerate(zip(ids, nxt)) if p != q), min(len(ids), len(nxt)))
        st = {}
        out = eng.generate(ids, min(a.n, a.max_len - len(ids) - 32), st, eos=eos, cache_at=[m])
        text = tk.decode(out)
        print(text)
        msgs.append({"role": "assistant", "content": text})
        hit = eng.pcache.last_hit if eng.pcache is not None else 0
        print(f"[prompt {len(ids)} tok ({hit} from cache), prefill {st['prefill_s']*1e3:.0f} ms | {len(out)} tok, "
              f"{(st['tokens']-1)/max(st['decode_s'], 1e-9):.0f} tok/s | cache {eng.pcache.bytes / 2**20:.0f} MB]", file=sys.stderr)
    sys.exit(0)

text = tk.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True) if a.chat else a.prompt
ids = tk.encode(text)
eng = SpecEngine(k=a.k, max_len=max(1024, len(ids) + a.n + 64), prefix_cache_mb=a.cache_mb)
st = {}
out = eng.generate(ids, a.n, st, eos=eos)
print(tk.decode(out))
print(f"\n[prompt {len(ids)} tok, prefill {st['prefill_s']*1e3:.0f} ms | {len(out)} tok, "
      f"{(st['tokens']-1)/st['decode_s']:.0f} tok/s, {st['accepted']/max(st['rounds'],1):.2f} drafts accepted/round]", file=sys.stderr)
