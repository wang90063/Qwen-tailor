"""Generation engine: prefill + compiled, pipelined greedy decode."""
from __future__ import annotations
import time
import numpy as np
import mlx.core as mx
from .model import Cfg, load_weights, new_state, forward, logits_of
from . import fast
from .head_prune import argmax_head
from .prefill import forward as pforward
from .prefix_cache import PrefixCache

TAIL_ROWS = 128          # >= prefill.GEMM_MIN_M so the tail rows stay on the NAX GEMMs

BUCKET = 256


class Engine:
    def __init__(self, max_len=4096, compile=True, fused=True, prefix_cache_mb=0, like=None):
        """`like`: another Engine whose (read-only) weights are shared instead of loading a second copy."""
        self.cfg = Cfg()
        self.W = like.W if like is not None else load_weights(self.cfg)
        self.fused = fused
        self.F = (like.F if like is not None and getattr(like, "fused", True) == fused else fast.prepare(self.W, self.cfg)) if fused else None
        self.max_len = max_len
        self.compile = compile
        self._steps = {}
        self._sel0 = mx.zeros((1,), dtype=mx.int32)
        self.pcache = PrefixCache(prefix_cache_mb) if prefix_cache_mb else None

    def _step_fn(self, kv_len):
        cfg = self.cfg

        def fstep(F, tok, pos, state):
            ids, _, new = fast.verify(cfg, F, tok.reshape(1), state, self._sel0, pos, kv_len)
            return ids.reshape(1, 1), new

        def step(W, tok, pos, state):
            h, new = forward(cfg, W, tok, state, pos, kv_len)
            nxt = mx.argmax(logits_of(W, h[:, -1]), axis=-1).astype(mx.int32)
            return nxt.reshape(1, 1), new

        fn = fstep if self.fused else step
        return mx.compile(fn) if self.compile else fn

    def step(self, tok, pos, state, kv_len):
        b = min(self.max_len, -(-kv_len // BUCKET) * BUCKET)
        fn = self._steps.get(b)
        if fn is None:
            fn = self._steps[b] = self._step_fn(b)
        return fn(self.F if self.fused else self.W, tok, pos, state)

    def prefill(self, ids, state=None, chunk=2048, cache_at=()):
        """Returns (first token [1,1], state, len(ids)).  With a prefix cache (Engine(prefix_cache_mb=...)) the prefill resumes from
        the deepest cached checkpoint that is a prefix of `ids` and leaves checkpoints behind (chunk boundaries, the end of the prompt,
        `cache_at` positions, and the overlap with an earlier prompt; see prefix_cache.py)."""
        g = self.prefill_iter(ids, state, chunk, cache_at)
        while True:
            try:
                next(g)
            except StopIteration as e:
                return e.value

    def prefill_iter(self, ids, state=None, chunk=2048, cache_at=(), chunk_fn=None, h_out=None):
        """`prefill` as a generator that yields after every evaluated chunk (the scheduler of a continuous-batching server runs one
        chunk between two decode steps) and returns (first token, state, len(ids)) through StopIteration.value.  `chunk_fn()`, when
        given, is called before each chunk to pick its size (a server prefills big chunks while nothing is decoding).
        `h_out` (a list): the final-norm hidden state of EVERY prompt position ([len(ids), hidden] bf16) is appended to it when the
        generator finishes -- what the speculative mode builds its MTP K / V from.  The last layer then cannot skip the non-tail rows
        (~4% more prefill work; the state and the first token are bit-identical, tests/prefill_tail.py), and the prefix cache keeps
        the rows next to the checkpoints (`extra["h"]`; a cached prefix without them is not used)."""
        S = len(ids)
        assert 0 < S <= self.max_len, (S, self.max_len)
        pc = self.pcache if state is None else None
        pos, node, split, toks = 0, None, 0, None
        hs = []
        if pc is not None:
            toks = np.asarray(ids, dtype=np.int32)
            node, split = pc.lookup(toks)
            if node is not None and h_out is not None:
                chain = node.chain()
                if any((c.extra or {}).get("h") is None for c in chain):
                    node = None
                else:
                    hs.append(pc.cat_rows(chain, lambda c: c.extra["h"], node.pos, 0))
            if node is not None:
                state, _ = pc.restore(node, self.max_len)
                pos = node.pos
                if pos == S:                                     # the whole prompt is cached, first token included
                    mx.eval(state)
                    if h_out is not None:
                        h_out.append(hs[0])
                    return node.last["nxt"], state, S
        state = state or new_state(self.cfg, self.max_len)
        marks = sorted({m for m in (*cache_at, split) if pos < m < S}) if pc is not None else []
        h = None
        while pos < S:
            ck = chunk_fn() if chunk_fn is not None else chunk
            end = min(S, pos + ck, *[m for m in marks if m > pos])
            part = mx.array([ids[pos:end]])
            # only the last token's hidden state is used: the last layer skips the other rows (non-final chunks: all of them)
            h, state = pforward(self.cfg, self.W, part, state, pos, end,
                                tail=None if h_out is not None else TAIL_ROWS if end >= S else 0)
            pos = end
            mx.eval(state)
            if h_out is not None:
                hs.append(h[0])
            if pc is not None and pos < S:
                node = pc.insert(node, toks, pos, state, extra={"h": h[0]} if h_out is not None else None)
            if pos < S:
                yield pos
        if self.F is not None:
            nxt = argmax_head(self.W["embed"], self.F.get("head_tab"), h[0, -1:]).reshape(1, 1)
        else:
            nxt = mx.argmax(logits_of(self.W, h[:, -1]), axis=-1).astype(mx.int32).reshape(1, 1)
        if pc is not None:
            mx.async_eval(nxt)
            pc.insert(node, toks, S, state, last={"nxt": nxt}, extra={"h": h[0]} if h_out is not None else None)
        if h_out is not None:
            h_out.append(mx.concatenate(hs) if len(hs) > 1 else hs[0])
        return nxt, state, pos

    def generate(self, ids, n, stamps=None):
        t0 = time.perf_counter()
        tok, state, pos = self.prefill(ids)
        mx.eval(tok)
        out = [tok]
        if stamps is not None: stamps.append(time.perf_counter() - t0)
        # pipelined: launch step i+1 before reading token i
        for i in range(n - 1):
            posa = mx.array(pos, dtype=mx.int32)
            nxt, state = self.step(tok, posa, state, pos + 1)
            mx.async_eval(nxt)
            pos += 1
            tok = nxt
            out.append(tok)
            if stamps is not None:
                mx.eval(out[-2]); stamps.append(time.perf_counter() - t0)
        res = [int(t.item()) for t in out]
        if stamps is not None: stamps.append(time.perf_counter() - t0)
        return res
