"""Self-speculative decoding with the native MTP head, entirely GPU-resident (one compiled graph per round)."""
from __future__ import annotations
import time
import numpy as np
import mlx.core as mx

from . import fast, mtp
from .head_prune import argmax_head
from .model import Cfg, load_weights, new_state as ref_state
from .prefill import forward as ref_forward
from .prefix_cache import PrefixCache

BUCKET = 256


class SpecEngine:
    def __init__(self, k=2, max_len=4096, draft_vocab=16384, compile=True, prefix_cache_mb=0, like=None):
        """`like`: another SpecEngine whose (read-only) weights, draft head and draft vocabulary are shared instead of loading again."""
        self.cfg = Cfg()
        self.k = k
        self.M = k + 1
        self.max_len = max_len
        self.compile = compile
        if like is not None:
            self.W, self.F, self.MT = like.W, like.F, like.MT
            self.Wsub, self.idmap, self.nvalid = like.Wsub, like.idmap, like.nvalid
        else:
            W = load_weights(self.cfg)
            self.W = W                   # reference layout (used for prefill)
            self.F = fast.prepare(W, self.cfg)
            self.MT = mtp.load_mtp(self.cfg)
            self.Wsub, self.idmap, self.nvalid = mtp.draft_vocab(W["embed"], draft_vocab)
        self._rounds = {}
        self.pcache = PrefixCache(prefix_cache_mb) if prefix_cache_mb else None

    # -- one round ------------------------------------------------------------------------------
    def _round_fn(self, kv_len):
        cfg, k, M = self.cfg, self.k, self.M

        def rnd(F, MT, Wsub, idmap, nvalid, carry):
            st, sel, mkv, Hprev, toks_prev, ids_prev, pos_prev = carry
            n = sel[0]
            x0 = mx.take(ids_prev, sel)                       # [1]
            pos = pos_prev + n + 1                            # position of x0
            # ---- draft: MTP batch over the previous verify's positions (repairs its KV), then chain ----
            tb = mx.concatenate([toks_prev[1:], ids_prev[M - 1:M]])
            tb = mx.where(mx.arange(M) == n, x0, tb)
            xb, mkv = mtp.step(cfg, MT, F["embed"], tb, Hprev, mkv, pos_prev, kv_len)
            x_row = mx.take(xb, sel, axis=0)                  # [1,H]
            drafts = []
            d, hn = mtp.draft_token(cfg, MT, Wsub, idmap, nvalid, x_row)
            drafts.append(d)
            for j in range(1, k):
                xj, mkv = mtp.step(cfg, MT, F["embed"], d, hn, mkv, pos + j - 1, kv_len)
                d, hn = mtp.draft_token(cfg, MT, Wsub, idmap, nvalid, xj)
                drafts.append(d)
            toks = mx.concatenate([x0] + drafts)              # [M]
            # ---- verify ----
            ids, Hn, st2 = fast.verify(cfg, F, toks, st, sel, pos, kv_len)
            eq = (toks[1:] == ids[:k]).astype(mx.int32)
            n_new = mx.sum(mx.cumprod(eq, axis=0)).astype(mx.int32).reshape(1)
            carry2 = (st2, n_new, mkv, Hn, toks, ids, pos)
            return carry2, (ids, n_new)

        return mx.compile(rnd) if self.compile else rnd

    def round(self, carry, kv_len):
        b = min(self.max_len, -(-kv_len // BUCKET) * BUCKET)
        fn = self._rounds.get(b)
        if fn is None:
            fn = self._rounds[b] = self._round_fn(b)
        return fn(self.F, self.MT, self.Wsub, self.idmap, self.nvalid, carry)

    # -- prefill (reference layout) ---------------------------------------------------------------
    def prefill(self, ids, chunk=2048, cache_at=()):
        """Main-model prefill (reference layout, all hidden rows kept for the MTP draft KV) -> (carry, first token, len(ids)).  With a
        prefix cache (SpecEngine(prefix_cache_mb=...)) it resumes from the deepest cached checkpoint that is a prefix of `ids`; a
        checkpoint carries the main state, the MTP K / V rows and the last hidden row (see prefix_cache.py)."""
        cfg = self.cfg
        S = len(ids)
        assert 0 < S <= self.max_len, (S, self.max_len)
        pc = self.pcache
        pos, node, split, toks, last_h, mkv = 0, None, 0, None, None, None
        if pc is not None:
            toks = np.asarray(ids, dtype=np.int32)
            node, split = pc.lookup(toks)
        if node is not None:
            st, chain = pc.restore(node, self.max_len)
            pos, last_h = node.pos, node.last["h"]
            if any(c.extra.get("mkv") is not None for c in chain):
                mkv = tuple(pc.cat_rows(chain, lambda c, j=j: None if c.extra.get("mkv") is None else c.extra["mkv"][j], self.max_len, 2)
                            for j in (0, 1))
        else:
            st = ref_state(cfg, self.max_len)
        marks = sorted({m for m in (*cache_at, split) if pos < m < S}) if pc is not None else []
        while pos < S:
            end = min(S, pos + chunk, *[m for m in marks if m > pos])
            part = mx.array([ids[pos:end]])
            h, st = ref_forward(cfg, self.W, part, st, pos, end)
            Hc = h[0]                                                     # [end-pos, H]
            mx.eval(st)
            # MTP K / V rows for positions p whose next token is known inside this chunk: pos-1 .. end-2 (row pos-1 needs the previous
            # chunk's last hidden row); position S-1 is covered by the first round
            if pos == 0:
                Hrow, tk, start = Hc[:-1], ids[1:end], 0
            else:
                Hrow, tk, start = (mx.concatenate([last_h, Hc[:-1]]) if end - pos > 1 else last_h), ids[pos:end], pos - 1
            if Hrow.shape[0]:
                mkv = mtp.prefill_kv(cfg, self.MT, self.W["embed"], Hrow, mx.array(tk, dtype=mx.int32), self.max_len, kv=mkv, start=start)
            last_h = Hc[-1:]
            pos = end
            if pc is not None:
                lo = max((node.pos if node is not None else 0) - 1, 0)
                seg = mx.stack([mkv[0][:, :, lo:pos - 1], mkv[1][:, :, lo:pos - 1]]) if mkv is not None and pos - 1 > lo else None
                node = pc.insert(node, toks, pos, st, last={"h": last_h}, extra={"mkv": seg} if seg is not None else {})
        first = argmax_head(self.W["embed"], self.F.get("head_tab"), last_h)        # [1]
        Hn = last_h
        if mkv is None:
            mkv = (mx.zeros((1, cfg.n_kv, self.max_len, cfg.head_dim), mx.bfloat16),) * 2
        M = self.M
        rep = lambda a: mx.concatenate([a] * M, axis=0)
        st = {"kv": st["kv"], "conv": [rep(c) for c in st["conv"]], "ssm": [rep(c) for c in st["ssm"]]}
        pad = lambda a, n, v=0: mx.concatenate([a, mx.full((n - a.shape[0],) + a.shape[1:], v, a.dtype)]) if a.shape[0] < n else a
        carry = (st, mx.array([0], dtype=mx.int32), mkv,
                 pad(Hn[-1:], M), pad(mx.array(ids[-1:], dtype=mx.int32), M), pad(first, M), mx.array(S - 1, dtype=mx.int32))
        mx.eval(carry)
        return carry, int(first.item()), S

    def generate(self, ids, n_tokens, stats=None, eos=(), cache_at=()):
        t0 = time.perf_counter()
        carry, first, S = self.prefill(ids, cache_at=cache_at)
        out = [first]
        t_prefill = time.perf_counter() - t0
        pend = []
        rounds = acc = 0
        M = self.M
        emitted = 1
        t1 = time.perf_counter()

        done = [False]

        def drain():
            nonlocal emitted, acc
            ids_, n_ = pend.pop(0)
            n_i = int(n_.item())
            new = ids_.tolist()[: n_i + 1]
            out.extend(new)
            emitted += n_i + 1
            acc += n_i
            if eos and any(t in eos for t in new):
                done[0] = True

        while emitted < n_tokens and not done[0]:
            need = S + emitted + M * (len(pend) + 2)
            if need > self.max_len:
                break
            carry, o = self.round(carry, need)
            mx.async_eval(o[0], o[1])
            pend.append(o)
            rounds += 1
            if len(pend) > 1:
                drain()
        while pend:
            drain()
        dt = time.perf_counter() - t1
        if stats is not None:
            stats.update(prefill_s=t_prefill, decode_s=dt, rounds=rounds, accepted=acc, tokens=len(out))
        out = out[:n_tokens]
        for i, t in enumerate(out):
            if t in eos:
                return out[:i]
        return out
