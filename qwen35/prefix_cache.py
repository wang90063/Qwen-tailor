"""Prefix cache for the exact (bf16) prefill path: resume a prefill from the state after a token prefix seen before.

A hybrid model cannot truncate its state to "the first p tokens": the 18 GatedDeltaNet layers keep one recurrent state each (ssm
[1,16,128,128] fp32 ~1 MiB + conv [1,3,6144] bf16) that only describes the position it was captured at.  So the cache holds
*checkpoints* (nodes): the complete state after `pos` tokens, plus the tokens themselves, and a lookup finds the deepest checkpoint
that is a prefix of the new prompt (longest common prefix over the stored token arrays, not a fixed grid).  The 6 full-attention layers' K / V are
stored once per token as *segments* (rows [parent.pos : pos) of every K / V buffer), so a chain of checkpoints of one long prompt costs
12 KiB / token + 19 MiB per checkpoint, not a full copy per checkpoint; a restore concatenates the chain's segments into fresh
contiguous max_len buffers (~0.5 ms for 2k tokens).

Checkpoints appear for free wherever a prefill has a chunk boundary and at the end of every prompt (multi-turn chat: the next turn
extends the last prompt).  A prompt that partly overlaps a stored checkpoint (same system prompt, different question) also gets an
extra checkpoint at the shared length, so the *next* request with that system prompt hits ("split learning", one extra chunk
boundary ~5 ms once); `cache_at=[n, ...]` forces checkpoints at known positions.  Eviction: least recently used leaf first, memory budget in MB.

Numerics: a hit continues exactly like a chunked prefill split at the checkpoint (the cached prefix state is bit-identical to what
the uncached run computed; the suffix runs through the same kernels, and MLX's SDPA tiles by query length (< 1024 vs >= 1024), so
the suffix rows can differ from a single-pass prefill at the bf16 rounding level; tests/prefix_cache.py bounds that and checks that
generation is token-identical).  bf16 path only: with QWEN35_W8A8 set, the cache is bypassed (with a warning).
"""
from __future__ import annotations
import sys
import numpy as np
import mlx.core as mx

from . import gemm_i8 as I8

_warned = []


class Node:
    __slots__ = ("parent", "pos", "toks", "kv", "conv", "ssm", "last", "extra", "nbytes", "stamp", "nchild", "uid")

    def __init__(self):
        self.parent = None
        self.nchild = 0

    def chain(self):
        out, n = [], self
        while n is not None:
            out.append(n)
            n = n.parent
        out.reverse()
        return out


def _nb(*arrs):
    return sum(a.nbytes for a in arrs if a is not None)


def _own(d):
    """Copy the arrays in dict `d` into buffers of their own: a slice of a big activation (the last hidden row, a conv window) would keep
    the whole activation alive for as long as the cache entry lives (a 2k-token prompt pinned ~0.6 GiB through the 18 conv windows)."""
    return None if d is None else {k: (mx.array(v) if isinstance(v, mx.array) else v) for k, v in d.items()}


class PrefixCache:
    MIN_SPLIT = 32        # a shared prefix shorter than this is not worth a checkpoint of its own
    MIN_GAIN = 32         # ... nor one that saves fewer tokens than this over the best existing hit

    def __init__(self, budget_mb=1024):
        self.budget = int(budget_mb * (1 << 20))
        self.nodes: dict[int, Node] = {}
        self.bytes = 0
        self._clock = 0
        self._uid = 0
        self.stats = dict(lookups=0, hits=0, hit_tokens=0, prompt_tokens=0, inserted=0, evicted=0, splits=0)
        self.last_hit = 0                      # pos of the checkpoint used by the latest lookup (0 = miss)

    # ---- helpers ----------------------------------------------------------------------------------
    @staticmethod
    def active():
        if I8.NAMES:
            if not _warned:
                print("prefix cache: bypassed (bf16 exact path only; QWEN35_W8A8 is set)", file=sys.stderr)
                _warned.append(1)
            return False
        return True

    def _touch(self, node):
        self._clock += 1
        while node is not None:
            node.stamp = self._clock
            node = node.parent

    def __len__(self):
        return len(self.nodes)

    # ---- lookup -----------------------------------------------------------------------------------
    def lookup(self, ids):
        """ids: token ids of the new prompt.  Returns (node, split): the deepest checkpoint that is a prefix of ids (None = miss; a
        checkpoint covering the whole prompt only counts if it carries the first token), and a position (0 = none) at which the new
        prefill should add a checkpoint because the prompt overlaps a stored one up to there."""
        if not self.active():
            return None, 0
        t = np.asarray(ids, dtype=np.int32)
        S = len(t)
        best, part = None, 0
        for nd in self.nodes.values():
            n = min(nd.pos, S)
            eq = nd.toks[:n] == t[:n]
            lcp = n if eq.all() else int(eq.argmin())
            if lcp == nd.pos:
                if (nd.pos < S or nd.last is not None) and (best is None or nd.pos > best.pos):
                    best = nd
            elif lcp < S:
                part = max(part, lcp)
        base = best.pos if best is not None else 0
        split = part if (part >= self.MIN_SPLIT and part - base >= self.MIN_GAIN) else 0
        st = self.stats
        st["lookups"] += 1
        st["prompt_tokens"] += S
        self.last_hit = base
        if best is not None:
            st["hits"] += 1
            st["hit_tokens"] += base
            self._touch(best)
        if split:
            st["splits"] += 1
        return best, split

    # ---- restore ----------------------------------------------------------------------------------
    @staticmethod
    def cat_rows(chain, get, total, axis):
        """Concatenate the per-node segments `get(node)` ([..., n, ...] arrays or None) along `axis`, zero-padded to `total` rows."""
        parts = [g for g in (get(c) for c in chain) if g is not None]
        have = sum(p.shape[axis] for p in parts)
        if total > have:
            shp = list(parts[0].shape)
            shp[axis] = total - have
            parts.append(mx.zeros(shp, parts[0].dtype))
        return mx.concatenate(parts, axis=axis) if len(parts) > 1 else parts[0]

    def restore(self, node, max_len):
        """State dict of the checkpoint ({"kv": [(K, V)] x6 [1,2,max_len,256], "conv": [...], "ssm": [...]}); the recurrent arrays are
        shared with the cache (MLX arrays are immutable values), the K / V buffers are fresh and contiguous."""
        assert node.pos <= max_len, (node.pos, max_len)
        chain = node.chain()
        nb = node.kv.shape[0]
        pad = mx.zeros((1, node.kv.shape[2], max_len - node.pos, node.kv.shape[4]), mx.bfloat16) if max_len > node.pos else None
        bufs = []
        for j in range(nb):
            parts = [c.kv[j] for c in chain] + ([pad] if pad is not None else [])
            bufs.append(mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        st = {"kv": [(bufs[2 * i], bufs[2 * i + 1]) for i in range(nb // 2)],
              "conv": [node.conv[i] for i in range(node.conv.shape[0])], "ssm": [node.ssm[i] for i in range(node.ssm.shape[0])]}
        return st, chain

    # ---- insert -----------------------------------------------------------------------------------
    def insert(self, parent, toks, pos, state, last=None, extra=None):
        """Checkpoint the live `state` (valid after `pos` tokens, K / V rows [parent.pos : pos) are the new ones).  `toks`: the prompt's
        token ids as an int32 array (>= pos long).  `last`: optional dict with what a hit at pos == len(prompt) needs (first token ...);
        `extra`: dict of per-segment arrays for the owning engine (SpecEngine: MTP K / V rows, last hidden row).  Returns the node
        (an existing one for identical tokens)."""
        if not self.active():
            return parent
        toks = np.asarray(toks, dtype=np.int32)
        for nd in self.nodes.values():
            if nd.pos == pos and np.array_equal(nd.toks, toks[:pos]):
                self._touch(nd)
                if nd.last is None and last is not None:
                    nd.last = _own(last)
                    mx.async_eval(*[v for v in nd.last.values() if isinstance(v, mx.array)])
                    nd.nbytes += _nb(*nd.last.values())
                    self.bytes += _nb(*nd.last.values())
                return nd
        a = parent.pos if parent is not None else 0
        assert pos > a
        nd = Node()
        nd.parent, nd.pos, nd.toks = parent, pos, toks[:pos].copy()
        nd.kv = mx.stack([x[:, :, a:pos] for k, v in state["kv"] for x in (k, v)])        # [12, 1, 2, pos-a, 256], a real copy
        nd.conv, nd.ssm = mx.stack(state["conv"]), mx.stack(state["ssm"])                # [18, ...] real copies
        nd.last, nd.extra = _own(last), extra
        nd.nbytes = _nb(nd.kv, nd.conv, nd.ssm) + _nb(*(nd.last or {}).values()) + _nb(*(extra or {}).values())
        if nd.nbytes > self.budget:
            return parent
        self._uid += 1
        nd.uid = self._uid
        self.nodes[nd.uid] = nd
        if parent is not None:
            parent.nchild += 1
        self.bytes += nd.nbytes
        self.stats["inserted"] += 1
        self._touch(nd)
        mx.async_eval(nd.kv, nd.conv, nd.ssm, *[v for d in (nd.last, extra) for v in (d or {}).values() if isinstance(v, mx.array)])
        self._evict(protect=nd)
        return nd

    def _evict(self, protect=None):
        keep = set()
        n = protect
        while n is not None:                    # the chain being built / used stays
            keep.add(n.uid)
            n = n.parent
        while self.bytes > self.budget:
            leaves = [n for n in self.nodes.values() if n.nchild == 0 and n.uid not in keep]
            if not leaves:
                break
            v = min(leaves, key=lambda n: n.stamp)
            del self.nodes[v.uid]
            if v.parent is not None:
                v.parent.nchild -= 1
            self.bytes -= v.nbytes
            self.stats["evicted"] += 1

    def clear(self):
        self.nodes.clear()
        self.bytes = 0
