"""Continuous-batching server on top of batch.py: slots, admission / eviction, chunked prefill between decode steps.

  * A fixed set of S slots (pools of batch.Pools).  Requests wait in a FIFO queue; whenever a slot is free the head of the queue is
    prefilled (Engine.prefill_iter -> prefix cache, NAX GEMMs, exact like the solo path) and its state is copied into the slot.
  * Every loop iteration launches ONE batched decode step for all running sequences (ragged positions), then reads the tokens of the
    PREVIOUS step (so the host round trip hides behind the GPU), then runs ONE prefill chunk.  The chunk size is the knob between the
    inter-token latency of the running sequences (a chunk blocks the GPU for chunk/11.5k s) and the time-to-first-token of the
    newcomer.  Chunking changes the prefill numerics at the bf16 level (NAX tile configs, SDPA tiling), so by default the chunk
    size depends on the prompt only (a request's tokens are then reproducible: Engine.prefill(chunk=same) + solo decode gives
    them exactly); `idle_chunk` (e.g. 2048) opts into bigger chunks while nothing is decoding, at the price of load-dependent
    rounding.
  * A sequence that hits EOS / max_new is noticed one step late (its extra step is computed and dropped); its slot is recycled at
    once -- GPU work is ordered, so the new occupant's state copy lands after the stale step.
  * policy="static" is the classic baseline: fill the slots, run until EVERY sequence of the batch is done (finished rows keep
    being computed), only then admit the next batch.

Request life cycle (Request.state; every change is appended to Request.events as (state, t) and checked against LEGAL):

    arrival --> QUEUED --> PREFILL --> DECODE --> DONE
                  |            |__________________^   (first token is EOS / max_new == 1)
                  +--> REJECTED   (prompt empty or longer than max_len; or the queue already holds `max_queue` requests)

  QUEUED -> PREFILL   admission: strictly FIFO (arrival order); needs (a) a free slot and (b) no other prefill running (the GPU is
                      serial, a second concurrent prefill would only slow the first); static policy: only while the batch is filling.
                      The slot is a RESERVATION of the request's whole footprint (recurrent state + KV for max_len), so an admitted
                      request can never fail for lack of memory and nothing is ever preempted or swapped: overload shows up purely as
                      queueing delay (QUEUED time), and `max_queue` turns an unbounded backlog into an explicit REJECTED.
  PREFILL             one chunk per loop iteration (chunk size = the TTFT-vs-ITL knob); the prefix cache is consulted before the first
                      chunk; the finished state is copied into the slot and the first token is produced.
  PREFILL -> DECODE   first token emitted (t_first); -> DONE at once when it is EOS or max_new == 1.
  DECODE              the sequence is a row of every batched step until it ends; one step per loop iteration.
  DECODE -> DONE      EOS (finish_reason "eos"), max_new tokens ("max_new"), or the context is full ("length": max_new was clipped to
                      max_len - len(prompt) + 1).  continuous: the slot is free again at once (a step already in flight for it is
                      computed and dropped); static: the slot stays held until the whole batch has drained.

Memory: `Server(mem_budget=B)` is an upper bound on EVERYTHING the process holds on the GPU side -- model weights (+ the int8 head,
compressed copies), the prefix cache budget, the MLX free-buffer cache limit, the prefill job's private state and scratch, the slot
pools -- and `slots="auto"` is whatever number of slots fits in what is left (capped at `max_slots`: beyond ~16 rows the GEMVs are ALU
bound, more slots only add latency).  `plan_memory` returns the itemised plan; `Server.plan` keeps it.

Speculative mode (`Server(eng, spec=SpecEngine(like=...))`, batch_spec.py): while few sequences run, every loop
iteration launches one batched speculative ROUND (MTP drafts + verify of k+1 tokens per sequence, ~2.4 tokens per sequence per round
at B <= 4: x1.7 at B=2, x1.45 at B=4 over the plain step) instead of one plain step; above that the plain batched step is the faster
one and takes over.  The mode follows the load AND the measured acceptance: speculation runs while at most `spec_max` sequences run and
the running estimate of tokens per sequence per round beats the round's extra cost (batch_spec.SPEC_COST: x1.2 at B=1 ... x2.3 at B=8; on noise
text that estimate falls to ~1 and the server stays plain, re-probing every ~second).  The two modes share the main pools (K / V, and the recurrent state of a sequence that is not mid-round lives in
`Pools`); a speculative sequence additionally owns a snapshot entry (M recurrent snapshots) and the MTP layer's K / V.  Switching
modes (rare: drain the step in flight, ~ms of state copies per sequence) is what the hidden-state history is for: every prefill chunk
and every plain step records the final hidden state of its positions (`SpecPools.hist`), and a sequence that enters (or re-enters)
speculative mode rebuilds the missing MTP K / V rows from it.  The emitted tokens are the same as without speculation -- drafts decide
only how many tokens a round yields -- and `Request.out` is truncated at EOS / max_new inside a round.  Entering needs `spec_dwell`
consecutive iterations at a speculation-worthy load (hysteresis); leaving is immediate.

Time is real: `Request.arrival` is seconds after the start of `run`, the loop sleeps until the next arrival when idle, and all
latencies are wall-clock.  Greedy decoding only; the generated tokens equal Engine.generate's exactly (tests/batch.py).
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import mlx.core as mx

from . import batch as Bt
from . import batch_spec as BS
from . import mtp as Mtp

DEFAULT_BUCKETS = tuple(range(1, 17))      # one compiled step per batch size: a padded row costs ~0.6-1 ms, and the whole warmup is ~0.5 s

QUEUED, PREFILL, DECODE, DONE, REJECTED = "QUEUED", "PREFILL", "DECODE", "DONE", "REJECTED"
LEGAL = {"": {QUEUED, REJECTED}, QUEUED: {PREFILL, REJECTED}, PREFILL: {DECODE, DONE}, DECODE: {DONE}, DONE: set(), REJECTED: set()}
MIB = 1 << 20
SPEC_PAD = 8                                 # rows the pools carry beyond max_len: rounds still in flight when a sequence ends write up to 2M-3 of them
PREFILL_SCRATCH_BASE = 450 * MIB             # prefill scratch above the job's own state = BASE + PER_TOKEN * chunk; fitted to the measured
PREFILL_SCRATCH_PER_TOKEN = int(0.4 * MIB)   # peaks of chunks 128..2048 (bench/serve_memory.py: 0.38-0.59 GiB .. 1.1 GiB, +10% margin)


@dataclass
class Request:
    rid: int
    prompt: list
    max_new: int
    arrival: float = 0.0                         # seconds after the start of the run
    eos: frozenset = frozenset()
    tag: str = ""
    # ---- results
    out: list = field(default_factory=list)
    t_first: float = None                        # first token available
    t_done: float = None
    t_admit: float = None                        # left the queue (slot assigned, prefill started)
    t_tokens: list = field(default_factory=list)  # host-side observation time of every token
    hit_tokens: int = 0                          # prompt tokens served from the prefix cache
    state: str = ""                              # QUEUED / PREFILL / DECODE / DONE / REJECTED (empty before arrival)
    events: list = field(default_factory=list)   # [(state, seconds after the start of the run), ...]
    finish_reason: str = ""                      # "eos" | "max_new" | "length" | rejection reason
    clipped: bool = False                        # max_new was cut to fit the context

    def to(self, state, t, reason=""):
        assert state in LEGAL[self.state], (self.rid, self.state, state)
        self.state = state
        self.events.append((state, t))
        if reason:
            self.finish_reason = reason


def seq_state_bytes(cfg, max_len):
    """One sequence's recurrent state + KV at capacity `max_len` (what Engine.prefill hands over)."""
    return Bt.Pools.slot_bytes(cfg, max_len, double_conv=False)


def plan_memory(eng, mem_budget=None, max_len=None, chunk=512, cache_limit=1 << 30, slots="auto", max_slots=16,
                pcache_mb=None, workspace_per_token=None, slack=256 * MIB, spec=None):
    """Itemised GPU-memory plan.  mem_budget: bytes the whole process may hold (default: half of the physical memory).

        budget = weights + prefix cache + MLX buffer cache + prefill job (private state x2 + scratch(chunk)) + slack
                 + (slots + 1 trash slot) * slot [+ (nspec + 1) * speculative snapshot entry]

    spec=(nspec, M): the speculative mode's pools are included -- per slot the MTP K / V and the hidden-state history, plus the
    (nspec + 1) snapshot entries of M recurrent states (SPEC_PAD extra K / V rows per slot).

    `weights` is what is resident now (call before the pools exist): the bf16 tables, the compressed and int8 copies, MTP, ...
    minus whatever the prefix cache already holds.  slots="auto" -> the largest count that fits, at most `max_slots`.
    Returns a dict (all sizes in bytes) with the itemisation, `slots`, and `total` (the planned footprint)."""
    cfg = eng.cfg
    max_len = max_len or eng.max_len
    if mem_budget is None:
        mem_budget = mx.device_info()["memory_size"] // 2
    pc = eng.pcache
    pcache = int(pcache_mb * MIB) if pcache_mb is not None else (pc.budget if pc is not None else 0)
    weights = mx.get_active_memory() - (pc.bytes if pc is not None else 0)
    wpt = PREFILL_SCRATCH_PER_TOKEN if workspace_per_token is None else workspace_per_token
    job = 2 * seq_state_bytes(cfg, max_len) + PREFILL_SCRATCH_BASE + int(wpt * chunk)
    Lp = max_len + (SPEC_PAD if spec else 0)
    per_slot = Bt.Pools.slot_bytes(cfg, Lp)
    spec_entries = 0
    if spec:
        per_slot += BS.SpecPools.mtp_bytes(cfg, Lp)
        spec_entries = (spec[0] + 1) * BS.SpecPools.entry_bytes(cfg, spec[1]) + (spec[0] + 1) * spec[1] * 2 * cfg.hidden
    fixed = weights + pcache + (cache_limit or 0) + job + slack + spec_entries
    fit = (mem_budget - fixed) // per_slot - 1                      # minus the trash slot
    n = max(0, min(int(fit), max_slots)) if slots == "auto" else int(slots)
    plan = dict(budget=mem_budget, weights=weights, pcache=pcache, buffer_cache=cache_limit or 0, prefill_job=job, slack=slack,
                per_slot=per_slot, slots=n, pools=(n + 1) * per_slot, max_len=max_len, spec_entries=spec_entries)
    plan["total"] = fixed + plan["pools"]
    if n < 1 or plan["total"] > mem_budget:
        raise MemoryError(f"memory budget {mem_budget / MIB:.0f} MiB cannot hold {max(n, 1)} slot(s) of max_len {max_len}: "
                          f"{plan['total'] / MIB:.0f} MiB needed ({describe_plan(plan)})")
    return plan


def describe_plan(p):
    g = lambda b: f"{b / MIB:.0f}"
    return (f"budget {g(p['budget'])} MiB = weights {g(p['weights'])} + prefix cache {g(p['pcache'])} + buffer cache {g(p['buffer_cache'])}"
            f" + prefill job {g(p['prefill_job'])} + slack {g(p['slack'])} + pools {g(p['pools'])} ({p['slots']} slots + trash x"
            f" {g(p['per_slot'])} MiB, max_len {p['max_len']})" + (f" + speculative entries {g(p['spec_entries'])}" if p.get("spec_entries") else "")
            + f" -> planned {g(p['total'])} MiB")


class _Seq:
    __slots__ = ("req", "slot", "pos", "pending", "done", "ss", "mtp_len", "par", "n_last")

    def __init__(self, req, slot, pos):
        self.req, self.slot, self.pos, self.pending, self.done = req, slot, pos, 0, False
        self.ss = -1                 # speculative snapshot entry (-1: plain mode)
        self.mtp_len = 0             # MTP K / V rows [0, mtp_len) are valid
        self.par = 0                 # speculative: rounds launched mod 2 (the conv buffer the next round reads)
        self.n_last = 0              # speculative: tokens accepted by the last handled round (its snapshot is the live state)


class _PrefillJob:
    def __init__(self, req, slot, gen):
        self.req, self.slot, self.gen = req, slot, gen
        self.h = []                  # speculative: the prompt's hidden rows (Engine.prefill_iter h_out)


class Server:
    def __init__(self, eng, slots=8, max_len=None, chunk=512, idle_chunk=None, buckets=DEFAULT_BUCKETS, policy="continuous",
                 cache_limit=1 << 30, mem_budget=None, max_slots=16, max_queue=None, spec=None, spec_max=6, spec_dwell=6, spec_prior=2.3):
        """`cache_limit`: cap on MLX's free-buffer cache (process-wide).  Prefill chunks allocate activations of many different
        sizes that are never reused, so the unbounded default cache grows to ~8.5 GB on a 16 GB machine and the final chunk of a
        prompt then stalls 0.2-1.5 s (memory pressure): measured on the mixed trace, TTFT p95 3.4-5.3 s -> 0.8 s once capped.

        `spec`: a spec.SpecEngine (k=2, built with like=<the SpecEngine / weights of `eng`>) enables the speculative mode for loads of at
        most `spec_max` sequences (see the module docstring); `spec_dwell`: iterations at a speculation-worthy load before plain ->
        speculative; `spec_prior`: the initial / relaxation target of the acceptance estimate (tokens per sequence per round)."""
        assert policy in ("continuous", "static")
        if cache_limit is not None:
            mx.set_cache_limit(cache_limit)
        self.eng, self.cfg, self.F = eng, eng.cfg, eng.F
        self.max_len = max_len or eng.max_len
        assert self.max_len <= eng.max_len
        self.max_queue = max_queue
        self.spec = spec
        self.Lp = self.max_len + (SPEC_PAD if spec is not None else 0)
        self.plan = None
        pspec = (spec_max, spec.M) if spec is not None else None
        if slots == "auto" or mem_budget is not None:
            self.plan = plan_memory(eng, mem_budget, self.max_len, chunk, cache_limit, slots, max_slots, spec=pspec)
            slots = self.plan["slots"]
            mx.set_memory_limit(self.plan["budget"])
        self.S = slots
        self.chunk, self.idle_chunk, self.policy = chunk, idle_chunk, policy
        self.buckets = tuple(b for b in buckets if b < slots) + (slots,)
        self.P = Bt.Pools(self.cfg, slots, self.Lp)
        self.tree = self.P.as_tree()
        self.cur = mx.zeros((slots + 1,), mx.int32)
        self._fns = {}
        self._sfns = {}
        self.stats = {}
        self.tpr = 0.0
        if spec is not None:
            self.nspec = max(1, min(spec_max, slots, len(BS.SPEC_COST)))
            self.spec_dwell, self.spec_prior, self.tpr = spec_dwell, spec_prior, spec_prior
            self.SP = BS.SpecPools(self.cfg, self.nspec, slots, spec.M, self.Lp)
            self.sp_tree = self.SP.as_tree()
            self.C = dict(self.SP.carry)
            self.sbuckets = tuple(b for b in buckets if b < self.nspec) + (self.nspec,)

    # ---- the compiled step per padded batch size ----------------------------------------------------
    def _fn(self, B):
        fn = self._fns.get(B)
        if fn is None:
            cfg = self.cfg
            if self.spec is not None:
                fn = mx.compile(lambda F, P, cur, sl, ps, hist: Bt.step(cfg, F, P, cur, sl, ps, hist))
            else:
                fn = mx.compile(lambda F, P, cur, sl, ps: Bt.step(cfg, F, P, cur, sl, ps))
            self._fns[B] = fn
        return fn

    def _sfn(self, B):
        fn = self._sfns.get(B)
        if fn is None:
            cfg, k = self.cfg, self.spec.k
            fn = self._sfns[B] = mx.compile(lambda F, MT, Ws, im, nv, P, SP, C, sl, ssl, par:
                                            BS.spec_round(cfg, k, F, MT, Ws, im, nv, P, SP, C, sl, ssl, par))
        return fn

    def _bucket(self, n):
        return next(b for b in self.buckets if b >= n)

    def _call(self, B, sl, ps):
        if self.spec is None:
            ids, self.cur = self._fn(B)(self.F, self.tree, self.cur, sl, ps)
            return ids, (ids, self.cur)
        ids, self.cur, tok = self._fn(B)(self.F, self.tree, self.cur, sl, ps, self.SP.hist)
        return ids, (ids, self.cur, tok)

    def launch(self, rows):
        """rows: list of (slot, pos).  Queues one decode step, returns the (lazy) ids [Bpad]."""
        B = self._bucket(len(rows))
        sl = np.full((B,), self.S, dtype=np.int32)             # padding rows point at the trash slot
        ps = np.zeros((B,), dtype=np.int32)
        for i, (s, p) in enumerate(rows):
            sl[i], ps[i] = s, p
        ids, outs = self._call(B, mx.array(sl), mx.array(ps))
        mx.async_eval(*outs)
        return ids

    def launch_spec(self, rows):
        """rows: list of (slot, entry, par).  Queues one speculative round, returns the (lazy) (ids [Bpad, M], n_new [Bpad])."""
        B = next(b for b in self.sbuckets if b >= len(rows))
        sl = np.full((B,), self.S, dtype=np.int32)
        ssl = np.full((B,), self.nspec, dtype=np.int32)         # padding rows: trash slot + trash entry
        par = np.zeros((B,), dtype=np.int32)
        for i, (s, e, q) in enumerate(rows):
            sl[i], ssl[i], par[i] = s, e, q
        sp = self.spec
        ids, nn, self.C = self._sfn(B)(self.F, sp.MT, sp.Wsub, sp.idmap, sp.nvalid, self.tree, self.sp_tree, self.C,
                                       mx.array(sl), mx.array(ssl), mx.array(par))
        mx.async_eval(ids, nn)
        return ids, nn

    def warmup(self):
        """Compile + JIT every bucket (a first call per shape costs ~0.2-1 s) on the trash slot."""
        for b in self.buckets:
            sl = mx.full((b,), self.S, dtype=mx.int32)
            ps = mx.zeros((b,), dtype=mx.int32)
            ids, outs = self._call(b, sl, ps)
            mx.eval(*outs)
        if self.spec is not None:
            sp = self.spec
            for b in self.sbuckets:
                sl = mx.full((b,), self.S, dtype=mx.int32)
                ssl = mx.full((b,), self.nspec, dtype=mx.int32)
                par = mx.zeros((b,), dtype=mx.int32)
                ids, nn, self.C = self._sfn(b)(self.F, sp.MT, sp.Wsub, sp.idmap, sp.nvalid, self.tree, self.sp_tree, self.C, sl, ssl, par)
                mx.eval(ids, nn)
            self._build_mtp(self.S, 0, 8, [0] * 9)

    def spec_ok(self, n, margin):
        """Is speculation worth it for n running sequences: few enough, and the acceptance estimate beats `margin` x the round's cost."""
        return 1 <= n <= self.nspec and self.tpr >= margin * BS.SPEC_COST[n - 1]

    # ---- plain <-> speculative state moves (host-synchronous; the caller has drained the steps in flight) ------------
    def _build_mtp(self, slot, a, b, toks):
        """MTP K / V rows [a, b) of `slot` from the hidden-state history + the tokens (position p's row needs the token at p + 1)."""
        cfg, sp, SP = self.cfg, self.spec, self.SP
        n_kv, D, Lp = cfg.n_kv, cfg.head_dim, self.Lp
        while a < b:
            e = min(b, a + 512)
            kv0 = (SP.mK[slot][None], SP.mV[slot][None])
            K, V = Mtp.prefill_kv(cfg, sp.MT, self.eng.W["embed"], SP.hist[slot, a:e], mx.array(toks[a + 1:e + 1], dtype=mx.int32),
                                  Lp, kv=kv0, start=a)
            mx.eval(Bt.blit(SP.mK, K, (e - a) * D, slot * n_kv * Lp * D + a * D, a * D, ny=n_kv, dstride=Lp * D, sstride=Lp * D),
                    Bt.blit(SP.mV, V, (e - a) * D, slot * n_kv * Lp * D + a * D, a * D, ny=n_kv, dstride=Lp * D, sstride=Lp * D))
            a = e

    def _to_spec(self, sq, ss):
        """Plain -> speculative for a sequence whose steps are all read: MTP K / V up to position pos - 2, the recurrent state into
        snapshot 0 / conv buffer 0 of entry `ss`, and the round carry (previous verify = the single row at pos - 1, accepted 0)."""
        r, SP, P, M = sq.req, self.SP, self.P, self.spec.M
        hid = self.cfg.hidden
        toks = r.prompt + r.out
        pos = len(toks) - 1                                       # position of the token to feed next (r.out[-1])
        assert pos == sq.pos, (pos, sq.pos)
        self._build_mtp(sq.slot, sq.mtp_len, pos - 1, toks)
        sq.mtp_len = pos - 1
        outs = []
        for i in range(len(SP.conv)):
            outs.append(Bt.blit(SP.conv[i], P.conv[i], 3 * Bt.CD, (ss * 2 * M) * 3 * Bt.CD, (sq.slot * 2 + (pos & 1)) * 3 * Bt.CD))
            outs.append(Bt.blit(SP.ssm[i], P.ssm[i], Bt.HV * Bt.DV * Bt.DK, ss * M * Bt.HV * Bt.DV * Bt.DK, sq.slot * Bt.HV * Bt.DV * Bt.DK))
        C = self.C
        C["H"][ss, 0] = SP.hist[sq.slot, pos - 1]
        C["toks"][ss, 0] = toks[pos - 1]
        C["ids"][ss, 0] = toks[pos]
        C["pos"][ss] = pos - 1
        C["sel"][ss] = 0
        mx.eval(outs, C)
        sq.ss, sq.par, sq.n_last = ss, 0, 0

    def _to_plain(self, sq):
        """Speculative -> plain: the live snapshot (index n_last of the last round's buffer) goes back into the plain pools, the last
        round's hidden rows into the history; MTP K / V stay valid below the last round's first position."""
        r, SP, P, M = sq.req, self.SP, self.P, self.spec.M
        hid = self.cfg.hidden
        ss, n = sq.ss, sq.n_last
        pos = len(r.prompt) + len(r.out) - 1                      # position of r.out[-1]
        p0 = pos - n - 1                                          # first position of the last round's verify
        outs = []
        for i in range(len(SP.conv)):
            outs.append(Bt.blit(P.ssm[i], SP.ssm[i], Bt.HV * Bt.DV * Bt.DK, sq.slot * Bt.HV * Bt.DV * Bt.DK, (ss * M + n) * Bt.HV * Bt.DV * Bt.DK))
            outs.append(Bt.blit(P.conv[i], SP.conv[i], 3 * Bt.CD, (sq.slot * 2 + (pos & 1)) * 3 * Bt.CD, ((ss * 2 + sq.par) * M + n) * 3 * Bt.CD))
        outs.append(Bt.blit(SP.hist, self.C["H"], (n + 1) * hid, (sq.slot * self.Lp + p0) * hid, ss * M * hid))
        self.cur[sq.slot] = r.out[-1]
        mx.eval(outs, self.cur)
        sq.pos, sq.mtp_len, sq.ss = pos, p0, -1

    # ---- the scheduler -------------------------------------------------------------------------------
    def run(self, requests):
        """Replay `requests` (arrival times relative to now) to completion; fills Request.out / t_* in place.  Returns run stats.

        policy="static": the batch is formed from whatever is waiting when the server is idle, prefilled, and decoded until every
        sequence is done; a finished sequence leaves the compute at once (a lower bound for the cost of static batching) but its slot
        stays occupied, and nothing new is admitted, until the whole batch has drained."""
        S = self.S
        spec_on = self.spec is not None
        static = self.policy == "static"
        reqs = sorted(requests, key=lambda r: r.arrival)
        pending = deque(reqs)
        waiting = deque()
        free = list(range(S))
        held = []                                # static: slots of finished sequences, released when the batch drains
        running: dict[int, _Seq] = {}            # slot -> seq (in the decode batch)
        free_ss = list(range(self.nspec)) if spec_on else []
        job = None
        inflight = None                          # ("plain", list[_Seq], ids) | ("spec", list[_Seq], ids, n_new)
        fill = True                              # static: still admitting
        mode_spec = False                        # the decode mode of the next launch
        calm = 0                                 # consecutive iterations at a speculation-sized load while in plain mode
        t0 = time.perf_counter()
        now = lambda: time.perf_counter() - t0
        steps = rows_total = nfinished = 0
        rounds = spec_rows = spec_tokens = switches = 0

        def end_reason(r, tok):
            if tok in r.eos:
                return "eos"
            return ("length" if r.clipped else "max_new") if len(r.out) >= r.max_new else None

        def finish(sq, t, reason):
            nonlocal nfinished
            sq.done = True
            sq.req.t_done = t
            sq.req.to(DONE, t, reason)
            running.pop(sq.slot, None)
            if sq.ss >= 0:
                free_ss.append(sq.ss)
                sq.ss = -1
            (held if static else free).append(sq.slot)
            nfinished += 1

        def handle(inf):
            nonlocal spec_tokens
            kind, rows, ids = inf[0], inf[1], inf[2]
            arr = np.array(ids)                  # blocks until that step is done
            nv = np.array(inf[3]) if kind == "spec" else None
            t = now()
            got = []
            for i, sq in enumerate(rows):
                sq.pending -= 1
                if sq.done:                      # finished earlier: this step was already in flight
                    continue
                r = sq.req
                if kind == "plain":
                    new = [int(arr[i])]
                else:
                    sq.n_last = int(nv[i])
                    new = [int(x) for x in arr[i, :sq.n_last + 1]]
                    spec_tokens += len(new)
                    got.append(sq.n_last + 1)
                for tok in new:
                    r.out.append(tok)
                    r.t_tokens.append(t)
                    why = end_reason(r, tok)
                    if why:
                        finish(sq, t, why)
                        break
            if got:                              # acceptance estimate (a sequence cut short by EOS / max_new still counts what the round gave)
                self.tpr += 0.1 * (sum(got) / len(got) - self.tpr)

        def drain():
            nonlocal inflight
            if inflight is not None:
                handle(inflight)
                inflight = None

        def switch_plain():
            nonlocal mode_spec, calm, switches
            drain()
            for sq in list(running.values()):
                if sq.ss >= 0:
                    free_ss.append(sq.ss)
                    self._to_plain(sq)
            mode_spec, calm = False, 0
            switches += 1

        def switch_spec():
            nonlocal mode_spec, calm, switches
            drain()
            for sq in list(running.values()):
                self._to_spec(sq, free_ss.pop())
            mode_spec, calm = True, 0
            switches += 1

        while nfinished < len(reqs):
            t = now()
            while pending and pending[0].arrival <= t:
                r = pending.popleft()
                why = ("prompt_too_long" if not 0 < len(r.prompt) <= self.max_len else
                       "queue_full" if self.max_queue is not None and len(waiting) >= self.max_queue else None)
                if why:
                    r.to(REJECTED, r.arrival, why)
                    nfinished += 1
                else:
                    r.to(QUEUED, r.arrival)
                    waiting.append(r)

            # 0) decode mode: speculative while few sequences run and the drafts are accepted often enough
            if spec_on and not (static and fill):
                n = len(running)
                if mode_spec:
                    if n and not self.spec_ok(n, 1.0):
                        switch_plain()
                else:
                    if n == 0:
                        if self.spec_ok(1, 1.1):
                            mode_spec, calm = True, 0
                    elif self.spec_ok(n, 1.1):
                        calm += 1
                        if calm >= self.spec_dwell:
                            switch_spec()
                    else:
                        calm = 0
                    if n:                        # no round, no news: drift back towards the prior so that plain mode re-probes
                        self.tpr += 0.003 * (self.spec_prior - self.tpr)

            # 1) one decode step / round for everything running -- launched BEFORE the previous one's tokens are read
            batch = [] if (static and fill) else [sq for sq in running.values() if len(sq.req.out) + sq.pending < sq.req.max_new]
            new_inflight = None
            if batch:
                if mode_spec:
                    ids, nn = self.launch_spec([(sq.slot, sq.ss, sq.par) for sq in batch])
                    for sq in batch:
                        sq.pending += 1
                        sq.par ^= 1
                    new_inflight = ("spec", batch, ids, nn)
                    rounds += 1
                    spec_rows += len(batch)
                else:
                    ids = self.launch([(sq.slot, sq.pos) for sq in batch])
                    for sq in batch:
                        sq.pos += 1
                        sq.pending += 1
                    new_inflight = ("plain", batch, ids)
                    steps += 1
                    rows_total += len(batch)

            # 2) the tokens of the previous step
            if inflight is not None:
                handle(inflight)
            inflight = new_inflight

            # 3) admission + one prefill chunk
            if job is None and waiting and free and (not static or fill):
                r = waiting.popleft()
                slot = free.pop()
                r.t_admit = now()
                r.to(PREFILL, r.t_admit)
                cap = self.max_len - len(r.prompt) + 1
                r.clipped = r.max_new > cap
                r.max_new = max(1, min(r.max_new, cap))
                pc = self.eng.pcache
                job = _PrefillJob(r, slot, None)
                job.gen = self.eng.prefill_iter(
                    r.prompt, chunk_fn=lambda: self.idle_chunk if (self.idle_chunk and not running) else self.chunk,
                    h_out=job.h if spec_on else None)
                job.hit_before = pc.stats["hit_tokens"] if pc is not None else 0
            if job is not None:
                try:
                    next(job.gen)
                except StopIteration as e:
                    nxt, st, n = e.value
                    mx.eval(nxt)
                    r = job.req
                    if self.eng.pcache is not None:
                        r.hit_tokens = self.eng.pcache.stats["hit_tokens"] - job.hit_before
                    self.P.load(job.slot, st, n)
                    if spec_on:
                        hid = self.cfg.hidden
                        mx.eval(Bt.blit(self.SP.hist, job.h[0], n * hid, job.slot * self.Lp * hid, 0))
                    self.cur[job.slot] = nxt.reshape(()).astype(mx.int32)
                    tok = int(nxt.item())
                    tt = now()
                    r.out.append(tok)
                    r.t_tokens.append(tt)
                    r.t_first = tt
                    sq = _Seq(r, job.slot, n)
                    why = end_reason(r, tok)
                    if why:
                        finish(sq, tt, why)
                    else:
                        r.to(DECODE, tt)
                        running[job.slot] = sq
                        if mode_spec:
                            if self.spec_ok(len(running), 1.0):
                                self._to_spec(sq, free_ss.pop())
                            else:
                                switch_plain()
                    job = None
            if static:
                if fill and job is None and (running or held) and (not waiting or not free):
                    fill = False                 # batch formed
                if not fill and not running and inflight is None:
                    free.extend(held)            # batch drained
                    held.clear()
                    fill = True

            # idle: nothing to compute until the next arrival
            if not running and job is None and inflight is None and not waiting and pending:
                dt = pending[0].arrival - now()
                if dt > 0:
                    time.sleep(dt)
        if inflight is not None:
            handle(inflight)
        return dict(wall=now(), steps=steps, mean_batch=(rows_total + spec_rows) / max(steps + rounds, 1), rounds=rounds, mode_switches=switches,
                    spec_rows=spec_rows, spec_tokens=spec_tokens, tokens_per_round=spec_tokens / max(spec_rows, 1),
                    mean_spec_batch=spec_rows / max(rounds, 1), tpr=self.tpr)
