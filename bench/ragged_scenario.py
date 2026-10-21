"""Ragged-batch scenario: a mixed local-server workload replayed in real time against sequential / static-batch / continuous-batch serving.

usage: ragged_scenario.py [trace] [--slots N|auto] [--mem-budget GiB] [--max-queue Q] [--dur S] [--rate R] [--seed K] [--chunks 256,512,2048] [--verify] [--no-cache] [--policies a,b,c]

Traces (every one is ragged in prompt length, generation length and arrival time):
  mixed  open-loop mix of four request classes arriving as a Poisson process plus one burst
           chat    prompt 20-120,    gen 40-160    (short questions)
           rag     prompt 1200-3000, gen 30-100    (long prompt; half of them reuse one of 3 shared 1200-token documents)
           agent   700-token shared system prompt + 40-300 suffix, gen 20-60   (tool-calling turns: prefix-cache hits)
           long    prompt 50-200,    gen 300-500   (a long generation that holds a slot while the others come and go)
  burst  16 requests at t=0, prompt lengths log-uniform 16-3000, gen 32-256  (maximally ragged: lengths differ by 200x, sequences
           finish at 16 different steps)
  churn  2 very long generations (600) + 60 short requests (gen 8-24) spread over the run (slot recycling under load)

Policies: `sequential` (solo Engine, FCFS, the no-batching system), `static` (fill the slots, decode until the whole batch drained),
`continuous` (this work; chunked prefill between decode steps, one run per --chunks value).
Metrics per class and overall: TTFT (arrival -> first token), ITL (gap between consecutive tokens of a request), end-to-end latency,
queueing delay, aggregate generated tokens/s over the makespan, mean decode batch.  --verify checks every generated sequence against
solo decoding (same prefill chunking), token for token.  The GPU is shared and noisy: numbers are one replay each (see
bench/quiet.py for the A/B methodology used elsewhere); the ordering between policies is what the scenario is for.
"""
import sys, time, argparse
import numpy as np
import mlx.core as mx
from qwen35.engine import Engine
from qwen35.serve import Server, Request, describe_plan

ap = argparse.ArgumentParser()
ap.add_argument("trace", nargs="?", default="mixed", choices=["mixed", "burst", "churn"])
ap.add_argument("--slots", default="8", help="number of slots, or `auto`: as many as --mem-budget allows (at most --max-slots)")
ap.add_argument("--mem-budget", type=float, default=None, help="GiB for everything on the GPU side incl. weights (default: half the RAM)")
ap.add_argument("--max-slots", type=int, default=16)
ap.add_argument("--max-queue", type=int, default=None, help="reject arrivals while this many requests wait")
ap.add_argument("--dur", type=float, default=30.0)
ap.add_argument("--rate", type=float, default=2.5, help="mean arrivals/s of the Poisson part (mixed)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunks", default="256,512,2048")
ap.add_argument("--policies", default="sequential,static,continuous")
ap.add_argument("--verify", action="store_true")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--spec", type=int, default=0, metavar="N", help="continuous/static: enable the speculative mode for loads of at most N sequences (0 = off)")
a = ap.parse_args()
SLOTS = a.slots if a.slots == 'auto' else int(a.slots)
BUDGET = None if a.mem_budget is None else int(a.mem_budget * (1 << 30))


# ------------------------------------------------------------------------------------------------ traces
def build_trace(kind, seed, dur, rate):
    r = np.random.default_rng(seed)
    tok = lambda n: [int(t) for t in r.integers(1000, 30000, n)]
    docs = [tok(1200) for _ in range(3)]
    system = tok(700)
    reqs = []

    def add(tag, prompt, gen, t):
        reqs.append(Request(rid=len(reqs), prompt=prompt, max_new=int(gen), arrival=float(t), tag=tag))

    def make(cls, t):
        if cls == "chat":
            add("chat", tok(int(r.integers(20, 121))), r.integers(40, 161), t)
        elif cls == "rag":
            if r.random() < 0.5:
                add("rag", docs[int(r.integers(0, 3))] + tok(int(r.integers(20, 101))), r.integers(30, 101), t)
            else:
                add("rag", tok(int(r.integers(1200, 3001))), r.integers(30, 101), t)
        elif cls == "agent":
            add("agent", system + tok(int(r.integers(40, 301))), r.integers(20, 61), t)
        else:
            add("long", tok(int(r.integers(50, 201))), r.integers(300, 501), t)

    if kind == "mixed":
        deck = []                                          # class mix drawn from shuffled decks: 9 chat, 3 rag, 6 agent, 2 long
        def draw():
            if not deck:
                deck.extend(r.permutation(["chat"] * 9 + ["rag"] * 3 + ["agent"] * 6 + ["long"] * 2).tolist())
            return deck.pop()
        t = 0.0
        while True:
            t += r.exponential(1 / rate)
            if t > dur:
                break
            make(draw(), t)
        tb = dur * 0.4                                     # a burst of 6 requests inside 150 ms
        for i in range(6):
            make(draw(), tb + 0.03 * i)
    elif kind == "burst":
        for i in range(16):
            n = int(np.exp(r.uniform(np.log(16), np.log(3000))))
            add("burst", tok(n), r.integers(32, 257), 0.0)
    else:
        add("long", tok(100), 600, 0.0)
        add("long", tok(150), 600, 1.0)
        for t in np.sort(r.uniform(0.2, dur, 60)):
            add("short", tok(int(r.integers(20, 200))), r.integers(8, 25), t)
    reqs.sort(key=lambda q: q.arrival)
    for i, q in enumerate(reqs):
        q.rid = i
    return reqs


def fresh(reqs, max_new=None, arrival=None):
    out = []
    for q in reqs:
        c = Request(rid=q.rid, prompt=q.prompt, max_new=q.max_new if max_new is None else min(max_new, q.max_new),
                    arrival=q.arrival if arrival is None else arrival, eos=q.eos, tag=q.tag)
        out.append(c)
    return out


# ------------------------------------------------------------------------------------------------ the sequential baseline
def run_sequential(eng, reqs, chunk=2048):
    """FCFS, one request at a time, solo decode pipelined like Engine.generate (step i+1 is launched before token i is read)."""
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    for q in sorted(reqs, key=lambda q: q.arrival):
        dt = q.arrival - now()
        if dt > 0:
            time.sleep(dt)
        q.t_admit = now()
        q.max_new = n = max(1, min(q.max_new, eng.max_len - len(q.prompt) + 1))
        pc = eng.pcache
        h0 = pc.stats["hit_tokens"] if pc is not None else 0
        tok, st, pos = eng.prefill(q.prompt, chunk=chunk)
        mx.eval(tok)
        if pc is not None:
            q.hit_tokens = pc.stats["hit_tokens"] - h0
        q.t_first = now()
        T = [tok]
        for i in range(n - 1):
            nxt, st = eng.step(T[i], mx.array(pos, dtype=mx.int32), st, pos + 1)
            mx.async_eval(nxt)
            pos += 1
            T.append(nxt)
            mx.eval(T[i])
            q.t_tokens.append(now())
        mx.eval(T[-1])
        q.t_tokens.append(now())
        q.t_tokens[0] = q.t_first                      # token 0 was observed at the end of the prefill
        q.out = [int(t.item()) for t in T]
        q.t_done = q.t_tokens[-1]
    return dict(wall=now(), steps=0, mean_batch=1.0)


# ------------------------------------------------------------------------------------------------ metrics
def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def report(name, all_reqs, stats, extra=""):
    print(f"\n== {name} {extra}")
    reqs = [q for q in all_reqs if q.state == "DONE" or q.t_done is not None]
    rej = [q for q in all_reqs if q.state == "REJECTED"]
    reasons = {}
    for q in reqs:
        reasons[q.finish_reason or "-"] = reasons.get(q.finish_reason or "-", 0) + 1
    if rej or len(reasons) > 1 or "length" in reasons:
        print(f"   finished {len(reqs)} {reasons}, rejected {len(rej)} {sorted({q.finish_reason for q in rej})}")
    toks = sum(len(q.out) for q in reqs)
    wall = max(q.t_done for q in reqs)
    print(f"   {len(reqs)} requests, {toks} generated tokens, makespan {wall:.1f} s -> {toks / wall:6.1f} tok/s   "
          f"decode batch mean {stats['mean_batch']:.2f}")
    if stats.get("rounds"):
        print(f"   speculative: {stats['rounds']} rounds ({stats['tokens_per_round']:.2f} tokens / sequence / round, mean batch {stats['mean_spec_batch']:.2f}), "
              f"{stats['steps']} plain steps, {stats['mode_switches']} mode switches (ITL below counts the 2-3 tokens of a round as arriving together)")
    hdr = f"   {'class':8s} {'n':>3s} | {'TTFT p50':>9s} {'p95':>8s} | {'ITL p50':>8s} {'p95':>8s} {'p99':>8s} | {'e2e p50':>8s} {'p95':>8s} | {'queue p50':>9s} {'p95':>8s} | cache"
    print(hdr)
    for tag in sorted({q.tag for q in reqs}) + ["ALL"]:
        rs = [q for q in reqs if tag == "ALL" or q.tag == tag]
        ttft = [(q.t_first - q.arrival) * 1e3 for q in rs]
        itl = [d * 1e3 for q in rs for d in np.diff(q.t_tokens)] if rs else []
        e2e = [(q.t_done - q.arrival) * 1e3 for q in rs]
        qw = [(q.t_admit - q.arrival) * 1e3 for q in rs]
        hit = sum(q.hit_tokens for q in rs) / max(1, sum(len(q.prompt) for q in rs))
        print(f"   {tag:8s} {len(rs):3d} | {pct(ttft, 50):9.0f} {pct(ttft, 95):8.0f} | {pct(itl, 50):8.1f} {pct(itl, 95):8.1f} {pct(itl, 99):8.1f} |"
              f" {pct(e2e, 50):8.0f} {pct(e2e, 95):8.0f} | {pct(qw, 50):9.0f} {pct(qw, 95):8.0f} | {hit * 100:4.0f}%")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ main
if a.verify:
    a.no_cache = True            # a cache hit resumes the prefill from another chunk boundary (bf16-level rounding), so tokens would
                                 # depend on arrival order; the cache path has its own test (tests/batch.py)
sp = None
if a.spec:
    from qwen35.spec import SpecEngine
    sp = SpecEngine(k=2, max_len=a.max_len)
eng = Engine(max_len=a.max_len, prefix_cache_mb=0 if a.no_cache else 2048, like=sp)
mx.set_cache_limit(1 << 30)      # same cap for every policy: an unbounded MLX buffer cache grows to ~8.5 GB under varied prefill shapes
                                 # and stalls prompts' last chunks 0.2-1.5 s (Server does this by default too)
trace = build_trace(a.trace, a.seed, a.dur, a.rate)
cls_n = {t: sum(q.tag == t for q in trace) for t in sorted({q.tag for q in trace})}
print(f"trace {a.trace}: {len(trace)} requests {cls_n}, prompt tokens {sum(len(q.prompt) for q in trace)}, "
      f"gen tokens {sum(q.max_new for q in trace)}, last arrival {trace[-1].arrival:.1f} s, slots {a.slots}, "
      f"prefix cache {'off' if eng.pcache is None else 'on'}")


def reset():
    if eng.pcache is not None:
        eng.pcache.clear()
    mx.clear_cache()


def solo_gen(prompt, n, ck):
    """Engine.generate with a chosen prefill chunk."""
    tk, st, pos = eng.prefill(prompt, chunk=ck)
    T = [tk]
    for i in range(n - 1):
        nx, st = eng.step(T[i], mx.array(pos, dtype=mx.int32), st, pos + 1)
        mx.async_eval(nx)
        pos += 1
        T.append(nx)
    return [int(t.item()) for t in T]


oracle = {}
if a.verify:
    for ck in sorted({int(c) for c in a.chunks.split(",")} | {2048}):
        reset()
        outs = {}
        for q in trace:
            n = max(1, min(q.max_new, a.max_len - len(q.prompt) + 1))
            outs[q.rid] = solo_gen(q.prompt, n, ck)
        oracle[ck] = outs
        print(f"  oracle (solo, prefill chunk {ck}) computed")


def check(name, reqs, ck):
    if not a.verify:
        return
    bad = [q.rid for q in reqs if q.out != oracle[ck][q.rid]]
    print(f"   verify vs solo (chunk {ck}): {'ALL ' + str(len(reqs)) + ' sequences token-exact' if not bad else 'MISMATCH ' + str(bad)}")


policies = a.policies.split(",")
if "sequential" in policies:
    reset()
    run_sequential(eng, fresh(trace, max_new=2, arrival=0.0))        # warm every prefill shape
    reset()
    rq = fresh(trace)
    st = run_sequential(eng, rq)
    report("sequential (solo Engine, FCFS)", rq, st)
    check("sequential", rq, 2048)
for pol in ("static", "continuous"):
    if pol not in policies:
        continue
    for ck in ([int(c) for c in a.chunks.split(",")] if pol == "continuous" else [512]):
        srv = Server(eng, slots=SLOTS, max_len=a.max_len, chunk=ck, policy=pol, mem_budget=BUDGET, max_slots=a.max_slots,
                     max_queue=a.max_queue, **(dict(spec=sp, spec_max=a.spec) if sp is not None else {}))
        if srv.plan is not None:
            print("\nmemory plan:", describe_plan(srv.plan))
        srv.warmup()
        reset()
        srv.run(fresh(trace, max_new=2, arrival=0.0))             # warm prefill shapes + every batch bucket
        reset()
        rq = fresh(trace)
        st = srv.run(rq)
        report(f"{pol} batching, {srv.S} slots, prefill chunk {ck}" + (f", speculative up to {srv.nspec} sequences" if sp is not None else ""), rq, st,
               f"[GPU memory peak {mx.get_peak_memory() / (1 << 20):.0f} MiB"
               + (f" of planned {srv.plan['total'] / (1 << 20):.0f} / budget {srv.plan['budget'] / (1 << 20):.0f}" if srv.plan else "") + "]")
        check(pol, rq, ck)
        del srv
        mx.clear_cache()
