"""Tune the residual-fused GEMMs (EPI_ADD: out_proj, down) -> results/gemm_cfg.json entries "<name>+add" (config or "mlx" =
mx.matmul followed by the separate add).  Candidates are timed on a quiet GPU (min over quiet samples), the best few are then
re-timed interleaved against the mx.matmul baseline.
usage: gemm_tune_add.py [M list] [names]"""
import sys, time, json, statistics
from pathlib import Path
import mlx.core as mx
from quiet import Probe
from qwen35 import gemm_nax as G

OUT = Path(__file__).resolve().parent.parent / "results" / "gemm_cfg.json"
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] != "all" else [128, 256, 512, 1024, 2048, 4096]
SHAPES = {"out_proj": (2048, 1024), "down": (3584, 1024)}
sel = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(SHAPES)
CFGS = [(32, 256, 256, 1, 4), (32, 128, 256, 1, 4), (32, 128, 256, 1, 2), (64, 128, 256, 2, 4), (64, 128, 512, 2, 4),
        (64, 256, 256, 2, 4), (64, 256, 512, 2, 4), (64, 256, 256, 1, 4), (64, 128, 256, 2, 2), (64, 128, 256, 1, 4),
        (128, 128, 256, 4, 4), (128, 128, 512, 4, 4), (128, 128, 256, 2, 4), (128, 128, 256, 4, 2),
        (128, 256, 256, 4, 4), (128, 256, 512, 4, 4), (128, 256, 256, 2, 4), (256, 128, 256, 4, 4), (64, 128, 128, 2, 4),
        (64, 64, 256, 2, 2), (32, 64, 256, 1, 2), (32, 64, 256, 2, 2)]
R = 6
probe = Probe(tol=1.18)


def timed(f):
    t = time.perf_counter(); mx.eval(f()); return (time.perf_counter() - t) / R


def qmin(f, n):
    """min of n quiet samples (falls back to the best unquiet one)"""
    best, got, loose = None, 0, None
    for _ in range(n * 6):
        dt, ok = probe.sample(lambda: timed(f), tries=10)
        loose = dt if loose is None else min(loose, dt)
        if ok:
            best = dt if best is None else min(best, dt); got += 1
            if got >= n: break
    return best if best is not None else loose * 1.3


table = json.loads(OUT.read_text()) if OUT.exists() else {}
for name in sel:
    K, N = SHAPES[name]
    ws = [(mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16) for _ in range(R)]
    mx.eval(ws)
    for M in MS:
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        r = (mx.random.normal((M, N)) * 0.5).astype(mx.bfloat16)
        mx.eval(x, r)
        ref = lambda: [r + x @ w.T for w in ws]
        mx.eval(ref())
        res = []
        for cfg in CFGS:
            tm = -(-M // cfg[0])
            for swz in (0, 1, 2):
                if swz > 0 and tm <= (1 << swz): continue
                f = lambda: [G.matmul_nt(x, w, epi=G.EPI_ADD, r=r, cfg=cfg, swizzle_log=swz) for w in ws]
                try: mx.eval(f())
                except Exception as e:
                    print(f"  {name} M={M} {cfg}: FAILED {str(e).splitlines()[0][:80]}", flush=True); break
                res.append((qmin(f, 2), cfg, swz, f))
        res.sort(key=lambda t: t[0])
        top = res[:4]
        # stage 2: interleaved, medians of 7 quiet samples against the mx.matmul + add baseline
        cand = {"mlx+add": ref}
        for t, cfg, swz, f in top: cand[f"{cfg}/s{swz}"] = f
        ts = {k: [] for k in cand}
        for _ in range(7):
            for k, f in cand.items():
                dt, ok = probe.sample(lambda: timed(f))
                if ok: ts[k].append(dt)
        med = {k: statistics.median(v) for k, v in ts.items() if v}
        best_k = min(med, key=med.get)
        flops = 2.0 * M * N * K
        print(f"{name:9s} M={M:5d}: mlx+add {med.get('mlx+add', float('nan')) * 1e3:.3f} ms | " +
              ", ".join(f"{k} {v * 1e3:.3f}" for k, v in sorted(med.items(), key=lambda kv: kv[1]) if k != "mlx+add") +
              f"  -> {best_k}", flush=True)
        if best_k == "mlx+add":
            table.setdefault(name + "+add", {})[str(M)] = "mlx"
        else:
            for t, cfg, swz, f in top:
                if f"{cfg}/s{swz}" == best_k:
                    table.setdefault(name + "+add", {})[str(M)] = list(cfg) + [swz]
        OUT.write_text(json.dumps(table, indent=1))
