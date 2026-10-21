"""Autotune the NAX GEMM tile configuration per (layer GEMM, M).  Writes results/gemm_cfg.json:
   {"<name>": {"<M>": [BM, BN, BK, WM, WN, swizzle_log], ...}, ...}
usage: gemm_tune.py [M list comma separated] [shape names comma separated]"""
import sys, time, json
from pathlib import Path
import mlx.core as mx
from qwen35 import gemm_nax as G

OUT = Path(__file__).resolve().parent.parent / "results" / "gemm_cfg.json"
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] != "all" else [128, 256, 512, 1024, 2048, 4096]
SHAPES = {                      # name: (K, N, epilogue)
    "in_proj": (1024, 8224, G.EPI_PLAIN),
    "qkv": (1024, 5120, G.EPI_PLAIN),
    "out_proj": (2048, 1024, G.EPI_PLAIN),
    "gate_up": (1024, 7168, G.EPI_SWIGLU),
    "down": (3584, 1024, G.EPI_PLAIN),
}
sel = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(SHAPES)
CFGS = [(32, 256, 256, 1, 4), (32, 128, 256, 1, 4), (32, 128, 256, 1, 2), (64, 128, 256, 2, 4), (64, 128, 512, 2, 4),
        (64, 256, 256, 2, 4), (64, 256, 512, 2, 4), (64, 256, 256, 1, 4), (64, 128, 256, 2, 2), (64, 128, 256, 1, 4),
        (128, 128, 256, 4, 4), (128, 128, 512, 4, 4), (128, 128, 256, 2, 4), (128, 128, 256, 4, 2),
        (128, 256, 256, 4, 4), (128, 256, 512, 4, 4), (128, 256, 256, 2, 4), (256, 128, 256, 4, 4), (64, 128, 128, 2, 4),
        (64, 64, 256, 2, 2), (32, 64, 256, 1, 2), (16 * 2, 64, 256, 2, 2)]
R = 6


def timeit(f, n=5):
    mx.eval(f()); ts = []
    for _ in range(n):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


table = json.loads(OUT.read_text()) if OUT.exists() else {}
for name in sel:
    K, N, epi = SHAPES[name]
    ws = [(mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16) for _ in range(R)]
    mx.eval(*ws)
    for M in MS:
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16); mx.eval(x)
        t_ref = timeit(lambda: [x @ w.T for w in ws]) / R if epi == G.EPI_PLAIN else None
        res = []
        for cfg in CFGS:
            bm, bn = cfg[0], cfg[1]
            tm = -(-M // bm)
            for swz in (0, 1, 2):
                if swz > 0 and tm <= (1 << swz):
                    continue
                try:
                    t = timeit(lambda: [G.matmul_nt(x, w, epi=epi, cfg=cfg, swizzle_log=swz) for w in ws]) / R
                except Exception as e:
                    t = None; print(f"  {name} M={M} {cfg} swz{swz}: FAILED {str(e).splitlines()[0][:80]}", flush=True); break
                res.append((t, cfg, swz))
        res.sort(key=lambda r: r[0])
        best_t, best_cfg, best_swz = res[0]
        flops = 2.0 * M * N * K
        ref_s = f"mlx {t_ref * 1e3:.3f} ms -> " if t_ref else ""
        print(f"{name:9s} M={M:5d}: {ref_s}best {best_t * 1e3:.3f} ms ({flops / best_t / 1e12:.1f} TFLOPS) cfg {best_cfg} swz {best_swz}"
              f" | next: " + ", ".join(f"{c}/s{s} {t * 1e3:.3f}" for t, c, s in res[1:4]), flush=True)
        table.setdefault(name, {})[str(M)] = list(best_cfg) + [best_swz]
        OUT.write_text(json.dumps(table, indent=1))
