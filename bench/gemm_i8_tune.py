"""Autotune the int8 (W8A8) NAX GEMM tile configuration per (layer GEMM, M).  Writes results/gemm_i8_cfg.json:
   {"<name>": {"<M>": [BM, BN, BK, WM, WN, swizzle_log], ...}, ...}
usage: gemm_i8_tune.py [M list comma separated] [shape names comma separated]"""
import sys, time, json
from pathlib import Path
import mlx.core as mx
from qwen35 import gemm_i8 as I
from quiet import Probe

OUT = Path(__file__).resolve().parent.parent / "results" / "gemm_i8_cfg.json"
MS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] != "all" else [128, 256, 512, 1024, 2048, 4096]
SHAPES = {"in_proj": (1024, 8224, I.EPI_PLAIN), "qkv": (1024, 5120, I.EPI_PLAIN), "gate_up": (1024, 7168, I.EPI_SWIGLU)}
sel = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(SHAPES)
CFGS = [(32, 256, 256, 1, 4), (32, 128, 256, 1, 4), (64, 128, 256, 2, 4), (64, 128, 512, 2, 4), (64, 256, 256, 2, 4),
        (64, 256, 512, 2, 4), (64, 128, 256, 2, 2), (64, 128, 256, 1, 4), (128, 128, 256, 4, 4), (128, 128, 512, 4, 4),
        (128, 128, 256, 2, 4), (128, 256, 256, 4, 4), (128, 256, 512, 4, 4), (256, 128, 256, 4, 4), (64, 128, 128, 2, 4),
        (64, 64, 256, 2, 2), (128, 128, 128, 4, 4)]
R = 6
probe = Probe(tol=1.15)

table = json.loads(OUT.read_text()) if OUT.exists() else {}
for name in sel:
    K, N, epi = SHAPES[name]
    wq, sw = I.quant_rows((mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16))
    mx.eval(wq, sw)
    for M in MS:
        qs = [I.quant_rows((mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)) for _ in range(R)]
        mx.eval(qs)
        res = []
        for cfg in CFGS:
            tm = -(-M // cfg[0])
            for swz in (0, 1, 2):
                if swz > 0 and tm <= (1 << swz):
                    continue
                def f():
                    t = time.perf_counter()
                    mx.eval([I.matmul_i8(q, s, wq, sw, epi=epi, cfg=cfg, swizzle_log=swz) for q, s in qs])
                    return (time.perf_counter() - t) / R
                try:
                    f(); t = sorted(probe.sample(f)[0] for _ in range(5))[2]
                except Exception as e:
                    print(f"  {name} M={M} {cfg} swz{swz}: FAILED {str(e).splitlines()[0][:80]}", flush=True); break
                res.append((t, cfg, swz))
        res.sort(key=lambda r: r[0])
        best_t, best_cfg, best_swz = res[0]
        print(f"{name:8s} M={M:5d}: best {best_t * 1e3:.3f} ms ({2.0 * M * N * K / best_t / 1e12:.1f} TOPS) cfg {best_cfg} swz {best_swz}"
              f" | next: " + ", ".join(f"{c}/s{s} {t * 1e3:.3f}" for t, c, s in res[1:4]), flush=True)
        table.setdefault(name, {})[str(M)] = [*best_cfg, best_swz]
        OUT.write_text(json.dumps(table, indent=1))
