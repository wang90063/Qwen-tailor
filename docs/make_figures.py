"""Regenerate the figures in docs/figures/ used by README.md.

usage: python docs/make_figures.py          (needs matplotlib, which is NOT a dependency of the package)

The numbers are copied from docs/PLAN.md (the measured results, one list per figure below); nothing is measured here.
The only derived quantities are the roofline walls, whose inputs are in `WALL` and `BATCH_FLOOR`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

for name in ["PingFang SC", "Hiragino Sans GB", "STHeiti", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
else:
    print("warning: no CJK font found, Chinese labels will render as boxes")
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100, "savefig.dpi": 150})

BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"

# ------------------------------------------------------------------------------------------- roofline inputs
BW = 124.5                       # GB/s, best measured streaming read (bench/bandwidth.py); nominal 153, median 106
WALL = {                         # bytes read per token (GB) -> wall tok/s = BW / GB
    "bf16 全部权重": 1.504,        # 497.6M trunk + 254.3M head params x 2 B
    "+ int8 头预筛": 0.995 + 0.254,
    "+ 12 bit 主干": 0.749 + 0.254,
}
WALL_TOK = {k: BW / v for k, v in WALL.items()}   # 82.8 / 99.7 / 124.1

# batched decode: weights + per-sequence recurrent state (18 layers x 1 MiB fp32 ssm, read + write) + KV read at pos ~ 900
W_BYTES = 1.003e9
STATE_BYTES = 18 * 2 * 2**20
KV_BYTES = 6 * 2 * 2 * 256 * 2 * 900            # 6 full-attn layers x (K,V) x 2 kv heads x 256 x bf16 x 900 positions
def batch_floor(B):                             # aggregate tok/s at the memory floor
    return B * BW * 1e9 / (W_BYTES + B * (STATE_BYTES + KV_BYTES))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------- 1. decode by milestone
def decode_progress():
    stages = ["mlx-lm\n基线", "M1\n自研框架", "M2\n融合GEMV\n+DeltaNet", "M5 前", "M5\nint8 头预筛", "M6 前", "M6\n12 bit 权重"]
    plain = [71.4, 71.8, 78.2, 84.4, 96.0, 98.7, 109.7]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = range(len(stages))
    a.plot(x, plain, "o-", color=BLUE, lw=2, label="decode 实测（无推测，贪心，逐 token 一致）")
    for i, v in zip(x, plain):
        a.annotate(f"{v:g}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
    cols = [RED, ORANGE, GREEN]
    for (k, v), c in zip(WALL_TOK.items(), cols):
        a.axhline(v, ls="--", color=c, lw=1.2)
        a.text(-.3, v + .8, f"{k}：墙 {v:.1f} tok/s", color=c, ha="left", fontsize=9)
    a.set_xticks(list(x)); a.set_xticklabels(stages, fontsize=8)
    a.set_xlim(-.5, len(stages) - .5); a.set_ylim(60, 132); a.set_ylabel("tok/s"); a.set_title("无推测 decode：实测 vs 内存带宽墙（124.5 GB/s ÷ 每 token 读的字节）")

    m = ["M4", "M5", "M6"]
    k2, k3 = [159, 189.7, 224.4], [172, 201.5, 240.5]
    b.plot(m, k2, "o-", color=BLUE, lw=2, label="k=2")
    b.plot(m, k3, "s-", color=ORANGE, lw=2, label="k=3")
    for ys, c, dy in [(k2, BLUE, -16), (k3, ORANGE, 8)]:
        for xx, yy in zip(m, ys):
            b.annotate(f"{yy:g}", (xx, yy), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=9, color=c)
    b.axhline(71.4, ls=":", color=GREY); b.text(2.05, 74, "mlx-lm 71.4", color=GREY, ha="right", fontsize=9)
    b.set_ylim(60, 265); b.set_ylabel("tok/s"); b.legend(loc="upper left")
    b.set_title("MTP 自推测 decode（6 条 prompt 平均）")
    save(fig, "decode_progress.png")


# ------------------------------------------------------------------------------------------- 2. prefill
def prefill():
    fig, a = plt.subplots(figsize=(7.2, 4.6))
    a.plot([512, 2048], [5465, 6822], "o-", color=GREY, lw=2, label="mlx-lm 基线")
    a.plot([128, 512, 2048], [128 / .0192, 512 / .0463, 11550], "o-", color=BLUE, lw=2, label="本项目（默认，无损）")
    a.plot([128, 512, 2048], [8200, 13200, 14600], "s--", color=ORANGE, lw=2, label="本项目 + W8A8（有损，opt-in）")
    a.axhspan(13000, 15000, color=RED, alpha=.12)
    a.text(130, 15150, "bf16 GEMM 算力墙 13–15k（≈15 TFLOPS ÷ 1.0 GFLOP/token）", color=RED, fontsize=9)
    a.vlines(2048 * 1.04, 2048 / .172, 2048 / .165, color="k", lw=5, alpha=.45)
    a.text(2048 * 1.07, 12150, "无损地板\n165–172 ms", fontsize=8, va="center")
    for xs, ys, c in [([512, 2048], [5465, 6822], GREY), ([128, 512, 2048], [128 / .0192, 512 / .0463, 11550], BLUE),
                      ([128, 512, 2048], [8200, 13200, 14600], ORANGE)]:
        for xx, yy in zip(xs, ys):
            a.annotate(f"{yy / 1000:.1f}k", (xx, yy), textcoords="offset points", xytext=(0, -14 if c != ORANGE else 8), ha="center", fontsize=8, color=c)
    a.set_xscale("log", base=2); a.set_xticks([128, 512, 2048]); a.set_xticklabels(["128", "512", "2048"])
    a.set_xlim(100, 3100); a.set_ylim(3000, 16500)
    a.set_xlabel("prompt 长度（token）"); a.set_ylabel("prefill 吞吐（tok/s，含首 token）")
    a.set_title("prefill 吞吐 vs prompt 长度"); a.legend(loc="lower right")
    save(fig, "prefill.png")


# ------------------------------------------------------------------------------------------- 3. batching
def batching():
    Bs = [1, 2, 4, 6, 8, 12, 16]
    meas = [115, 221, 400, 510, 617, 744, 804]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(Bs, meas, "o-", color=BLUE, lw=2, label="连续批处理 decode 实测")
    xs = list(range(1, 17))
    a.plot(xs, [batch_floor(B) for B in xs], "--", color=RED, lw=1.5, label="内存地板（权重 + 状态 + KV 的字节 ÷ 124.5 GB/s）")
    a.plot(xs, [109.7] * len(xs), ":", color=GREY, label="单序列 decode 109.7（串行服务 B 个请求的上限）")
    for B, v in zip(Bs, meas):
        a.annotate(f"{v}（{v / batch_floor(B) * 100:.0f}%）", (B, v), textcoords="offset points", xytext=(14, -16) if B < 4 else (0, -24),
                   ha="left" if B < 4 else "center", fontsize=8, color=BLUE, bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
    a.set_xlabel("并发序列数 B"); a.set_ylabel("聚合吞吐 tok/s"); a.set_ylim(0, 1250)
    a.set_title("连续批处理：聚合 decode 吞吐 vs 批大小"); a.legend(fontsize=8, loc="upper left")

    Bs2 = [1, 2, 3, 4, 6, 8]
    spec = [285.7, 458.7, 609.5, 721.9, 760.8, 785.0]
    plain = [116.9, 226.1, 326.1, 414.6, 527.8, 647.2]
    b.plot(Bs2, plain, "o-", color=GREY, lw=2, label="plain 批 decode")
    b.plot(Bs2, spec, "o-", color=GREEN, lw=2, label="批量 MTP 推测解码（k=2）")
    for B, s, p in zip(Bs2, spec, plain):
        b.annotate(f"{s / p:.2f}×", (B, s), textcoords="offset points", xytext=(-14, 6), ha="center", fontsize=9, color=GREEN,
                   bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
    b.set_xlabel("并发序列数 B"); b.set_ylabel("聚合吞吐 tok/s"); b.set_ylim(0, 900); b.set_xlim(.4, 8.6)
    b.set_title("批量推测解码 vs plain（代码文本，服务器级）"); b.legend(loc="lower right")
    save(fig, "batching.png")


# ------------------------------------------------------------------------------------------- 4. prefix cache
def prefix_cache():
    suf = ["1", "16", "64", "128", "512"]
    hit = [14.1, 16.0, 16.8, 22.9, 54.6]
    miss = [185, 188, 188, 196, 229]
    speedup = [13.2, 11.7, 11.2, 8.6, 4.2]             # as recorded in PLAN.md (computed from unrounded times)
    fig, a = plt.subplots(figsize=(7.2, 4.4))
    w = .38
    xs = range(len(suf))
    a.bar([x - w / 2 for x in xs], miss, w, color=GREY, label="无缓存")
    a.bar([x + w / 2 for x in xs], hit, w, color=BLUE, label="命中 2048 token 前缀")
    for x, h, m, sp in zip(xs, hit, miss, speedup):
        a.text(x + w / 2, h + 3, f"{h:.1f} ms\n{sp}×", ha="center", fontsize=8, color=BLUE)
        a.text(x - w / 2, m + 3, f"{m}", ha="center", fontsize=8, color=GREY)
    a.set_xticks(list(xs)); a.set_xticklabels(suf)
    a.set_xlabel("前缀之后新增的 token 数"); a.set_ylabel("TTFT（ms）"); a.set_ylim(0, 270)
    a.set_title("前缀缓存：TTFT（2048 token 前缀）"); a.legend(loc="upper left")
    save(fig, "prefix_cache.png")


# ------------------------------------------------------------------------------------------- 5. roofline summary
def roofline_summary():
    rows = [  # (label, fraction of wall, (lo, hi) or None, colour)
        ("decode 无推测 109.7 / 124（墙：12 bit 主干 + int8 头）", 109.7 / WALL_TOK["+ 12 bit 主干"], None, BLUE),
        ("mlx-lm 基线 71.4 / 83（墙：bf16 全部权重）", 71.4 / WALL_TOK["bf16 全部权重"], None, GREY),
        ("prefill@2048 11.55k（无损地板 165–172 ms）", 0.5 * (165 / 177 + 172 / 177), (165 / 177, 172 / 177), BLUE),
        ("prefill@2048 11.55k / bf16 GEMM 墙 13–15k", 11550 / 14000, (11550 / 15000, 11550 / 13000), BLUE),
        ("批 decode B=1（内存地板）", 115 / batch_floor(1), None, GREEN),
        ("批 decode B=4", 400 / batch_floor(4), None, GREEN),
        ("批 decode B=8", 617 / batch_floor(8), None, GREEN),
        ("批 decode B=16（ALU / launch 间隙受限）", 804 / batch_floor(16), None, GREEN),
    ]
    fig, a = plt.subplots(figsize=(9.6, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, f, rng, c) in zip(ys, rows):
        a.barh(y, f * 100, color=c, alpha=.85)
        a.text(max(f * 100, 100) + 1.5, y, f"{f * 100:.0f}%" + (f"（{rng[0] * 100:.0f}–{rng[1] * 100:.0f}%）" if rng else ""), va="center", fontsize=9)
    a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows], fontsize=9)
    a.axvline(100, color="k", lw=1)
    a.set_xlim(0, 118); a.set_xlabel("达到的屋顶百分比（%）"); a.grid(axis="y", alpha=0)
    a.set_title("与 roofline 的距离")
    save(fig, "roofline_summary.png")


if __name__ == "__main__":
    print("walls (tok/s):", {k: round(v, 1) for k, v in WALL_TOK.items()})
    print("batch floor:", {B: round(batch_floor(B), 1) for B in (1, 2, 4, 6, 8, 12, 16)})
    decode_progress(); prefill(); batching(); prefix_cache(); roofline_summary()
