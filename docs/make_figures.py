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


# ------------------------------------------------------------------------------------------- 0. software stack (top of README)
# Pure drawing, no numbers.  Chip = (title, files / note); the colour says who wrote it.
DONE, REUSE, TODO = "done", "reuse", "todo"
STYLE = {                       # face, edge, title colour, sub colour, linestyle
    DONE: ("#c6f6d5", "#2f855a", "#1c4532", "#2f6f4f", "-"),
    REUSE: ("#bee3f8", "#2b6cb0", "#1a365d", "#2c5282", "-"),
    TODO: ("#f7fafc", "#a0aec0", "#5f6b7a", "#8591a3", (0, (4, 3))),
}


def architecture():
    from matplotlib.patches import FancyBboxPatch, Rectangle
    W, H = 12.0, 9.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.grid(False)
    X0, X1, GAP, CH, RG = 0.25, W - 0.25, 0.10, 0.70, 0.09       # margins / chip gap / chip height / row gap
    BAND = 0.55                                                   # width of the vertical layer label

    def box(x, y, w, h, fc, ec, ls="-", lw=1.2, r=0.06, z=1):
        ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r + .02}",
                                    fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))

    def chip(x, y, w, kind, title, sub):
        fc, ec, tc, sc, ls = STYLE[kind]
        box(x, y, w, CH, fc, ec, ls, 1.2 if kind != TODO else 1.0, z=3)
        head, _, desc = title.partition("\n")
        ax.text(x + 0.09, y + CH - 0.075, head, ha="left", va="top", fontsize=9.6, fontweight="bold", color=tc, zorder=4)
        if desc:
            ax.text(x + 0.09, y + CH - 0.275, desc, ha="left", va="top", fontsize=8.4, color=tc, zorder=4, linespacing=1.15)
        ax.text(x + 0.09, y + 0.07, sub, ha="left", va="bottom", fontsize=7.6, color=sc, zorder=4, linespacing=1.15)

    def layer(top, name, en, rows, ncols=4, fc="#ffffff", ec="#4a5568", band="#2d3748"):
        h = len(rows) * CH + (len(rows) - 1) * RG + 0.2
        y0 = top - h
        box(X0, y0, X1 - X0, h, fc, ec, lw=1.4, z=1)
        ax.add_patch(Rectangle((X0 + 0.03, y0 + 0.03), BAND, h - 0.06, fc=band, ec="none", zorder=2))
        ax.text(X0 + 0.03 + BAND / 2, y0 + h / 2, name, rotation=90, ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=3)
        ax.text(X0 + 0.03 + BAND / 2 + 0.17, y0 + h / 2, en, rotation=90, ha="center", va="center", fontsize=6.8, color="#cbd5e0", zorder=3)
        cx0 = X0 + BAND + 0.18
        cw = (X1 - 0.12 - cx0 - (ncols - 1) * GAP) / ncols
        for r, row in enumerate(rows):
            y = top - 0.1 - (r + 1) * CH - r * RG
            for c, item in enumerate(row):
                chip(cx0 + c * (cw + GAP), y, cw, *item)
        return y0

    ax.text(W / 2, H - 0.12, "Qwen-tailor 的软件栈：哪些是我们写的，哪些复用现成的，哪些没做", ha="center", va="top", fontsize=14.5, fontweight="bold")
    # legend
    lx, ly = W / 2 - 3.55, H - 0.62
    for kind, label, w in ((DONE, "本仓库实现（文件名见小字）", 2.35), (REUSE, "复用 / 内嵌的现成组件", 2.2), (TODO, "没有实现", 1.35)):
        fc, ec, _, _, ls = STYLE[kind]
        box(lx, ly - 0.17, 0.36, 0.22, fc, ec, ls, 1.1, r=0.03)
        ax.text(lx + 0.46, ly - 0.06, label, ha="left", va="center", fontsize=9.5)
        lx += w + 0.35

    top = H - 0.95
    fw = layer(top, "框架", "FRAMEWORK", [
        [(DONE, "函数式模型 + 静态缓存\n整步 mx.compile 的 decode", "model.py · engine.py · fast.py"),
         (DONE, "prefill\n分块 · 末层只算末尾行", "prefill.py"),
         (DONE, "可证明无损的\nint8 lm_head 预筛", "head_prune.py"),
         (DONE, "无损 12 bit/权重压缩\n（权重字节 −25%）", "wcomp.py")],
        [(DONE, "MTP 自推测解码\n草稿 → 批量校验 → 状态回滚", "spec.py · mtp.py"),
         (DONE, "前缀缓存\nDeltaNet 状态的检查点树", "prefix_cache.py"),
         (DONE, "连续批处理服务器\n+ 批量推测解码", "batch.py · batch_spec.py · serve.py"),
         (DONE, "命令行\n聊天模板 · 多轮 REPL", "run.py")],
        [(TODO, "采样", "temperature / top-p / top-k（只做了贪心）"),
         (TODO, "HTTP / OpenAI 兼容接口", "调度器内核有，网络层没有"),
         (TODO, "其它模型 · int4 · 分页 KV\n其它 Apple 芯片", "全部为 Qwen3.5-0.8B × M5 特化"),
         (REUSE, "HF tokenizer\n权重文件读取", "transformers · mx.load")],
    ])
    top = fw - 0.12
    cp = layer(top, "AI 编译器", "COMPILER", [
        [(DONE, "图 IR\nembed … deltanet … head_argmax", "compiler.py · build_graph（9 种算子）"),
         (DONE, "融合 pass\nnorm→linear · linear→+res · swiglu", "fuse：261 → 123 节点"),
         (DONE, "权重布局 pass\ngate/up 交错 · 12 bit 编码", "layout"),
         (DONE, "静态调度 lowering\n直线型 kernel 调用序列", "lower → 整体交给 mx.compile")],
        [(TODO, "通用前端\n从 HF / ONNX 导入任意图", "只有手写的 Qwen3.5 图"),
         (TODO, "自动 codegen / autotune", "tile 表来自离线调参脚本\n→ results/gemm_cfg.json"),
         (TODO, "prefill 与 MTP 草稿路径\n没有走编译器", "仍是手写；整图 compile 对 prefill 无收益"),
         (REUSE, "mx.compile\n惰性图捕获 · 逐元素融合", "MLX")],
    ])
    top = cp - 0.12
    kn = layer(top, "Kernel", "KERNEL", [
        [(DONE, "融合 GEMV 家族（decode）\n融合 norm · residual · SwiGLU · argmax", "kernels.py"),
         (DONE, "DeltaNet 单 dispatch kernel（decode）\n卷积 + 归一化 + 门 + 递推 + 门控 norm", "kernels.py"),
         (DONE, "分块 DeltaNet（NAX，prefill）\ntensor op 做块内矩阵乘 · 精确前代求逆", "gdn_nax.py"),
         (DONE, "NAX GEMM（prefill）\n带 SwiGLU / ADD epilogue", "gemm_nax.py · results/gemm_cfg.json")],
        [(DONE, "int8 预筛 K1–K4\n区间 → 候选 → 精确重算", "head_prune.py"),
         (DONE, "12 bit 权重 SWAR 解码\n寄存器里还原精确 bf16", "wcomp.py · kernels.py"),
         (DONE, "批处理 / 批推测 kernel\nragged 位置 · 快照池", "batch.py · batch_spec.py"),
         (DONE, "int8 GEMM（W8A8，有损 opt-in）\n+ 逐元素胶水 kernel", "gemm_i8.py · prefill.py · mslib.py")],
        [(REUSE, "DeltaNet 递推 kernel\n（内嵌，源自 mlx-lm）", "gated_delta.py · MIT"),
         (REUSE, "NAX mainloop 头文件\n（内联，kernel 自带 epilogue）", "nax.py · 来自 MLX"),
         (REUSE, "SDPA · rms_norm · rope · conv1d\n6 层 full-attention 用 MLX 的", "mx.fast.* · mx.conv1d"),
         (TODO, "自写 attention / paged attention\nint4 · fp8 等其它 dtype 的 kernel", "attention 一律用 MLX 的 SDPA")],
    ])
    top = kn - 0.12
    h = 0.78
    box(X0, top - h, X1 - X0, h, STYLE[REUSE][0], STYLE[REUSE][1], lw=1.4, z=1)
    ax.add_patch(Rectangle((X0 + 0.03, top - h + 0.03), BAND, h - 0.06, fc="#2a4365", ec="none", zorder=2))
    ax.text(X0 + 0.03 + BAND / 2, top - h / 2, "MLX", rotation=90, ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=3)
    ax.text(X0 + BAND + 0.2, top - 0.1, "张量运行时与 Metal 后端（MLX）　全部复用，没有改动 MLX 的源码", ha="left", va="top", fontsize=10.2, fontweight="bold", color="#1a365d")
    ax.text(X0 + BAND + 0.2, top - 0.4, "惰性计算图 · mx.compile · mx.fast.metal_kernel（运行时 JIT，不需要 Xcode）· 融合算子库 · 统一内存分配器 · Metal 命令缓冲提交",
            ha="left", va="top", fontsize=8.8, color="#2c5282")
    top = top - h - 0.1
    h = 0.5
    box(X0, top - h, X1 - X0, h, "#e2e8f0", "#4a5568", lw=1.4, z=1)
    ax.text(W / 2, top - h / 2, "Apple M5　·　10 核 GPU + NAX 张量单元　·　16 GB 统一内存（标称 153 GB/s，实测流式读 106–124 GB/s）　·　bf16 GEMM ≈ 15 TFLOPS",
            ha="center", va="center", fontsize=9.3, color="#2d3748", fontweight="bold")
    fig.savefig(OUT / "architecture.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "architecture.png")


if __name__ == "__main__":
    print("walls (tok/s):", {k: round(v, 1) for k, v in WALL_TOK.items()})
    print("batch floor:", {B: round(batch_floor(B), 1) for B in (1, 2, 4, 6, 8, 12, 16)})
    architecture(); decode_progress(); prefill(); batching(); prefix_cache(); roofline_summary()
