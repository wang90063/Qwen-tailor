# Qwen-tailor

**一次实验：让 coding agent 为一个具体的模型、一块具体的芯片写出模型特化、性能逼近硬件上限的推理服务。**
目标是 **Qwen3.5-0.8B（bf16，不量化）× Apple M5**，基于 [MLX](https://github.com/ml-explore/mlx)；本仓库是这次实验的全部产物：
手写 Metal kernel、一个只服务于这个模型的小型图编译器、MTP 自推测解码、前缀缓存、连续批处理服务器，以及全部的基准、测试和开发记录。

默认路径与 mlx-lm 的贪心输出**逐 token 一致**，下文标「有损」的选项都是显式 opt-in。

> 目录：[1. 实验目的](#1-实验目的) · [2. 实验过程](#2-实验过程)（[2.1 实验配置](#21-实验配置) · [2.2 框架功能 / kernel / 编译器](#22-框架支持的功能--实现的-kernel--编译器的改动) · [2.3 实验结果](#23-实验结果)）· [3. 观察](#3-观察难度在哪里) · [4. 使用与复现](#4-使用与复现) · [5. 仓库结构](#5-仓库结构) · [6. 局限](#6-局限)

## 1. 实验目的

通用推理框架（mlx-lm、llama.cpp、vLLM……）要服务成百上千种模型，所以对任何单个模型都留有余地。反过来，如果只服务**一个**模型、**一块**芯片，
可以把形状、数据类型、融合方式、调度全部写死，贴着硬件上限走——但这需要同时懂 GPU kernel、编译器和推理系统。

我想验证的是：**让 coding agent 来写一个模型特化、并且性能良好的推理服务，难度到底是什么样的？**

具体来说，这个实验想回答：

1. agent 能不能从原样的 mlx-lm 出发，把性能推到离硬件 roofline 不远的位置（而不是只做出「能跑」的版本）？
2. 这条路上的难点在哪里：写 kernel、做性能归因、保证正确性，还是别的？
3. 哪些东西是 agent 能独立走通的，哪些需要外部的验收标准、测量基础设施或人的判断？

Qwen3.5-0.8B + M5 这个组合的好处是问题有一个**明确可算的天花板**：模型小（每 token 只需读约 1.5 GB 权重），decode 是纯内存带宽受限，
prefill 是纯算力受限，两者的 roofline 都可以实测；同时它是 GatedDeltaNet（线性注意力）与全注意力的混合结构，递推状态让前缀缓存、推测解码都比纯 Transformer 多出需要解决的问题。

## 2. 实验过程

### 2.1 实验配置

**硬件与软件**

| 项 | 配置 |
|---|---|
| 芯片 | Apple M5，10 核 GPU，16 GB 统一内存，标称内存带宽 153 GB/s |
| 系统 | macOS 26.5.1，只装了 Command Line Tools（**没有 Xcode**：没有 `metal` 编译器，也没有 GPU 抓帧 / Instruments） |
| 软件 | Python 3.12，mlx 0.32.4，mlx-lm 0.32.2，transformers 5.20.2（只用分词器） |
| 模型 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，bf16；24 层 = 18 层 GatedDeltaNet（递推状态 fp32 `[16,128,128]`）+ 6 层全注意力（GQA，头维 256）；隐层 1024，词表 248320，输出头与 embedding 绑定；带 1 层 MTP 头 |
| 基线 | 原样的 mlx-lm 0.32.2，贪心解码 |
| 开发方式 | 所有 kernel 用 `mx.fast.metal_kernel` 在运行时 JIT 编译（没有 Xcode 也能写 Metal），NAX（Neural Accelerator）部分内联 MLX 自带的 NAX 头文件 |

**Coding agent 与人的分工**（整理自开发时的对话记录）

| 项 | 内容 |
|---|---|
| Coding agent | Claude Code（桌面 App，2.1.x）；全程没有使用子 agent |
| 模型 | 开头约 1 小时的选题讨论用 Claude Haiku 5.5，此后全程 Claude Sonnet 5.5（`claude-sonnet-5-5`），推理强度从 2026-10-20 晚起设为 xhigh |
| 时间 | 2026-10-20 至 2026-10-21，同一条对话，墙钟约 31.5 小时（含夜间空档）；按相邻消息间隔不超过 30 分钟累计，有效工作时间约 12 小时，其中最初约 3 小时（墙钟）是选题与调研 |
| 规模 | 约 1170 次工具调用，绝大多数是 shell 命令（跑基准 / 测试、用 heredoc 写文件）；上下文自动压缩续接了 20 多次 |
| 人发的消息 | 三四十条自然语言消息，其中近 10 条只是「继续 / 重试」；对话里人没有贴过代码或补丁 |

人的介入集中在**选题和改目标**，没有给过具体的 kernel 或算法实现提示：

1. **选题**：先后提过 AlphaEvolve 式的进化搜索（放弃）、把 27B int4 跑在这台机器上（agent 调研后没有可行方案）；最后指定「从编译器 / kernel / 框架三个层面优化一个 1B 左右的模型」，选定 Qwen3.5-0.8B bf16，其余「都按你的」，并要求「给个很高的性能目标」、三层都要改，而不是「从开源下下来跑通就行」。
2. **持续推进**：用 `/goal`（「按照你的 plan 和目标持续推进」「向着性能目标持续迭代吧」）让 agent 自己连续工作；里程碑划分（M0–M10）、目标数字、设计取舍都是 agent 定的。
3. **几次改方向**：改成「朝着 roofline 为目标」；追问「是不是已经到上限了」「有 prefix caching 吗」，随后要求加入前缀缓存、先只做 bf16；嫌「怎么老是干 decode」，要求更激进的打法；要求在 bs>1 时有 continuous batching，并设计一个测 ragged batch 的场景；要求给「包含模型权重的整体内存上限」、由它自动算 slot 数，并写清请求的调度状态流转规则；功能完成后「接着迭代性能」，批量推测解码也是人点头后做的；最后问「目前离 roofline 还有多远」。
4. **约束**：有损优化只能 opt-in，默认路径必须与 golden 逐 token 一致。
5. **环境**：人同意过安装 Xcode（为了 Metal 编译器和 profiler），但没有装成——没有 Apple ID，App Store 上当时的 Xcode 也要求更新的 macOS——所以下面的全部工作都是在没有 Xcode 的条件下完成的。

仓库里能看到的工作流：先把物理上限和基线实测下来、写成目标（[`docs/PLAN.md`](docs/PLAN.md) §1–§3），再按里程碑 M0–M10 逐个推进，每一步都留下测量数字、做法和「没成的尝试」。

**Roofline 的实测值**（不是标称值，脚本在 `bench/`）

| 项 | 实测 | 脚本 |
|---|---|---|
| 流式读带宽 | 中位 106 GB/s / 最好 124.5 GB/s（标称的 70–81%） | `bandwidth.py` |
| lm_head GEMV（M=1） | 中位 109 / 最好 122 GB/s | `bandwidth.py` |
| bf16 GEMM 峰值 | 约 15 TFLOPS（prefill 形状 13–14） | `compute.py` |
| NAX int8 GEMM | 约 24–26 TOPS（bf16 的 1.5–1.7×） | `nax_ops.py`、`gemm_i8_tune.py` |
| 每 token 必须读的权重 | **1.504 GB**（主干 497.6M + 输出头 254.3M 参数 × 2 B），其中输出头占 34% | 由 config 推算 |
| 单个小 GEMV 一次 eval | 0.25 ms（≈30 GB/s）：单次提交的延迟开销很大 | `bandwidth.py` |

由此得到几堵「墙」：

- **无损 decode 的带宽墙**：124.5 GB/s ÷ 1.504 GB ≈ **82.8 tok/s**（按中位带宽算约 70）。mlx-lm 基线 71.4 tok/s，已经在这堵墙的 86%。
- **prefill 的算力墙**：主干约 1.0 GFLOP/token × 13–15 TFLOPS ⇒ **13–15k tok/s**。mlx-lm 基线 6.8k（约一半）。

结论是 decode 在「每 token 读完整个 bf16 模型」的前提下只剩约 10% 空间，所以要么**少读字节**（且保证结果不变），要么**换算法**（推测解码、批处理）；prefill 则还有约 2× 的空间。这就是后面几层优化的出发点。

**测量方法**

- **正确性红线**：默认路径对 `results/baseline_mlx_lm.json` 里的 golden 贪心 token **逐 token 一致**；所有「无损」优化各有逐比特对拍的测试；推测路径输出恒等于逐 token 贪心，草稿只影响接受率；有损的 W8A8 默认关闭。
- **共享 GPU 的抖动**：GPU 与 WindowServer 等共用，墙钟可抖 2×。可信的对比都用 `bench/quiet.py` 的 `Probe`（只接受前后探针 GEMM 都跑在最佳速度附近的样本）和进程内交替 A/B。
- **没有 profiler 的性能归因**：靠自己写的微基准，以及「在整图里把某个 kernel 多执行一次 / 换成廉价桩」的边际耗时法（`bench/prefill_marginal.py`、`bench/step_ablate.py`）。独立链式微基准会因并发 dispatch 低估小算子，所以小算子以整图边际耗时为准。
- **数字的来源**：下文所有结果取自 [`docs/PLAN.md`](docs/PLAN.md) 的记录（2026-10-21，开发机），误差 ±2% 左右。复现请以自己机器上 `bench/decode.py` 的结果为准。

### 2.2 框架支持的功能 / 实现的 kernel / 编译器的改动

整个优化分成四层：**框架层**（自己实现计算图、静态缓存）、**kernel 层**（手写 Metal）、**编译器层**（把 decode 步编译成静态的 dispatch 序列）、**算法层**（突破带宽墙：推测解码、批处理、缓存）。代码量：`qwen35/` 约 5.2k 行 Python（含内嵌的 Metal 源码），`tests/` 1.9k 行，`bench/` 3.5k 行。

#### 2.2.1 框架支持的功能

| 功能 | 做法 | 入口 |
|---|---|---|
| 单序列贪心 decode | 自己实现 Qwen3.5 的函数式模型（不用 mlx-lm 的模型代码），状态和 KV 静态预分配，整步 `mx.compile`，采样（argmax）融进输出头 | `Engine`（`model.py`、`engine.py`、`fast.py`） |
| prefill | NAX GEMM + 分块 DeltaNet，长 prompt 分块（chunk 2048），最后一层只对要用的末尾行计算 | `prefill.py` |
| **可证明无损的 int8 lm_head 预筛** | tied embedding 占每 token 流量的 34%，而 decode 只要 argmax：用 int8 副本给每个词算严格的 logit 区间，只对可能是最大值的少数候选词用 bf16 行精确重算，结果与读完整张表**逐比特相同**（含平局取最小下标） | `head_prune.py` |
| **无损 12 bit/权重压缩** | bf16 权重的指数位熵只有约 2.7 bit：每个权重存 1 字节（符号+尾数）加 4 位指数码，GEMV 里在寄存器中还原出精确的 bf16 位型，读的字节少 25%，结果逐比特不变 | `wcomp.py` |
| **MTP 自推测解码** | 用模型自带的 MTP 头起草（草稿头用裁剪到 16K 词表的 int8 子集），整轮「草稿 → 批量校验 → DeltaNet 状态快照回滚」在一张 GPU 常驻的编译图里完成 | `SpecEngine`（`spec.py`、`mtp.py`） |
| bf16 前缀缓存 | DeltaNet 的递推状态不能像 KV 那样截断，所以缓存的是**检查点树**；支持分裂学习、LRU、多轮对话、整条 prompt 重复 | `prefix_cache.py` |
| 连续批处理服务器 | slot 池、ragged 位置、分块 prefill 与 decode 交错、`QUEUED→PREFILL→DECODE→DONE` 状态机、总显存上限自动算 slot 数；批内每一行与单序列 decode **逐比特一致** | `batch.py`、`serve.py` |
| 批量推测解码 | 连续批处理 + MTP 自推测，快照池原位更新；按「每序列每轮产出 token 数 vs 成本比」在 plain / 推测间自适应切换 | `batch_spec.py`、`serve.py` |
| W8A8 prefill（**有损**，opt-in） | NAX int8 GEMM，激活 per-token 对称量化，Hadamard 旋转压低离群值误差 | `gemm_i8.py` |
| 命令行 | 生成、聊天模板、多轮 REPL（带前缀缓存） | `run.py` |

#### 2.2.2 实现的 kernel

全部通过 `mx.fast.metal_kernel` 运行时编译。

| 路径 | kernel | 要点 | 文件 |
|---|---|---|---|
| decode | **融合 GEMV 家族** | 前置 RMSNorm / 门控 RMSNorm，后置 residual 加 / SwiGLU / argmax；M=1…16 行，M 按寄存器预算分块；可选 12 bit 压缩权重的 SWAR 解码 | `kernels.py` |
| decode | **DeltaNet 单 dispatch kernel** | 卷积窗 + q/k 归一化 + 门 + 递推 + 门控 RMSNorm 一次 dispatch；M 行时每个 token 位置写一份状态快照 | `kernels.py` |
| decode | 输出头 argmax、MTP 草稿子词表 argmax | GEMV 与 argmax 融合；草稿头用 int8 | `kernels.py` |
| 无损字节削减 | int8 预筛 K1–K4 | K1 int8 近似区间（M≥6 时改用 NAX 张量算子）→ K2 候选收集 → K3 bf16 精确重算 → K4 终归约；254 MB 约 2.0 ms（≈127 GB/s，贴墙） | `head_prune.py` |
| 无损字节削减 | 12 bit 权重编解码 | 每行连续 `A 平面 + E 平面`，0.30% 的例外块走旁表；解码每个 bf16 对约 5 条整数指令 | `wcomp.py`、`kernels.py` |
| prefill | **分块 DeltaNet NAX kernel** | 阶段 A 对（16-token 块, head）并行：KKᵀ / QKᵀ 用 tensor op，(I+L)⁻¹ 用 fp32 前代精确求；阶段 B 每个（head, dv 切片）一个 simdgroup，128 条独立链；conv_prep 与阶段 A 融合 | `gdn_nax.py` |
| prefill | **NAX GEMM 家族** | 复用 MLX 的 NAX mainloop 并自带 epilogue：`EPI_SWIGLU`（gate/up 行交错，寄存器里算完 SwiGLU，省掉 2048×7168 的中间张量）、`EPI_ADD`（残差加融进 out_proj / down）；每个（层 GEMM, M）的 tile 配置由调参脚本写入 `results/gemm_cfg.json` | `gemm_nax.py` |
| prefill | int8 GEMM + `norm_q` | int8 输入、int32 累加；`norm_q` 在 RMSNorm kernel 里顺手做 Hadamard 旋转与逐行量化 | `gemm_i8.py`、`prefill.py` |
| prefill | 逐元素胶水 | add_norm、conv_prep、gated_norm 等，已贴 DRAM 带宽墙；bf16 逐元素语义与 MLX 逐比特一致（`sigmoid`/`softplus` 的精确舍入结构） | `prefill.py`、`mslib.py` |
| 批处理 | `deltanet_b`、`attn_b`、`blit` | 每行 slot / pos 不同的 DeltaNet 与注意力（算术与 M=1 完全一致）、原地拷贝；双缓冲 conv 状态 | `batch.py` |
| 批推测 | `deltanet_bm`、`verify_b`、`qsubset_argmax_b`、`mtp_step_b` | M 个链式 token 的快照池，接受 n 个就选第 n 份，不回滚不重算 | `batch_spec.py` |
| 基础设施 | NAX 头文件内联、bf16 逐位语义库 | 让自写 kernel 与 MLX 的 bf16 舍入逐比特对齐 | `nax.py`、`mslib.py` |

#### 2.2.3 编译器的改动

`qwen35/compiler.py` 是一个只服务于这个模型的小型图编译器，覆盖 decode / verify 一步：

```
build_graph(cfg)  图 IR：embed / rmsnorm / linear / add / swiglu / deltanet / gated_norm / attn_core / head_argmax
fuse(graph)       模式重写：rmsnorm→linear (48) · gated_norm→linear (18) · linear→+residual (48) · linear→swiglu (24)；261 个节点 → 123 个
layout(graph, W)  权重布局：gate/up 行交错；对所有融合后的 linear 做 12 bit 压缩，丢掉原权重
lower(graph, …)   静态调度：一串直线型的 kernel 调用，整体交给 mx.compile
```

- 编译器输出与手写 `fast.verify` **逐比特一致**（`tests/compiler.py`），速度持平（≈78 tok/s，当时）；关闭融合的 lowering 为 74 tok/s，融合本身约 +5%。
- 压缩权重和 int8 头预筛是通过 `layout` pass 与 `head_argmax` 算子接进编译器路径的。
- **没有覆盖的部分**：prefill 路径和 MTP 草稿路径仍是手写；`mx.compile` 整图用于 prefill 没有收益（已放弃）。

### 2.3 实验结果

#### 总览（同一台机器，基线是原样的 mlx-lm 0.32.2）

| 指标 | mlx-lm 基线 | 本项目 |
|---|---|---|
| decode，无推测，贪心 | 71.4 tok/s | **109–110 tok/s**（逐 token 一致） |
| decode，MTP 自推测（6 条 prompt 平均） | 71.4 tok/s | **224**（k=2）/ **235–240**（k=3）tok/s，单条最高 314 |
| prefill，2048 token | 6.8k tok/s | **11.55k tok/s**（可选 W8A8，有损：14.6k） |
| TTFT，512 token | 94 ms | **46 ms**（W8A8：39 ms） |
| TTFT，命中 2048 token 前缀缓存 + 16 token 后缀 | 188 ms | **16 ms**（11.7×） |
| 连续批处理，decode 聚合吞吐 | 109.7 tok/s（B=1，串行） | B=4 **400** / B=8 **617** / B=16 **804** tok/s |
| 批量推测解码相对 plain 批 decode | — | B=1 **2.44×** … B=4 1.74× … B=8 1.21×（代码文本） |
| 混合负载回放（79 请求）TTFT p50 / p95 | 串行 FCFS 16.5 s / 47.6 s | continuous batching **0.13 s / 0.72 s**（开前缀缓存 0.055 s / 0.38 s） |

#### 和 roofline 比

![与 roofline 的距离](docs/figures/roofline_summary.png)

| 场景 | 屋顶 | 达到 | 占比 |
|---|---|---|---|
| decode 无推测 | 带宽墙 (0.749 GB 压缩主干 + 0.254 GB int8 头) ÷ 124.5 GB/s ≈ 124 tok/s | 109.7 tok/s（≈110 GB/s 的权重流量） | **88%** |
| decode 无推测（基线） | bf16 带宽墙 82.8 tok/s | 71.4 tok/s | 86% |
| prefill @2048 | 无损地板 165–172 ms（GEMM 在 NAX bf16 峰值，norm / conv / gnorm 在 DRAM 带宽墙，GDN 阶段 B 受 NAX 吞吐限制） | 177 ms（含 ≈2 ms int8 头） | **93–97%** |
| prefill @2048 | bf16 GEMM 算力墙 13–15k tok/s | 11.55k tok/s | 77–89% |
| 批 decode B=1 / 4 / 8 / 16 | 内存地板（权重 + 状态 + KV 的字节 ÷ 124.5 GB/s，见下） | 115 / 400 / 617 / 804 tok/s | 97% / 96% / 86% / **72%** |
| MTP 自推测 | 没有固定的带宽墙，取决于每轮接受的 token 数（代码 / 事实类约 2.8 token/轮；故事类在 M4 时只有 111–134 tok/s） | 224–240 tok/s ≈ 无推测的 2.0–2.2× | — |

说明与口径：

- **墙取的是实测最好带宽 124.5 GB/s，是偏保守的估计**：M5 优化前曾实测过 84.4 tok/s × 1.504 GB ≈ 127 GB/s 的有效读带宽，所以上面的百分比应按偏乐观的口径理解；实测是在机器有负载时做的，安静状态下墙可能略高。
- **「墙」是会动的**：82.8 → 99.7 → 124 tok/s 这三堵墙不是三种测量，而是同一个带宽除以不同的「每 token 读的字节数」（1.504 → 1.249 → 1.003 GB）——后两步靠无损字节削减把墙本身搬高了。
- **批 decode 的内存地板是我按 PLAN 里的状态大小推算的**：每步读一遍权重 1.003 GB，每个序列再读写一遍 18 层 fp32 递推状态（18 × 2 MiB），再读 KV（6 层，12 KiB/token × pos≈900），都按 124.5 GB/s 折算。推算脚本见 [`docs/make_figures.py`](docs/make_figures.py)。B ≤ 4 已贴地板；B=16 剩下的差距来自 GEMV 的 ALU 余量（≈2.5 ms）和小 kernel 的 launch 间隙（见 PLAN M9 的边际消融）。

#### decode：无推测 与 MTP 自推测

![decode 性能曲线](docs/figures/decode_progress.png)

无推测 decode 的逐阶段记录（权重字节是每 token 必须读的权重，不含 KV、状态和激活）：

| 阶段 | 做了什么 | 每 token 权重字节 | tok/s | 有效带宽 |
|---|---|---|---|---|
| mlx-lm 基线 | — | 1.504 GB | 71.4 | 107 GB/s |
| M1 | 自研模型 + 静态缓存 + 整步编译 | 1.504 GB | 71.8 | 108 GB/s |
| M2 | 融合 GEMV 家族 + DeltaNet 单 dispatch kernel | 1.504 GB | 78.2 | 118 GB/s |
| M5 前 | M2 之后、M5 之前的小幅累计（PLAN 未单独记账） | 1.503 GB | 84.4 | 127 GB/s |
| M5 | int8 lm_head 预筛（无损） | 1.249 GB | 96 | 120 GB/s |
| M6 | 主干 12 bit 压缩（无损） | 1.003 GB | 109.7 | 110 GB/s |

读法：**无推测 decode 的 1.54× 加速里，约 1.50× 来自少读的字节**（1.504 → 1.003 GB），有效带宽几乎没变（107 → 110 GB/s）。框架重写本身（M1）收益约等于零，手写 kernel（M2）让带宽从 107 提到 118 GB/s 左右，而最大的两步都是「在保证结果逐比特不变的前提下少读数据」。随着字节变少，有效带宽反而降到 110 GB/s：每 token 固定的开销（DeltaNet 状态读写、KV、小 N GEMV 的启动延迟）不随权重缩小，离墙还剩 12%（9.1 ms 一步里约 1 ms），PLAN 记录的主要差距在小 N 的 down / out_proj（延迟受限）以及 DeltaNet 与全注意力的胶水算子。

MTP 自推测在此之上再前进：k=2 / k=3 的 6 条 prompt 平均吞吐 M4 159 / 172 → M5 189.7 / 201.5 → M6 224.4 / 240.5 tok/s（M4 时每轮耗时 15–17 ms）。

#### prefill

![prefill 吞吐](docs/figures/prefill.png)

| prompt | mlx-lm 基线 | 本项目（默认，无损） | 本项目 + W8A8（有损） |
|---|---|---|---|
| 128 | 未测 | 6.7k tok/s（19.2 ms） | 8.2k tok/s（15.6 ms） |
| 512 | 5.5k tok/s（94 ms） | 11.1k tok/s（46.1–46.6 ms） | 13.2k tok/s（38.7 ms） |
| 2048 | 6.8k tok/s | 11.55k tok/s（≈177 ms） | 14.6k tok/s（139.9 ms） |

prefill 的演进：6.96k →（M3：分块 DeltaNet NAX kernel + 自写 NAX GEMM + SwiGLU / 残差 epilogue 融合）11.0k →（M7：末层 tail 路径、conv_prep 与阶段 A 融合、首 token 走 int8 头）11.55k。GEMM 已在 NAX bf16 峰值（M=2048 约 15.2–15.6 TFLOPS，占总时间约 70%），无损的地板在 165–172 ms，12.5k 在 bf16 下够不到。W8A8 把 in_proj / qkv / gate_up 换成 int8（≈25 TOPS），代价是 logits 相对误差 1.6–3.4%（无损路径约 0.7%），所以是显式 opt-in。

#### 前缀缓存、连续批处理、批量推测

![前缀缓存](docs/figures/prefix_cache.png)

整条 prompt 重复（首 token 也缓存）时 TTFT 只有 1.7 ms（@2048）。命中后的输出与「在检查点处分块的 prefill」逐比特一致；与整段一次 prefill 在 bf16 舍入层面不逐比特一致（MLX 的 SDPA 按 query 长度分档），首 token 一致，生成只在 bf16 近平局处可能分叉。

![连续批处理与批量推测](docs/figures/batching.png)

| B | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| 批 decode 步耗时（ms） | 8.69 | 9.03 | 10.00 | 11.77 | 12.98 | 16.13 | 19.91 |
| 聚合吞吐（tok/s） | 115 | 221 | 400 | 510 | 617 | 744 | 804 |
| 内存地板（tok/s，推算） | 118 | 226 | 416 | 577 | 715 | 940 | 1117 |
| 达到的比例 | 97% | 98% | 96% | 88% | 86% | 79% | 72% |

大 B 的 500 → 802 tok/s 来自一组全部与旧 kernel 逐比特一致的优化：拆分 norm、认证 int8 头的近似阶段改用 NAX 张量算子、按 B 查表的 GEMV 分块、K=1024 的输入保持 fp32、权重每线程只解码一次。批量推测解码（k=2，代码文本，服务器级，B 个并发请求）：

| B | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| plain（tok/s） | 116.9 | 226.1 | 326.1 | 414.6 | 527.8 | 647.2 |
| 推测（tok/s） | 285.7 | 458.7 | 609.5 | 721.9 | 760.8 | 785.0 |
| 加速 | 2.44× | 2.03× | 1.87× | 1.74× | 1.44× | 1.21× |

收益取决于草稿接受率：约 1 token/轮时推测反而更慢（B=1 约 −15%，B=4 约 −38%），所以服务器按每序列每轮产出的 token 数自动在 plain 与推测之间切换。

## 3. 观察：难度在哪里

以下是从开发记录（[`docs/PLAN.md`](docs/PLAN.md)）里能读出的几点，不是对「难度」的最终判断。

1. **能做到什么**：从原样 mlx-lm 出发，在保持输出逐 token 一致的前提下，decode 到内存带宽墙的约 88%，无损 prefill 到地板的 93–97%，并在算法层（MTP 自推测、批处理、前缀缓存）把端到端指标再推 2–12×。
2. **难点不在「把带宽吃满」，而在「找到不改变输出的少做事办法」**。基线 mlx-lm 已经在 bf16 墙的 86%；无推测 decode 的收益几乎全部来自少读字节（int8 预筛需要一个 Cauchy–Schwarz 区间证明，12 bit 压缩需要观察到指数位的熵只有约 2.7 bit 并设计可并行解码的布局）。纯框架重写（M1）收益约等于零。
3. **验收标准决定了能走多远**。golden 贪心输出、逐比特对拍的 16 个测试脚本、teacher-forced 的 bf16 平局容差，让每一步优化都能被自动判对错——「可证明无损」这类优化只有在这样的判据下才敢做。
4. **没有 profiler 时，经验判断不可靠，必须自建归因工具**。开发记录里有几次被测量推翻的判断：M5 时认为「M>1 的校验是 GEMV 内循环 ALU / 延迟受限」，M6 的测量证明这是错的（额外代价来自 gnorm 前置的 out_proj 在每个线程组里重复算门控 RMSNorm，以及 DeltaNet 每多一行要多写一份 1 MB 的状态快照）；full-attn 层的 q/k norm + rope、`o*sigmoid(gate)` 胶水融成 kernel 逐比特一致、但整图没有收益（MLX 的并发 dispatch 本来就让独立的小算子并行，独立链微基准会高估胶水成本）；`mx.compile` 整图用于 prefill 无收益。
5. **测量本身是一项成本**。共享 GPU 的 2× 抖动，让「安静 GPU 探针 + 进程内交替 A/B」成了必需品；MLX 空闲缓存不封顶会涨到 8.6 GiB，让 TTFT p95 在 2.8–5.3 s 间乱跳，一度被误读成「chunk 256 比 512 好」。
6. **一些没成的尝试**（别重试清单，细节在 PLAN）：自写 DeltaNet 递推 kernel（预取、每 simdgroup 多行）慢于 mlx-lm 的 packed kernel；两个 GPU stream 并行 DeltaNet 与 GEMM 无重叠收益；阶段 B 操作数预取多用 26 个寄存器、反而慢 2.5×；SDPA 按 query 分块慢于 MLX 的融合 NAX attention；6 bit / 5 bit 头需要位平面或再加一级精筛，收益 ≤ 0.5–0.75 ms/token，没做。

> **待补（作者填写）**：对「coding agent 写模型特化推理服务的难度」的总体结论，以及人在其中承担的角色。

## 4. 使用与复现

### 环境要求

- **Apple M5**。prefill / 批处理的 kernel 依赖 M5 GPU 的 Neural Accelerator（Metal 4 tensor op），并内联 MLX 自带的 NAX 头文件。**只在 M5 上测试过**；其它芯片上代码没有做硬件检测，很可能在 kernel 编译处直接报错，或者给出没有验证过的结果。
- macOS 26.x（开发和测试在 26.5.1）；Python ≥ 3.10（测试用的是 3.12）。
- 依赖固定在开发时的版本区间：`mlx>=0.32.4,<0.33`、`mlx-lm>=0.32.2,<0.33`、`transformers>=5.7`、`numpy`。kernel 内联了 MLX 的头文件、复用了 mlx-lm 的 gated-delta 代码，换大版本很可能失效。
- 内存：开发机是 16 GB。权重加各种派生副本（压缩权重、int8 头、融合权重）约占 2.4 GiB。

### 安装

权重、调参表和 golden 输出都是按仓库目录定位的，所以请**从克隆下来的仓库里**安装：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

权重不随仓库分发（遵循 Qwen 自己的 Apache-2.0 许可，`models/` 与 `*.safetensors` 已被 git 忽略）。没装 PyTorch 时 transformers 会打印 “PyTorch was not found” 的提示，可以忽略：这里只用它的分词器。

### 快速开始

```bash
# 自推测解码生成（默认 k=3）
python run.py "def fibonacci(n):" -n 128

# 套用聊天模板
python run.py "用三句话介绍一下 Metal" --chat -n 256

# 多轮对话（stdin 一行一条用户消息，空行或 EOF 退出；带前缀缓存，每轮只 prefill 新增部分）
python run.py --repl

# 打开 W8A8（有损）prefill
python run.py "..." --w8a8
```

完整参数见 `python run.py -h`：`-n` 生成长度，`--k` 草稿长度，`--cache-mb` 前缀缓存预算，`--max-len`（`--repl` 的上下文长度）。每次运行结束会在 stderr 打印 prefill 用时、decode 速度和每轮接受的草稿数。

### Python API

```python
from transformers import AutoTokenizer
from qwen35.model import MODEL_DIR
from qwen35.engine import Engine
from qwen35.spec import SpecEngine

tk = AutoTokenizer.from_pretrained(str(MODEL_DIR))
ids = tk.encode("The capital of France is")

# 无推测 decode：逐 token 与 mlx-lm 的贪心输出一致
eng = Engine(max_len=2048)
print(tk.decode(eng.generate(ids, 64)))

# MTP 自推测
sp = SpecEngine(k=3, max_len=2048)
stats = {}
print(tk.decode(sp.generate(ids, 128, stats)))
print(stats["accepted"] / stats["rounds"], "个草稿 token / 轮被接受")
```

连续批处理服务器（`serve.py`）是一个调度器，不是 HTTP 服务：`Server.run(requests)` 按各请求的 `arrival` 时间回放一组请求并就地填好 `out` / `t_first` / `t_done` 等字段。

```python
from qwen35.serve import Server, Request

sp = SpecEngine(k=2, max_len=1024)
eng = Engine(max_len=1024, like=sp)          # like=：共用同一份权重
srv = Server(eng, slots=4, max_len=1024, spec=sp)   # 不传 spec 就是纯 plain 批 decode
srv.warmup()
reqs = [Request(rid=i, prompt=tk.encode(p), max_new=64) for i, p in enumerate(["Hello", "def f(x):", "1 + 1 ="])]
srv.run(reqs)
for r in reqs:
    print(r.rid, r.finish_reason, tk.decode(r.out))
```

调度策略、请求状态机、内存预算的细节见 `qwen35/serve.py` 文件头和 [docs/PLAN.md](docs/PLAN.md) 的 M9 / M10。

### 开关（环境变量）

默认值就是最快且精确的配置，下面的开关主要用于 A/B 对照和排查。

<details>
<summary>展开开关表</summary>

| 变量 | 默认 | 作用 |
|---|---|---|
| `QWEN35_W8A8` | 空（关） | 逗号分隔的 GEMM 名，如 `in_proj,qkv,gate_up`：这些 prefill GEMM 走 int8（**有损**，logits 相对误差约 2–3%）；`run.py --w8a8` 等价于全开 |
| `QWEN35_W8A8_ROT` | `1` | W8A8 下的 Hadamard 旋转；`0` 关闭 |
| `QWEN35_HEAD_PRUNE` | `1` | 可证明无损的 int8 lm_head 预筛；`0` 回到读完整张 bf16 表 |
| `QWEN35_WCOMP` | `1` | 主干权重的无损 12 bit 压缩；`0` 回到 bf16 权重 |
| `QWEN35_DRAFT_Q8` | `1` | MTP 草稿头的 int8 词表子集 |
| `QWEN35_GDN` / `QWEN35_GDN_FLAGS` | `nax` / 空 | prefill 的 DeltaNet 实现（`nax` 分块 kernel 或 `ref` 即 mlx-lm 的顺序 kernel）及其精度 flag |
| `QWEN35_GEMM` / `QWEN35_GEMM_OFF` | `nax` / 空 | prefill 投影用自写 NAX GEMM 还是 `mx.matmul`；或指定哪几个留在 `mx.matmul` |
| `QWEN35_FUSE_ADD` / `QWEN35_PREPA` | `1` / `1` | prefill 的残差加融合、conv+qk-norm 与 DeltaNet 阶段 A 融合（`0` 为逐比特等价的未融合版本） |
| `QWEN35_PREP` / `QWEN35_PREP_TB` | `v2` / `16` | conv_prep kernel 版本与 token 分块 |
| `QWEN35_HEAD_NAX_MIN` | `6` | 批 ≥ 该值时 int8 头的近似阶段用 NAX（`0` 为永不） |
| `QWEN35_F32X_MIN` | `4` | 批 ≥ 该值时 K=1024 的 GEMV 以 fp32 读归一化后的输入 |

</details>

前缀缓存只在精确（bf16）路径上工作，`QWEN35_W8A8` 打开时会绕过并警告一次。

### 正确性与测试

- **默认路径**对 `results/baseline_mlx_lm.json` 里的 golden 贪心 token **逐 token 一致**；融合 / 压缩 / 预筛这些「无损」优化各自有逐比特对拍的测试。
- **推测路径**的输出恒等于逐 token 的贪心，草稿只影响接受率；但批量校验的数值与单 token 路径不同，在 bf16 恰好平局的位置可能分叉（golden 的第 3 条 prompt 就是），测试会报告这类分叉并检查 logit 差在 bf16 一格以内。
- **分块 prefill / 前缀缓存命中**与「整段一次 prefill」在 bf16 舍入层面不逐比特一致（MLX 的 SDPA 按 query 长度分档、NAX tile 配置不同），首 token 一致，生成只会在近平局处分叉；缓存命中与「在检查点处分块的 prefill」是逐比特一致的。
- **W8A8** 是有损的，精度见 docs/PLAN.md 的 M7。

运行测试（需要权重；全部跑完需要几分钟，GPU 会满载，建议关掉其它占 GPU 的程序）：

```bash
python tests/run_all.py            # 全部
python tests/run_all.py golden gemm   # 按脚本名子串筛选
python tests/run_all.py --list
```

`tests/*.py` 也都可以单独当脚本运行（`python tests/golden.py`），失败时以非 0 退出码结束。`tests/spec.py` 只是报告，不判定成败。这些脚本不是 pytest 用例：它们在 import 时就执行，所以特意没有叫 `test_*.py`。

### 复现实验里的数字与图

- 各个数字对应的基准脚本见 [`bench/README.md`](bench/README.md)（按物理上限、decode、MTP、prefill、NAX kernel、前缀缓存与批处理分类）；开发记录与每个里程碑的消融在 [`docs/PLAN.md`](docs/PLAN.md)。
- 本文的图由 [`docs/make_figures.py`](docs/make_figures.py) 生成（需要 matplotlib，它不是本包的依赖）；脚本里的数字抄自 PLAN，不重新测量。

## 5. 仓库结构

```
qwen35/        核心实现
  model.py       函数式模型（无 nn.Module）、权重加载与融合
  engine.py      prefill + 编译后的流水线 decode（Engine）
  spec.py mtp.py MTP 自推测（SpecEngine）与 MTP 头
  kernels.py     手写 Metal kernel：融合 GEMV 家族、DeltaNet、argmax 等
  compiler.py    小型图编译器
  fast.py        用融合 kernel 搭的 decode / verify 前向
  prefill.py gemm_nax.py gdn_nax.py gemm_i8.py nax.py mslib.py   prefill 路径与 NAX kernel
  head_prune.py wcomp.py     无损 int8 头预筛；无损权重压缩
  prefix_cache.py            前缀缓存
  batch.py serve.py batch_spec.py   连续批处理、调度器、批量推测解码
run.py         命令行入口
tests/         正确性测试（run_all.py 一键运行）
bench/         基准、调参和实验脚本，见 bench/README.md
results/       运行时数据与参考输出：golden 贪心输出、调好的 tile 配置表、MTP 草稿头的词频表
docs/PLAN.md   设计与开发记录：测量方法、每个里程碑的做法、数字、没成的尝试
docs/figures/  本文的图（docs/make_figures.py 生成）
models/        权重放这里（git 忽略）
```

## 6. 局限

- 只针对 Qwen3.5-0.8B 这一个模型的形状（层数、头数、维度写在 `Cfg` 里并被 kernel 特化），换别的模型需要重新调 kernel。这是实验的设定，不是缺陷。
- 只在 M5 上验证；其它 Apple 芯片未测试，没有运行时保护。
- 只做贪心（argmax）解码，没有采样。
- 批处理与服务器是调度器内核，不带 HTTP / OpenAI 兼容接口。
- 批量路径与单序列逐比特一致的保证限于上下文 < 4096（见 docs/PLAN.md M9）。
- 开发时的 GPU 是和系统共用的，基准数字有抖动；复现时请关掉其它占 GPU 的程序。
- 这是一次单模型、单机器、单次运行的实验：数字说明在这个设定下能做到什么，不能直接外推到别的模型或别的芯片。

## 致谢

- [MLX](https://github.com/ml-explore/mlx) 与 [mlx-lm](https://github.com/ml-explore/mlx-lm)：本项目基于 MLX 的 Metal 后端，复用了 NAX mainloop 头文件，并以 mlx-lm 的输出作为 golden 基线。
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B)：模型与权重，遵循其自身的 Apache-2.0 许可，不随本仓库分发。

## 许可

[MIT](LICENSE)
