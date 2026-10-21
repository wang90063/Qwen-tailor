# Qwen3.5-0.8B (bf16) on Apple M5 —— 编译器 / Kernel / 框架 三层优化

机器：Apple M5 (10 核 GPU)，16GB 统一内存，标称带宽 153GB/s。模型：`Qwen/Qwen3.5-0.8B`，bf16，不量化。
基线 = 原样的 mlx-lm 0.32.2。所有数字来自本仓库 `bench/`，结果在 `results/`。

## 0. 当前成绩（2026-10-21，同一台机器，负载 ≈5 时有 ±2 抖动）

| 指标 | 基线 | 目标 | 冲刺 | 现在（默认，精确） | 可选 W8A8（有损，M7） |
|---|---|---|---|---|---|
| decode，无推测 | 71.4 | ≥ 79 | 82 | **109–110**（M6 之后；M5 时 96–98.7；M5 之前 78–84） | — |
| decode，MTP 自推测（6 prompt 平均 / 最好） | 71.4 | ≥ 150 | 180 | **224（k=2）/ 235–240（k=3）/ 314**（M6 之后；M5 时 189–202 / 197–217；M5 之前 159 / 172 / 226） | — |
| prefill 2048（Engine 路径，含 int8 头） | 6.8k | ≥ 11k | 12.5k | **11.55k**（≈ 175 ms，未达冲刺） | **14.6k**（139.9 ms） |
| TTFT @512 | 94 ms | ≤ 55 ms | 45 ms | **46.1–46.6 ms**（@128：19.2 ms；未达冲刺） | **38.7 ms**（@128：15.6 ms） |
| **前缀缓存**（M8，bf16，默认关，`Engine(prefix_cache_mb=…)`）：TTFT，命中 2048 token 前缀 + 16 / 128 token 后缀 | 187.6 / 196.2 ms（无缓存） | ≤ 25 ms | 15 ms | **16.0 / 22.9 ms**（11.7× / 8.6×；+512 后缀 54.6 ms，4.2×） | 不支持（旁路） |
| 前缀缓存：整条 prompt 重复（首 token 也缓存） | 187 ms | ≤ 3 ms | — | **1.7 ms**（@256：1.1 ms） | — |
| **连续批处理**（M9，bs>1，`qwen35/serve.py`）：批 decode 聚合吞吐（ragged 位置，逐 token 与单序列一致） | 109.7（B=1） | B=4 ≥ 350；B=8 ≥ 450；B=16 ≥ 600 | B=16 ≥ 800 | **B=4 400 / B=8 617 / B=12 744 / B=16 804 tok/s**（B=4/8/16 目标与冲刺 800 全部达成，见 M9） | 不涉及 |
| **批量推测解码**（M10，`Server(spec=…)`，代码文本，服务器级，同一进程交替）：B 个并发请求的 decode 聚合吞吐，推测 / 纯 plain | 同上 plain：116.9 / 226.1 / 326.1 / 414.6 / 527.8 / 647.2（B=1/2/3/4/6/8） | B=2–4 再涨 ≥ 40% | — | **285.7 / 458.7 / 609.5 / 721.9 / 760.8 / 785.0 tok/s = 2.44× / 2.03× / 1.87× / 1.74× / 1.44× / 1.21×**；逐 token 与单序列一致；自适应切回 plain（接受率不够时） | 不涉及 |
| ragged 场景 `mixed`（79 请求，4 类，泊松 + 突发，8 slot）：TTFT p50 / p95，ITL p99 | 串行 FCFS：16.5 s / 47.6 s，11.5 ms | TTFT p95 ≤ 1 s | — | static 2.5 s / 6.8 s；**continuous 0.13 s / 0.72 s，60 ms**（chunk 512）；开前缀缓存 0.055 s / 0.38 s | 不涉及 |

默认路径对 golden 逐 token 一致；W8A8 只在 `QWEN35_W8A8=in_proj,qkv,gate_up` 时打开（见 M7），是显式 opt-in，不会成为默认。

入口：`run.py`（命令行生成；`--cache-mb N` 打开前缀缓存，`--repl` 多轮对话，检查点放在“下一轮不再相同”的位置，即聊天模板去掉历史回答里 `<think>` 之前）、`tests/*.py`（kernel/verify/编译器/golden/推测对拍/权重压缩）、`bench/*.py`。

## 1. 物理上限（实测，不是标称）

| 项 | 实测 | 来源 |
|---|---|---|
| 流式读带宽 | 中位 106 / 最好 124.5 GB/s（标称的 70–81%） | `bench/bandwidth.py` |
| lm_head GEMV (M=1) | 中位 109 / 最好 122 GB/s | 同上 |
| 单个小 GEMV 一次 eval | 0.25 ms（约 30 GB/s） | 同上，说明单次提交的延迟开销很大 |
| bf16 GEMM 峰值 | 约 15 TFLOPS（4096³），prefill 形状 13–14 | `bench/compute.py` |
| 每 token 必须读的权重 | **1.504 GB**（主干 497.6M + 输出头 254.3M 参数，× 2B） | 由 config.json 推算 |

注意：输出头（tied embedding，词表 248320）占每 token 流量的 **34%**。

推论：
- **无损、单 token 的 decode 墙**（每 token 读全部 bf16 权重）≈ 124.5 / 1.504 ≈ **82 tok/s**（中位数口径约 70）。这堵墙里输出头占 34%；M5 用可证明无损的 int8 预筛不再整张读输出头，墙移到 (0.995 GB 主干 + 0.254 GB int8 头) / 124.5 GB/s ≈ 10.0 ms ≈ **99.7 tok/s**，实测 96。
- **prefill 的 GEMM 墙**：主干约 1.0 GFLOP/token → 13–15 TFLOPS 对应 **13–15k tok/s**。
- 以上测量时机器负载较高（load ≈ 4.8，交换区在用），**安静状态下需要复测**，墙可能略高。

## 2. 基线（mlx-lm，bf16）

| 指标 | 基线 |
|---|---|
| decode，ctx 64，贪心 | **71.4 tok/s**（66.3–73.3），等效 107 GB/s，占标称 70% |
| prefill 512 | 5465 tok/s（TTFT ≈ 94 ms） |
| prefill 2048 | 6822 tok/s（≈ 7 TFLOPS，约为 GEMM 墙的一半） |

结论：**decode 在无推测的前提下只剩约 10% 空间，必须靠算法层面绕开带宽墙；prefill 还有约 2× 的空间。**

## 3. 目标

| 指标 | 基线 | 目标 | 冲刺 | 说明 |
|---|---|---|---|---|
| decode，无推测 | 71.4 | ≥ 79 tok/s | 82 | 贴着内存墙，靠框架+kernel+编译器去掉调度和多余流量（冲刺目标由 M5 超额完成） |
| **decode，MTP 自推测** | 71.4 | **≥ 150 tok/s（2.1×）** | 180 | 有效吞吐，接受长度依赖文本；用模型自带的 MTP 头（`mtp_num_hidden_layers=1`）；校验保持无损 |
| prefill 2048 | 6.8k | ≥ 11k tok/s（1.6×） | 12.5k | 约 GEMM 墙的 80%；超过 13.6k（2×）在 bf16 下基本不可能 |
| TTFT @512 | 94 ms | ≤ 55 ms | 45 ms | |
| **前缀缓存（bf16）**：命中 2048 token 前缀 + ≤128 token 后缀的 TTFT | 无缓存 ≈190–200 ms | ≤ 25 ms（≥ 8×） | 15 ms | 多轮对话 / 共同 system prompt / 重复 prompt；命中 ≡ 在检查点处分块的 prefill（已接受的一类），状态与无缓存的前缀计算逐比特相同；每 2k token 条目 ≈ 43 MB；W8A8 不在范围内 |
| 前缀缓存：完全重复的 prompt | ≈190 ms | ≤ 3 ms | — | 只做恢复（拷 KV），不跑模型 |
| **连续批处理（bs>1）** | 串行 109.7 tok/s | 聚合吞吐 B=4 ≥ 350、B=8 ≥ 450、B=16 ≥ 600 tok/s；ragged 场景 TTFT p95 ≤ 1 s | B=16 ≥ 800 | 每个 slot 独立位置、动态准入/回收；批内每行与单序列 decode **逐比特一致**（正确性红线不变）；总显存上限 + slot 数自动计算；请求状态机见 M9 |

正确性红线：对 `results/baseline_mlx_lm.json` 里的 golden 贪心 token，无推测路径要**逐 token 一致**；推测路径允许因批量数值差异在近似平局处分叉，但要报告匹配率。

## 4. 三层路线

**框架层**：自己实现 Qwen3.5 的计算图（不用 mlx-lm 的模型代码）、静态预分配的状态和 KV cache（只有 6 层全注意力，GQA 2 个 KV 头，头维 256，KV 很小）、整步编译、融合采样。目标：先把无推测 decode 推到 74+。

**Kernel 层**：
- DeltaNet decode 融合 kernel（卷积窗 + q/k 归一化 + 门 + 递推 + 门控 RMSNorm 一次 dispatch），头配置 (16,16)，MLX PR #4020 未覆盖。
- 输出头 GEMV 与 argmax 融合；bf16 GEMV 针对 M=1 的专用 kernel。
- prefill：分块 DeltaNet、融合 MLP，用 Metal 4 tensor ops 用上 Neural Accelerator。

**编译器层**：为这一个模型写一个小型整图编译器：图 IR → 算子融合 → 生成 Metal 源码（运行时编译）→ 静态调度的 dispatch 序列，把每 token 的提交开销压到最低。

**算法层（突破带宽墙）**：MTP 自推测。难点有三个：
1. DeltaNet 是递推状态，不能像 KV cache 一样直接截断回滚，需要为每个草稿位置保存状态快照。
2. 草稿每步都要读一遍 0.5GB 的输出头，所以草稿用**裁剪词表**的输出头（高频 token 子集），校验仍用完整词表。
3. 批量校验 k+1 个 token 要让 GEMV 变成小 M 的 GEMM，kernel 要对 M=1…8 都高效。

## 5. 环境备忘

- 无 Xcode（只有 Command Line Tools）：没有 `metal`/`metallib` 命令，也没有 GPU 抓帧 / Instruments。运行时编译 Metal 源码（`mx.fast.metal_kernel` JIT）完全可用；缺的只是 GPU profiler，所以性能归因靠自己写的微基准和“整图内多执行一次某个算子”的边际耗时法（`bench/prefill_marginal.py`、`bench/step_ablate.py`）。
- 基准抖动：GPU 与 WindowServer 等共用，墙钟可在 190→330 ms 间摆动、微基准可差 2×。一律用 `bench/quiet.py` 的 `Probe`（只接受前后各一次探针 GEMM 都跑在最好速度附近的样本）+ 进程内交替 A/B（`bench/prefill_cmp.py`）。独立链式微基准会因并发 dispatch 把小的逐元素 kernel 的耗时藏起来（add_norm 微基准 0.15 ms，整图里实际 0.12 ms/次且串行），小算子以整图边际耗时为准。
- 开发环境：Python 3.12，mlx 0.32.4 / mlx-lm 0.32.2，macOS 26.5.1。模型权重放在 `models/Qwen3.5-0.8B/`（`hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B`），开发时已与 Hugging Face 上的 SHA256 校验。

## 6. 里程碑

- [x] M0 环境、基线、实测上限、golden
- [x] M1 自己的模型实现 + 静态缓存 + 整步编译：71.8 tok/s（与 mlx-lm 持平，目标 74 未达；纯框架层收益≈0，瓶颈在 dispatch 与 GEMV 之外的小算子）
- [x] M2 DeltaNet / 输出头 / GEMV kernel：融合 GEMV 家族（norm 前置、residual/SwiGLU/argmax 后置）+ DeltaNet 单 dispatch kernel + 门控 norm 融入 out_proj → **decode 78.2 tok/s**（+9.5%，≈117 GB/s 有效，贴墙；prefill 见 M3，输出头见 M5）
- [x] M3 prefill 与图层融合 → **prefill@2048 6.96k → 11.0k tok/s（1.58×），TTFT@512 49.8 ms，@128 21.0 ms**（`qwen35/prefill.py`、`gdn_nax.py`、`gemm_nax.py`、`nax.py`、`mslib.py`）：
  - **DeltaNet 分块（chunked/WY）NAX kernel**（`gdn_nax.py`）：阶段 A 对 (16-token 块, head) 并行——KKᵀ/QKᵀ 用 Neural Accelerator 的 tensor op，(I+L)⁻¹ 用 fp32 前代精确求（不用 Neumann 近似），写 T/A/门；阶段 B 每个 (head, 16 宽 dv 切片) 一个 simdgroup，128 条独立链 × 128 块步，每步 14 个 NAX 运算。bf16 逐元素语义与 MLX 逐比特一致（`mslib.py`）。DeltaNet 递推 41 ms → 约 11 ms（18 层，0.61 ms/层，A 0.23 + B 0.38）。阶段 B 线程组大小 TG=128（按核均衡，standalone 0.69 → 0.60 ms/层，整图 -1 ms）。
  - **自写 NAX GEMM 家族**（`gemm_nax.py`）：复用 MLX 的 NAX mainloop（运行时内联其头文件，`nax.py`），自带 epilogue——`EPI_SWIGLU`（gate/up 权重按 16 行交错，silu(g)*u 在寄存器里算完，省掉 2048×7168 中间张量，是最大的单项收益）、`EPI_ADD`（残差加融进 out_proj/down，`D = bf16(R + bf16(acc))`，舍入与分开加完全一致）。每个 (层 GEMM, M) 的 tile 配置由 `bench/gemm_tune.py` / `gemm_tune_add.py` 调出，存 `results/gemm_cfg.json`（`<name>+add` 为残差融合版；某些 (name, M) 上 mx.matmul 不输就保留 mx.matmul）。结果确定且与配置无关；与 MLX 非 NAX kernel 只差累加顺序噪声（小 M、M<1500 的 down）。
  - **整图边际耗时（S=2048，ms，机器偏忙时测的，基线 200）**：gate_up GEMM 1.98 ms/层 ≈ 15.2 TFLOPS（已贴 NAX 峰值）；conv_prep 7.2（0.40 ms/层 = 50MB / 125GB/s，已是 DRAM 带宽下限）；gdn 10.9；gated_norm 2.9（带宽下限）；add_norm×48 5.9；SDPA 约 11（MLX 融合 NAX attention，1.7–1.9 ms/层，约峰值 60%）；lm_head 约 4（508MB，带宽墙）。GEMM 合计约 134 ms ≈ 总时间 70%，GEMM FLOPs 屋顶（15.6 TFLOPS）= 131 ms。
  - **残差加融合**：out_proj/down 的 epilogue 直接写 `x+r`，原来的 add_norm（读 x、r，写 xs、h）换成只做 RMSNorm 的 kernel（读 x，写 h），舍入逐比特一致（`bench/fuse_check.py`，S∈{1…2048}）。整图 -1.6~-2 ms（@2048 188.3→186.3），@512 50.0→49.8，@128 21.5→21.0。要点：M=2048 时 out_proj/down 原先走 mx.matmul，必须给 EPI_ADD 单独调 tile 才有收益（用 mx.matmul + 单独加法反而持平）。
  - **NAX 硬件实测**（`bench/nax_ops*.py`）：16×16×32 与 16×32×16 每核每次 ≈13–14.6 ns（≈12 TFLOPS/核集），16×16×16 ≈8.2 ns，操作数 dtype / 转置不影响。阶段 B 每链每步 ≈244 ns，其中 NAX 本身 ≈176 ns（72%）。寄存器压力是硬约束：给阶段 B 加操作数预取（+26 寄存器）反而慢 2.5×；EPI_ADD 在主循环前预取残差 tile 慢 1.7 ms；线程组大小要让 10 个核均衡（4 个 simdgroup/TG 优于 8）。
  - **已验证无效/放弃**：自写递推 kernel（预取、每 simdgroup 多行、qk 合并）慢于 mlx-lm packed kernel；两个 GPU stream 并行 gdn 与 GEMM 无重叠收益；`mx.compile` 整图无收益；conv_prep v2（逐比特一致，无加速，`QWEN35_PREP`）；阶段 B 操作数预取（见上）；SDPA 按 query 分块（4.0 vs 1.94 ms/层，MLX 的无分块调用本来就是融合 NAX attention）；EPI_ADD 残差 tile 预取。
  - **精度底线**：prefill 对 golden 用 teacher-forced 容差（TOL 0.13）把关（`tests/golden.py`、`tests/prefill_long.py`）；GEMM 逐元素在 float64 参照的 1 个 bf16 ulp + fp32 累加噪声内（`tests/gemm.py`）。
  - 剩余空间（不再追，除非有新办法）：full-attn 层的 q/k rmsnorm+rope+转置、`o*sigmoid(gate)` 胶水可各融成一个 kernel（估计 -3 ms）；conv_prep 与阶段 A 融合（-2 ms，复杂）；自写 D=256 SDPA（-2 ms，工作量大）；lm_head 带宽墙见 M5。
- [x] M3b 图编译器 `qwen35/compiler.py`：IR（embed/rmsnorm/linear/add/swiglu/deltanet/gated_norm/attn_core/head_argmax）→ 模式重写融合 pass（rmsnorm→linear 48、gated_norm→linear 18、linear→+residual 48、linear→swiglu 24；261 个节点 → 123 个）→ 权重布局 pass（gate/up 行交错）→ 静态调度 + `mx.compile`。输出与手写 `fast.verify` **逐比特一致**（tests/compiler.py），速度持平（≈78 tok/s，机器负载 5 时抖动 ±2）；关闭融合的 lowering 74 tok/s。目前 prefill 路径和 MTP 草稿路径仍是手写。
- [x] M4 MTP 自推测（GPU 常驻单图一轮：MTP 批量 KV 修复 → 链式草稿(静态 top-16K 词表头) → M=k+1 校验 → DeltaNet 快照 `sel` 回滚）：6 条 prompt、192 token、贪心，**k=2: 159 tok/s；k=3: 172 tok/s**（接受 2.8 token/轮 on 代码/事实，故事类低至 111–134），轮耗时 15–17 ms；golden 前 64 token 与基线一致（仅 prompt 3 在 bf16 精确平局处分叉，与无推测融合路径同位置）。词表 8K→148、16K→159、32K→157（k=2）。
  - 接受率实测（早先）：首草稿 84.7%，链式第 2 草稿 82.6%（用 MTP 自身 post-norm 隐状态；pre-norm 仅 36.7%）；输入顺序 [embed, hidden]。
- [x] M5 **输出头的可证明无损 int8 预筛**（`qwen35/head_prune.py`，`tests/head_prune.py`，开关 `QWEN35_HEAD_PRUNE=0` 关闭）：tied embedding 508 MB 占每 token 流量的 34%，而 decode 只要 argmax。额外存一份 int8 拷贝（每行对称量化，254 MB，`build_tables`），先用 int8 给每个词一个**严格**的 logit 区间，只对区间可能含最大值的少数候选词用原 bf16 行精确重算——结果与读全表的 `kernels.head_argmax` **逐比特相同**（包括平局取最小下标）。
  - **区间**：bf16 logit 的精确值 v_i = BF(fp32 点积)。近似值 ℓ̃_i 由 int8 行算出；每行预存 cp_i = ‖e_i − s_i·q_i‖₂·(1+1e-5) + 1.3e-4·‖e_i‖₂（第二项盖住 fp32 累加误差界 γ₁₀₂₃≈6.2e-5，对精确点积和近似点积各一次）。Cauchy–Schwarz：|v_i − ℓ̃_i| ≤ rad_i = 1.004·cp_i·‖h‖₂ + 0.004·|ℓ̃_i| + 1e-30（0.004 盖住 bf16 舍入 2⁻⁸，1.004 = 1+2⁻⁸ 向上取）。T = max_j (ℓ̃_j − rad_j)；候选 = {i : ℓ̃_i + rad_i ≥ T}，必含 argmax 及其全部并列项。候选数没有上限假设：全词表平局（h=0 或所有行相同）时 248320 个候选走 grid-stride 路径，6.2 ms（普通 kernel 3.8 ms），结果仍正确。
  - **四个 kernel**：K1 int8 近似（写每行上界 ub、每线程组的 lb/ub 最大值）→ K2 收集（每个 m 一个 1024 线程的 TG：由 lbmax 得 T，扫 ubmax，再扫命中 TG 的 ub，原子追加候选）→ K3 精确（与 `_fused_gemv` 完全相同的算术：T=128、R=4、`BF(s)`；grid (G·128, M)，G=64，按候选 grid-stride，TG 内按(值降序, 下标升序)归并）→ K4 终归约（INT_MAX→0）。`tests/head_prune.py` 验证：精确段对全部 248320 行与 `fused_gemv` 的 logit 逐比特相同；区间对每一行成立（真实隐状态、10⁻³…500 的各种尺度随机向量、稀疏/离群向量、与某行量化残差同号的对抗向量）；argmax 与普通 kernel 相等（M=1…4）；重复行/全零/全相等/N 不是 R 的倍数/全负。
  - **候选数（真实隐状态，严格界）**：int8 中位 3 / p99 22 / 最大 48；7 bit 中位 9 / p99 102；6 bit 中位 56 / p99 1197 / 最大 3057；5 bit 中位 3198 / p99 34902；4 bit ≈ 整个词表。实际误差只有 C–S 上界的 ≈2.5%——概率界能用更少的 bit，但那不再是证明，不做。信息论估计该界需要约 7 bit/元素；6 bit（192 MB）还能省 ≈0.5 ms/token，但需要 4+2 位平面或 3 字节打包（ALU 更重），5 bit 需要再加一级精筛，未实现。
  - **K1 性能**：254 MB 约 2.0 ms（≈127 GB/s，与 bf16 kernel 的 ≈132 GB/s 同级，已贴墙）；(T∈{32,64}, R∈{2,4,8,16}) 不敏感，生产配置 T=64、R=8、每线程每行 uint4=16 个 int8、h 放寄存器。M=4 配 T=32 反常地慢（11–19 ms），T=128 对 int8 K=1024 无效。
  - **接入**：`fast.verify` 与编译器的 `head_argmax` 算子末尾、`Engine.prefill` 首 token、`SpecEngine` 首 token、prefill 基准都走 `head_prune.argmax_head`（`prepare`/`flatten_weights` 里建表；无表时回落到 `kernels.head_argmax`）。额外显存：int8 表 254 MB（+ `gate_up_il` 的 352 MB）。
  - **结果**（同机、同 prompt、前后对比）：无推测 decode **84.4 → 95.5–96.3 tok/s**（+13%）；MTP 自推测 k=2 均值 174.3 → 189.7、k=3 188.1 → 201.5（单条最高 274）；`verify` 单轮 M=1 13.60 → 10.56 ms、M=4 16.76 → 14.11 ms；TTFT @512 49.2 → **47.0 ms**、@128 20.6 → **18.5 ms**（进程内交替 A/B，14 轮中位数；省的就是 2.1 ms 的头读取）。全部测试通过：tests/head_prune / gemm / kernels / verify / compiler（编译器路径与手写逐比特一致）/ golden / prefill_long；spec 与 golden 的分叉位置不变（prompt 3 在 bf16 精确平局处）。
  - **没做/没成**：6 bit / 5 bit 头（见上，需要位平面或再加一级精筛，收益 ≤ 0.5–0.75 ms/token）；概率界（比严格界省 bit，但不再是证明）；MTP 草稿头（静态 top-16K 词表，0.26 ms/步）int8 化——不能证明无损且收益边际。
  - **剩余空间（M6 之前的判断，已被 M6 推翻）**：no-spec 96 tok/s，bf16 墙 99.7 tok/s；当时认为再往上只有主干权重无损压缩和 6 bit 头两条路，且 M>1 校验是“GEMV 内循环 ALU/延迟受限”——后一个判断是错的：融合 GEMV 家族在 M=4 和 M=1 一样受内存带宽限制，M>1 的额外代价来自 gnorm 前置的 out_proj 在每个线程组里重复算门控 RMSNorm（R=2→R=8 修掉，verify M=4 −0.39 ms，位级一致）以及 DeltaNet 每多一行要多写 1 MB fp32 状态快照并串行递推（M=1 的 DeltaNet 已在带宽墙 ≈17 µs/层）。
- [x] M6 **主干权重的无损 12 bit/权重压缩**（`qwen35/wcomp.py`，`tests/wcomp.py`，开关 `QWEN35_WCOMP=0` 关闭）：decode 是纯带宽受限的（主干 0.995 GB bf16 每 token 读一遍），而训练出的 bf16 权重指数位熵只有约 2.7 bit（每个矩阵 99.7–99.98% 的权重落在 15 个连续指数 109..123 内）。把每个权重存成 **1 字节（符号|7 位尾数）+ 4 位指数码（exp−109）**，GEMV 内循环在寄存器里用 SWAR 重建**精确的 bf16 位型**，再走原来不变的 fp32 累加——结果与 bf16 kernel **逐比特相同**，读的字节少 25%。
  - **布局**（每行连续，`uint32 Wc[N, 3K/8]` = A 平面 K/4 字 | E 平面 K/8 字）：一个“块”= 一行里连续 8 个权重 = bf16 kernel 里一个线程的一次 uint4 加载。A 平面每块 2 字（`x = b0|b2<<8|b1<<16|b3<<24`，`y` 同理接 b4..b7），E 平面每块 1 字（第 k 对权重的码 `code(2k)<<7 | code(2k+1)<<23`，再循环左移 4k 位）。解码每个 bf16 对 ≈5 条整数指令（`(a&0x007F007F)|((a<<8)&0x80008000)` 取符号和尾数，`(rotate(e,s)&0x07800780)+0x36803680` 取指数并加回偏移 109），仍然带宽受限。
  - **例外块**：任一权重指数落在窗口外的块（占 0.30%，多是 |w|<2⁻¹⁸ 的极小值，全模型 184946 / 62201856 块）原样存进旁表 `X[nx+1,4]`（8 个 bf16 = 4 字），E 字存标记 0xFFFFFFFF，A.x 存 `X` 下标；解码里一个发散分支多一次相关加载——实测代价很小。码 15 保留。总大小 0.995 → **0.749 GB（75.3%）**；各矩阵 75.25–75.34%。
  - **kernel**：`kernels._fused_gemv(..., comp=True)` 只改权重加载（`uint2 ab[R][CPT]` + `uint eb[R][CPT]`）和取权重的一行（`dec_chunk(...)` 代替 uint4），其余（前置 norm、累加顺序、simd_sum、epilogue）原样；`fused_gemv` 接受 bf16 矩阵或 `(Wc, X)` 元组。`fast.prepare` 对 in_proj/out_proj/qkv/o_proj/gate_up_il/down 做压缩（≈2.7 s，numpy 编码；额外 0.75 GB，prefill 仍用原 bf16 `W`），编译器 `layout` pass 对所有融合 linear 节点做同样的压缩。输出头仍是 M5 的 int8 预筛。
  - **单 kernel 实测**（真实权重含例外，M=1…4，位级一致；进程内交替、12 份权重）：in_proj 1.24–1.40×、gate_up 1.28–1.40×、down 1.24–1.27×、out_proj 1.17–1.20×。小 N 的 down/out_proj 受启动/延迟限制（≈80–90 GB/s），对 T/R 不敏感，所以比 25% 带宽理论值（1.33×）低。
  - **整图结果**（compiled verify 交替 A/B，14 轮中位数，质量探针过滤，输出位级相同）：M=1 11.09 → 9.23 ms、M=2 11.60 → 9.71、M=3 12.40 → 10.27、M=4 12.26 → 10.58（每个 M 约 −1.7…−2.1 ms）。decode（5 次中位）**98.7 → 109.7 tok/s（+11%）**；MTP 自推测 6 prompt 均值 k=2 **201.9 → 224.4**、k=3 **217.1 → 240.5**（单条最高 314）；golden / tests/spec 的 token 序列与关压缩时完全相同（prompt 3 仍在 bf16 平局处分叉）。
  - **测试**：`tests/wcomp.py`（编解码往返含随机位型/NaN/Inf/±0/非规格数/全例外/无例外；压缩 GEMV ≡ bf16 GEMV 逐比特，覆盖 M=1..4 × {none, rms, rms+swiglu, residual} × N 非 R 倍数 × 多种 T/R）+ tests/verify（hidden/ssm/conv/kv maxdiff 0.0）+ tests/compiler（编译器与手写逐比特）+ golden/spec/head_prune/gemm/kernels 全部通过。
  - **没成/没做**：权重加载提前（hoist）到前置 norm 之前——无收益，已还原；DeltaNet 惰性快照（≈0.4 ms，复杂）；MTP 草稿头（≈40 MB）压缩——最初测不出收益而还原，后来在草稿步 GEMV 上重做并保留（`mtp.load_mtp` 里 `M[k+"_c"] = wcomp.encode(...)`，每轮 −0.33 ms；批量 prefill 仍用原 bf16）；6 bit 头（−0.5 ms）。
  - **剩余空间**：no-spec 现在 109.7 tok/s；压缩后主干 0.749 GB + int8 头 0.254 GB = 1.003 GB → 带宽墙（124.5 GB/s）≈ 124 tok/s，当前达到 ≈88%；差距主要在小 N 的 down/out_proj（延迟受限）、DeltaNet 与 full-attn 的胶水算子。进一步压到 ≈10.5 bit/权重需要熵编码（变长，难以并行解码），不做。
- [x] M7 **prefill 再推一轮：无损部分触底 + 可选 W8A8 int8 路径**（`qwen35/prefill.py`、`gemm_i8.py`、`tests/w8a8.py`、`tests/gdn_fuse.py`、`bench/gemm_i8_tune.py`）
  - **无损部分（默认）**：prefill@2048 Engine 路径 11.2k → 11.55k tok/s（≈177 ms 含 ≈2 ms int8 头；forward 本体 ≈172 ms），TTFT@512 47.0 → 46.1–46.6 ms。内容：① 最后一层 tail 路径（`forward(..., tail=R)`，最后一层只对要用的末尾 R=128 行算 q/o_proj/MLP，所有行的 K/V 照写；`tests/prefill_tail.py` 对 chunked 与整段逐比特一致）；② conv_prep 与 GDN 阶段 A 融合成 `gdn_nax.prep_a`（q/k 不再落内存再读，`tests/gdn_fuse.py` 验证与分开的 kernel、与整个 forward 开关前后逐比特一致）；③ 首 token 走 M5 的 int8 头。`tests/prefill_long.py` 的语料冻结为 mlx_lm 自带源码（之前依赖本项目源文件，文件一变 prompt 就变，造成假的 BAD）。
  - **确认的无损下限**：GEMM 在 NAX bf16 峰值（M=2048 约 15.2–15.6 TFLOPS，≈129–135 ms），tile 重调已饱和；norm / conv_prep / gnorm 在 DRAM 带宽墙；GDN 阶段 B 受 NAX 吞吐限制（≈72%）。所以无损 prefill 的地板是 ≈165–172 ms，12.5k（164 ms）在 bf16 下够不到。
  - **没成（别重试）**：① full-attn 层 q/k rmsnorm+rope+转置、`o*sigmoid(gate)` 胶水融合成 kernel——逐比特一致，但整图没有收益，已删：MLX 的并发 dispatch encoder 本来就让互相独立的小算子（q/k/v 分支、与 SDPA 并行的 `sigmoid(gate)`）并行跑，融合并不缩短关键路径；SDPA 输出物理布局就是 [S,H,D]，`transpose(0,2,1,3).reshape` 是免费视图；独立链微基准会高估胶水成本。② SLC 驻留/跨 kernel 重叠——无可用杠杆。
  - **W8A8（opt-in，有损）**：NAX 的 int8 路径实测 ≈24–26 TOPS，是 bf16 的 1.5–1.7×，且 int32 累加精确。`gemm_i8.py` 复用 MLX 的 NAX mainloop（int8 输入，`NAXTile<int>` 累加），epilogue 反量化 `bf16((float)acc * (sx[row]*sw[col]))`，或 SwiGLU 版（`mx_silu_bf16`，gate/up 16 行交错，与 bf16 路径同一套舍入）。权重按输出通道对称 int8（scale = amax/127，`prepare_w8a8` 一次性量化，+约 0.35 GB），激活按 token 对称 int8，由前一个 norm 的 kernel（`prefill.norm_q`：与 `norm` 同样的算术，再做行内 amax 归约和量化）顺手产出，不多一趟内存。只覆盖由 norm 直接喂的三个 GEMM：`in_proj`（GDN 层）、`qkv`（full-attn 层）、`gate_up`；`out_proj`/`down` 的输入来自 gated_norm/SwiGLU，需要额外一趟 amax，且 `down` 输入有离群值，精度明显更差（rel-L2 5–7% vs 2–3%），收益估计只有 5–9 ms，不做。最后一层（tail 路径）保持精确。
    - **int8 tile 配置**（`bench/gemm_i8_tune.py` → `results/gemm_i8_cfg.json`，按 (name, M) 最近邻，与 bf16 的 `gemm_cfg.json` 同结构）：M=2048 时 in_proj 1.313 ms（26.3 TOPS，bf16 2.315）、qkv 0.843（25.5，bf16 1.375）、gate_up+swiglu 1.171（25.7，bf16 1.958）；默认的 128×128 tile 反而慢，最优是 64×128×256 的各种 simdgroup 布局。
    - **Hadamard 旋转（QuaRot 式，默认开，`QWEN35_W8A8_ROT=0` 关）**：per-token int8 的误差主要来自激活离群通道（只量化权重时隐状态误差 2.1%，只量化激活 4.2%）。`norm_q` 在量化前对 K=1024 维做 Walsh–Hadamard 变换（线程内 2 级 + simd_shuffle_xor 5 级 + 线程组内存 3 级，/32 归一），权重离线存成 W·H（`gemm_i8.quant_weight`，H 对称正交，乘积不变），把离群值摊到所有通道。不需要标定数据。真实 kernel 的 logits rel-L2（teacher-forced）从 2.8–4.5% 降到 **1.6–3.4%**，隐状态误差降约 40%（真实文本 S=700：6.1%/4.8% → 3.7%/2.6%）；叠加 SmoothQuant 式逐通道缩放（折进 norm 权重）收益只有 0.1–0.2 个百分点，没做。代价：24 次 norm_q 共 ≈ +1.1 ms（@2048）。
    - **整图（进程内交替 A/B，12 轮中位数）**：S=2048 forward 173.8 → **138.1 ms**（−35.6 ms，14.8k tok/s）；S=512 44.7 → **36.1 ms**；S=128 17.6 → 约 13.6 ms。关旋转时 137.0 / 35.4 ms。单开（默认 tile）：in_proj −12.9、qkv −2.2、gate_up −14.3（调优后更多）。Engine 路径（含 int8 头）：128/512/2048 = 8.2k/13.2k/14.6k tok/s，TTFT 15.6/38.7/139.9 ms。chunked prefill（3000、2500 token，chunk 2048）+ 24 token 生成与精确路径完全一致。
    - **精度**（真实 kernel，teacher-forced，48 步解码；脚本思路同 `tests/prefill_long.py`）：开旋转，prefill 末位 logits 相对 L2 误差 code(n=130/500/2000) 2.0/3.4/1.6%，重复段落(512/2000) 1.7/1.7%；首 token 全部一致；解码 argmax 不一致只出现在 logit 差 ≤ 0.125（bf16 精度 1 格）的近平局处（code n=500 的 3/48，其余 0）。关旋转：3.6/4.5/3.6/2.8/3.4%，不一致 4/48 + 1/48（差 ≤ 0.19）。对照：无损路径 rel-L2 ≈ 0.7%。仍比无损路径大约 2–4 倍，所以是显式 opt-in：`QWEN35_W8A8=in_proj,qkv,gate_up`（也可只选其中几个）；短于 `GEMM_MIN_M=128` 行的 prefill 自动回落到精确路径。
    - **测试**：`tests/w8a8.py`——int8 GEMM（plain / swiglu，M∈{37,130,300,2000,2048}，5 种 tile）与精确参照逐比特相同；`norm_q` 的 scale = amax/127、int8 值与（旋转后）逐行量化的 fp64 参照一致（≤1 格且 >99.9% 相同），`quant_weight` 在半个量化步内；整图隐状态相对误差上限；W8A8 跑完后精确路径不变。
  - **剩余空间**：W8A8 之后 forward ≈ 137 ms：out_proj + down 仍是 bf16（≈35 ms），GDN 阶段 A/B ≈ 20 ms，SDPA ≈ 11 ms，norm/conv/gnorm 在带宽墙。再往下要么给 down/out_proj 上 int8（输入是 gated_norm / SwiGLU 产出，需多一趟 amax+量化的 kernel，净收益估计 down ≈ 4–5 ms、out_proj ≈ 2 ms，精度更差；down 的 K=3584=7×512 可用 512 块对角 Hadamard），要么自写 D=256 SDPA（≤ 1.8 ms，工作量大）。

- [x] M8 **bf16 前缀缓存（prefix caching）**（`qwen35/prefix_cache.py`，`Engine.prefill` / `SpecEngine.prefill` 接入，`tests/prefix_cache.py`，`bench/prefix_cache_bench.py`；`Engine(prefix_cache_mb=…, like=…)`，默认关；`QWEN35_W8A8` 打开时旁路并警告一次）
  - **难点**：混合模型的 18 个 GatedDeltaNet 层只有一个递推状态（ssm [1,16,128,128] fp32 ≈ 1 MiB/层 + conv [1,3,6144]），不能像 KV 那样截断到“前 p 个 token”，所以只有**恰好在某个位置做过检查点**的前缀才能复用；6 个全注意力层的 KV 则可以按 token 切段共享。
  - **设计**：检查点树（`Node` = 位置 pos 的完整递推状态 + 本段 KV 行 [parent.pos, pos) + token 数组 + 可选 `last`）。查找 = 对所有节点的 token 数组求最长公共前缀（不是固定网格），取最深的、`lcp == node.pos` 的节点；恢复 = 把链上各节点的 KV 段 `mx.concatenate` 成新的连续 max_len 缓冲（自写 Metal kernel 要求行连续），conv/ssm 直接引用缓存里的数组（MLX 数组是不可变值，不用拷）。整条 prompt 的末尾节点带 `last`（Engine：首 token；SpecEngine：末位隐状态行 + MTP KV），所以完全重复的 prompt 零计算。检查点来源：prefill 的分块边界（chunk=2048）、prompt 末尾（多轮对话下一轮就是上一轮的延伸）、`cache_at=[…]` 手动指定、以及“分裂学习”——新 prompt 与已有节点部分重叠（≥32 token 且比现有命中多 ≥32）时在重叠长度处多加一个检查点（一次性多一个分块边界 ≈ 5 ms），下一个共用 system prompt 的请求就命中。淘汰：按最近使用的叶子 LRU，预算以 MB 计；超过预算的单个节点不存。SpecEngine 额外缓存 MTP 草稿 KV 段（只影响草稿接受率，输出由校验决定）。
  - **数值（关键发现）**：命中后的输出不是与“整段一次 prefill”逐比特一致，而是与“在检查点处分块的 prefill”逐比特一致。原因是 MLX 融合 SDPA（D=256）的配置按 query 长度分档（≤8 / 9–1023 / ≥1024），后缀 <1024 行时和整段 prefill 的舍入不同（末位隐状态相对 L2 ≈1.4e-2，状态 1.1–1.4e-2；GEMM（M≥96，NAX）与 GDN 与 M 无关）；在 [1024,1024] 处切分则与整段逐比特相同。缓存的前缀状态本身与无缓存的前缀计算逐比特相同。这属于已接受的“分块 prefill”一类（`tests/prefill_tail`/`tests/prefill_long` 同类），首 token 全部一致，生成只在 bf16 精确平局处分叉（teacher-forced TOL 0.13）。
  - **内存陷阱（已修）**：conv 状态是 [1,2051,6144] 激活的**切片**，直接存引用会让整块激活（25 MB × 18 层 ≈ 0.6 GiB / 2k token）一直活着，实际占用是账面的 10 倍（基准被 OOM 杀）。现在 `insert` 把 conv/ssm `mx.stack` 成各自一块、`last` 里的数组 `mx.array` 拷贝，账面 = 实际（每个 2k token miss +0.06 GiB，条目 42.6 MiB）。
  - **结果**（`bench/prefix_cache_bench.py`，进程内交替 + 质量探针，10/10 样本安静，同一进程）：2048 token 前缀命中 + 1 / 16 / 64 / 128 / 512 token 后缀的 TTFT = **14.1 / 16.0 / 16.8 / 22.9 / 54.6 ms**，无缓存 185 / 188 / 188 / 196 / 229 ms（**13.2× / 11.7× / 11.2× / 8.6× / 4.2×**）；1024 前缀 13.8–50.7 ms vs 93–131 ms；256 前缀 13.7–49.4 vs 32–73 ms。整条 prompt 重复：@256 1.1 ms、@1024 1.35 ms、@2048 1.7 ms（恢复 ≈ 1.2–1.6 ms ≈ 85 MB 读写，贴近带宽下限；lookup 0.07 ms）。miss 时顺带存档的额外开销 ≈ 0.2–1 ms（异步拷贝）。后缀的 prefill 在 n<96 时走 mx.matmul 回退，权重按 bf16 读一遍（≈10 ms 的地板）。
  - **测试**（`tests/prefix_cache.py`，46 项全过）：重复 prompt token/状态逐比特；延伸命中 ≡ 手动按检查点分块；与无缓存整段 prefill 首 token 相同、状态相对 L2 < 0.06；生成等价（teacher-forced 平局容差）；别名保护（从恢复的状态解码不改变缓存节点）；4500 token 链（2048/4096/4500）和 KV 段字节数；分裂学习、`cache_at`、多轮命中、LRU 淘汰在预算内、超大节点不存、W8A8 旁路；SpecEngine 的重复命中 / miss 路径 ≡ 无缓存 / +13、+150、+900 生成 / 草稿 KV 链恢复。`like=` 让多个 Engine 共用一份权重（测试里 ~10 个 Engine 否则会把 16 GB 机器压进交换区）。
  - **没做**：生成出来的 token 的检查点（多轮时上一轮的回答仍要重算，约 n_answer 个 token 的后缀 prefill）；W8A8 路径的缓存（范围限定为 bf16）；命令行多轮（`run.py --repl`）已接入，但回答本身的 token 每轮仍要重新 prefill；GDN 阶段 B 内部快照以获得免费的细粒度检查点；后缀 <96 token 时用压缩权重的 GEMV 以降低 10 ms 地板。

- [x] M9 **连续批处理（continuous batching）+ ragged 场景**（`qwen35/batch.py`、`qwen35/serve.py`、`tests/batch.py`、`bench/ragged_scenario.py`、`bench/serve_memory.py`；`Engine.prefill_iter`）
  - **池与 kernel**（`batch.py`）：每个 slot 一份状态，三个池原地更新（不拷）：ssm `[S+1,16,128,128]` fp32；conv `[(S+1)*2,3,6144]` bf16，**双缓冲**（读 `pos&1`，写另一半——原地移位会在线程组间竞争）；K/V `[S+1,2,Lmax,256]`，`attn_b` 在原位追加。slot S 是填充行的垃圾槽，`cur[S+1]` 存每个 slot 的下一个输入 token。`deltanet_b`（每行 slot/pos 不同，算术与 M=1 `deltanet` 完全一致）、`attn_b`（一个 1024 线程 TG 对一个 q 头，与 MLX `sdpa_vector` 同一套 simd 归约，除以 sum_exp；1-pass，任意长度）、`blit`（原地拷贝）。批步 `step(cfg,F,P,cur,slots,pos)` = 融合 GEMV 家族（行与 M=1 逐比特一致，`_default_T` 同 T）+ 上述 kernel + int8 certified `argmax_head`；每个 padded 批大小 `mx.compile` 一次（桶 1,2,3,4,6,8,12,16）。顺序约束：MLX 只对数据依赖插 barrier，原位写的先后靠 token 链（下一步消费上一步的 token）保证。
  - **调度器**（`serve.py`）：循环每次迭代 = ① 对所有 DECODE 序列发射一步批 decode（`async_eval`）→ ② 读上一步的 token（主机往返藏在 GPU 后面）→ ③ 推进一个 prefill chunk（chunk 大小是 TTFT↔ITL 的旋钮）。slot 回收即时，EOS 晚一步发现（多算的那步丢弃）。`policy="static"` 作对照（凑满一批，排空后才进下一批）。
  - **请求状态机**：`QUEUED → PREFILL → DECODE → DONE`，另有 `REJECTED`；每次转换记入 `Request.events` 并由 `LEGAL` 表检查。准入严格 FIFO，条件 = 有空闲 slot 且没有别的 prefill 在跑；slot 是**整段 max_len 状态的预留**，所以已准入的请求不会因内存失败，也不做抢占/换出，过载只表现为排队延迟，`max_queue` 把无限积压变成显式拒绝（`queue_full`；prompt 为空或超 max_len → `prompt_too_long`）。结束原因 `eos` / `max_new` / `length`（max_new 被上下文截短）；首 token 即 EOS 时 PREFILL 直接到 DONE。
  - **总显存上限 + 自动 slot 数**：`Server(mem_budget=B, slots="auto")`，预算 = 权重（bf16 + 压缩副本 + int8 头，实测 2395 MiB）+ 前缀缓存预算 + MLX 空闲缓存上限（1 GiB）+ prefill 任务（私有状态 ×2 + 临时空间 450 MiB + 0.4 MiB/chunk token，`bench/serve_memory.py` 标定）+ slack 256 MiB + (slots+1)×slot；slot = 18 层 ssm+双缓冲 conv ≈ 18.7 MiB + KV 12 KiB/token，max_len 4096 时 ≈ 67 MiB。自动 slot 数取 min(放得下的数, 16)（B>16 时 GEMV 已被 ALU 卡住，多 slot 只增加延迟）；放不下 1 个就抛 `MemoryError` 并列出明细。实测 5 GiB 预算：计划 5069 MiB（8 slot），真实峰值 4134 MiB。默认预算 = 物理内存的一半。
  - **MLX 缓冲缓存必须封顶**（`Server(cache_limit=1<<30)`，`bench/ragged_scenario.py` 对所有策略同样设置）：prefill chunk 的激活尺寸多变、无法复用，默认不封顶的空闲缓存涨到 **8.6 GiB**（16 GB 机器），每个 prompt 的最后一个 chunk 随机卡 0.2–1.5 s（内存压力），TTFT p95 在 2.8–5.3 s 间乱跳，还一度被误读成“chunk 256 比 512 好”。封顶后 chunk 扫描稳定。
  - **数值（关键发现）**：分块 prefill 与一次性 prefill 在 bf16 层面不逐比特一致（NAX tile 配置、SDPA 按 query 长度分档），所以服务器默认 chunk 只取决于 prompt（`idle_chunk=None`），对拍基准用同一 chunk 的 `Engine.prefill`。SDPA 在 N≥~4096 切 2-pass，与 `attn_b` 的 1-pass 不再逐比特一致——所以与单序列逐 token 一致的保证限于 N<4096（`max_len=4096` 的默认配置下恰好是上限）。前缀缓存命中改变分块边界，同样改变舍入，所以 `--verify` 关闭缓存；缓存路径由 `tests/batch.py` 单独验证。
  - **GEMV/head 的 M 分块**：`kernels._tiles` 把 M 按寄存器预算切成 ≤ 7//cpt 行一组（M=8 → 4+4），`_reduce_scatter` 是与 `simd_sum` 逐比特相同的 xor 蝶形；`head_prune._approx_kernel` 同样 M 分块（M≥8 时旧版寄存器溢出崩溃：lm_head 3.45 ms@M=8、6.37 ms@M=16，结果仍精确，区间与求和顺序无关所以分块免费）。
  - **正确性**（`tests/batch.py`，全绿）：kernel 与单序列逐比特一致（ragged 位置、slot 乱序、未触碰的 slot/行不被写）；批步 5 条 ragged 序列 + 填充行，编译 + 异步流水后的 token == `Engine.generate`；调度器（continuous / static，11 请求 3 slot、分块 prefill、EOS、slot 复用、1 slot 退化为串行）每个请求 == 同 chunk 的单序列；前缀缓存 + 调度器；状态机（非法转换被拒、`REJECTED`、`queue_full`、结束原因、FIFO、事件序列合法）；内存计划（`slot_bytes×(S+1) == Pools.nbytes()`、自动 slot 数、`max_slots` 封顶、预算不足抛错）。大 B 的 12 / 18 条 ragged 序列 + 填充（M=13 / M=19，覆盖拆分 norm、fp32 x、wf、R=2 分块、NAX 头）与逐条单序列 decode 逐比特一致；`tests/head_prune.py` 的 NAX 段（M∈{6,7,8,12,16,17,24,32}，真实数据 / 缩放 / 稀疏 / 离群 / 对抗 / N=1003 等尾部形状 / 重复行 / 全零）验证区间包含全量 fp32 结果。回归（`tests/*` 十余项）全绿。
  - **ragged 场景**（`bench/ragged_scenario.py`，真实时间回放，`--verify` 对每条序列与单序列 decode 逐 token 比对）：`mixed` = chat（prompt 20–120 / 生成 40–160）、rag（1200–3000 token，一半复用 3 份共享文档）、agent（700 token 共享 system prompt + 40–300 后缀）、long（生成 300–500）按洗牌牌堆混合，泊松到达 + 一次 6 请求突发；`burst` = 16 请求同时到达，prompt 16–3000 对数均匀；`churn` = 2 个 600 token 长生成 + 60 个短请求持续进出。对比 sequential（单序列 FCFS）/ static / continuous（chunk 扫描）。
  - **结果**（8 slot；seed 0；`--verify` 下三种策略每条序列都与单序列逐 token 一致：mixed 79 条、burst 16 条、churn 62 条）：
    - `mixed`（79 请求 / 43.7k prompt token / 8.2k 生成 token）：串行 FCFS makespan 82.6 s，TTFT p50/p95 = 16.5 / 47.6 s（队列积压）；static 2.5 / 6.8 s，ITL p99 18 ms；**continuous chunk 512：0.126 / 0.72 s，ITL p50/p95/p99 = 10.7 / 36.7 / 60 ms，端到端 p50 1.2 s**；chunk 256：0.151 / 0.76 s，ITL p99 42.5 ms。chunk 扫描（128/256/512/1024）：TTFT p50 210/149/128/115 ms，ITL p99 39/43/60/103 ms，吞吐不变——chunk 大 ⇒ TTFT 低、ITL 差。开前缀缓存（chunk 512）：命中 49% prompt token，agent 类 TTFT p50 131 → 55 ms，整体 p95 0.72 → 0.38 s。（负载受到达速率限制，吞吐 ≈ 239 tok/s；`--rate` 调高可压满。）
    - `churn`：static 把短请求堵在长生成后面，TTFT p95 4.79 s、e2e p95 5.43 s；continuous 0.079 s、0.296 s（60× / 18×）。
    - `burst`（16 个 ragged 请求同时到达）：串行 104 tok/s、TTFT p50 9.1 s；static 324 tok/s、2.0 s；continuous 356 tok/s、0.9 s。
    - **批 decode 步时**（`bench/batch_step_bench.py 4096 1200 1,…,16`，pos 600–1200，安静采样中位数，逐行与单序列逐比特一致）：B=1 8.69 ms（115 tok/s）、B=2 9.03（221）、B=4 10.00 ms（400）、B=6 11.77（510）、B=8 12.98（617）、B=12 16.13（744）、B=16 19.91 ms（**804 tok/s**）；基线 B=16 32.3 ms（≈500）、B=4 10.58。批大小 5/7/9–11/13–15 也各有自己的编译桶（`DEFAULT_BUCKETS = 1..16`，整套预热 ≈0.4 s），不再向上填充（一个填充行 ≈0.6–1 ms）。
  - **大 B 的无损优化（500 → 802 tok/s，全部与旧 kernel 逐比特一致；逐项累计 B=16：500 → 568 → 638 → 719 → 745 → 802）**：① 拆分 norm（`rms_rows`/`gnorm_rows`，B≥`SPLIT_MIN`=2；B=4：10.52 → 9.85 ms）：融合的 rms/gnorm 前缀在每个线程组里重算，占 ALU 受限 GEMV 的 ~25%；② 认证 int8 头的 K1 改用 NAX 张量算子（`head_prune.approx_nax`，M∈[6,32]，`QWEN35_HEAD_NAX_MIN`）：int8→bf16 片段精确、乘积在 fp32 里精确，区间与求和顺序无关所以最终 argmax 仍与全量 argmax 逐比特相同，B=16 头 6.3 → 2.8 ms；③ GEMV 分块表 `gemv_tiling(B)`：R=2（每 TG 2 行）+ 均衡的 M 分块（B≥8：mtmax 4 或 5），`bench/gemv_tune_m.py` 扫出（编译器的寄存器分配对 (R, mtmax) 极不单调：同一 kernel bf16 权重 + fp32 x 在 R=2 时 427 µs vs 254 µs）；④ K=1024 的几类 GEMV（in_proj/qkv/gate_up）输入保持 fp32（`rms_rows(f32=True)` + `fused_gemv(xf32)`，每个元素省一次 bf16→fp32 解包，K>1024 的 kind 反而变慢故不用）；⑤ `wf`：权重每个线程只解码一次（`float wfl[R][CPT][8]`），编译器不会跨行块外提解码，B=16 GEMV 再快 10–20%。
  - **与内存地板的距离**（`bench/step_ablate.py`：把某个 kernel 换成廉价桩的边际消融，B=16 的 19.2 ms）：GEMV 9.2 ms（in_proj 2.7 / gate_up 3.4 / down 2.4 / qkv 0.6 / o_proj 0.3 / out_proj 余量）、deltanet_b 4.6 ms、头 ≈2.8 ms、attn_b 1.0 ms、其余（rope/q-k norm/门控/embed）≈1.6 ms。deltanet 读写 fp32 状态 B×18×2 MiB = 576 MiB / 4.6 ms ≈ 130 GB/s，头读 254 MB int8 表 ≥2.0 ms，权重流 ≈6 ms：B≤4 已贴内存地板（B=1 8.4–8.7 ms ≈ 6 + 2 + 0.3），B=16 只剩 GEMV 的 ALU 余量（≈2.5 ms）和小 kernel 的 launch 间隙。**没做/待做**：NAX K1 头的 epilogue 融合（省 dq 往返 ≈0.3 ms）；GEMV 结构（reduce-scatter 开销、K>1024 的 x 转换、wf 下更大的 mtmax）；NAX GEMM 批路径（非逐比特，需走 golden/TOL 判据，只能 opt-in）；（批量推测解码 → M10）；单序列 Engine 的原位 KV（每步省 ≈0.6 ms 的整缓冲拷贝）；N≥4096 的 ragged 注意力；抢占/换出（现在靠预留避免需要）。

- [x] M10 **批量推测解码（continuous batching + MTP 自推测）**（`qwen35/batch_spec.py`、`serve.py` 的推测模式、`tests/batch_spec.py`、`tests/serve_spec.py`、`bench/batch_spec_bench.py` / `batch_spec_parts.py` / `serve_spec_bench.py`）
  - **设计**：每个序列一轮 = MTP 起草 k=2 个 token + 一次 M=3 的批验证（`verify_b`），接受前缀后只推进被接受的位置；输出恒等于逐 token greedy（草稿/MTP 只影响接受率，不影响结果）。M 个链式 token 的递归状态用**快照池**（ssm `[NS+1,M,16,128,128]`、conv `[(NS+1)*2,M,3,6144]`）原位更新：`deltanet_bm` 每个 token 位置写一份快照，接受 n 个就选第 n 份，不需要回滚/重算；`attn_b(M=3)` 在原位追加 K/V，被拒绝的 K/V 行之后被覆盖；`qsubset_argmax_b`（草稿用的 int8 子词表头，一遍扫表出 B 行）与 `mtp_step_b`。每个序列带一份 carry（H 隐状态、token、ids、pos、sel），整轮 `spec_round(…)` 一次 `mx.compile`，每个填充批大小一份。所有行与单序列 `SpecEngine` 逐比特一致（`tests/batch_spec.py`）。
  - **服务器模式**（`serve.py`）：plain ↔ 推测两种解码模式，切换是主机同步的（先排空在途步）。进入推测需要“划算的负载”连续 `spec_dwell`（6）次迭代（空服务器直接进入），离开立即；`hist`（每 slot 的隐状态历史，给 MTP 起草用）由 prefill（blit）、plain 步（`hrec` 原位写）、推测→plain（carry 的 H 行）三处维护；pad 行用垃圾 slot/垃圾 entry（carry 的 pos 钳为 0）。内存计划把推测池算进去（每 slot 的 MTP K/V + hist，外加 `nspec+1` 份 M 个状态的快照 entry；6 GiB 预算、max_len 1024：16 slot → 8 slot，entry 289 MiB）。
  - **成本比与自适应策略**：一轮推测的用时 / plain 一步（k=2，M=3；pos 300–600，`bench/batch_spec_bench.py`）：n=1..8 ≈ **1.17 / 1.41 / 1.47 / 1.61 / 1.86 / 1.90 / 2.09 / 2.29**（plain 一步 8.74 / 9.06 / 9.64 / 10.00 / 10.46 / 11.66 / 12.30 / 12.82 ms；一轮 10.20 / 12.77 / 14.14 / 16.05 / 19.43 / 22.21 / 25.74 / 29.30 ms），所以每序列每轮产出 > 成本比 才划算。代码文本约 2.4–2.9 token/轮，贪心英文/随机 prompt 也有 2.0–2.8（贪心很快陷入重复，MTP 猜得中）。服务器用“每序列每轮 token 数”的滑动估计（0.1 步长）对照 `SPEC_COST` 表：进入需 ≥ 1.1×成本比 + dwell，离开 < 1.0×成本比；在 plain 模式下估计按 0.3%/迭代回归先验 2.3（≈2 s），自动重新探测。默认 `spec_max`=6。
  - **正确性**（`tests/serve_spec.py` 全绿）：7 个 ragged 请求 4 slot（spec_max 2，EOS 落在轮内）、单请求长生成、`max_new` = 1/2/3/7/8 在轮内截断、上下文上限（在途轮越过末尾）、10 请求 6 slot 高流失、抖动（3 波×4 请求，spec_max 1，dwell 1，6 次切换）、static 策略、前缀缓存命中（共享 300 token 前缀）、内存计划、自适应（噪声 prompt 退回 plain；估计被强制压低后在代码上自己恢复）——每个请求的 token 都等于单序列 plain greedy。`bench/ragged_scenario.py mixed --spec 4 --verify`：79 条序列全部逐 token 一致（10 次模式切换）。
  - **服务器级结果**（`bench/serve_spec_bench.py`，代码文本 100 token prompt，生成 200–300，窗口 = 全部 B 个都在 decode 的区间，3 次中位数）：B=1/2/3/4/6/8 推测 285.7 / 458.7 / 609.5 / 721.9 / 760.8 / 785.0 tok/s vs plain 116.9 / 226.1 / 326.1 / 414.6 / 527.8 / 647.2 = **2.44× / 2.03× / 1.87× / 1.74× / 1.44× / 1.21×**（贪心英文 METADATA 文本：B=1 2.26×、B=4 1.64×、B=8 1.15×）。收益取决于接受率：约 1 token/轮时推测反而慢（B=1 ≈ −15%，B=4 ≈ −38%），自适应策略就是为此兜底。
  - **ragged `mixed`**（79 请求，8 slot，chunk 512，前缀缓存开，`--spec 4`；负载由到达速率限制，所以吞吐只能 250.7 → 257.9 tok/s）：e2e p50 / p95 = **810 / 3801 ms → 626 / 2910 ms**（chat 1122 → 711，long 4160 → 2925，agent 480 → 302）；代价是 TTFT p50 / p95 50 / 228 → 56 / 262 ms（一轮比一步长，prefill chunk 与轮交错；模式切换要排空在途步），ITL p99 41.4 → 46.9 ms（一轮的 2–3 个 token 在主机上同时到达，ITL p50 从 9.6 → 10.2 ms 只是度量方式）。GPU 峰值 5782 → 6371 MiB（预算 8 GiB 内）。1304 轮（2.70 token/序列/轮，平均批 1.55）+ 429 个 plain 步，6 次模式切换。
  - **没做/待做**：k=3 的批推测（单序列 k=3 比 k=2 略好，批里成本更高）；R=B·M 验证行的 GEMV 分块调优（现在复用 R=2 表）；按序列自适应（现在是全局估计、全局模式）；N≥4096 的推测；NAX K1 头 epilogue 融合。
