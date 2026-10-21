# bench/ —— 基准、调参与实验脚本

这里是开发过程中用到的全部脚本，`docs/PLAN.md` 里每个里程碑的数字都来自它们。每个脚本文件头的 docstring 写明了用途和参数，
直接 `python bench/<名字>.py [参数]` 运行即可（需要先 `pip install -e .`，或在仓库根目录加 `PYTHONPATH=.`）。

几点约定：

- **这是一台共享 GPU 的笔记本**：GPU 与 WindowServer 等共用，墙钟抖动可达 2×。可信的对比都用 `quiet.py` 里的 `Probe`
  （只接受前后探针 GEMM 都跑在最佳速度附近的样本）和进程内交替 A/B。想复现 `PLAN.md` 里的数字，请关掉其它占 GPU 的程序。
- **会写文件的脚本**：`baseline.py` → `results/baseline_mlx_lm.json`（golden 贪心输出，`tests/golden.py` 等依赖它）；
  `gemm_tune.py` / `gemm_tune_add.py` → `results/gemm_cfg.json`，`gemm_i8_tune.py` → `results/gemm_i8_cfg.json`（运行时按 (层, M) 查表的 tile 配置）；
  `vocab_freq.py` → `results/token_freq_{corpus,heldout}.npy`（MTP 草稿头的词频表；依赖本机语料，重新生成不保证逐位相同，仓库里提交的才是草稿头实际使用的）。其余脚本只打印。
- 带 `M1` / `M3` 等字样的是早期里程碑的探索脚本，保留是为了让 `PLAN.md` 的结论可查证，不是推荐用法。

## 物理上限与基线

| 脚本 | 内容 |
|---|---|
| `bandwidth.py` | 在 decode 实际用到的形状下实测可达内存带宽 |
| `compute.py` | bf16 GEMM 吞吐（prefill 的算力屋顶） |
| `baseline.py` | 原样 mlx-lm 的基准，并写出 golden 输出 |
| `quiet.py` | 工具模块：安静 GPU 采样（`Probe`） |

## decode（单序列）

| 脚本 | 内容 |
|---|---|
| `decode.py` | 无推测 decode 吞吐（5 次中位数）及与 golden 的分叉 |
| `breakdown.py` | M1 时期的 decode 拆分：主干 / lm_head / argmax |
| `ablate.py` | M1 时期的消融（已被 `step_ablate.py`、`prefill_marginal.py` 取代） |
| `verify_marginal.py` | 校验图里每类算子的边际耗时（M=1 vs M=4） |
| `compiler_bench.py` | 图编译器输出（融合 / 未融合）vs 手写 `fast.verify` |
| `gemv_bench.py`, `gemv_chain.py`, `gemv_m.py` | GEMV kernel 的 tile 扫描、冷缓存链式测量、M=1/2/4 行的融合 GEMV 家族 |
| `dn_m.py` | DeltaNet decode kernel 在 M=1/2/4 行的耗时 |
| `head_prune_bench.py`, `head_prune_stats.py` | 可证明无损的 int8 lm_head 预筛：正确性 + 耗时；真实隐状态上的候选数统计 |

## MTP 自推测

| 脚本 | 内容 |
|---|---|
| `spec_bench.py` | 6 条 prompt 的自推测 decode：tok/s、每轮接受数、轮耗时 |
| `round_breakdown.py` | 一轮推测各部分（校验 M=1/2/4、MTP 步、16K 子词表草稿头）的耗时 |
| `mtp_accept.py`, `mtp_chain.py` | MTP 草稿接受率（贪心；链式草稿） |
| `vocab_freq.py` | 为裁剪词表草稿头生成词频表（`results/token_freq_*.npy`） |

## prefill

| 脚本 | 内容 |
|---|---|
| `prefill.py` | Engine 路径 prefill 吞吐 / TTFT |
| `prefill_cmp.py`, `prefill_ab.py` | 进程内交替对比 prefill 变体 / GEMM 实现 |
| `prefill_marginal.py`, `prefill_ablate.py` | 整图内的边际耗时 / 消融归因 |
| `prefill_breakdown.py`, `prefill_breakdown2.py` | 早期的逐层拆分（stock mlx-lm 算子 / `qwen35.prefill` 辅助函数） |
| `prefill_blocks.py` | 把 prefill 的层内算子按 token 分块是否更快 |
| `attn_breakdown.py`, `fullattn_glue.py`, `sdpa_blocks.py` | 全注意力层：各部分拆分、胶水算子成本、SDPA 分块 |
| `fuse_check.py`, `prep_check.py` | 融合 / 改写必须与原版逐比特一致的检查 |
| `epi_add.py` | 残差 epilogue（`EPI_ADD`）的独立成本 |

## NAX GEMM 与 DeltaNet 分块 kernel

| 脚本 | 内容 |
|---|---|
| `nax_ops.py`, `nax_ops2.py` | NAX 张量算子各形状 / 数据类型的吞吐与延迟 |
| `nax_probe.py`, `nax_probe2.py` | NAX MMA 精度探针（fp32 / bf16 操作数、tf32 软件舍入、hi-lo 拆分） |
| `gemm_nax.py`, `gemm_swiglu.py` | 自写 NAX GEMM vs `mx.matmul`；融合 SwiGLU epilogue |
| `gemm_tune.py`, `gemm_tune_add.py`, `gemm_i8_tune.py` | tile 配置自动调优（bf16 / 残差融合 / int8），写 `results/` |
| `gdn_nax.py`, `gdn_variants.py` | 分块 DeltaNet NAX kernel 的精度与速度；各 flag 变体 |
| `gdn_nax_err.py`, `gdn_nax_real.py`, `gdn_f64.py` | 误差归因；真实层输入上的精度；与 float64 递推比谁更准 |
| `gdn_native.py`, `gdn_native2.py` | MLX 原生 `mx.fast.gated_delta_update` 的速度与精度 |
| `gate_exact.py` | `sigmoid` / `softplus` 的精确 bf16 舍入结构（逐比特对齐 mlx-lm 需要） |
| `overlap.py` | 两个 GPU stream 上 GDN 与 GEMM 能否重叠（结论：不能） |

## 前缀缓存、连续批处理、批量推测

| 脚本 | 内容 |
|---|---|
| `prefix_cache_bench.py` | 有 / 无前缀缓存的 TTFT |
| `batch_step_bench.py` | 批 decode 步耗时与聚合吞吐 vs 批大小 B |
| `step_ablate.py` | 批 decode 步的边际消融（离内存地板还有多远） |
| `step_split_sweep.py`, `step_tile_sweep.py`, `gemv_tune_m.py` | 批 GEMV 的拆分 norm / fp32 输入 / tile 参数扫描与调优 |
| `batch_spec_bench.py`, `batch_spec_parts.py` | 批量推测一轮 vs plain 一步；一轮里时间花在哪 |
| `serve_spec_bench.py` | 服务器级推测 vs plain 吞吐（B 个并发请求） |
| `ragged_scenario.py` | 混合负载回放：串行 / static / continuous 的 TTFT、ITL 对比；`--verify` 逐 token 对拍 |
| `serve_memory.py` | 标定服务器内存模型里 prefill chunk 的临时空间 |
