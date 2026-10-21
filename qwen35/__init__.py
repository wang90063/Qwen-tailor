"""Qwen3.5-0.8B on Apple M5 (MLX): hand-written Metal kernels, a small graph compiler, MTP speculative decoding,
a lossless int8 lm_head prefilter and a continuous-batching server.  See README.md and docs/PLAN.md.

Entry points:  qwen35.engine.Engine (plain decode),  qwen35.spec.SpecEngine (MTP speculative decode, prefix cache),
               qwen35.serve (continuous-batching server),  qwen35.batch_spec (batched speculative decoding)."""
