"""The gated delta rule recurrence (GatedDeltaNet) and the q/k normalization, cut out of mlx-lm.

Vendored from mlx_lm/models/gated_delta.py (mlx-lm 0.32.2, MIT License, Copyright (c) 2025 Apple Inc.; see
THIRD_PARTY_LICENSES.md) and cut down to the one case this model has: scalar gate, Dk = 128, fp32 state, no padding mask.
The gate / normalization math and the packed kernel are kept verbatim, so the results are bit-identical to mlx-lm's
(tests/gated_delta.py checks that against an installed mlx-lm); the generic kernels, the vector-gate / masked variants and the
op-by-op fallback are dropped, and anything outside that case raises instead of silently taking another path.
"""
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def normalize_qk(q: mx.array, k: mx.array, *, inv_scale: float, eps: float) -> Tuple[mx.array, mx.array]:
    """L2-normalize q/k and fold the ``head_dim**-0.5`` readout scale into q.

    ``eps`` is the reference l2norm eps, which is added to sum(x^2).
    ``mx.fast.rms_norm`` adds it to mean(x^2), so scale it by ``inv_scale**2``.
    """
    rms_eps = eps * inv_scale**2
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, rms_eps)
    k = inv_scale * mx.fast.rms_norm(k, None, rms_eps)
    return q, k


def _make_packed_kernel():
    """The scalar-gate Dk=128 specialization: eight independent value rows per SIMD-group, four lanes per row, each lane
    keeping 32 contiguous state elements in registers.  Its reductions use an explicit ascending butterfly (xor 1, 2, 4, 8, 16),
    so y and the state are bit-identical to mlx-lm's unpacked simd_sum kernel on current Apple GPUs."""
    source = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_kernel = _make_packed_kernel()


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Gated delta rule recurrence over T tokens.

    q, k: [B, T, Hk, Dk] (already through ``normalize_qk``); v: [B, T, Hv, Dv]; a, b: [B, T, Hv]; state: [B, Hv, Dv, Dk] fp32
    (zeros if None).  Returns (y [B, T, Hv, Dv], new state).
    """
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if Dk != 128 or Dv % 8 or g.ndim != 3 or g.dtype != mx.float32 or state.dtype != mx.float32:
        raise ValueError(f"only the scalar-gate, Dk=128, fp32-state case is supported "
                         f"(Dk={Dk}, Dv={Dv}, g {g.shape} {g.dtype}, state {state.dtype})")
    return _kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv // 8, B * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
