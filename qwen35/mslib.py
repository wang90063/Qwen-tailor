"""Metal helpers that reproduce MLX's bf16 elementwise ops bit-exactly.

MLX evaluates every elementwise op on a bf16 array in bf16, i.e. rounds after each one (also inside the sigmoid/softplus
implementations), so a fused kernel that rounds once at the end differs from the unfused graph in ~3% of the elements.
Each function takes/returns a float holding a bf16-exact value (verified exhaustively by bench/gate_exact.py)."""

BF16_MATH = r"""
#ifndef BF
#define BF(v) ((float)static_cast<bfloat16_t>(v))
#endif
inline float mx_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == 1.0f) return x;
    return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}
// mx.sigmoid(bf16): 1 / (1 + exp(|x|)), mirrored for x >= 0
inline float mx_sigmoid_bf16(float x) {
    float y = BF(1.0f / BF(1.0f + BF(metal::precise::exp(fabs(x)))));
    return x < 0.0f ? y : BF(1.0f - y);
}
// nn.silu(bf16) = x * sigmoid(x)
inline float mx_silu_bf16(float x) { return BF(x * mx_sigmoid_bf16(x)); }
// nn.softplus(bf16) = logaddexp(x, 0)
inline float mx_softplus_bf16(float x) {
    return BF(max(x, 0.0f) + BF(mx_log1p(BF(metal::precise::exp(-fabs(x))))));
}
"""
