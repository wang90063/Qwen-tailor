"""Lossless 12 bit/weight storage of bf16 weight matrices for the decode / verify GEMV (-25% bytes streamed).

A bf16 weight is sign | 8-bit exponent | 7-bit mantissa.  Trained weights use very few exponents (entropy ~2.7 bit; 99.7-99.98% of
the weights of every matrix lie in the 15 values 109..123), so a weight is stored as
    byte  = sign | mantissa          (8 bit, "A plane")
    code  = exponent - BASE          (4 bit, "E plane"; 0..14, 15 reserved)
and the GEMV rebuilds the exact bf16 bit pattern in registers (SWAR, ~5 integer ops per weight) before the unchanged fp32
accumulation, so the result is bit-identical to the bf16 kernel.

Row-contiguous layout of a [N, K] matrix: uint32 array Wc [N, 3K/8] = per row  A plane (K/4 words) | E plane (K/8 words).
A "chunk" is 8 consecutive weights of a row (what one thread loads as one uint4 in the bf16 kernel):
    A (2 words) : x = b0 | b2<<8 | b1<<16 | b3<<24 ,  y = b4 | b6<<8 | b5<<16 | b7<<24      (b_i = byte of weight i)
    E (1 word)  : for pair k (weights 2k, 2k+1):  rotl( code(2k)<<7 | code(2k+1)<<23 , 4k )
Chunks with any exponent outside the window (0.25-0.5% of the chunks, mostly tiny |w| < 2^-18) are stored verbatim in the side
table X [nx+1, 4] (8 bf16 as 4 words); their E word is the marker 0xFFFFFFFF (never produced by a normal chunk) and A.x holds the
index into X.  A weight matrix in this format is the tuple (Wc, X); `kernels.fused_gemv` accepts it wherever it takes a bf16 matrix."""
import os
import numpy as np
import mlx.core as mx

BASE = 109
ENABLED = os.environ.get("QWEN35_WCOMP", "1") != "0"
MARK = 0xFFFFFFFF

# Metal: rebuild the 4 bf16 pairs (= the original uint4 chunk) from (A, E); X is the verbatim table for marker chunks
DEC = r"""
inline uint4 dec_chunk(uint2 a, uint e, const device uint4* X) {
    if (e == 0xFFFFFFFFu) return X[a.x];
    const uint K109 = 0x36803680u, MK = 0x07800780u;
    uint4 w;
    w.x = ((a.x & 0x007F007Fu) | ((a.x << 8) & 0x80008000u)) | ((e & MK) + K109);
    w.y = (((a.x >> 8) & 0x007F007Fu) | (a.x & 0x80008000u)) | ((rotate(e, 28u) & MK) + K109);
    w.z = ((a.y & 0x007F007Fu) | ((a.y << 8) & 0x80008000u)) | ((rotate(e, 24u) & MK) + K109);
    w.w = (((a.y >> 8) & 0x007F007Fu) | (a.y & 0x80008000u)) | ((rotate(e, 20u) & MK) + K109);
    return w;
}
"""
assert BASE == 109            # K109 above = (109 << 7) | (109 << 23)


def is_comp(W):
    return isinstance(W, tuple)


def shape_of(W):
    """(N, K) of a matrix given either as a bf16 array or as a compressed (Wc, X) tuple"""
    if is_comp(W):
        return W[0].shape[0], W[0].shape[1] * 8 // 3
    return W.shape


def _rotl32(v, n):
    n %= 32
    return ((v << np.uint32(n)) | (v >> np.uint32(32 - n))) if n else v


def encode(w):
    """bf16 [N, K] (mx array) -> (Wc uint32 [N, 3K/8], X uint32 [nx+1, 4]), lossless."""
    u = np.array(w.view(mx.uint16)).astype(np.uint32)
    N, K = u.shape
    assert K % 8 == 0
    nch = K // 8
    code = ((u >> 7) & 0xFF).astype(np.int64) - BASE
    bad = (code < 0) | (code > 14)
    chbad = bad.reshape(N, nch, 8).any(-1)
    code = np.where(bad, 0, code).astype(np.uint32).reshape(N, nch, 8)
    byte = (((u >> 8) & 0x80) | (u & 0x7F)).reshape(N, nch, 8)
    ax = byte[..., 0] | (byte[..., 2] << 8) | (byte[..., 1] << 16) | (byte[..., 3] << 24)
    ay = byte[..., 4] | (byte[..., 6] << 8) | (byte[..., 5] << 16) | (byte[..., 7] << 24)
    e = np.zeros((N, nch), np.uint32)
    for k in range(4):
        e |= _rotl32((code[..., 2 * k] << 7) | (code[..., 2 * k + 1] << 23), 4 * k)
    idx = np.flatnonzero(chbad.reshape(-1))
    nx = len(idx)
    X = np.zeros((nx + 1, 4), np.uint32)
    if nx:
        uu = u.reshape(N, nch, 8).reshape(-1, 8)[idx]
        X[:nx] = uu[:, 0::2] | (uu[:, 1::2] << 16)
        ef, axf, ayf = e.reshape(-1), ax.reshape(-1), ay.reshape(-1)
        ef[idx] = MARK
        axf[idx] = np.arange(nx, dtype=np.uint32)
        ayf[idx] = 0
    a = np.stack([ax, ay], -1).reshape(N, nch * 2)
    return mx.array(np.concatenate([a, e], axis=1)), mx.array(X)


def decode(Wc, X):
    """numpy reference decoder: (Wc, X) -> bf16 bit patterns uint16 [N, K] (for tests)"""
    Wc = np.array(Wc); X = np.array(X)
    N = Wc.shape[0]
    K = Wc.shape[1] * 8 // 3
    nch = K // 8
    a = Wc[:, :K // 4].reshape(N, nch, 2)
    e = Wc[:, K // 4:]
    out = np.zeros((N, nch, 8), np.uint32)
    for k in range(4):
        aw = a[..., k // 2]
        if k % 2 == 0:
            sm = (aw & 0x007F007F) | ((aw << np.uint32(8)) & 0x80008000)
        else:
            sm = ((aw >> np.uint32(8)) & 0x007F007F) | (aw & 0x80008000)
        ec = (_rotl32(e, 32 - 4 * k) & 0x07800780) + 0x36803680
        p = sm | ec
        out[..., 2 * k] = p & 0xFFFF
        out[..., 2 * k + 1] = p >> 16
    mk = e == MARK
    if mk.any():
        xi = a[..., 0][mk]
        xv = X[xi]
        out[mk] = np.stack([xv & 0xFFFF, xv >> 16], -1).reshape(-1, 8)
    return out.reshape(N, K).astype(np.uint16)
