"""Helper: inline MLX's own NAX (Neural Accelerator / Metal 4 tensor-op) headers so our JIT kernels
(`mx.fast.metal_kernel`) can use the same fragment/MMA wrappers as MLX's precompiled GEMM kernels."""
import re
from pathlib import Path

import mlx

_KDIR = Path(mlx.__path__[0]) / "include"
_RX = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"\s*$')
_cache = {}


def _inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (_KDIR / rel).read_text().splitlines():
        m = _RX.match(line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "\n".join(out)


_DESC = re.compile(r"(transpose_b,\s*)true(,)")


def header(relaxed=True, extra=()):
    """Source of nax.h (+ gated_delta_nax_ops.h macros) with all mlx includes inlined.
    `relaxed` selects matmul2d's relaxed_precision flag (MLX hardcodes true)."""
    key = (relaxed, tuple(extra))
    if key not in _cache:
        seen = set()
        parts = [_inline("mlx/backend/metal/kernels/steel/gemm/nax.h", seen),
                 _inline("mlx/backend/metal/kernels/gated_delta_nax_ops.h", seen)]
        parts += [_inline(e, seen) for e in extra]
        src = "\n".join(parts)
        src, n = _DESC.subn(lambda m: m.group(1) + ("true" if relaxed else "false") + m.group(2), src)
        # the 16x16x16 helper zero-fills the unused half of B with a double literal; make it valid for bf16 operands
        src = src.replace("ct_b[BaseNAXFrag::kElemsPerFrag + i] = 0.0;", "ct_b[BaseNAXFrag::kElemsPerFrag + i] = static_cast<BType>(0.0f);")
        _cache[key] = src + "\nusing namespace mlx::steel;\n"
    return _cache[key]
