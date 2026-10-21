# 第三方许可

本仓库自己的代码是 [MIT](LICENSE)。下面这些代码来自第三方，保留其原许可声明。

## mlx-lm

`qwen35/gated_delta.py` 取自 [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.32.2 的 `mlx_lm/models/gated_delta.py`
（`compute_g`、`normalize_qk`、packed 递推 kernel 逐字保留，其余通用路径已删去）。mlx-lm 的许可全文如下：

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MLX

`qwen35/nax.py` 在运行时读取已安装的 MLX 包里的 NAX 头文件并内联进自己的 kernel，不复制这些文件；MLX 本身是 MIT 许可
（Copyright © 2023 Apple Inc.），作为 pip 依赖安装。
