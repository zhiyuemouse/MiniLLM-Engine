# MiniLLM-Engine

用 PyTorch 手写 Qwen3-0.6B 的自回归生成和 KV Cache 实验，在 CPU 上验证分块存储，再到 Nano-vLLM 中修改并测量真实请求调度。

| 阶段 | 实现与验证 | 入口 |
|---|---|---|
| M0 | 手写无缓存贪心生成，对照 Hugging Face 的完整 token IDs | [生成基线](src/simple_autogressive.py) |
| M1 | 接入 Transformers DynamicCache，比较有无缓存的生成结果和延迟，核对 KV 数据量 | [缓存生成](src/my_kv_cache.py)、[容量核对](src/accumulate_kvcache.py)、[计时](src/benchmark_kv_cache.py) |
| M2 | 实现块分配、回收、增长和地址映射，在 CPU 上验证分块读写与单头 attention | [块管理](src/block_manager.py)、[分块存储](src/block_storage.py)、[attention 对照](src/block_attention.py) |
| M3–M5 | 在 Nano-vLLM 中实现可选的连续 prefill 轮数限制，验证调度行为并整理复现实验 | [运行说明](https://github.com/zhiyuemouse/nano-vllm/blob/main/experiments/README.md)、[结果与原始数据](https://github.com/zhiyuemouse/nano-vllm/blob/main/experiments/results/README.md) |

M2 的 attention 实验先按逻辑顺序收集 K、V，再计算 attention，用于验证分块存储的正确性。M3 的改动落在 Nano-vLLM 的配置与调度器上，复用上游的 GPU attention 和缓存实现。

## 目录结构

| 路径 | 用途 | 如何获得 |
|---|---|---|
| `src/` | 手写生成循环、KV Cache 实验和 CPU 分块模拟 | 随本仓库提供 |
| `models/Qwen3-0.6B/` | 模型权重、配置和 tokenizer | 从 [Qwen3-0.6B 模型页](https://huggingface.co/Qwen/Qwen3-0.6B) 下载到此目录 |
| `upstream/nano-vllm/` | 调度器修改、回归测试和 GPU benchmark | 单独克隆 [我的 Nano-vLLM fork](https://github.com/zhiyuemouse/nano-vllm) |

`models/` 和 `upstream/` 是本地运行目录，需要按上表准备。`upstream/` 是这里的目录名，其中的 Nano-vLLM fork 有自己的 Git 历史。要运行调度实验，在项目根目录执行：

```bash
git clone https://github.com/zhiyuemouse/nano-vllm.git upstream/nano-vllm
```

## 运行入口

在项目根目录，用 Python 3.11 运行块管理测试。分块存储与 attention 实验需要 PyTorch，可在 CPU 上运行。

```bash
python src/block_manager.py
python src/block_storage.py
python src/block_attention.py
```

M0/M1 的模型脚本从项目根目录读取本地 `models/Qwen3-0.6B`，并使用 CUDA。Nano-vLLM 实验在 WSL、RTX 3050 Ti Laptop 上验证过。安装版本、模型路径、调度回归测试与 GPU 命令见 [实验运行说明](https://github.com/zhiyuemouse/nano-vllm/blob/main/experiments/README.md)。

## 调度实验结论

策略默认关闭。启用后，连续 prefill 达到设定轮数且有 running 请求可追加时，调度器优先尝试 decode。对于 A、B、C 分别在 `0 / 10 / 50 ms` 到达的三个短请求，正反两种运行顺序下，A/B 的首次 token 间隔均缩短，第二次间隔约翻倍，C 的首 token 延迟上升。两批请求总延迟中位数均下降，降幅分别为 5.83%–6.64% 和 1.24%–1.76%。每种策略在每批记录五轮，结论限于该负载。

调度改动基于 GeeeekExplorer/nano-vllm 的 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 开发，保留 [上游 MIT 许可证](https://github.com/zhiyuemouse/nano-vllm/blob/main/LICENSE)。固定时间实验的源码和数据保存在 fork 的提交 `adb66ac`。

## 算子实验与引擎集成

配套仓库包含 CUDA RowSum、Softmax、GEMM/GEMV 和 Triton RMSNorm 的独立实现、正确性校验与微基准。[MiniLLM-Kernels](https://github.com/zhiyuemouse/MiniLLM-Kernels)

### 接入与正确性

在 Nano-vLLM fork 中，将 Triton fused residual add + RMSNorm 接入 Qwen3 decoder 的 post-attention 层。启用实验开关后，满足形状、设备、BF16 dtype 和连续布局检查的输入走 Triton 路径，返回归一化输出和更新后的 residual。其他输入走原有 RMSNorm 路径。[集成源码与验证脚本](https://github.com/zhiyuemouse/nano-vllm/tree/2a473e9) · [Triton RMSNorm 实验说明](https://github.com/zhiyuemouse/nano-vllm/blob/main/experiments/README.md#triton-post-attention-rmsnorm)

- **算子对照**：按 Nano-vLLM 编译路径观测到的 BF16 舍入顺序实现 Triton 版本。固定输入检查归一化输出误差，并单独检查更新后的 residual。
- **整模型对照**：两组使用相同 prompt，并用 baseline 的贪心 token 推进 decode。比较采样前的 logits 和 top-1。验证覆盖完整 prefill、连续 KV-cache decode，以及 scheduler 分配第二个 KV block 时的 `256 → 257` 边界。所测模型用例的 top-1 一致。
- **CUDA Graph**：验证跨块 decode 的实际 replay。对 batch size 1 和 2，正式 capture 均命中 28/28 层 Triton post-attention RMSNorm。

### 性能结果

整模型实验使用 Qwen3-0.6B、BF16 和三个短请求。A/B 固定调度策略，按相同调度步提交请求。整模型差值为 Triton 中位数减去 baseline 中位数。

| 测量范围 | 结果 |
|---|---|
| 独立算子 | 在 `1/2/14/17/18 × 1024` 五种形状上，配对差中位数均为负，节省约 9–27 µs/次 |
| Eager decode forward | batch size 1 和 2 在正反两种运行顺序下均出现中位数上升，差值约 +0.36 至 +1.23 ms |
| CUDA Graph decode forward | 两种 batch size、两种运行顺序的差值约 +10 至 +22 µs（0.1–0.3%），两组基本持平 |

这组工作负载下，独立算子的收益未转化为可观察的整模型 decode 收益，因此 Triton 路径默认关闭。

### 复现 CUDA Graph A/B

在准备好模型和 GPU 环境的 Nano-vLLM fork 根目录执行。两个命令使用相同的 `step` 到达模式和调度配置。脚本预热 2 轮、测量 5 轮，并在 JSON 中保存按阶段和行数标记的 forward 样本。

```bash
common=(--model-path ../../models/Qwen3-0.6B --max-consecutive-prefills 0
        --arrival-mode step --cuda-graph --measure-forward)

python experiments/benchmark_scheduler_policy.py "${common[@]}" \
  --output experiments/output/rmsnorm_graph_baseline.json
python experiments/benchmark_scheduler_policy.py "${common[@]}" \
  --use-triton-post-attn-norm \
  --output experiments/output/rmsnorm_graph_triton.json
```

反序复测时先跑 Triton，再跑 baseline，并使用不同的输出文件。若要对照 eager 路径，两组都去掉 `--cuda-graph`。
