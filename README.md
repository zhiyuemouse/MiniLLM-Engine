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
