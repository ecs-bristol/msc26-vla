# 本机 VLA/VLM Benchmark 框架

这个文件夹用于在个人电脑 RTX 5060 上先跑通小规模实验。目标不是训练大模型，而是先建立一条可复用的实验链路：

```text
图片 + 任务指令 -> VLM/VLA-like 模型 -> 文本动作/动作标签 -> 记录延迟、显存、输出
```

后面 Jetson Orin Nano Super 到货后，可以复用同一套输入、prompt、日志格式，对比：

```text
Laptop RTX 5060 baseline
Jetson 原始推理
Jetson 量化/优化后推理
```

## 目录结构

```text
Local_VLA_Benchmark_Framework/
  configs/
    models.yaml          # 候选模型和加载参数
    prompts.yaml         # 测试指令和动作输出要求
  data/
    sample_images/       # 放测试图片
    tasks.csv            # 每张图片对应的 prompt 和 expected action
  results/               # benchmark 输出
  scripts/
    check_env.ps1        # Windows 环境检查
    run_sample.ps1       # 单张图片试跑入口
  src/vla_bench/
    env_check.py         # Python/CUDA/GPU 检查
    run_inference.py     # 单次推理
    benchmark.py         # 批量 benchmark
    action_schema.py     # 动作输出格式说明
```

## 推荐先跑的模型

第一阶段建议先从小模型开始：

1. `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
2. `google/paligemma2-3b-mix-224`
3. `Qwen/Qwen2.5-VL-3B-Instruct`

RTX 5060 通常是 8GB 显存，完整 OpenVLA 可能会比较吃紧。建议等云 GPU、学校 GPU 或 Jetson 环境准备好以后，再把 OpenVLA 放进后续实验。

## 安装环境

建议用 Conda 或 venv 创建独立环境。Windows 上可以先这样做：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Local_VLA_Benchmark_Framework"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

先按 PyTorch 官网的 Start Locally 页面安装 Windows + Pip + CUDA 版本的 PyTorch：

```text
https://pytorch.org/get-started/locally/
```

然后安装其余依赖：

```powershell
pip install -r requirements.txt
```

这样可以避免误装成 CPU-only PyTorch。

## 第一步：检查环境

```powershell
.\scripts\check_env.ps1
```

你应该看到：

```text
CUDA available: True
GPU name: NVIDIA GeForce RTX 5060 ...
```

## 第二步：放入测试图片

把 2-5 张图片放进：

```text
data/sample_images/
```

例如：

```text
data/sample_images/desk_cup.jpg
data/sample_images/blocks.jpg
```

## 第三步：单张图片试跑

```powershell
.\scripts\run_sample.ps1 -ImagePath .\data\sample_images\desk_cup.jpg -Prompt "Pick up the red cup."
```

默认模型可以在 `configs/models.yaml` 里改。

## 任务表

当前任务表在：

```text
data/tasks.csv
```

每行包含：

```text
task_id, image, prompt, expected_target, expected_action, success_criteria, notes
```

它用于把每张图片和具体任务绑定起来，避免所有图片都使用同一个通用 prompt。

## 第四步：批量 benchmark

```powershell
python -m src.vla_bench.benchmark `
  --image-dir .\data\sample_images `
  --prompt "Pick up the red cup." `
  --model-key smolvlm2_500m
```

结果会写入：

```text
results/benchmark_results.jsonl
results/benchmark_summary.csv
```

如果想直接跑两个轻量 baseline：

```powershell
.\scripts\run_two_baselines.ps1 -Prompt "Identify the target object and give a short robot action plan."
```

当前两个 baseline 是：

```text
smolvlm2_500m -> HuggingFaceTB/SmolVLM2-500M-Video-Instruct
qwen2_vl_2b   -> Qwen/Qwen2-VL-2B-Instruct
```

## 当前阶段建议记录的指标

- model name
- device
- precision
- input image size
- prompt
- output text/action
- latency
- peak GPU memory
- whether the run succeeded

这些指标后面可以直接放进最终报告和 Jetson 对比表。
