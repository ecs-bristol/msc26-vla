# 面向 Jetson Orin Nano Super 的 VLA 模型部署、优化与基准测试项目总结

日期：2026-06-06

## 项目暂定方向

中文题目：

**面向 Jetson Orin Nano Super 的视觉-语言-动作模型部署、优化与基准测试**

英文题目：

**Benchmarking and Optimising Vision-Language-Action Model Deployment on Jetson Orin Nano Super**

## 核心想法

本项目研究 VLA 模型，也就是 Vision-Language-Action 模型，能否部署到资源受限的嵌入式 GPU 平台上，例如 Jetson Orin Nano Super Developer Kit。

项目第一阶段不接真实机械臂，重点先放在：

- VLA 模型能不能在 Jetson 上跑通
- 哪些模型能跑，哪些模型跑不动
- 推理延迟、内存占用、GPU/CPU 使用率如何
- 通过低精度、量化、输入分辨率调整、模型选择等方式能不能优化
- 最后如果时间允许，再接入主机上的仿真器，形成闭环：仿真器给图像和任务指令，Jetson 输出 action，仿真器执行 action

## 为什么这个方向有价值

现在 VLA 模型在机器人和 embodied AI 里很热门，例如 OpenVLA、SmolVLA、Octo、π0 / OpenPI 等。但这些模型通常很大，很多是在服务器 GPU 上运行的。

Jetson Orin Nano Super 虽然是嵌入式 GPU，但资源仍然有限。它适合 edge AI 和机器人部署，但能不能真正跑 VLA 模型、能跑多快、瓶颈在哪里，并不是一个简单问题。

所以我们的创新点不是“提出一个新的 VLA 模型”，而是：

**在资源受限的嵌入式 GPU 上，对多种 VLA 模型进行系统化部署、性能分析、优化和 benchmark。**

这可以给出一个实际结论：哪些 VLA 模型适合 Jetson，哪些只能远程推理，如何优化部署流程。

## 推荐项目范围

### 第一阶段：Jetson 本地部署和 Benchmark

我们先尝试在 Jetson Orin Nano Super 上部署不同规模的 VLA 或 VLA-style 模型，例如：

- SmolVLA：轻量模型，最优先尝试
- Octo-Small / Octo-Base：参数量较小，适合作为 baseline
- OpenVLA-7B：经典开源 VLA，但很重，作为 stress test
- OpenPI / π0 / π0-FAST：前沿模型，但部署风险较高
- NanoVLA / LiteVLA-Edge：如果代码可用，可以作为 edge-optimised 方向参考

我们不应该一开始承诺 OpenVLA-7B 一定能在 Jetson 上本地实时运行。如果跑不动，这本身也是有价值的 negative result。

### 第二阶段：优化实验

对能跑通的模型做优化：

- FP32 vs FP16 / BF16
- 8-bit 或 4-bit quantization，如果模型支持
- 不同输入图像分辨率
- 不同 batch size，主要是 batch size = 1
- 不同 Jetson power mode
- CPU/GPU memory usage
- 模型加载方式和推理框架对比
- local inference vs remote inference

### 第三阶段：可选仿真闭环

如果时间允许，再做 host simulator + Jetson policy server：

```text
主机模拟器
生成图像 + 任务指令
        ↓
通过网络发送给 Jetson
        ↓
Jetson 运行 VLA 模型
输出 action
        ↓
返回主机模拟器
模拟器执行 action
生成下一帧
```

这个可以用来评估：

- end-to-end latency
- closed-loop control frequency
- action output 是否稳定
- 模型在模拟任务中的基本表现
- Jetson 推理延迟对闭环控制的影响

真实机械臂可以作为最后的 extension，不作为核心承诺。

## 可能的系统架构

第一阶段的离线 benchmark 架构：

```text
输入：
公开机器人数据集 / 摄像头图像 / 仿真器图像
+
语言任务指令

        ↓

Jetson Orin Nano Super
运行 VLA / VLA-style 模型

        ↓

输出：
action vector / action tokens / robot command representation

        ↓

Benchmark logger
记录延迟、内存、GPU 使用率、CPU 使用率、功耗、是否 OOM、action 输出格式
```

如果加入仿真：

```text
Host simulator
→ image + instruction
→ Jetson VLA inference server
→ predicted action
→ simulator executes action
→ next observation
```

## Benchmark 指标

### 部署可行性

- 是否能安装环境
- 是否能加载模型
- 是否 OOM
- 模型加载时间
- 依赖复杂度
- 是否需要 swap

### 推理性能

- 单次 action prediction latency
- average latency
- p95 latency
- actions per second
- GPU utilisation
- CPU utilisation
- RAM / VRAM usage
- power mode 下的性能变化

### 优化效果

- FP16 / INT8 / 4-bit 后的速度提升
- 显存下降幅度
- action 输出是否变化明显
- 输入分辨率对速度和输出稳定性的影响
- 本地推理 vs 远程推理延迟对比

### 如果有仿真闭环

- closed-loop frequency
- simulator step latency
- action validity
- task success rate，如果仿真器支持
- 网络传输延迟
- Jetson 处理时间占总循环时间比例

## 可参考的现有案例

我们已经找到一些相关项目，可以作为依据：

- OpenVLA：开源 VLA 模型，支持 image + instruction → 7-DoF action
- LeRobot async inference：robot client 和 policy server 通过 gRPC 传 observation 和 action
- Tether / Reflex：声称支持把 SmolVLA、π0、GR00T 等 VLA policy 部署到 Jetson Orin 等 edge GPU
- LiteVLA-Edge：研究 VLA 在 Jetson Orin-class hardware 上的 on-device inference
- NanoVLA：专门讨论 VLA 在 Jetson Orin Nano 等资源受限设备上的部署挑战
- Characterizing VLA Models：分析 VLA 在 Jetson Orin / Thor 上的性能瓶颈，指出 action generation 是主要瓶颈之一

这些说明我们的方向不是孤立的，而是当前 edge robotics / embodied AI 里很新的问题。

## 项目贡献点

### 1. 部署贡献

在 Jetson Orin Nano Super 上尝试部署多个 VLA/VLA-style 模型，记录哪些能跑、哪些不能跑，以及具体原因。

### 2. Benchmark 贡献

建立一套系统化 benchmark，包括延迟、内存、GPU/CPU 使用率、功耗、模型加载时间和 action 输出稳定性。

### 3. 优化贡献

尝试低精度、量化、输入分辨率调整、power mode 调整等方法，分析不同优化策略的 trade-off。

### 4. 可选系统贡献

如果时间允许，构建主机仿真器 + Jetson policy server 的闭环测试系统。

## 项目风险

主要风险有：

- OpenVLA-7B 太大，Jetson 本地可能跑不动
- VLA 模型依赖复杂，ARM + CUDA + PyTorch 版本可能不好配
- 有些模型代码只适合 x86 GPU，不适合 Jetson
- 模型输出 action 但没有机械臂时，难以验证真实执行效果
- 仿真闭环可能需要额外时间适配 action space

## 应对策略

我们可以这样控制风险：

- 先从轻量模型开始，例如 SmolVLA、Octo
- OpenVLA-7B 作为 stress test，不作为必须成功项
- 先做 offline dataset replay，不一开始做闭环仿真
- 优先 benchmark 部署可行性和性能指标
- 仿真和真实机械臂都作为 extension
- 如果本地跑不动，可以做 local vs remote inference 对比

## 建议最终项目表述

本项目研究视觉-语言-动作模型在 Jetson Orin Nano Super 这类资源受限嵌入式 GPU 上的部署可行性、性能瓶颈和优化策略。我们将比较多个 VLA/VLA-style 模型，分析模型大小、推理延迟、内存占用、功耗和 action 输出稳定性，并探索低精度、量化和输入分辨率调整等优化方法。如果时间允许，将进一步构建主机仿真器与 Jetson policy server 的闭环 benchmark。

## 一句话版本

我们先不急着接机械臂，而是先把 VLA 模型部署到 Jetson Orin Nano Super 上，系统化测试哪些模型能跑、跑多快、占多少内存、怎么优化；如果时间够，再让主机仿真器给 Jetson 发图像和指令，Jetson 返回 action，形成模拟闭环。
