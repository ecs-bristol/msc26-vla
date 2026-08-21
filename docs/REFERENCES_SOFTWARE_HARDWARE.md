# 软件与硬件优化参考清单

> 说明：整理自 `docs/LITERATURE_VLA_OPTIMIZATION.md` 与硬件优化选型时的检索。
> 软件线对应已完成/进行中的实验 A（步数）与实验 B（量化）；硬件线对应
> TensorRT + CUDA Graphs 方案。标注“官方文档/教程”的条目不是论文，引用时按资源类型处理。

## 1. 软件层优化

### 1.1 步数 / 蒸馏（Flow Matching）

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation | arXiv:2604.05656 | 支撑“去噪占端到端 80% 延迟”与 `num_steps` 缩减的动机；SmolVLA 专属加速 |

### 1.2 量化

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| ActQuant: Sub-4-bit Action-Guided Quantization for VLA Models | arXiv:2605.24011 | LIBERO 上 3 bpw 保持 95% 成功率，量化方法直接支撑 |
| SQIL: Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control | arXiv:2505.15304 (ICCV 2025) | 4-bit 量化 + Jetson 实测加速/能效 |
| BitVLA: 1-Bit Vision-Language-Action Models for Robotics Manipulation | arXiv:2506.07530 | 极低比特量化可行性边界 |
| LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics | arXiv:2603.03380 | Jetson 上 4-bit GGUF 路线先例（也属硬件线） |

### 1.3 剪枝 / Token 缩减

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| EfficientVLA: Training-Free Acceleration and Compression for VLA Models | arXiv:2506.10100 (NeurIPS 2025) | 免训练层/路径剪枝 |
| EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration | arXiv:2602.00780 (ICML 2026) | 通道剪枝 + 推理调度 |
| ADP: Action-aware Dynamic Pruning for Efficient VLA Manipulation | arXiv:2509.22093 (ICLR 2026) | LIBERO 同基准的动态剪枝 |
| SpecPrune-VLA: Accelerating VLAs via Action-Aware Self-Speculative Pruning | arXiv:2509.05614 (ICML 2026) | 免训练两级剪枝 |
| DepthCache: Depth-Guided Training-Free Visual Token Merging | arXiv:2603.10469 | 视觉 token 合并 |
| Drop-Then-Recovery: How Redundant Are VLA Models? | arXiv:2606.27755 | 语言主干冗余度最高，支撑语言侧优化 |
| Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered | arXiv:2510.08464 | 剪枝负面证据，报告需谨慎引用 |

### 1.4 综述与基线模型

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| A Survey on Efficient Vision-Language-Action Models | arXiv:2510.24795 | related work 主线引用 |
| SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | arXiv:2506.01844 (Microsoft) | 项目基线模型本身 |

## 2. 硬件层优化

### 2.1 TensorRT / INT8（Jetson 官方加速路径）

| 资源 | 类型 | 与本项目的关系 |
| --- | --- | --- |
| NVIDIA TensorRT 官方文档 | 官方文档 | Jetson/Orin 部署与 INT8 PTQ 标准流程 |
| GR00T-N1.6-bridge-INT8-Edge（Hugging Face） | 官方示例模型 | TensorRT INT8 VLA 部署示例 |
| Seeed Studio Jetson GR00T TensorRT 教程 | 教程 | Jetson 上 VLA + TensorRT 实操参考 |
| SQIL（arXiv:2505.15304） | 论文 | Jetson AGX Orin 上量化部署与能效实测 |

### 2.2 CUDA Graphs

| 资源 | 类型 | 与本项目的关系 |
| --- | --- | --- |
| NVIDIA CUDA Graphs 官方文档 | 官方文档 | 减少 kernel launch / CPU-GPU 同步开销的标准方法 |
| Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs | arXiv:2312.10351，2024 | 以顺序 CUDA Graph 为基准，说明其消除 launch 开销；也用于报告 CUDA Graphs 的参考 |

### 2.3 GGUF / llama.cpp 路线（备选）

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics | arXiv:2603.03380 | Jetson 4-bit GGUF 推理先例；TensorRT 不可行时的备选 |

### 2.4 DLA / 工作负载映射（评估后排除）

| 文献 | 出处 | 与本项目的关系 |
| --- | --- | --- |
| Deep Learning Workload Mapping Optimization on Jetson Platforms（JDIMO） | ACM TACO 22(2)，2025，DOI: 10.1145/3736175 | DLA/硬件分区优化研究；作为“DLA 不适合 SmolVLA 动态算子”的排除依据 |

### 2.5 边缘 VLA 架构与功耗测量

| 资源 | 类型 | 与本项目的关系 |
| --- | --- | --- |
| NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies | arXiv:2510.25122 | Orin Nano 上 VLA 优化的 motivation 支撑 |
| EdgeVLA-Tiny / MiniVLA | 社区模型 | 更轻模型对照讨论（非正式文献） |
| NVIDIA `nvpmodel` / `tegrastats` | 官方工具 | 功耗、温度、GPU 利用率采集 |

## 3. BibTeX 简版

```bibtex
@article{actquant,
  title={ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models},
  author={Akbari, Arash and others},
  journal={arXiv:2605.24011}, year={2026}}
@article{snapflow,
  title={SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation},
  author={Luan, Wuyang and others},
  journal={arXiv:2604.05656}, year={2026}}
@article{sqil,
  title={Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control},
  journal={arXiv:2505.15304}, year={2025}}
@article{bitvla,
  title={BitVLA: 1-Bit Vision-Language-Action Models for Robotics Manipulation},
  journal={arXiv:2506.07530}, year={2025}}
@article{litevlaedge,
  title={LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics},
  author={Williams, Justin and others},
  journal={arXiv:2603.03380}, year={2026}}
@article{efficientvla,
  title={EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models},
  journal={Advances in Neural Information Processing Systems}, year={2025}}
@article{ecovla,
  title={EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models},
  journal={arXiv:2602.00780}, year={2026}}
@article{adp,
  title={Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation},
  author={Pei, Xiaohuan and others},
  journal={arXiv:2509.22093}, year={2025}}
@article{specprunevla,
  title={SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning},
  journal={arXiv:2509.05614}, year={2025}}
@article{depthcache,
  title={DepthCache: Depth-Guided Training-Free Visual Token Merging for Vision-Language-Action Model Inference},
  journal={arXiv:2603.10469}, year={2026}}
@article{dtr,
  title={Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models?},
  journal={arXiv:2606.27755}, year={2026}}
@article{gluestick,
  title={Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered},
  journal={arXiv:2510.08464}, year={2025}}
@article{evla-survey,
  title={A Survey on Efficient Vision-Language-Action Models},
  journal={arXiv:2510.24795}, year={2025}}
@article{smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  journal={arXiv:2506.01844}, year={2025}}
@article{nanovla,
  title={NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies},
  journal={arXiv:2510.25122}, year={2025}}
@article{opara,
  title={Opara: Exploiting Operator Parallelism for Expediting DNN Inference on GPUs},
  journal={arXiv:2312.10351}, year={2024}}
@article{jdimo,
  title={Deep Learning Workload Mapping Optimization on Jetson Platforms},
  author={Wang, Farui and Hao, Meng and Yang, Shuang and Zhang, Wei},
  journal={ACM Transactions on Architecture and Code Optimization},
  volume={22}, number={2}, year={2025},
  doi={10.1145/3736175}}
```

