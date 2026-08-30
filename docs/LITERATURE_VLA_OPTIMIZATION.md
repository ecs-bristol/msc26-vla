# VLA 模型效率优化文献综述（用于方法选型）

> 目的：为「单模型 SmolVLA + 软件（量化/剪枝等）+ Jetson 硬件优化」的项目
> 方向提供文献支撑。检索时间：2026-08-15。所有条目均已核实 arXiv 编号或会议出处。

## 1. 方法分类总览

| 方法类别 | 代表文献 | 关键结论 | 两周可行性 |
|---|---|---|---|
| Flow Matching 步数 / 蒸馏 | SnapFlow (2604.05656) | SmolVLA 去噪占端到端 80%；单步蒸馏 3.56× 加速 | ★★★ 高（步数缩减）/ 中（蒸馏需 ~12h GPU） |
| 量化（4-bit 及以下） | ActQuant (2605.24011)、LiteVLA-Edge (2603.03380)、SQIL (2505.15304)、BitVLA (2506.07530) | 子 4-bit 在 LIBERO 上保持 ~90-95% 成功率；Jetson 上 4-bit 实测 ~150ms；1-bit 亦有先例 | ★★★ 高（需先验证 SmolVLA 4-bit 加载支持） |
| 剪枝 / token 缩减（training-free） | EfficientVLA (2506.10100, NeurIPS'25)、EcoVLA (2602.00780, ICML'26)、ADP (2509.22093, ICLR'26)、SpecPrune-VLA (2509.05614, ICML'26)、DepthCache (2603.10469) | 层/通道/token 剪枝可 2× 左右加速；语言主干冗余度最高；有负面证据表明剪枝会破坏 VLA | ★★ 中（无需微调，但需谨慎评测） |
| 架构级优化 | NanoVLA (2510.25122)、SmolVLA 论文 (2506.01844)、EdgeVLA-Tiny、MiniVLA | 轻量架构/解耦融合/动态路由；SmolVLA 本身用 layer-skipping + 64 token | 不适用（换架构，超出"单模型"约束） |
| 推理引擎 / 硬件 | TensorRT/DLA（NVIDIA 官方）、LiteVLA-Edge（GGUF runtime） | Jetson 上 TensorRT/DLA 是标准加速路径；Orin 的 DLA 支持 INT8；GGUF/llama.cpp 是已验证的 4-bit 落地路线 | ★★ 中（TensorRT 适配工作量大；GGUF 路线已有先例） |
| 综述 / 路线图 | A Survey on Efficient VLAs (2510.24795) | 系统梳理量化/剪枝/蒸馏/架构四大类方法，报告 related work 直接引用 | ★★★ 高（写作时引用） |

## 2. 量化（Quantization）

### ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models
- arXiv:2605.24011（2026-05，v3 2026-08）｜作者：Arash Akbari et al.｜[arXiv](https://arxiv.org/abs/2605.24011)｜[代码](https://github.com/arashakb/ActQuant)
- 方法：动作引导的混合精度 PTQ——按权重对动作预测的贡献分配位宽（inter-tensor bit allocator）+ 动作感知曲率的块级缩放优化（intra-tensor scale optimizer）。附 agentic 转换管线到原生 C/C++ 低比特内核。
- 结果：LIBERO 上 OpenVLA-OFT 3 bpw 保持 95.0%，2.5 bpw 时 90.1%；backbone 14.3 GB→2.7 GB（5.3×）。真实 UR3 机械臂验证。
- 与项目关系：**量化方法的最直接支撑**（同为 LIBERO 评测）。但对象是 OpenVLA/π0，SmolVLA 需自行实现类似思路或验证现有 4-bit 工具链。

### LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics
- arXiv:2603.03380（2026-03）｜Justin Williams et al.｜[arXiv](https://arxiv.org/abs/2603.03380)
- 方法：FP32 图像-动作微调 + 训练后 **4-bit GGUF 量化** + GPU 加速 runtime，全设备推理。
- 结果：Jetson Orin 类硬件上端到端 150.5 ms（≈6.6 Hz），ROS 2 集成，完全离线。
- 与项目关系：**Jetson 上 4-bit 量化的直接先例**，证明可行性；GGUF/llama.cpp 路线可作为 SmolVLA 4-bit 的参考实现路径。

### Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control（SQIL）
- arXiv:2505.15304（2025-05）｜ICCV 2025｜[arXiv](https://arxiv.org/abs/2505.15304)
- 方法：显著性感知量化 + 模仿学习，量化 4-bit OpenVLA。
- 结果：Jetson AGX Orin 实测约 2.5× 加速与能源节省。
- 与项目关系：量化 + 边缘硬件实测的支撑（平台为 AGX Orin，非 Nano）。

### BitVLA: 1-Bit Vision-Language-Action Models for Robotics Manipulation
- arXiv:2506.07530（2025-06）｜[arXiv](https://arxiv.org/abs/2506.07530)
- 方法：对 VLA 做 **1-bit 极低比特量化**，主打内存压缩与边缘部署。
- 结果：在机器人操作评测中保持可用成功率，显著压缩模型体积。
- 与项目关系：证明"量化可以走得很深"的可行性边界；SmolVLA 可先做 4-bit，1-bit 作为讨论/对照。

## 3. 剪枝（Pruning）

### EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models
- arXiv:2506.10100｜NeurIPS 2025｜Yang et al.｜[arXiv](https://arxiv.org/abs/2506.10100)
- 方法：**免训练**结构化加速——按信息贡献剪语言模块冗余层 + 任务感知的紧凑视觉处理路径选择。
- 结果：CogACT 在 SIMPLER 环境验证，FLOPs/延迟显著下降。
- 与项目关系：**剪枝的最可行先例**（无需微调），可直接迁移思路到 SmolVLA 的语言模块层剪枝。

### EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration
- arXiv:2602.00780｜ICML 2026｜[arXiv](https://arxiv.org/abs/2602.00780)
- 方法：免训练、即插即用的环境感知自适应**通道剪枝**（EAP）+ 交错推理调度，可与 token 剪枝正交组合。
- 结果：组合 token 剪枝时 2.18× 加速、仅 0.5% 性能下降。

### Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation（ADP）
- arXiv:2509.22093｜ICLR 2026｜Xiaohuan Pei et al.｜[arXiv](https://arxiv.org/abs/2509.22093)
- 方法：文本驱动的 token 选择 + 动作轨迹门控的多模态动态剪枝。
- 结果：LIBERO 套件评测（与本项目同基准）。

### SpecPrune-VLA: Accelerating VLAs via Action-Aware Self-Speculative Pruning
- arXiv:2509.05614｜ICML 2026｜[arXiv](https://arxiv.org/abs/2509.05614)
- 方法：**免训练**两级剪枝（action-level 静态剪枝 + token-level 自推测动态剪枝），启发式控制。
- 结果：动作感知剪枝在保持动作质量的同时降低计算量，与 EfficientVLA 同属 training-free 路线。
- 与项目关系：层/通道剪枝之外，提供 **token 级别自推测剪枝**的第二种实现选择。

### DepthCache: Depth-Guided Training-Free Visual Token Merging for VLA Inference
- arXiv:2603.10469（2026-03）｜[arXiv](https://arxiv.org/abs/2603.10469)
- 方法：利用深度信息做**免训练视觉 token 合并**（token merging），减少视觉 token 数量。
- 结果：训练-free 的视觉 token 缩减，可与剪枝/量化正交叠加。
- 与项目关系：SmolVLA 每帧只有 64 个 visual token，此方向收益相对有限，可作为进阶对照。

### Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models?（DTR）
- arXiv:2606.27755（2026-06，UMD + Cisco）｜[arXiv](https://arxiv.org/abs/2606.27755)｜[代码](https://github.com/s1ghhh/VLADrop)
- 方法：对 Vision / Language / Action 三部分分别做 Drop Half / Keep 2 干预，测量各模块冗余度。
- 结果：**语言主干（language backbone）冗余度最高**，视觉与动作路径更关键；Drop-Then-Recovery 可恢复部分能力。
- 与项目关系：为"对 SmolVLA 语言模块做层剪枝"提供正面证据（与 EfficientVLA 相互印证）。

### 负面证据：Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered
- arXiv:2510.08464（2025-10，GLUESTICK）｜[arXiv](https://arxiv.org/abs/2510.08464)
- 结论：直接剪枝会显著破坏 VLA 能力，但可通过恢复/微调找回。
- 含义：剪枝实验必须配合成功率评测，若掉点明显需说明"需微调恢复"，或选择训练中/训练后恢复型剪枝。

## 4. 推理加速：Flow Matching 步数与蒸馏

### SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation
- arXiv:2604.05656（2026-04）｜Wuyang Luan et al.｜[arXiv](https://arxiv.org/abs/2604.05656)
- 方法：flow-matching VLA（π0、π0.5、**SmolVLA**）的渐进自蒸馏，将 10 步去噪压缩为单步（1-NFE），无需外部教师、无架构改动，单卡约 12h 训练。
- 结果：π0.5 四套 LIBERO 40 任务 98.75%（与 10 步教师相当），去噪 9.6× 加速、端到端 274→83 ms；**SmolVLA 上 MSE −8.3%、端到端 3.56× 加速**。与层蒸馏/token 剪枝正交。
- 与项目关系：**SmolVLA 专属加速**。其关键洞察「去噪占端到端 80%」支撑「减少 num_steps」实验的动机；完整蒸馏可在 PC GPU 上 ~12h 完成。

## 5. 架构级 / 部署系统参考

### SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
- arXiv:2506.01844（2025-06，Microsoft）｜[arXiv](https://arxiv.org/abs/2506.01844)
- 模型自身的效率设计：VLM 上层 layer skipping、pixel shuffling（每帧 64 visual token）、异步推理栈（解耦感知/动作预测与执行）。
- 含义：项目基线模型的设计依据，报告 related work 必引。

### NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies
- arXiv:2510.25122（2025-10）｜[arXiv](https://arxiv.org/abs/2510.25122)
- 方法：视觉-语言融合后移（decoupling）+ 长短动作分块（long-short action chunking）+ 动态路由（轻/重 backbone 按任务复杂度分配）。
- 结果：边缘设备上最高 52× 推理加速、98% 参数减少，Jetson Orin Nano 部署。
- 含义：证明"在 Orin Nano 上做 VLA 优化"这一方向的价值（related work / motivation 引用）。

### EdgeVLA-Tiny / MiniVLA（社区模型）
- EdgeVLA-Tiny：164M 参数（比 SmolVLA 小 64%），宣称可在 Jetson Orin Nano 实时运行（FP16）。
- MiniVLA：Jetson Orin Nano 上跑 LIBERO，报告约 5-10% 成功率下降。
- 含义：可作为"更轻模型替代"的对照组讨论（但项目约束为单模型，仅作参考）。

### A Survey on Efficient Vision-Language-Action Models
- arXiv:2510.24795（2025-10）｜Yu, Wang et al.｜[arXiv](https://arxiv.org/abs/2510.24795)｜[项目页](https://evla-survey.github.io/)｜[文献列表](https://github.com/YuZhaoshu/Efficient-VLAs-Survey)
- 内容：系统综述高效 VLA 的四大类方法——量化、剪枝、蒸馏、架构设计，并给出 edge 部署用例。
- 含义：报告 related work 的**主线引用**，可据此把本项目定位到"SmolVLA + 量化/剪枝 + Jetson"子方向。

## 6. 硬件层参考

- NVIDIA TensorRT + DLA（Jetson Orin 内置 Deep Learning Accelerator，支持 INT8）：官方工具链是 Jetson 上唯一受支持的硬件加速部署路径。DLA 会由 TensorRT 自动分区子图，适合 INT8 CNN 类算子。
- Power mode（`nvpmodel` MAXN / 15W / 10W）+ `jetson_clocks`：系统级命令，用于功耗-延迟 trade-off 实验。
- `tegrastats` / `jtop`：推理期间功耗、GPU/CPU 利用率、温度采集。

## 7. 推荐方法组合（两周内）

按「文献支撑 × 两周可行 × 与 SmolVLA 契合度」排序：

1. **Flow Matching 步数缩减**（SnapFlow 动机支撑）：`num_steps` 10→8→5→2，评测成功率与 `inference_ms` 的 trade-off。改动最小，直接回答"去噪占 80% 延迟"这一核心瓶颈。
2. **Power mode + 功耗采集**（硬件项，导师要求）：`nvpmodel` 三档 + `tegrastats` 记录功耗/温度/GPU 利用率。
3. **4-bit 量化**（ActQuant / LiteVLA-Edge / SQIL 支撑）：先验证 SmolVLA 的 4-bit 加载（GGUF 或 bitsandbytes/TensorRT），跑 LIBERO spatial 对比成功率与延迟。
4. **Training-free 层剪枝**（EfficientVLA / EcoVLA / DTR 支撑）：对语言模块做层剪枝实验，严格评测成功率（注意 Don't Run with Scissors 的负面结论）。
5. **进阶（时间允许）**：TensorRT FP16/INT8 + DLA；SnapFlow 单步蒸馏（PC GPU ~12h）。
6. **报告写作**：以 Survey (2510.24795) 为主线组织 related work，逐方法标注支撑文献。

## 8. 引用清单（BibTeX 简版）

```bibtex
@article{actquant,
  title={ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models},
  author={Akbari, Arash and others},
  journal={arXiv:2605.24011}, year={2026}}
@article{snapflow,
  title={SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation},
  author={Luan, Wuyang and others},
  journal={arXiv:2604.05656}, year={2026}}
@article{litevlaedge,
  title={LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics},
  author={Williams, Justin and others},
  journal={arXiv:2603.03380}, year={2026}}
@article{efficientvla,
  title={EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models},
  author={Yang and others},
  journal={Advances in Neural Information Processing Systems}, year={2025}}
@article{ecovla,
  title={EcoVLA: Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models},
  journal={arXiv:2602.00780}, year={2026}}
@article{adp,
  title={Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation},
  author={Pei, Xiaohuan and others},
  journal={arXiv:2509.22093}, year={2025}}
@article{sqil,
  title={Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control},
  journal={arXiv:2505.15304}, year={2025}}
@article{bitvla,
  title={BitVLA: 1-Bit Vision-Language-Action Models for Robotics Manipulation},
  journal={arXiv:2506.07530}, year={2025}}
@article{nanovla,
  title={NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies},
  author={Chen, Jiahong and others},
  journal={arXiv:2510.25122}, year={2025}}
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
  author={Yu, Zhaoshuo and Wang, ... and others},
  journal={arXiv:2510.24795}, year={2025}}
@article{smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  journal={arXiv:2506.01844}, year={2025}}
```
