# Jetson VLA Project: Top 10 Highly Relevant Papers

Date: 2026-06-25

## Selection Logic

These papers were selected for the project theme: deploying, benchmarking, and optimising Vision-Language-Action (VLA) or VLA-style robot policies on a Jetson-class embedded platform for low-cost manipulation. Priority was given to papers that support at least one of:

- VLA baseline selection: OpenVLA, SmolVLA, Octo, RT-2.
- Embedded/Jetson feasibility: local inference, edge bottlenecks, quantisation, edge-cloud split.
- Low-cost robot evaluation: SO-101/ALOHA-style hardware, ACT baseline, failure analysis.
- Benchmark design: latency, memory, control frequency, task success, failure taxonomy.

## Top 10 Papers

### 1. OpenVLA: An Open-Source Vision-Language-Action Model

- Authors: Moo Jin Kim et al.
- Year: 2024
- Link: https://arxiv.org/abs/2406.09246
- Why it matters: This is the main open-source VLA baseline for the project. It is a 7B-parameter VLA trained on large-scale robot demonstrations and supports fine-tuning and quantised serving. Use remote OpenVLA inference as a strong baseline; treat full local Jetson deployment as a high-risk stress test.

### 2. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

- Authors: Mustafa Shukor et al.
- Year: 2025
- Link: https://arxiv.org/abs/2506.01844
- Why it matters: This is one of the closest matches to the project because it targets affordable robotics and efficient deployment. Its asynchronous inference stack and smaller model size make it a strong candidate for comparison against remote OpenVLA.

### 3. LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics

- Authors: Justin Williams et al.
- Year: 2026
- Link: https://arxiv.org/abs/2603.03380
- Why it matters: Directly relevant to Jetson-class on-device VLA inference. It reports a practical pipeline using quantisation and ROS2-style integration, making it useful for defining your Jetson benchmark metrics and optimisation experiments.

### 4. Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures

- Authors: Manoj Vishwanathan, Suvinay Subramanian, Anand Raghunathan
- Year: 2026
- Link: https://arxiv.org/abs/2603.02271
- Why it matters: This paper gives a systems-level reason for your project: VLA deployment is constrained by latency and memory bottlenecks, especially during action generation. It is highly useful for justifying latency, p95 latency, GPU/RAM use, and control-loop frequency measurements on Jetson.

### 5. QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization

- Authors: Yuhao Xu et al.
- Year: 2026
- Link: https://arxiv.org/abs/2602.03782
- Why it matters: Useful for the optimisation part of the project. It argues that generic LLM quantisation is not enough for VLA because small action deviations can compound into robot failures. This supports measuring action stability and task success after quantisation, not just speed/memory.

### 6. Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis

- Authors: Yi Yu, Xinchuan Qiu
- Year: 2026
- Link: https://arxiv.org/abs/2606.08881
- Why it matters: Very close to your low-cost robot arm setting. It evaluates VLA and imitation-learning policies on an affordable real-world SO-101 platform and adds failure/recovery analysis beyond binary success rate. Use it to design failure taxonomy and recovery-aware metrics.

### 7. Octo: An Open-Source Generalist Robot Policy

- Authors: Octo Model Team et al.
- Year: 2024
- Link: https://arxiv.org/abs/2405.12213
- Why it matters: Strong open-source generalist robot policy baseline trained on Open X-Embodiment. It is useful for discussing cross-embodiment adaptation, new sensors, new action spaces, and fine-tuning to a low-cost arm.

### 8. Open X-Embodiment: Robotic Learning Datasets and RT-X Models

- Authors: Open X-Embodiment Collaboration et al.
- Year: 2023
- Link: https://arxiv.org/abs/2310.08864
- Why it matters: Provides the dataset foundation behind many generalist robot policies. It helps explain why large VLA models generalise across robots, but also why action-space and embodiment mismatch become major issues when deploying to a low-cost arm.

### 9. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

- Authors: Anthony Brohan et al.
- Year: 2023
- Link: https://arxiv.org/abs/2307.15818
- Why it matters: Foundational VLA paper. It introduced the idea of co-training vision-language models with robot trajectories and representing robot actions as model outputs. Use it for background and motivation rather than as a deployable baseline.

### 10. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
- Year: 2023
- Link: https://arxiv.org/abs/2304.13705
- Why it matters: Important low-cost manipulation and ACT baseline paper. Even though it is not a VLA paper, it is very useful for your project because it supports low-cost hardware as a serious research platform and gives a non-VLA imitation-learning baseline for comparison.

## Suggested Reading Order

1. OpenVLA
2. SmolVLA
3. LiteVLA-Edge
4. Characterizing VLA Models
5. Benchmarking VLA Models on SO-101
6. QVLA
7. Octo
8. Open X-Embodiment
9. RT-2
10. ALOHA / ACT

## How To Use These In The Report

- Background and motivation: RT-2, Open X-Embodiment, OpenVLA.
- Main baselines: OpenVLA, SmolVLA, Octo, ACT.
- Embedded deployment contribution: LiteVLA-Edge, Characterizing VLA Models, QVLA.
- Low-cost robot benchmark design: Benchmarking VLA Models on SO-101, ALOHA/ACT.
- Metrics to adopt: task success, grasp success, latency, p95 latency, model loading time, RAM/VRAM use, power mode, control-loop frequency, OOM rate, action validity, failure type, and recovery success.

