# Related Work and Baselines for the OpenVLA + Jetson + Low-Cost Arm Project

Date: 2026-06-06

## Most Relevant References

### 1. OpenVLA: Open-source VLA baseline

- Project: https://openvla.github.io/
- GitHub: https://github.com/openvla/openvla
- Paper: https://arxiv.org/abs/2406.09246

Why it matters:

- Main VLA baseline for this project.
- 7B-parameter open-source vision-language-action model for robotic manipulation.
- Provides code, model checkpoints, fine-tuning pipeline, and REST API serving for integration with robot control stacks.

How to use:

- Use remote OpenVLA inference as the first VLA baseline.
- Treat full local OpenVLA-7B on Jetson as a stretch goal, not the minimum deliverable.

### 2. SmolVLA / LeRobot: efficient and affordable robotics baseline

- SmolVLA model card: https://huggingface.co/lerobot/smolvla_base
- Paper: https://arxiv.org/abs/2506.01844
- LeRobot GitHub: https://github.com/huggingface/lerobot

Why it matters:

- Compact VLA direction designed for affordable robotics.
- Strong alternative if OpenVLA-7B is too heavy for local embedded inference.
- Closely aligned with low-cost robot arms such as SO-100/SO-101.

How to use:

- Use as a local or lightweight-policy extension.
- Compare remote OpenVLA vs smaller local policy if time allows.

### 3. Octo: open-source generalist robot policy

- Project: https://octo-models.github.io/
- Paper: https://arxiv.org/abs/2405.12213

Why it matters:

- Open-source generalist robot policy trained on large robot datasets.
- Useful for comparing VLA/generalist-policy approaches.

How to use:

- Literature review reference.
- Possible alternative baseline if OpenVLA integration is difficult.

### 4. RT-1 / RT-2: foundational VLA and robotics transformer work

- RT-1 blog: https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/
- RT-2 paper: https://arxiv.org/abs/2307.15818

Why it matters:

- RT-1 established scalable transformer policies for real-world robot control.
- RT-2 popularised the vision-language-action framing by transferring web-scale VLM knowledge to robotic actions.

How to use:

- Background literature for why VLA models matter.
- Not ideal as a direct baseline because models/tooling are less open than OpenVLA/LeRobot.

### 5. ALOHA / Mobile ALOHA / ACT: low-cost robot manipulation and action chunking

- Mobile ALOHA project: https://mobile-aloha.github.io/
- ALOHA paper PDF: https://robots-that-learn.github.io/resources/Aloha23.pdf
- Example low-cost ACT implementation: https://github.com/Shaka-Labs/ACT

Why it matters:

- Shows that capable manipulation systems can be built with low-cost hardware.
- ACT provides a strong imitation-learning baseline for robot control.

How to use:

- Literature review for low-cost manipulation.
- Alternative non-VLA baseline if OpenVLA action mapping becomes difficult.

### 6. Jetson Orin Nano Super and Jetson robotics stack

- Jetson Orin Nano Super: https://www.nvidia.com/en-sg/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- Jetson Orin Nano user guide: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/index.html
- Jetson software for robotics: https://developer.nvidia.com/embedded/develop/software

Why it matters:

- Official embedded GPU platform for the project.
- NVIDIA describes Jetson as an edge AI and robotics platform; Orin Nano Super is suitable for real-time perception and robot-side control.

How to use:

- Use as the hardware platform justification.
- Benchmark CPU/GPU/RAM and power mode on Jetson.

### 7. NVIDIA Isaac ROS Benchmark / ros2_benchmark

- ros2_benchmark GitHub: https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark
- Isaac ROS Benchmark docs: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_benchmark/index.html
- Isaac ROS GitHub org: https://github.com/NVIDIA-ISAAC-ROS

Why it matters:

- Provides a formal way to benchmark ROS2 graphs on Jetson/NVIDIA platforms.
- Useful if the project uses ROS2 for camera/arm integration.

How to use:

- Use as benchmark methodology inspiration.
- Optional if the project stays Python-only rather than ROS2-heavy.

### 8. Jetson AI Lab OpenVLA container

- Page: https://tokk-nv.github.io/jetson-generative-ai-playground/openvla.html

Why it matters:

- Shows NVIDIA community interest in OpenVLA on JetPack/Jetson-style environments.
- Useful for feasibility exploration around containers and LoRA training.

How to use:

- Check as a practical starting point for Jetson-compatible OpenVLA tooling.

### 9. Edge/VLA optimisation papers

- Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures: https://arxiv.org/abs/2603.02271
- LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics: https://arxiv.org/abs/2603.03380
- TinyVLA project: https://tiny-vla.github.io/

Why it matters:

- These are very close to the proposed novelty: VLA inference under edge/embedded constraints.
- They help justify optimisation, quantisation, and deployment benchmarking as research contributions.

How to use:

- Use as emerging related work.
- Compare their framing with this project's Jetson + low-cost arm benchmark.

### 10. Stereo/depth and safety references

- Intel RealSense ROS notes: https://nu-msr.github.io/ros_notes/ros2/realsense.html
- MoveIt2 GitHub: https://github.com/moveit/moveit2
- Depth camera based collision avoidance: https://www.sciencedirect.com/science/article/pii/S0278612514000417
- Empirical comparison of stereo depth cameras for robotics: https://arxiv.org/abs/2501.07421
- Depth-based visual servoing using low-accuracy arm: https://arxiv.org/abs/1612.03784

Why it matters:

- Supports the stereo safety wrapper idea.
- Gives practical background for depth reliability, robot collision avoidance, and low-accuracy arm control.

How to use:

- Use stereo depth to filter unsafe actions, define workspace boundaries, and analyse failure cases.

### 11. Robot-learning datasets and benchmarks

- Open X-Embodiment paper: https://arxiv.org/abs/2310.08864
- BridgeData V2 project: https://rail-berkeley.github.io/bridgedata/
- BridgeData V2 paper: https://arxiv.org/abs/2308.12952
- LIBERO GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO paper: https://arxiv.org/abs/2306.03310

Why it matters:

- OpenVLA, Octo, and related models build on large robot datasets.
- LIBERO can be used for simulation/literature benchmarking if real hardware integration becomes delayed.

How to use:

- Cite as background for robot policy pretraining and benchmark design.
- Use BridgeData V2 because it is collected on WidowX 250, a low-cost/tabletop manipulation setup close to the proposed project.

## Recommended Baseline Stack for This FYP

### Minimum practical baseline

1. Jetson Orin Nano Super + stereo/depth camera + low-cost arm.
2. Rule-based pick-and-place baseline.
3. Remote OpenVLA inference integrated with Jetson-side execution.
4. Benchmark logging: success, latency, resource usage, failure modes.

### Stronger research baseline

1. Rule-based baseline.
2. OpenVLA remote baseline.
3. OpenVLA + stereo safety wrapper.
4. Optional SmolVLA / lightweight local policy baseline.

## Best Papers to Read First

1. OpenVLA: An Open-Source Vision-Language-Action Model
2. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
3. Octo: An Open-Source Generalist Robot Policy
4. Open X-Embodiment: Robotic Learning Datasets and RT-X Models
5. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ALOHA
6. Characterizing VLA Models for Edge AI Architectures
7. LiteVLA-Edge

## Best Projects to Try First

1. OpenVLA GitHub: REST inference and action prediction examples.
2. LeRobot GitHub: low-cost arm software and efficient policy tooling.
3. ros2_benchmark / Isaac ROS Benchmark: benchmark methodology if ROS2 is used.
4. MoveIt2: optional arm planning/collision checking.
5. RealSense ROS wrapper or camera SDK: stereo/depth integration.
