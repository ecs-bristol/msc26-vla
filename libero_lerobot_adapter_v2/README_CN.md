# LeRobot / LIBERO 适配器 v2

这是原型的安全稳定版。默认仅预览命令；只有显式加入 `--run` 才会执行训练或评测。

## 快速使用

```bash
python libero_pipeline.py check
python libero_pipeline.py eval
python libero_pipeline.py eval --run
python libero_pipeline.py train --batch-size 4 --steps 1000 --run
python -m unittest -v
```

## 随机物品与指定抓取

```bash
python interactive_grasp.py --list
python interactive_grasp.py --count 4 --attempts 3 --run
```

第二条命令会随机列出 4 个 LIBERO 已知物品，然后等待输入中文或英文名称。也可以直接指定：

```bash
python interactive_grasp.py --count 4 --seed 12 --target 牛奶 --attempts 3 --run
```

可靠模式默认把单回合上限提高到 500 步，使用 `num_steps=10`、每一步重新规划，并对目标尝试 3 个不同初始状态。每次运行会建立带时间戳的新目录，避免旧结果被覆盖。需要更高完成概率时可使用 `--attempts 5`，但运行时间也会相应增加。

使用 `--target` 指定物品时，该物品会被强制加入本轮候选列表，其余候选仍随机生成。

全自动随机目标模式：

```bash
python interactive_grasp.py --count 4 --auto-random --attempts 3 --mode accurate --run
```

程序会自动随机选定物品、从 LIBERO 任务元数据确认名称、执行抓取并写入 `outputs/interactive_eval/grasp_history.csv`。这里的“识别”是任务元数据识别，不是独立的 RGB 视觉检测。

加入真正的 RGB 开放词汇检测：

```bash
python interactive_grasp.py --count 4 --auto-random --attempts 3 --mode accurate --vision --run
```

首次运行会下载 `IDEA-Research/grounding-dino-tiny`。程序在抓取前渲染主摄像头画面，识别物品并保存 `vision/initial_scene.png`、`vision/detected_scene.png` 和 `vision/detections.json`；只有视觉检测确认目标存在才继续抓取。8 GB 显存下检测结束后会释放检测模型，再加载机器人策略。

准确率优先模式依次尝试模型原生配置、10 步平滑动作块和每步重规划。只有当前策略达到全部回合成功才提前停止；部分成功会继续比较后续策略，并在汇总中选择成功率最高者。`--attempts 3` 表示每种策略最多 3 个回合，因此最坏情况会运行 9 个回合。只想快速验证时加入 `--mode fast`。

当前版本从 LIBERO 原生任务元数据识别物品，目标可靠且模型见过；它并非对任意陌生 3D 模型进行开放世界识别。

## 本版改进

- 所有数值、任务编号、模型与数据集标识均在启动前校验。
- 拒绝根目录输出，预览模式不创建任何目录。
- 中断时先优雅结束整个子进程组，超时后再强制结束，避免残留 GPU 进程。
- 每次真实运行写入 `last_run.json`，包含命令、退出码与耗时，便于复现和排错。
- 数据契约不再使用可能被 Python 优化参数禁用的 `assert`。
- worker 默认值随 CPU 调整且有上限；批量和并行参数必须为正数。

输出默认位于 `outputs/`。LIBERO 仍需在 Linux/WSL2 中运行；Windows 上可以执行预览和测试。
