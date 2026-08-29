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

完整评测 LIBERO Object 的 10 个任务（每个任务 3 次）：

```bash
python3 interactive_grasp.py --all-tasks --attempts 3 --mode accurate --seed 42 --run
```

批量模式默认令 `batch-size=attempts`，因此同一任务的 3 个回合会进行批量推理，
比旧版逐回合串行执行更快。显存不足时可加 `--batch-size 1` 回退到串行模式。
调试时还可加入 `--soft-reset` 缩短环境重建时间；正式复现实验建议继续使用默认硬重置。
一次加载策略后连续评测全部任务，结果目录会生成 `suite_summary.json`
（完整可复现实验记录）和 `task_summary.csv`（每个任务、每种策略的成功率）。
为保持批量基准一致性，`--all-tasks` 不与抓取前单场景检测的 `--vision` 同时使用；
需要检查 RGB 识别时仍使用上面的单目标命令。

单独测试速度与纠错折中的 `balanced`（4 步动作块）：

```bash
python3 interactive_grasp.py --all-tasks --attempts 3 --strategy balanced --seed 42 --run
```

`balanced` 是独立实验组，不会自动加入默认 `accurate`，避免默认实验从 60 回合增加到 90 回合。

使用历史实验自动为每个物品选择策略（应使用与历史数据不同的新种子验证）：

```bash
python3 interactive_grasp.py --all-tasks --attempts 3 --strategy router --seed 43 --run
```

路由器从旧的 `suite_summary.json` 汇总每个任务的策略成功率，并使用拉普拉斯平滑
`(成功次数+1)/(尝试次数+2)` 选择历史最优策略；无历史记录时回退到 `native`。
路由依据会写入新汇总的 `routing_decisions`，便于论文复现。不要使用形成路由规则的同一随机种子
报告最终性能，否则会产生数据泄漏。

只随机选择 3 个任务（固定 seed 后可复现）：

```bash
python3 interactive_grasp.py --random-tasks 3 --attempts 1 --strategy router --seed 43 --run
```

由用户选择任务时，先运行 `python3 interactive_grasp.py --list` 查看编号，再执行：

```bash
python3 interactive_grasp.py --task-ids 0 6 7 --attempts 1 --strategy router --seed 43 --run
```

`--random-tasks`、`--task-ids`、`--all-tasks`、`--target` 和 `--auto-random` 是互斥的任务选择方式。

混合恢复策略先按历史结果选择 `native` 或 `smooth`，首轮完全失败时自动切换另一策略重试一次：

```bash
python3 interactive_grasp.py --random-tasks 7 --attempts 1 --strategy hybrid --seed 44 --run
```

成功任务不会重复。汇总文件会同时记录 `task_completion_rate`、`recovered_tasks` 和
`recovery_success_rate`。用于论文时，应使用未参与建立路由历史的新随机种子。

首次运行会下载 `IDEA-Research/grounding-dino-tiny`。程序在抓取前渲染主摄像头画面，识别物品并保存 `vision/initial_scene.png`、`vision/detected_scene.png` 和 `vision/detections.json`；只有视觉检测确认目标存在才继续抓取。8 GB 显存下检测结束后会释放检测模型，再加载机器人策略。

若第一轮多物品检测漏掉目标，程序会自动用包装物专用提示词进行第二轮检测（例如 `milk carton`），阈值最低为 0.15。两轮均未确认目标时才安全停止，并在 `vision/target_retry/` 保存复检证据。

准确率优先模式比较模型原生配置和 10 步平滑动作块。实验确认当前检查点的原生配置与每步重规划产生完全相同的视频，因此重复的 responsive 策略已移除。只有当前策略达到全部回合成功才提前停止；部分成功会继续比较另一策略，并在汇总中选择成功率最高者。`--attempts 3` 表示每种策略最多 3 个回合，因此最坏情况运行 6 个回合。只想快速验证时加入 `--mode fast`。

默认的 `--strategy-order adaptive` 会扫描过去的 `accuracy_summary.json`，对每个物品分别累计策略成功次数，并用拉普拉斯平滑成功率安排优先顺序。可用 `--strategy-order fixed` 恢复固定的 native → smooth 顺序。

当前版本从 LIBERO 原生任务元数据识别物品，目标可靠且模型见过；它并非对任意陌生 3D 模型进行开放世界识别。

## 本版改进

- 所有数值、任务编号、模型与数据集标识均在启动前校验。
- 拒绝根目录输出，预览模式不创建任何目录。
- 中断时先优雅结束整个子进程组，超时后再强制结束，避免残留 GPU 进程。
- 每次真实运行写入 `last_run.json`，包含命令、退出码与耗时，便于复现和排错。
- 数据契约不再使用可能被 Python 优化参数禁用的 `assert`。
- worker 默认值随 CPU 调整且有上限；批量和并行参数必须为正数。

输出默认位于 `outputs/`。LIBERO 仍需在 Linux/WSL2 中运行；Windows 上可以执行预览和测试。
