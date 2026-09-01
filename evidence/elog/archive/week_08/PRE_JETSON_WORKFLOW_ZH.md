# Pre-Jetson 统一测试入口

这个入口用于把上 Jetson 之前要做的事情集中到一个地方：

- 在 `configs/pre_jetson_workflow.yaml` 里调整模型、任务、重复次数和输出目录。
- 用 `src.vla_bench.pre_jetson_runner` 统一运行 benchmark。
- 自动读取 `data/tasks.csv`。
- 自动输出 `trials.jsonl`、`trials.csv`、`summary.csv` 和 `metadata.json`。

## 1. 查看已有实验配置

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Local_VLA_Benchmark_Framework"
.\scripts\run_pre_jetson_workflow.ps1 -List
```

如果 Windows 禁止运行 `.ps1`，直接用 `.cmd` 版本：

```powershell
.\scripts\run_pre_jetson_workflow.cmd --list
```

## 2. 先做 dry run

不会加载模型，只检查将要运行哪些模型和任务。

```powershell
.\scripts\run_pre_jetson_workflow.ps1 -Experiment offline_vlm_smoke -DryRun
```

等价的 `.cmd` 命令：

```powershell
.\scripts\run_pre_jetson_workflow.cmd --experiment offline_vlm_smoke --dry-run
```

也可以临时覆盖模型和任务：

```powershell
.\scripts\run_pre_jetson_workflow.ps1 `
  -Experiment offline_vlm_smoke `
  -Models "qwen2_vl_2b" `
  -Tasks "desk_cup_pick,blue_can_pick" `
  -Repeats 1 `
  -DryRun
```

## 3. 正式运行 smoke test

```powershell
.\scripts\run_pre_jetson_workflow.ps1 -Experiment offline_vlm_smoke
```

输出目录类似：

```text
results/pre_jetson/offline_vlm_smoke_20260626_153000/
  metadata.json
  trials.jsonl
  trials.csv
  summary.csv
```

## 4. 跑完整轻量模型对比

```powershell
.\scripts\run_pre_jetson_workflow.ps1 -Experiment offline_vlm_compare
```

默认会跑：

- `smolvlm2_500m`
- `qwen2_vl_2b`
- `paligemma2_3b`

如果显存不足，可以先只跑一个：

```powershell
.\scripts\run_pre_jetson_workflow.ps1 -Experiment offline_vlm_compare -Models "qwen2_vl_2b" -Repeats 1
```

## 5. 配置文件怎么改

主要改这里：

```text
configs/pre_jetson_workflow.yaml
```

常用字段：

```yaml
model_keys:
  - qwen2_vl_2b
task_ids: all
repeats: 3
warmup: 1
max_new_tokens: 80
```

模型 key 来自：

```text
configs/models.yaml
```

任务 key 来自：

```text
data/tasks.csv
```

## 6. 输出字段

每次 trial 会记录：

- `model_key`
- `model_id`
- `task_id`
- `instruction`
- `expected_target`
- `expected_action`
- `raw_output`
- `parsed_actions`
- `action_valid`
- `expected_action_found`
- `target_mentioned`
- `auto_score_pass`
- `latency_sec`
- `peak_gpu_memory_gb`
- `model_load_sec`

注意：`auto_score_pass` 是自动粗评，不等价于真实机器人任务成功率。后面接 robosuite 或真实机械臂时，还需要记录 `task_success` 和 `failure_type`。
