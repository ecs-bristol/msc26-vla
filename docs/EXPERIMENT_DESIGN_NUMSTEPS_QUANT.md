# 实验设计方案：num_steps 扫描 + 4-bit 量化 + 动作块（PC 先行）

> 状态：A/B 已完成；实验 C（动作块）新增（2026-08-20）｜执行人：Huang Yuhang
> 约束：Jetson 板子暂不在手边，全部优化先在 PC 上验证“方法是否有效”；板子回来后只补部署与硬件测量。
> 文献支撑：见 `docs/LITERATURE_VLA_OPTIMIZATION.md`（SnapFlow 2604.05656、ActQuant 2605.24011、LiteVLA-Edge 2603.03380、SQIL 2505.15304）。

## 1. 目标与假设

### 实验 A：Flow Matching 步数扫描（`num_steps` 10 → 8 → 5 → 3 → 2）
- 动机：SmolVLA 动作生成是 flow matching，默认 10 步迭代去噪。SnapFlow 指出去噪占端到端延迟约 80%，且 SmolVLA 在单步（1-NFE）蒸馏下仍可保持成功率。直接减小 `num_steps` 是零训练成本、立即可测的延迟优化。
- 假设 H1：`num_steps` 从 10 降到 5，成功率不显著下降（≤5pp），单步推理延迟近似线性下降。
- 假设 H2：`num_steps` 降到 2 时成功率开始明显下降，形成“延迟-成功率”权衡曲线（报告的核心图）。

### 实验 B：4-bit 权重量化
- 动机：ActQuant / LiteVLA-Edge / SQIL 均证明 4-bit（含子 4-bit）量化可在 LIBERO 上保持 ~90-95% 成功率，并在 Jetson 类硬件上显著省内存/延迟。SmolVLA 权重体积的主要部分是 VLM 主干（SmolVLM2-500M）。
- 假设 H3：将 VLM 主干量化为 4-bit（NF4 权重量化）后，LIBERO spatial 成功率相对 fp16 基线掉点 ≤5pp，峰值显存下降约 3-4×。
- 说明（诚实预期）：NF4 是权重量化，forward 仍按 fp16 计算，**主要收益是显存/内存，不是计算延迟**。若需要计算延迟收益，后续在 Jetson 上走 GGUF Q4_K_M（LiteVLA-Edge 同路线）或 INT8 张量核。PC 上这一步的价值是：验证“量化后成功率不掉”这一前置条件。

## 2. 共同实验协议（保证可比性）

- 环境：WSL `~/vla/lerobot-libero`（lerobot v0.6.1）+ `HF_HOME=~/vla/hf-cache`，MuJoCo EGL。
- 模型：`HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，fp16。
- 评测套件：先 `libero_spatial`（10 任务，与基线同套件）；时间允许再补 `libero_object`。
- 评测协议：`lerobot-eval`（官方 harness），`eval.batch_size=1`、`env.max_parallel_tasks=1`、`episode_length=280`，输出 `eval_info.json`（`overall.pc_success` + 每任务 successes）。
- 种子/确定性：所有变体使用同一组 seed；基线（fp16、num_steps=10）用同一 harness 重跑一次，作为同条件对照（不与历史 8/10 直接比较，因为历史 run 与本次评测条件可能不同）。
- 样本量：筛选阶段 `N_EPISODES=1`（每任务 1 集）；最终上报的配置用 `N_EPISODES=5` 重跑，报告里标注样本量。
- 指标：
  - 成功率：overall + 每任务（对比表）。
  - 延迟：microbenchmark（见 4.3），取 mean / p95 `inference_ms`。
  - 显存：microbenchmark 期间 `torch.cuda.max_memory_allocated()` 峰值 + 权重内存 `sum(p.numel()*p.element_size())`。

## 3. 实验 A：num_steps 扫描

### 3.1 变量与固定项
- 自变量：`num_steps ∈ {10, 8, 5, 3, 2}`（10 为基线）。
- 固定：suite=libero_spatial、episode_length=280、n_episodes=1（筛选）、checkpoint/revision/fp16、seed。
- 注意：`num_steps` 只影响动作生成迭代次数，不影响观测与策略其余部分。

### 3.2 操作步骤
1. 先确认 checkpoint 默认值（预期 10）：

   ```bash
   source ~/vla/lerobot-libero/bin/activate
   python - <<'PY'
   from lerobot.configs.policies import PreTrainedConfig
   c = PreTrainedConfig.from_pretrained(
       "HuggingFaceVLA/smolvla_libero",
       revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
   )
   print("num_steps =", c.num_steps, "| n_action_steps =", c.n_action_steps, "| chunk =", c.chunk_size)
   PY
   ```

2. 给 `scripts/wsl/run_official_pc_local_eval.sh` 增加 `NUM_STEPS` 环境变量，追加 `--policy.num_steps="$NUM_STEPS"`（默认 10）。
3. 跑筛选（每档 1 episode/task）：

   ```bash
   for n in 10 8 5 3 2; do
     NUM_STEPS=$n SUITE=libero_spatial N_EPISODES=1 EPISODE_LENGTH=280 \
       MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
       bash scripts/wsl/run_official_pc_local_eval.sh
   done
   ```

4. 对 top-2 档位 + 基线（10）用 `N_EPISODES=5` 重跑确认。
5. 汇总 `eval_info.json` 到 `evidence/latest/pc_local/num_steps/`。

### 3.3 判定标准
- 通过：5 档成功率曲线单调或近似平台期，且 `num_steps=5` 相对 10 掉点 ≤5pp。
- 报告产出：`成功率 vs num_steps` 曲线 + `inference_ms vs num_steps` 曲线。

## 4. 实验 B：4-bit 量化

### 4.1 路线选择（按风险排序）

| 路线 | 做法 | 优点 | 缺点 | 结论 |
| --- | --- | --- | --- | --- |
| B1 int4_groupwise（自包含） | 对 VLM 主干（vision/text/connector 的 Linear）做 per-group absmax 4-bit 量化，运行时反量化 | 无外部内核依赖、跨平台稳定、方案可精确描述；post-load 直接替换 Linear | 计算仍是反量化后 fp32/fp16，无内核加速 | **默认路线**（已实现） |
| B2 bitsandbytes NF4 | VLM 主干 Linear 换 `bnb.nn.Linear4bit`（compute_dtype 跟随 use_amp） | 与文献（ActQuant/LiteVLA-Edge）一致 | 依赖 bnb CUDA 版本；compute dtype 与模型主精度耦合 | 实验性选项（已实现，需先 probe） |
| B3 GGUF Q4_K_M + llama.cpp | LiteVLA-Edge 同路线 | Jetson 上可复用、内存+延迟双收益 | 改动最大、要重写推理路径 | 板子回来后做部署时再考虑 |

### 4.2 实现位置（关键设计）
- 复用现有 **LeRobot 策略插件**机制（`plugins/lerobot_policy_remote_jetson` 已验证）：新建 `plugins/lerobot_policy_smolvla_int4`，注册策略类型 `smolvla_int4`。
- 插件职责：
  1. 参数：checkpoint、revision、`quant_method`（none/int4_groupwise/bnb_nf4）、`num_steps`、`n_action_steps`。
  2. 用 lerobot 原逻辑加载 SmolVLA（`PreTrainedConfig.from_pretrained` + `SmolVLAPolicy.from_pretrained`，CPU 加载后整体 `.to(device)`）。
  3. 加载后只量化 `policy.model.vlm_with_expert.vlm`（SmolVLM2 主干：vision_model/connector/text_model.layers 的 `nn.Linear`）；action expert（Gemma）与 embedding 保持原精度。
  4. `select_action`/`predict_action_chunk` 直接委托给内层 SmolVLA，噪声路径与 RNG 和基线完全一致；`use_amp` 从 checkpoint 镜像。
- 评测入口不变：`lerobot-eval --policy.type=smolvla_int4 ...`，输出仍是 `eval_info.json`，与基线格式一致。

### 4.3 延迟/显存 microbenchmark
- 新增 `scripts/wsl/bench_smolvla_latency.py`（复用插件加载逻辑的小脚本）：
  - 固定一组合成观测（LIBERO 格式：instruction + agentview/wrist 图像 + 8 维 state），跑 100 次 `select_action`。
  - 记录 mean/p95 `inference_ms`、`torch.cuda.max_memory_allocated()`、权重内存。
  - 分别测：fp16/num_steps=10、fp16/num_steps=5、int4/num_steps=10、int4/num_steps=5，形成 2×2 对比。

### 4.4 操作步骤
1. 安装插件（`scripts/wsl/install_smolvla_int4_policy.sh`）。
2. probe（0.5-1 天）：`python scripts/wsl/bench_smolvla_latency.py --quant-method int4_groupwise --probe`，确认能加载并跑通 1 个 batch；失败则按报错回退。
3. 跑 `bash scripts/wsl/run_int4_eval.sh`（spatial，N_EPISODES=1 先冒烟，再 N_EPISODES=5 正式）。
4. 跑 microbenchmark 2×2，记录显存与延迟。
5. 汇总到 `evidence/latest/pc_local/int4/`。

### 4.5 判定标准
- 通过：int4 相对 fp16 成功率掉点 ≤5pp，且权重内存下降 ≥3×。
- 若掉点 >5pp：记录掉点幅度与失败任务分布，报告中定性讨论（对齐 Don't Run with Scissors / ActQuant 的 trade-off 叙事），不强行洗白。

## 5. 结果汇总表模板

| 变体 | num_steps | 精度 | overall 成功率 | 每任务成败 | inference_ms mean/p95 | 峰值显存 | 权重内存 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 基线（重跑） | 10 | fp16 | ? | ? | ? | ? | ? |
| fp16 步数 8/5/3/2 | ... | fp16 | ? | ? | ? | ? | ? |
| int4（同 harness） | 10 / 5 | 4-bit NF4 | ? | ? | ? | ? | ? |

## 6. 实验 C：动作块大小（n_action_steps）扫描

> 来源：2026-08-20 与 Zhiqiang Que 的会议讨论。当前 `n_action_steps=1`，
> 每次推理只生成并执行一步，再和模拟器交互一次；Que 老师建议一次生成动作块
> （5/10/20 步）连续执行，减少 PC↔Jetson 交互与推理次数，缩短任务总时间，
> 代价是开环执行可能降低成功率。

### 6.1 变量与固定项

- 自变量：`n_action_steps ∈ {1, 5, 10, 20}`（1 为当前基线）。
- 固定：`num_steps` 取实验 A 的最优档（默认 2）、`libero_spatial`、
  `episode_length=280`、checkpoint/revision/fp16、seed。
- 筛选：`N_EPISODES=1` 冒烟；top 档位用 `N_EPISODES=5` 正式上报。

### 6.2 假设

- H6：增大 `n_action_steps` 会降低推理次数/交互次数与任务总时间，但成功率可能
  下降；存在成功率-完成时间的 Pareto 最优块大小。

### 6.3 指标

- 成功率：`overall.pc_success` + 每任务。
- 任务总时间：一次 episode 的墙钟时间（`eval_ep_s`）。
- 推理次数 / PC↔Jetson 交互次数：从 `remote_transport.jsonl`（Jetson 远程）或
  eval 日志（PC 本地）统计。
- 单步推理延迟：microbenchmark（同实验 A 协议）。
- 图件：成功率 vs 任务总时间的 Pareto 图，各 `n_action_steps` 为散点。

### 6.4 操作步骤

1. 确认 `n_action_steps` 与 `chunk_size` 的关系：SmolVLA 一次推理输出
   `chunk_size` 步，实际执行 `n_action_steps` 步后重新推理。已确认
   `src/libero_platform/policies/smolvla_policy.py` 会强制校验
   `n_action_steps <= chunk_size`，因此 `{1,5,10,20}` 档位只要
   checkpoint 的 `chunk_size ≥ 20` 就合法（运行时再读一次确认）。
2. 给 eval 入口追加 `N_ACTION_STEPS` 环境变量，覆盖 `--policy.n_action_steps`。
   官方 `lerobot-eval` 路径原生支持 `n_action_steps`；自研平台
   （`serve-policy` + `remote_jetson`）当前只用 `_first_action_values`
   返回第一步，Jetson 端实验 C 需要额外改造。
3. 筛选：

   ```bash
   for n in 1 5 10 20; do
     N_ACTION_STEPS=$n NUM_STEPS=2 SUITE=libero_spatial N_EPISODES=1 \
       MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
       bash scripts/wsl/run_official_pc_local_eval.sh
   done
   ```

4. top 档位用 `N_EPISODES=5` 重跑确认。
5. 汇总到 `evidence/latest/pc_local/action_chunk/`。

### 6.5 判定标准

- 通过：能画出成功率-完成时间的 Pareto 前沿，并给出推荐块大小。
- 若大块（10/20）成功率明显下降：如实报告开环执行代价，结论写成
  “块大小需按任务复杂度选择”。

### 6.6 风险

| 风险 | 影响 | 回退/缓解 |
| --- | --- | --- |
| `remote_jetson` 插件只支持单步返回（已确认：插件 `decode_action_response` 要求 `(7,)`，平台 adapter 用 `_first_action_values` 只取第一步） | 动作块实验在 Jetson 端跑不了 | 先在 PC 本地用官方 `lerobot-eval` 验证可行性；Jetson 端需同步扩展 server 返回块 + 插件解码多步 |
| `chunk_size` 与 `n_action_steps` 不一致 | 生成步数与执行步数错位 | 已确认平台会强制 `n_action_steps <= chunk_size`；运行时读 checkpoint 再核对，测试日志核对每 episode 推理次数 |
| 开环执行导致大块成功率崩 | 实验 C 结论负面 | 作为 trade-off 如实报告，并在报告里引出“自适应块大小”future work |

### 6.7 延伸实验 D：缩短 chunk_size（预测更短的动作块）

> 动机：当前 `n_action_steps < chunk_size=50` 时，模型仍每次生成完整 50 步，
> 只执行前 `n_action_steps` 步，其余是白算的。把 `chunk_size` 改小可以降低
> 单次推理的动作序列长度，可能进一步缩短单步延迟与任务总时间。

- 配对扫描（`num_steps=2` 固定）：

  | chunk_size | n_action_steps | 对比目标 |
  | --- | --- | --- |
  | 50 | 20 | 现有实验 C 最优档（基线） |
  | 20 | 20 | 预测与执行对齐，隔离 chunk 计算量 |
  | 10 | 10 | 更短的预测 + 更频繁重规划 |

- 先 1 集冒烟确认 `--policy.chunk_size` 覆盖生效且数值不崩，再对
  成功率/时间可接受的档位跑 `N_EPISODES=5`。
- 指标与实验 C 相同；额外记录单步推理延迟，确认“短 chunk 是否真的更快”。
- 风险：模型按 50 步 chunk 训练，更短 chunk 是分布偏移，成功率可能掉；
  机制上代码支持（噪声 shape 跟随 `config.chunk_size`），但需 probe 验证。

## 7. 工作量与排期（PC，2 周内）

| 天 | 内容 |
| --- | --- |
| D1 | Step 0 验证（num_steps 默认值 + Hydra 覆盖生效）；写 NUM_STEPS 脚本改动 |
| D2 | 实验 A 筛选跑批（5 档 × spatial，挂机）；同时开始 B1 可行性验证 |
| D3 | 实验 A 汇总 + 延迟 microbenchmark（5 档） |
| D4-5 | 实验 B 插件实现 + 冒烟 + 2×2 microbenchmark |
| D6 | 实验 B 正式评测（spatial，N_EPISODES=5）+ 汇总 |
| D7 | 两表两图 + 结论写入报告素材；若时间允许补 object 套件 |

并行策略：实验 A 跑批是纯挂机任务，与实验 B 的实现互不阻塞。

## 8. 风险与回退

| 风险 | 影响 | 回退/缓解 |
| --- | --- | --- |
| Hydra `--policy.num_steps` 覆盖不生效 | 实验 A 失效 | 先用 microbenchmark 验证 num_steps=10 vs 2 的延迟差异；不生效则改走平台 `SmolVLAInferenceSpec`（`smolvla_policy.py` 已支持 num_steps） |
| bnb 与 lerobot VLM 加载不兼容 | B2 走不通 | 默认走自包含 int4_groupwise（无外部依赖） |
| 单集成功率噪声大（8/10 vs 9/10） | 结论不稳 | 正式上报一律 N_EPISODES=5；报告标注样本量 |
| int4 掉点明显 | 实验 B 结论负面 | 按 trade-off 如实报告，作为“GGUF/Jetson 部署前需要 QAT 恢复”的动机 |
| PC 延迟与 Jetson 无可比性 | 延迟结论被质疑 | 报告中明确 PC 数据只用于“相对趋势”，绝对数值以板子回来后 Jetson 实测为准 |

## 9. 产出物（交付）
1. `evidence/latest/pc_local/num_steps/`：5 档 `eval_info.json` + 汇总 CSV。
2. `evidence/latest/pc_local/int4/`：fp16 vs int4 的 `eval_info.json` + microbenchmark CSV。
3. 报告素材：成功率-num_steps 曲线、延迟-num_steps 曲线、fp16/int4 内存对比表。
4. 代码：`NUM_STEPS` 脚本改动 + `plugins/lerobot_policy_smolvla_int4` + bench 脚本（提交与否听用户安排）。

## 10. 执行 Runbook（在 WSL 里按顺序跑）

所有命令在 WSL 内、项目根目录（`/mnt/d/.../LIBERO_Benchmark_Platform`）执行：

```bash
source ~/vla/lerobot-libero/bin/activate
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
export HF_HOME=~/vla/hf-cache
```

### Step 0：验证环境与覆盖生效（约 20 分钟）
```bash
# 确认 checkpoint 默认 num_steps
python - <<'PY'
from lerobot.configs.policies import PreTrainedConfig
c = PreTrainedConfig.from_pretrained(
    "HuggingFaceVLA/smolvla_libero",
    revision="$MODEL_REVISION",
)
print("num_steps =", c.num_steps, "| n_action_steps =", c.n_action_steps)
PY

# 冒烟：num_steps=2 跑 1 个任务，确认覆盖生效（延迟应明显低于默认）
NUM_STEPS=2 N_EPISODES=1 MODEL_REVISION=$MODEL_REVISION \
  bash scripts/wsl/run_official_pc_local_eval.sh
```

### Step 1：安装 int4 插件 + 量化后端
```bash
bash scripts/wsl/install_smolvla_int4_policy.sh
# 可选：python -m pip install bitsandbytes   # 若想试 bnb_nf4
python scripts/wsl/bench_smolvla_latency.py --quant-method int4_groupwise --probe
```

### Step 2：实验 A 跑批（挂机 ~3-4h，跑批期间做 Step 3 的准备）
```bash
SUITE=libero_spatial N_EPISODES=1 MODEL_REVISION=$MODEL_REVISION \
  bash scripts/wsl/run_numsteps_sweep.sh
```

### Step 3：实验 B 评测（等实验 A 跑完，GPU 独占）
```bash
# 冒烟 1 集
QUANT_METHOD=int4_groupwise NUM_STEPS=10 SUITE=libero_spatial N_EPISODES=1 \
  MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_int4_eval.sh

# 正式 5 集（fp16 基线 = 实验 A 的 num_steps=10 那一档）
QUANT_METHOD=int4_groupwise NUM_STEPS=10 SUITE=libero_spatial N_EPISODES=5 \
  MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_int4_eval.sh

# 组合最优档（num_steps=2，正式 5 集）
QUANT_METHOD=int4_groupwise NUM_STEPS=2 SUITE=libero_spatial N_EPISODES=5 \
  MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_int4_eval.sh
```

### Step 4：延迟/显存 microbenchmark（GPU 独占，3×2）
```bash
python scripts/wsl/bench_smolvla_latency.py --quant-method none --num-steps 10
python scripts/wsl/bench_smolvla_latency.py --quant-method none --num-steps 5
python scripts/wsl/bench_smolvla_latency.py --quant-method none --num-steps 2
python scripts/wsl/bench_smolvla_latency.py --quant-method int4_groupwise --num-steps 10
python scripts/wsl/bench_smolvla_latency.py --quant-method int4_groupwise --num-steps 5
python scripts/wsl/bench_smolvla_latency.py --quant-method int4_groupwise --num-steps 2
```

### Step 5：汇总
- 结果目录：`~/vla/results/libero_spatial_pc_local_*`（实验 A）与 `~/vla/results/libero_spatial_int4_*`（实验 B）。
- 把 `eval_info.json` 汇总到 `evidence/latest/pc_local/`，microbenchmark 输出存 CSV。

### Step 6：实验 C（动作块扫描，跑完 A/B 后）
```bash
# 冒烟 1 集
for n in 1 5 10 20; do
  N_ACTION_STEPS=$n NUM_STEPS=2 SUITE=libero_spatial N_EPISODES=1 \
    MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_official_pc_local_eval.sh
done

# top 档位正式 5 集
N_ACTION_STEPS=5 NUM_STEPS=2 SUITE=libero_spatial N_EPISODES=5 \
  MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_official_pc_local_eval.sh
N_ACTION_STEPS=10 NUM_STEPS=2 SUITE=libero_spatial N_EPISODES=5 \
  MODEL_REVISION=$MODEL_REVISION bash scripts/wsl/run_official_pc_local_eval.sh
```

- 结果目录：`~/vla/results/libero_spatial_pc_local_*`（需在目录名带
  `na<n_action_steps>` 标记，或汇总时按命令参数区分）。
- 汇总到 `evidence/latest/pc_local/action_chunk/`，输出成功率-完成时间 Pareto 图。
