# 可视化实验操纵台设计

日期：2026-07-17
状态：已批准
适用项目：Jetson VLA/VLM Benchmark Capstone Project

## 1. 目的

本设计定义一个本地、单用户、可视化的实验操纵台，用于从同一界面完成：

1. 设置实验环境；
2. 选择或批量比较实验模型；
3. 编辑并版本化任务要求；
4. 启动模型评估或轻量训练；
5. 通过原生 MuJoCo Viewer 观察实验或训练后的 rollout；
6. 自动记录、检查和比较实验数据。

操纵台不是简化的 preset 启动页，也不是把所有脚本按钮堆在一个页面里。它采用明确的三阶段工作流：

```text
设置实验 -> 运行与观察 -> 实验结果
```

这份设计取代 `2026-07-16-minimal-experiment-bench-design.md` 作为后续 Dashboard 产品方向。旧文档和对应分支保留为历史记录，其中后端兼容代码和验证逻辑可以选择性复用，但极简 preset 页面不作为最终界面。

## 2. 当前基础与问题

项目已经具备以下能力：

- `Final_Project/Benchmark_UI/server.py` 提供本地 HTTP 服务、白名单作业启动、状态查询和结果读取；
- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/unified_runner.py` 提供统一模型适配与实验执行；
- `unified_schema.py` 和 `unified_results.py` 定义 trial 字段及 `metadata.json`、`trials.csv`、`summary.csv`、`failures.csv`；
- `Final_Project/Robosuite_MuJoCo_Sim/` 提供 robosuite/MuJoCo rollout、原生 Viewer、帧记录和 Tiny BC 训练/评估；
- 当前 Dashboard 已能启动部分作业并读取多类结果。

现有问题不是缺少按钮，而是职责和流程混合：

- 环境、模型、任务和执行参数分散在不同入口；
- Unified Runner、Matrix、Tiny BC 和 Viewer 被当成相互独立的工具；
- Viewer debug 与正式实验没有统一 run 身份；
- 结果目录和字段可以读取，但缺少统一的运行清单和完整性状态；
- 页面同时展示过多控制、调试和结果内容，用户难以判断下一步操作。

## 3. 已批准的产品决策

### 3.1 总体交互

采用分阶段方案：

1. `设置实验`：构造并验证一次完整实验；
2. `运行与观察`：只关注当前运行、日志、指标和 Viewer；
3. `实验结果`：检查数据完整性、比较结果和回放 trial。

每个阶段只显示当前需要的信息。页面顶部保留阶段导航和当前运行目标状态，右侧可保留本次实验摘要，但不展示与当前阶段无关的大型工具面板。

### 3.2 环境设置

采用结构化环境配置，而不是只选 preset 或实现拖拽式场景编辑器。

第一版允许配置：

- 仿真套件和任务，例如 robosuite `PickPlace`；
- 机器人和控制器；
- 场景物体及受支持的初始位置参数；
- 相机和图像设置；
- 随机化开关和随机种子；
- 最大执行步数；
- 运行前场景预览。

第一版不提供在 Viewer 中拖动物体并保存场景的可视化编辑器。

### 3.3 模型选择

默认是单模型运行。用户显式开启“批量比较模式”后，可以选择多个模型。

批量比较必须：

- 共享同一份环境和任务版本快照；
- 为每个模型创建独立 child run；
- 用 parent batch 统一展示进度和比较结果；
- 单个模型失败后继续运行其他模型；
- 公共环境或输出路径失败时停止整个 batch。

### 3.4 任务要求

任务不是只有一段 prompt。任务版本至少包含：

- `task_id`；
- 版本号；
- 自然语言指令；
- 目标物体或目标区域；
- 可执行的成功条件；
- 最大步数；
- 可选说明。

用户从任务模板创建新版本。已经被 run 引用的任务版本不可原地覆盖。

### 3.5 数据记录

核心记录始终开启，不能被用户关闭：

- 实验配置和版本快照；
- Git commit；
- 随机种子；
- 模型和部署配置；
- trial 成败和失败类型；
- 推理、端到端延迟及可获取的资源指标；
- 作业日志和时间戳；
- 产物路径。

帧序列和视频属于可选的重型记录。用户可设置是否保存和 frame stride。

### 3.6 MuJoCo Viewer

采用“原生 Viewer + 网页状态/低频预览”的组合：

- 操纵台负责打开、关闭和重新打开受控的原生 MuJoCo Viewer；
- 原生 Viewer 在独立桌面窗口中显示实时仿真；
- 网页显示 Viewer 状态、当前 trial/step 和低频预览帧；
- 关闭或意外退出 Viewer 不会停止实验；
- 仿真进程失败和 Viewer 展示进程失败必须区分；
- 第一版不把连续截图冒充原生 Viewer，也不在网页中嵌入原生窗口。

### 3.7 训练范围

第一版支持项目现有的 Tiny BC/BC 类轻量训练，包含：

- 选择训练数据集；
- 设置该训练器真实支持的主要参数；
- 显示阶段、日志、训练/验证指标和 checkpoint；
- 训练完成后启动 MuJoCo rollout 验证策略。

界面只能显示执行器真实产生的训练指标。若 Tiny BC 训练是闭式或单阶段计算，界面显示阶段进度与最终 train/validation metrics，不伪造 epoch 或 loss 曲线。只有迭代式训练器发出 epoch/loss 事件时才绘制对应曲线。

### 3.8 部署目标

统一运行目标选择器包含：

- `pc_local`；
- `jetson_local`；
- `jetson_quantized`；
- `jetson_remote_client`；
- 需要兼容已有 schema 时显示相应 PC remote/server 模式。

当前不可用的目标仍可见，但必须禁用启动并显示原因，例如“Jetson 未连接”或“量化运行时未安装”。第一版不负责自动安装 Jetson 环境或进行任意远程管理。

## 4. 信息架构与阶段边界

### 4.1 设置实验

设置阶段只负责创建和验证一个 `ExperimentSpec`。

页面内容：

- 模式：模型评估或轻量训练；
- 运行目标；
- 环境、机器人、场景、相机、种子和最大步数；
- 单模型或批量模型选择；
- 模型运行精度和量化配置；
- 任务模板、任务版本及成功条件；
- trials、warmup 和记录方式；
- 训练模式下的数据集和训练器参数；
- 场景预览；
- 配置兼容性检查；
- 预计运行数和输出位置预览。

主要操作：

- `保存实验模板`；
- `验证配置`；
- `进入运行与观察`。

验证未通过时不能启动。前端永远不提交原始 shell 命令。

### 4.2 运行与观察

作业启动后 `ExperimentSpec` 锁定。修改配置必须复制为新实验。

页面内容：

- run id 或 batch id；
- 状态、当前阶段和运行时间；
- 当前模型、任务、trial、step；
- 已完成数量和总数；
- 日志尾部；
- 评估指标：成功状态、延迟、内存、失败信息；
- 训练指标：执行器真实提供的阶段、epoch、loss、train/validation metrics 和 checkpoint；
- Viewer 状态及低频预览。

主要操作：

- 打开、关闭或重新打开 Viewer；
- 刷新状态；
- 停止当前受控作业；
- 在终态后进入结果页。

停止动作只针对作业注册表中的已知子进程。停止后保留全部已有日志和部分结果。

### 4.3 实验结果

结果阶段先检查完整性，再展示指标。

页面内容：

- 作业状态与结果完整性两个独立状态；
- 必需文件存在性和解析状态；
- 单次运行成功率、延迟、内存和失败类型；
- batch 内相同环境/任务下的模型对比；
- trial 输入、动作、成败原因和逐步记录；
- 帧或视频回放；
- 训练指标、checkpoint 和验证 rollout；
- 原始产物路径。

`按此配置重新运行` 会复制旧 `ExperimentSpec` 并创建新 run id，绝不覆盖旧结果。

## 5. 模块化架构

保留轻量 Python HTTP 服务作为本地入口，但把职责从单一 `server.py` 中拆出。

### 5.1 前端模块

- `Setup Stage`：编辑、预览和验证 `ExperimentSpec`；
- `Run Stage`：显示作业状态、事件、日志和 Viewer 控制；
- `Results Stage`：完整性检查、归一化展示、比较和回放；
- `Shared Store`：保存当前 draft、active run id 和 catalog，不保存后端权威运行状态。

### 5.2 后端模块

#### Experiment Catalog

统一读取和返回：

- 部署 profile；
- MuJoCo 环境和受支持参数；
- 现有 model catalog；
- 任务模板与版本；
- 训练器和数据集；
- 记录能力。

Catalog 适配已有 YAML、CSV、JSON 和 policy metadata。前端不直接解析项目文件。

#### Config Validator

负责：

- schema 和数值边界；
- 环境、模型、部署模式和任务兼容性；
- 成功条件完整性；
- 模型或 checkpoint 可用性；
- 输出路径可写性；
- 可选设备和依赖预检；
- dry-run 摘要。

#### Experiment Orchestrator

负责：

- 创建 run id 和 batch id；
- 冻结 spec；
- 选择白名单 executor；
- 管理作业生命周期和子进程；
- 将执行器输出转换为统一事件；
- 为 batch 创建和调度 child runs。

Orchestrator 不包含模型推理、训练算法或 MuJoCo 控制逻辑。

#### Viewer Controller

负责：

- 为允许的仿真作业构建 Viewer 启动参数；
- 记录 Viewer 进程和状态；
- 防止同一 run 重复打开多个 Viewer；
- 独立处理 Viewer 退出；
- 不影响主作业状态。

#### Run Recorder

负责创建控制目录、原子更新 manifest、追加事件和日志，并记录 executor 的真实产物目录。

#### Result Reader

负责把 unified runner、robosuite benchmark 和训练器的现有结果映射为统一前端对象。它不重写原始 CSV，也不假设所有执行器拥有相同的可选指标。

### 5.3 现有执行器

第一版复用：

- Unified Runner；
- scripted/VLM/BC MuJoCo rollout；
- Tiny BC/BC 训练脚本；
- 后续 Jetson local、quantized 和 remote-client executor。

所有执行器通过白名单注册表接入。API payload 只引用 executor key 和结构化参数。

## 6. 核心数据契约

### 6.1 ExperimentSpec

`ExperimentSpec` 是唯一的实验输入，结构如下：

```json
{
  "schema_version": 1,
  "name": "smolvlm_blue_can_pc",
  "mode": "evaluation",
  "deployment_profile": "pc_local",
  "environment": {
    "suite": "robosuite",
    "task": "PickPlace",
    "robot": "Panda",
    "controller": "OSC_POSE",
    "scene_config": "stable_pick_v1",
    "camera": "agentview",
    "randomization": false,
    "seed": 42,
    "max_steps": 500
  },
  "model_selection": {
    "mode": "single",
    "model_keys": ["smolvlm2_500m"],
    "runtime_precision": "fp16",
    "quantization": "none"
  },
  "task": {
    "task_id": "blue_can_pick",
    "version": 3,
    "instruction": "Pick up the blue can and place it in the bin.",
    "expected_target": "blue_can",
    "success_condition": "object_in_bin",
    "max_steps": 500
  },
  "execution": {
    "trials": 5,
    "warmup": 1
  },
  "recording": {
    "core": true,
    "save_frames": true,
    "save_video": false,
    "frame_stride": 20
  },
  "training": null
}
```

训练模式把 `mode` 改为 `training`，并填写受选定 trainer schema 约束的 `training` 对象。评估和训练不共享无意义的字段。

### 6.2 RunManifest

`RunManifest` 是运行状态和产物索引，至少包含：

```json
{
  "schema_version": 1,
  "run_id": "run_20260717_120000_ab12cd34",
  "parent_batch_id": null,
  "status": "running",
  "result_integrity": "pending",
  "phase": "trial",
  "spec_path": "spec.json",
  "git_commit": "3267416",
  "executor_key": "unified_evaluation",
  "timestamps": {},
  "progress": {},
  "viewer": {
    "status": "open",
    "pid": 1234
  },
  "artifacts": [],
  "error": null
}
```

允许的作业状态：

- `queued`：等待执行；
- `running`：执行器仍在运行；
- `completed`：执行器正常结束；
- `failed`：执行器或公共基础条件失败；
- `stopped`：用户主动停止。

结果完整性独立记录为：

- `pending`：运行尚未到达可检查状态；
- `complete`：该执行器定义的全部必需产物存在且可解析；
- `partial`：存在可用产物，但部分必需产物缺失或损坏；
- `unavailable`：没有可解析的实验结果。

因此，失败作业可以同时表示为 `status=failed`、`result_integrity=partial`。界面必须同时展示两个字段，不能因为作业失败而隐藏已经产生的有效数据。

### 6.3 控制记录目录

```text
Final_Project/Benchmark_UI/state/runs/<run_id>/
  spec.json
  manifest.json
  events.jsonl
  job.log
```

该目录只保存控制状态和索引。执行器继续在现有 `results/` 或 `outputs/` 下保存：

- `metadata.json`；
- `trials.csv` 和/或 `trials.jsonl`；
- `summary.csv`；
- `failures.csv`；
- frames/videos；
- model/checkpoint artifacts。

`manifest.json` 记录真实路径。第一版不移动、覆盖或复制历史结果。

## 7. 状态与数据流

标准流程：

```text
编辑 draft
  -> POST validate
  -> validated ExperimentSpec
  -> create run / batch
  -> freeze spec + create manifest
  -> start executor
  -> append events and artifacts
  -> optionally open native Viewer
  -> terminal status
  -> normalize and display results
```

主作业状态机：

```text
draft -> validated -> queued -> running -> completed | failed | stopped
```

Viewer 使用独立子状态：

```text
closed -> opening -> open -> closed | error
```

Viewer 状态不得把主作业从 `running` 改成 `failed`。

## 8. API 设计

保留当前 endpoint 作为兼容层，并增加以 console 为边界的结构化 API：

| Endpoint | Method | 用途 |
| --- | --- | --- |
| `/api/console/catalog` | GET | 返回环境、模型、任务、部署和训练 catalog |
| `/api/console/validate` | POST | 校验 ExperimentSpec 并返回 dry-run |
| `/api/console/runs` | POST | 创建单 run 或 batch |
| `/api/console/runs` | GET | 列出 run 和 batch |
| `/api/console/run?id=<id>` | GET | 返回 manifest、事件摘要和日志尾部 |
| `/api/console/stop` | POST | 停止已注册的 run |
| `/api/console/viewer/open` | POST | 为指定 run 打开 Viewer |
| `/api/console/viewer/close` | POST | 关闭指定 run 的 Viewer |
| `/api/console/results?id=<id>` | GET | 返回归一化结果和完整性报告 |

API 不接受命令字符串、任意模块路径或工作目录。

## 9. 错误处理

### 9.1 预检错误

预检错误阻止启动，并按字段返回：

- 无效或不兼容配置；
- 缺失模型、checkpoint、数据集或场景；
- 设备不可用；
- 输出路径不可写；
- 任务成功条件不完整。

### 9.2 运行错误

运行失败时必须记录：

- 失败阶段；
- failure type；
- 返回码；
- 异常或错误摘要；
- 日志路径和尾部；
- 已完成进度；
- 已生成产物。

### 9.3 Batch 错误

- 模型加载、OOM、推理或模型特定失败：child run 失败，batch 继续；
- 环境无法创建、公共数据无效或输出根不可写：停止未开始的 child runs，batch 失败；
- batch 结果展示每个 child 状态，不能只显示一个总失败。

### 9.4 恢复规则

- 浏览器刷新后，从后端 manifest 恢复 active run；
- 服务重启后，无法确认仍存活的旧作业标记为 `failed`；若已有可读产物，则 `result_integrity` 标记为 `partial`，不得假装继续运行；
- manifest 使用临时文件加原子替换更新，events 使用 append-only JSONL；
- 损坏的单个结果文件不能阻止其他可读产物展示。

## 10. 安全与复现

- 所有 executor 和 Viewer 命令来自白名单注册表；
- 数值和枚举字段在后端再次验证；
- 所有路径解析后必须位于项目允许目录；
- stop 只操作作业注册表中的 PID；
- 旧结果永不被重新运行或结果页操作覆盖；
- 每次 run 固化 Git commit、配置版本、随机种子和命令等价信息；
- 不支持任意 shell、网页代码编辑、多用户权限或公网远程控制。

## 11. 界面设计规则

- 桌面端为主，目标宽度为 1280px 及以上；较窄屏幕允许纵向堆叠，但不承诺移动端运行实验；
- 顶部三阶段导航始终可见；
- 运行目标连接状态始终可见；
- 设置页使用分组表单和右侧实验摘要；
- 运行页让 Viewer/预览和进度成为主内容，日志和指标为辅助内容；
- 结果页先显示完整性，再显示结论；
- 高风险操作使用明确文字和确认；
- 不把 Matrix、Tiny BC、Viewer 当成互不相关的顶级入口；
- 调试详情可以折叠，但正式运行信息不能隐藏在 Debug Tools 中。

## 12. 第一版范围

### 12.1 必须实现

- 三阶段操纵台；
- 结构化环境配置和场景预览；
- 单模型与 batch 模型比较；
- 版本化任务编辑；
- PC local 真实运行；
- Jetson/remote profile 及可用性状态；
- Unified Runner 评估；
- Tiny BC/BC 轻量训练和验证 rollout；
- 原生 Viewer 生命周期控制；
- 自动核心记录和可选帧记录；
- 结果完整性、比较、失败和 trial 视图。

### 12.2 不在第一版

- 浏览器内嵌原生 Viewer；
- 拖拽式场景编辑器；
- 大型 VLA/VLM 完整微调平台；
- 自动安装或任意管理 Jetson；
- MCP；
- 多用户或云端控制；
- 自动归档为论文 Evidence。

## 13. 迁移策略

后续实现不直接合并极简实验 bench 作为最终页面，而是按以下顺序迁移：

1. 保留现有 `server.py` 为入口，先抽出 catalog、validator、orchestrator、recorder、viewer 和 result reader；
2. 用兼容适配器包裹当前 Unified Runner、MuJoCo benchmark 和训练脚本；
3. 引入 `ExperimentSpec` 和 `RunManifest`，不修改历史结果；
4. 实现三个独立前端阶段；
5. 旧 API 和旧页面在新流程验证完成前保留；
6. 新流程通过回归测试后，再把旧控件降级为兼容/debug 路径。

## 14. 测试策略

### 14.1 单元测试

- ExperimentSpec schema 和边界；
- catalog 归一化；
- 环境/模型/任务/部署兼容性；
- run 和 batch 状态机；
- manifest 原子更新和事件追加；
- result reader 对缺失字段的容错；
- Viewer 命令构建和状态隔离。

### 14.2 后端集成测试

- validate/dry-run；
- 创建 mock run；
- 创建真实 unified smoke run；
- 创建轻量训练 run；
- 停止作业并保留部分数据；
- batch 中一个模型失败后继续；
- Viewer open/close/reopen；
- 服务刷新后恢复 manifest；
- 读取现有历史结果。

### 14.3 浏览器流程测试

- 完成设置、预检、运行和结果三个阶段；
- 校验错误定位到具体字段；
- 单模型与 batch 切换；
- 运行期间锁定 spec；
- 刷新页面后恢复 active run；
- `status` 与 `result_integrity` 的组合状态正确；
- 结果完整性和 artifact path 正确。

### 14.4 原生人工验证

Windows 上至少验证：

- Viewer 从操纵台打开；
- Viewer 可关闭和重开；
- Viewer 关闭不终止主实验；
- 训练后 rollout 可在 Viewer 中观察；
- 多次操作不会遗留重复 Viewer 进程。

## 15. 验收标准

第一版完成时，用户无需编辑 YAML 或运行命令即可：

1. 配置一个结构化 MuJoCo 实验环境；
2. 选择一个模型或创建公平的多模型比较；
3. 编辑并保存带成功条件的任务版本；
4. 预检并启动一次模型评估或轻量训练；
5. 打开原生 Viewer 观察 rollout；
6. 查看实时状态、日志和真实指标；
7. 停止或完成运行后找到配置、trial、失败、指标、帧和 checkpoint；
8. 分别判断作业是 completed、failed 还是 stopped，以及结果是 complete、partial 还是 unavailable；
9. 用相同配置创建新 run，而不覆盖历史结果。

达到上述标准后，操纵台才算实现了“设置、执行、观察、记录、比较”这一完整实验闭环。
