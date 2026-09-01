# robosuite / MuJoCo 虚拟机械臂仿真框架

这个文件夹用于搭建一个轻量虚拟机械臂环境，作为后续 VLA/VLM 闭环实验的 simulator 端。

第一版目标：

```text
robosuite Panda Lift 环境
  -> 获取 RGB 相机图像
  -> 接收文本动作 token
  -> action adapter 转成机械臂控制量
  -> simulator step
  -> 保存观测图像和运行日志
```

## 为什么先用 robosuite

- 比 Isaac Sim 轻，适合先在笔记本 RTX 5060 上验证
- 内置 Panda / Sawyer / Kinova3 等机械臂
- 内置 Lift / PickPlace / Stack 等 manipulation 任务
- 可以直接获取相机图像，方便和 VLA/VLM 模型连接

## 推荐环境

robosuite 在 Windows 原生环境可能能跑，但最稳的是：

```text
Conda + Python 3.10
```

建议不要装进已有的 VLM `.venv`，单独建一个仿真环境：

```powershell
conda create -n vla_sim python=3.10
conda activate vla_sim
```

然后进入本文件夹：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Robosuite_MuJoCo_Sim"
```

安装依赖：

```powershell
pip install -r requirements-sim.txt
```

如果没有 Conda，也可以先试 venv，但 Python 3.10/3.11 会比 3.12 更稳：

```powershell
python -m venv .sim_venv
.\.sim_venv\Scripts\Activate.ps1
pip install -r requirements-sim.txt
```

## Windows 常见问题

robosuite 官方文档提到 Windows 上可能遇到 `C:\tmp\robosuite.log` 不存在的问题。可以先创建：

```powershell
mkdir C:\tmp
```

如果遇到渲染相关问题，脚本会默认设置：

```text
MUJOCO_GL=wgl
```

## 1. 检查环境

```powershell
python -m src.robot_sim.env_check
```

或：

```powershell
.\scripts\check_sim_env.ps1
```

## 2. 保存一张仿真相机图像

```powershell
python -m src.robot_sim.capture_lift_observation
```

输出会保存到：

```text
outputs/lift_frontview.png
```

## 3. 跑随机动作 demo

```powershell
python -m src.robot_sim.random_lift_demo --steps 50
```

## 4. 打开可视化窗口 demo

随机动作可视化：

```powershell
python -m src.robot_sim.visual_lift_demo --mode random --steps 200
```

文本动作可视化：

```powershell
python -m src.robot_sim.visual_lift_demo --mode text --actions move_forward move_down grasp move_up --repeat 20
```

也可以用 PowerShell 脚本：

```powershell
.\scripts\run_visual_lift_demo.ps1 -Mode random -Steps 200
```

或：

```powershell
.\scripts\run_visual_lift_demo.ps1 -Mode text -Actions move_forward,move_down,grasp,move_up -Repeat 20
```

## 5. 跑文本动作 demo

```powershell
python -m src.robot_sim.text_action_demo --actions move_forward move_down grasp --repeat 10
```

这个 demo 会把文本动作映射成连续控制量：

```text
move_forward -> 末端向前移动
move_left    -> 末端向左移动
move_up      -> 末端向上移动
grasp        -> 闭合夹爪
release      -> 打开夹爪
stop         -> 空动作
```

## 后续和 VLA/VLM 连接

后面可以把你已经跑通的 VLM baseline 接进来：

```text
robosuite camera image
  -> VLM/VLA model
  -> action text: move_forward / grasp / stop
  -> text_action_adapter
  -> robosuite env.step(action)
```

Jetson 到货后，可以把模型推理放到 Jetson，仿真仍然留在笔记本：

```text
Laptop simulator -> image + instruction -> Jetson inference -> action -> Laptop simulator
```
