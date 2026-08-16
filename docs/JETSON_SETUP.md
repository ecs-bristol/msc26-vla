# Jetson 环境配置指南

本文档说明如何在 Jetson Orin Nano 上搭建 SmolVLA 推理服务所需的环境。
仓库内的 `docker/jetson/Dockerfile` 和 `scripts/jetson/` 都假设以下环境已就绪。

> **如果这台 Jetson 已经由实验室/同学配置过**（Docker 镜像
> `libero-smolvla:jetson-0.1` 已构建、`~/vla/hf-cache` 已有模型缓存），你
> **不需要**重复执行第 0–5 节。直接用 SSH 登录（第 1 节）验证环境（第 6 节），
> 然后按 `docs/START_GUIDE.md` 第 4.2 节运行即可。
> 每台 PC 只需各自配置 WSL 环境（`docs/START_GUIDE.md` 第 3.1 节）。

## 0. 系统要求

- 设备：Jetson Orin Nano（或其他 Orin 系列）
- JetPack / L4T：与 `docker/jetson/Dockerfile` 基础镜像兼容的版本
  （基础镜像为 `nvcr.io/nvidia/pytorch:25.08-py3`，请按 NVIDIA 官方要求匹配）
- 存储：至少 20 GB 可用（Docker 镜像 + 模型缓存）
- 内存：建议 8 GB 以上

## 1. 启用 SSH

> 先把 Jetson 和 PC 连好并配好 IP：接线与两端静态 IP 配置见
> `docs/JETSON_PC_NETWORK.md`（默认 Jetson=`10.42.0.2`、PC=`10.42.0.1`，直连网段 `10.42.0.0/24`）。

```bash
sudo systemctl enable --now ssh
ip addr   # 记下 Jetson 的 IP（本文档示例固定为 10.42.0.2）
```

从 PC 验证：`ssh msc26vla@10.42.0.2`

## 2. 安装 Docker

在 Jetson 上安装 Docker（`docker.io` 或 JetPack 自带版本均可），并让当前用户可用：

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"   # 重新登录后生效
```

验证：`docker run --rm hello-world`

## 3. 安装 NVIDIA Container Toolkit（`--runtime nvidia`）

`scripts/jetson/run_container.sh` 依赖 `--runtime nvidia`，需要安装 NVIDIA
Container Toolkit（Jetson 使用 l4t 版本）。请按 NVIDIA 官方安装指南操作：

<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

安装完成后配置 Docker 运行时并重启：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 GPU 可用：

```bash
docker run --rm --runtime nvidia <基础镜像> nvidia-smi
```

> 如果验证失败，确认 JetPack 版本与基础镜像匹配，并重新按官方指南配置运行时。

## 4. 获取项目代码

```bash
mkdir -p ~/vla/project

# 方式 A：从 GitHub 克隆（推荐）
git clone -b smolvla-benchmark git@github.com:ecs-bristol/msc26-vla.git ~/vla/project

# 方式 B：从 PC 直接拷贝
scp -r /path/to/this/repo msc26vla@10.42.0.2:~/vla/project/
```

## 5. 构建服务镜像并准备模型缓存

```bash
cd ~/vla/project
docker build -f docker/jetson/Dockerfile -t libero-smolvla:jetson-0.1 .
mkdir -p ~/vla/hf-cache ~/vla/outputs
```

首次运行使用 `bootstrap` 模式联网下载模型到 `~/vla/hf-cache`，之后可
`offline` 模式离线运行（详见 `docs/START_GUIDE.md` 第 4.2 节）。

## 6. 验证

启动服务后，在 PC/WSL 上运行预检：

```bash
export JETSON_ENDPOINT=http://10.42.0.2:8081
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
bash scripts/wsl/run_jetson_remote_preflight.sh
```

输出 `remote preflight: ok` 即环境就绪。

## 7. 多人轮流使用同一台 Jetson

本项目设计为「一台 Jetson + 各自 PC/WSL」的轮流使用模式，请遵守以下约定：

- **同一时刻只允许一个人做 Jetson 远程推理**：SmolVLA 推理占用 GPU 显存和算力，
  并发会互相拖慢甚至 OOM。PC-local 模式不占用 Jetson，可以多人同时跑。
- **服务用完即关**：`start_smolvla_libero_service.sh` 在终端前台运行，按
  `Ctrl+C` 停止。不关闭会导致下一个人无法启动（端口 8081 被占用）。若提示
  `Address already in use`，在 Jetson 上执行 `docker ps` 查看残留容器。
- **Jetson 端 `~/vla/project` 是共享代码**：所有人都用它启动服务镜像，请保持
  与 GitHub `smolvla-benchmark` 分支同步：
  ```bash
  cd ~/vla/project && git pull
  ```
- **结果互不干扰**：每个人的评估结果写在各自 WSL 的 `~/vla/results/` 下，
  不会覆盖他人。
- **模型缓存共享**：`~/vla/hf-cache` 在 Jetson 上共享，首次 `bootstrap`
  下载一次即可，之后大家都能 `offline` 使用。

## 常见问题

- **`--runtime nvidia` 报错**：NVIDIA Container Toolkit 未安装或未配置，回到第 3 步。
- **Docker 拉取镜像失败**：确认网络可访问 NVIDIA 镜像仓库（`nvcr.io`）。
- **模型下载失败**：`bootstrap` 模式需要网络；离线模式要求模型已存在于
  `~/vla/hf-cache`（通过 `HF_HOME` 挂载进容器）。
