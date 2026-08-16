# Jetson ↔ PC 网络连接指南

本文档说明 **Jetson 端如何与电脑连在一起**（接线 + 两端 IP 配置 + 验证），
是 `docs/START_GUIDE.md` 4.2 节和 `docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md`
的前提。SSH 与推理服务的使用见上述文档。

## 1. 拓扑选择

本项目固定使用网段 `10.42.0.0/24`，两种接法：

| 方案 | 接线 | Jetson IP | PC IP | 适用 |
| --- | --- | --- | --- | --- |
| A：网线直连（推荐，默认） | PC ↔ Jetson 一根网线 | `10.42.0.2`（静态） | `10.42.0.1`（静态） | Jetson 网口只连这一台 PC |
| B：同一路由器/交换机 | 两者都插路由器 | 视路由器而定 | DHCP | Jetson 还要上网、多人轮流用 |

- 方案 A 是仓库所有脚本默认值（`JETSON_ENDPOINT=http://10.42.0.2:8081`），零配置即可跑通。
- 方案 B 需要先用 `ip addr` 查到 Jetson 实际 IP，并把 `ssh` / `JETSON_ENDPOINT` 里的地址替换掉。
- WSL2 默认走 NAT，**通过 Windows 宿主的网卡访问 `10.42.0.2`**，所以只要 Windows 网卡配好，WSL 里直接就能通，不需要在 WSL 里再配 IP。

## 2. Jetson 端：确认网卡并配置静态 IP

> 提示：如果 Jetson 的网口还要用来上网，建议直接用方案 B；直连方案会占用网口。

```bash
# 查看网卡名（通常是 eth0 或 en*）和当前 IP
ip -br addr

# JetPack 的 Ubuntu 用 NetworkManager 管网络，用 nmcli 配置静态 IP
nmcli device status                                    # 确认网卡名，下面以 eth0 为例
nmcli con add type ethernet ifname eth0 con-name jetson-pc-link \
  ipv4.method manual ipv4.addresses 10.42.0.2/24
nmcli con up jetson-pc-link

# 验证
ip addr show eth0                                      # 应显示 inet 10.42.0.2/24
```

如果网卡已有自动配置的连接（如 `Wired connection 1`），也可以直接改它：

```bash
nmcli con mod "Wired connection 1" ipv4.method manual ipv4.addresses 10.42.0.2/24
nmcli con up "Wired connection 1"
```

## 3. PC 端（Windows）：配静态 IP `10.42.0.1`

以管理员身份打开 PowerShell：

```powershell
# 找到接 Jetson 的以太网适配器名（把 "以太网" 换成实际名称）
Get-NetAdapter | Format-Table Name, Status, LinkSpeed

# 设置静态 IP（网段 10.42.0.0/24，PC 用 .1，Jetson 用 .2）
netsh interface ip set address name="以太网" static 10.42.0.1 255.255.255.0
```

验证：

```powershell
ping 10.42.0.2
```

> 如果之前是 DHCP，改成静态后记得在 Jetson 侧重启连接或用 `nmcli con up` 重连。

## 4. 端到端验证清单

按顺序检查，全部通过即可开始实验：

```text
1. Jetson：ip addr 显示 10.42.0.2/24
2. Windows：ping 10.42.0.2 通
3. WSL：   ping 10.42.0.2 通（走宿主网卡）
4. Jetson：启动推理服务（start_smolvla_libero_service.sh offline）
5. WSL：   curl -s http://10.42.0.2:8081/health   # 返回 ok
6. WSL：   bash scripts/wsl/run_jetson_remote_preflight.sh  # 输出 remote preflight: ok
```

## 5. 常见问题

- **ping 不通**：确认两端网卡都显示正确 IP；换根网线；Windows 防火墙临时关掉或允许
  ICMP 再试；确认 Jetson 网口名不是 wlan0。
- **SSH 通但 8081 不通**：服务没启动或端口被占用，在 Jetson 上 `docker ps` 看残留容器。
- **直连后 Jetson 上不了网**：网口被直连占用了，改用无线或方案 B（接路由器）。
- **多人轮流用**：每台 PC 都配 `10.42.0.1`，Jetson 侧 `10.42.0.2` 不变；同一时刻只允许
  一个人启动推理服务（见 `docs/JETSON_SETUP.md` 第 7 节）。
