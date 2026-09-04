# 机器人局域网连接说明

本文说明如何在同一局域网内确认机器人在线，并通过 SSH 或 Jupyter Notebook 连接机器人。

## 1. 连接参数

| 项目 | 参数 |
|---|---|
| 机器人 IP 地址 | `192.168.31.xxx` |
| 示例 IP 地址 | `192.168.31.212` |
| SSH 用户名 | `pi` |
| SSH 密码 | `pi` |
| Jupyter 端口 | `8888` |

> 使用前请将示例中的 `192.168.31.212` 替换为目标机器人的实际 IP 地址。

## 2. 确认机器人在线

确保电脑和机器人连接到同一个局域网，然后在电脑终端执行：

```bash
ping 192.168.31.212
```

若持续出现来自该 IP 的回复，说明机器人已经联网且网络可达。按 `Ctrl+C` 停止 Ping。

## 3. 通过 SSH 连接

在电脑终端执行：

```bash
ssh pi@192.168.31.212
```

首次连接时可能出现主机指纹确认：

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入：

```text
yes
```

随后输入密码：

```text
pi
```

密码输入过程中终端不会显示字符，这是正常现象。登录成功后，即可在机器人系统中执行 Linux 命令。

退出 SSH：

```bash
exit
```

## 4. 通过 Jupyter Notebook 连接

在电脑浏览器地址栏输入：

```text
http://192.168.31.212:8888
```

页面打开后，即可查看、编辑并运行机器人上的 Notebook。若页面要求 Token 或密码，应使用机器人镜像中配置的 Jupyter 凭据。

## 5. Kria PYNQ 登录

Kria PYNQ 的连接参数：

| 项目 | 参数 |
|---|---|
| Kria IP | `192.168.31.81` |
| Jupyter 地址 | `http://192.168.31.81:9090` |
| 当前身份 | `root@kria` |
| Python 环境 | `pynq-venv` |

在浏览器中打开：

```text
http://192.168.31.81:9090
```

进入 Jupyter 后，选择：

```text
New → Terminal
```

终端当前身份应为 `root@kria`，并已进入 `pynq-venv`。执行以下命令确认环境：

```bash
whoami
hostname
pwd
```

端口 `8080` 用于 FPGA 花朵分类服务，只有启动服务器后才会开放。

## 6. 安全建议

默认账号和密码 `pi/pi` 仅适合受信任的封闭局域网。建议完成初次连接后修改密码：

```bash
passwd
```

不要将 SSH 的 22 端口或 Jupyter 的 8888 端口直接暴露到公网。
