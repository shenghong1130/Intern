# Kria FPGA 花朵分类服务

## 连接信息

```text
Kria IP：192.168.31.81
Jupyter：http://192.168.31.81:9090
密码：xilinx
终端身份：root@kria
Python 环境：pynq-venv
部署目录：/root/jupyter_notebooks/fpga_flower_server
```

在浏览器打开 Jupyter 后，通过 `New → Terminal` 进入终端。

## 启动服务

先检查当前活动应用：

```bash
xmutil listapps
```

如果已有应用占用 FPGA，先卸载当前活动应用：

```bash
xmutil unloadapp
```

进入部署目录并启动服务：

```bash
cd /root/jupyter_notebooks/fpga_flower_server
python3 -u fpga_server_api_ready.py
```

看到以下输出表示启动成功：

```text
Overlay ready.
Running on http://192.168.31.81:8080
```

服务运行期间必须保持该终端开启。

## 当前接口约定

服务只提供一个分类接口：

```text
POST /predict
Content-Type: multipart/form-data
表单字段：image
```

服务会把上传图像转换为 `28 × 28` RGB 输入并送入当前 FPGA Overlay。成功响应同时包含 API 花名、中文花名、类别序号和置信度，主要字段为：

```json
{
  "flower": "yinghua",
  "flower_api": "yinghua",
  "flower_cn": "樱花",
  "predicted_class": "yinghua",
  "class_index": 0,
  "confidence": 0.95
}
```

缺少 `image` 字段或图像无法解码时，服务返回 `400`。当前 Flask 服务以 `threaded=False` 串行处理请求，端口固定为 `8080`。

TonyPi 的分类客户端默认请求地址为 `http://192.168.31.81:8080/predict`，HTTP 超时由机器人端配置控制；服务不可达、超时、`5xx`、`408` 或 `429` 都会被机器人任务层视为可恢复的分类服务异常，不等于目标 Tag 或 Screen 不存在。

## 关闭服务

### 1. 服务正在当前终端前台运行

如果当前终端仍然停留在：

```bash
python3 -u fpga_server_api_ready.py
```

直接按：

```text
Ctrl + C
```

即可停止前台运行的服务。这是最推荐、最简单的关闭方法。

### 2. 已经退出原来的终端，但服务仍然运行

首先查询进程：

```bash
ps aux | grep fpga_server_api_ready.py
```

找到对应的 Python 进程 PID，然后执行：

```bash
kill <PID>
```

如果普通 `kill` 无法结束进程，再使用：

```bash
kill -9 <PID>
```

### 3. 按脚本名直接关闭

也可以执行：

```bash
pkill -f fpga_server_api_ready.py
```

该命令会结束所有匹配此脚本名称的进程，因此使用前应确认没有其他不希望关闭的同名进程。

### 4. 检查服务是否已经关闭

执行：

```bash
ss -lntp | grep 8080
```

如果没有任何输出，表示 `8080` 端口已经没有进程监听，FPGA Flower Server 已经成功关闭。

如果仍然有输出，则根据输出中的 PID 查找并结束对应进程。

## 从 TonyPi 检查服务

在 TonyPi 终端检查 `8080`：

```bash
curl -i http://192.168.31.81:8080/predict
```

如果返回 `405 Method Not Allowed`，说明服务可达，但 `/predict` 只接受 POST 请求。

如果返回 `Connection refused`，说明服务尚未启动或服务进程已经退出。

确认机器人已人工放置在目标点并正对屏幕后，可执行安全的模拟换花测试：

```bash
cd /home/pi
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --skip-change
```

`--skip-change` 会使用真实相机和 FPGA 分类，但不会真实举手或发送 Worker 请求。
