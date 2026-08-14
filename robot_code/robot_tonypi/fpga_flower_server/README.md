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
