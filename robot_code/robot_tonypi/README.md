# TonyPi 比赛程序使用手册

本文件只讲“使用者如何准备、测试和运行”。程序内部各文件职责见 [FILES.md](FILES.md)，任务状态和判断分支见 [robot_decision_tree.html](robot_decision_tree.html)。

## 1. 程序做什么

TonyPi 会完成以下比赛流程：

```text
AprilTag 初始定位
→ 按当前 pose 选择最近的未处理 Tag/屏幕
→ 根据 Tag 四角判断西/东/南/北面
→ 生成 Tag 所在面中心正前方 15 cm 的四向任务位姿
→ 先导航到地图安全接近点，再小步对准 15 cm 和标准 yaw
→ 15 cm 到达几何门通过后确认 screen_id
→ 只对当前目标重新拍摄并由 FPGA 识别花朵
→ 已是目标花：记录，不换花
→ 不是目标花：在当前位置复核完整换花安全门
→ 举左手，向对应 Worker 发送换花请求
→ 收手站立
```

导航途中会继续框取屏幕并绑定左上 AprilTag，但不会裁剪花朵、调用分类器或投票，也不会为了观察其他屏幕而停靠或切换目标。每处理一个目标后，程序按最新 pose 到各 15 cm `task_target_xy` 的欧氏距离重新选最近目标；同距离按 screen/tag ID 升序。

地图、Tag 世界坐标和 AprilTag 定位算法没有变化。原约 34 cm `target_xy` 现在只作为 A* 可到达的内部安全接近点，不能宣布任务到达，也不能触发分类。最终任务位姿来自 Tag 固定 X/Y 平面，身体 yaw 只能是 `0°`、`-180°`、`+90°`、`-90°`。程序先通过只检查定位与几何的“到达几何门”允许分类；分类为非目标花后，再通过包含花朵状态和 Worker 映射的“完整换花安全门”允许举手和请求。

## 2. 使用前确认

以下命令均在机器人上执行。示例安装目录为：

```text
/home/pi/robot_tonypi
```

先登录机器人，例如：

```bash
ssh pi@192.168.31.220
cd /home/pi
```

确认三个目录存在：

```bash
ls /home/pi/robot_tonypi
ls /home/pi/robotall
ls /home/pi/test
```

停止可能同时控制机器人舵机的 TonyPi 默认服务：

```bash
sudo systemctl stop tonypi
```

确认 Python 可以导入依赖：

```bash
cd /home/pi
python3 -c "import cv2, numpy, robotall; print(robotall.__file__)"
```

注意 `robotall.__file__` 可能指向已安装的：

```text
/home/pi/.local/lib/python3.11/site-packages/robotall/
```

如果希望使用 `/home/pi/robotall` 中刚上传的版本，需要先明确确认版本，再单独安装；不要在比赛前临时覆盖一个已经可用的版本。

## 3. 检查动作组

普通导航动作由 TonyPi SDK 从这里读取：

```text
/home/pi/TonyPi/ActionGroups/
```

至少检查程序使用的关键动作组：

```bash
ls /home/pi/TonyPi/ActionGroups/turn_left_small_step_s80.d6a
ls /home/pi/TonyPi/ActionGroups/turn_right_small_step_s80.d6a
ls /home/pi/TonyPi/ActionGroups/go_forward_fast.d6a
```

仓库内附带的是自定义左右转动作组：

```text
/home/pi/robot_tonypi/action_groups/
```

如果 TonyPi 默认动作组目录缺少这些自定义转身文件，可在确认文件来源后复制：

```bash
cp /home/pi/robot_tonypi/action_groups/*.d6a /home/pi/TonyPi/ActionGroups/
```

## 4. 配置比赛参数

主要配置文件：

```text
/home/pi/robot_tonypi/config/competition_config.json
```

### 4.1 确认 screen → worker 编号

比赛编号规则已经确认：视觉绑定的 AprilTag ID、`screen_id` 和 NFC `worker_id` 完全相同。

```text
Tag 25 → screen_id 25 → worker_id 25
```

程序直接使用 `worker_id = screen_id`，不需要在配置文件中填写手动映射。

### 4.2 检查现场标定参数

默认交互参数包括：

```text
interaction_distance_cm = 15
interaction_distance_tolerance_cm = 4
interaction_yaw_tolerance_deg = 10
sensor_left_offset_cm = 5
interaction_lateral_tolerance_cm = 4
left_hand_body_offset_cm = 0
```

其中 `left_hand_body_offset_cm = 0` 只是未知机械尺寸的占位值。正式比赛前应现场确认左手相对身体中心的横向偏移、最佳读卡距离和容差。

## 5. 推荐执行顺序

不要直接从正式换花开始。推荐按以下顺序逐级测试。

### 第一步：运行无硬件测试

```bash
cd /home/pi/robot_tonypi
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

这些测试不会驱动机器人，主要检查交互安全门、Worker 异常收尾和屏幕左侧 AprilTag 绑定。

### 第二步：完全无硬件 dry-run

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --dry-run \
  --debug \
  --debug-host 0.0.0.0 \
  --debug-port 8090 \
  --time-limit-s 60
```

`--dry-run` 不连接真实相机、控制板、动作组、FPGA 或 Worker，适合检查程序能否启动和配置能否加载。它不能验证真实定位、视觉识别或运动精度。

因为 dry-run 没有真实相机/FPGA，它不会伪造花朵识别结果；用于验证启动、配置与“无硬件连接”语义，不用于跑完整目标完成数。

### 第三步：只测试定位

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode localize \
  --target-flower hehua \
  --debug \
  --debug-host 0.0.0.0 \
  --debug-port 8090
```

这个模式会真实打开相机、转动头部并检测 AprilTag。如果第一次扫描没有看到 Tag，程序可能执行搜索转身或后退动作，所以机器人周围仍需留出安全空间。

### 第四步：导航到最近目标后识别一次

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode harvest \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --debug \
  --debug-host 0.0.0.0 \
  --debug-port 8090
```

程序先定位，选择最近目标并真实导航；确认进入到达状态后，只对当前目标绑定的屏幕裁剪 28×28 图像并调用 FPGA 分类器。该模式不执行实体换花。机器人会移动，周围必须留出安全空间。

### 第五步：真实导航，但模拟换花

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red \
  --robot-id red-1 \
  --robot-secret 1234 \
  --skip-change \
  --debug \
  --debug-host 0.0.0.0 \
  --debug-port 8090 \
  --time-limit-s 600
```

`--skip-change` 与 `--dry-run` 不同：

| 行为 | `--dry-run` | `--skip-change` |
|---|---:|---:|
| 真实相机和 AprilTag | 否 | 是 |
| 真实 FPGA 分类 | 否 | 是 |
| 真实动作组和导航 | 否 | 是 |
| 真实最终对准 | 否 | 是 |
| 真实举左手 | 否 | 否 |
| 真实 `send_request` | 否 | 否 |

因此 `--skip-change` 仍会让机器人移动，只是最后的举手和 Worker 请求被模拟。`--skip-api` 是它的旧兼容别名。

### 第六步：正式运行

确认以下条件全部满足后，删除 `--skip-change`：

- 相机画面和 AprilTag 定位稳定；
- FPGA 分类服务可访问；
- 导航动作组都能执行；
- 已确认视觉绑定到正确的 `screen_id`；程序会使用相同编号作为 `worker_id`；
- team、robot-id 和 secret 与机器人注册信息一致；
- 15 cm、身体 yaw、横向读卡位置和左手偏移已现场验证；
- 场地周围安全，操作员可以随时停止机器人。

正式命令：

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team YOUR_TEAM \
  --robot-id YOUR_ROBOT_ID \
  --robot-secret YOUR_SECRET \
  --debug \
  --debug-host 0.0.0.0 \
  --debug-port 8090 \
  --time-limit-s 600
```

正式换花顺序是：

```text
最终位姿检查
→ stand
→ lift_left_hand(stand=False)
→ 再次检查位姿
→ send_request
→ 等待 Worker 响应
→ finally: stand
```

只有 `result["ok"] == True` 才会将屏幕标记为 `CHANGED`。

## 6. 打开 Debug 页面

启动参数必须包含：

```text
--debug --debug-host 0.0.0.0 --debug-port 8090
```

如果机器人 IP 是 `192.168.31.220`，在同一局域网电脑中打开：

```text
http://192.168.31.220:8090
```

在机器人上检查监听状态：

```bash
ss -lntp | grep 8090
```

Dashboard 会显示：

- 相机标注图和场地地图；
- 当前 pose、定位置信度和导航路线；
- 每个 Screen 的状态和识别结果；
- interaction target、distance/yaw/lateral error；
- interaction ready、左手状态、Worker 请求和响应；
- 定位、动作、恢复和失败事件。

程序退出后，8090 端口也会关闭。日志默认保存在：

```text
/home/pi/TonyPi/debug_runs/<timestamp>/
```

## 7. 停止和故障处理

正常停止使用：

```text
Ctrl+C
```

程序会执行清理流程。交互事务无论成功、失败或异常都会在 `finally` 中尝试执行 `stand`。

常见错误：

### `Action group not found`

说明 `/home/pi/TonyPi/ActionGroups/` 中缺少配置引用的 `.d6a` 文件。先检查文件是否存在，不要只检查仓库内的 `action_groups/`。

### `saw_any_tag=False`

说明当前扫描没有检测到 AprilTag。检查相机画面、Tag 是否进入视野、距离、光照和 Tag 清晰度。

### 无法打开 8090

确认程序仍在运行、启动时带了 `--debug-host 0.0.0.0`，并使用机器人当前 IP 访问。

## 8. 参数速查

| 参数 | 作用 |
|---|---|
| `--mode mission` | 完整任务 |
| `--mode localize` | 只完成初始定位 |
| `--mode harvest` | 定位后导航到最近目标，并在到达后只识别该目标；不实体换花 |
| `--target-flower` | 本队目标花的拼音 API 名 |
| `--classifier-url` | FPGA 分类服务地址 |
| `--team` | 注册队伍名称 |
| `--robot-id` | 注册机器人 ID |
| `--robot-secret` | 注册密钥 |
| `--dry-run` | 完全无硬件模式 |
| `--skip-change` | 真实识别和导航，模拟最后换花事务 |
| `--debug` | 保存调试内容并启用 Dashboard |
| `--debug-host` | Dashboard 监听地址，局域网访问使用 `0.0.0.0` |
| `--debug-port` | Dashboard 端口，默认 8090 |
| `--time-limit-s` | 运行时限 |
| `--max-screens` | 测试用：达到指定实际成功数后停止 |
| `--start-x/y/yaw` | 测试用手动起始位姿 |

可用花名：

```text
bailianhua chuju hehua juhua lamei lanhua meiguihua
shuixianhua taohua yinghua yuanweihua zijinghua
```

## 9. 使用者最终检查清单

```text
[ ] 默认 TonyPi 服务已停止
[ ] robotall 可以导入
[ ] 动作组在 /home/pi/TonyPi/ActionGroups 中存在
[ ] 相机可取帧
[ ] AprilTag 可定位
[ ] FPGA 地址可访问
[ ] screen → worker 映射正确
[ ] 交互参数已现场标定
[ ] --skip-change 流程已成功完成
[ ] 正式命令已删除 --skip-change
[ ] 操作员可以随时 Ctrl+C
```
