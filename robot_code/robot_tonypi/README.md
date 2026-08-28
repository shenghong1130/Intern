# TonyPi 比赛程序使用手册

本文只描述当前仓库的真实运行方式。模块职责见 [FILES.md](FILES.md)，完整状态与分支见 [robot_decision_tree.html](robot_decision_tree.html)，测试入口见 [tests/README.md](tests/README.md)。

## 1. 当前比赛流程

```text
启动并居中云台
→ 初始 AprilTag 定位
→ SELECT_NEAREST_TARGET
→ 锁定 TargetGoal(screen_id/tag_id/anchor/goal/yaw/generation)
→ 导航到 20 cm task target，并完成 cardinal yaw
→ 实时确认当前 Tag
→ 使用 15 秒内同 ID 的 Tag↔Screen 分类缓存，或有限拍摄新帧
→ flower == target：ALREADY_TARGET，不执行 final forward / NFC
→ flower != target：执行一次 interaction_forward_10cm
→ NFC Attempt 1
→ 成功：CHANGED
→ 失败：后退 10 cm、重新定位、最多 3 轮重新寻找当前目标并重新 FPGA
   → 当前目标已是 target：CHANGED，立即结束当前 Screen
   → 当前目标仍不是 target：重新导航到 20 cm task target，再执行 Attempt 2
   → 3 轮仍找不到当前目标：GAVE_UP 当前 Screen
→ Attempt 2 成功：CHANGED；失败：GAVE_UP
→ 如曾进入近距离交互位，完成后退和重新定位
→ MARK_TARGET_COMPLETE / 清理当前目标上下文
→ 重新选择最近目标
```

四条必须保持的不变量：

- `Localization success != Current target reacquired`：其他 Tag 可以更新机器人 Pose，但不能证明当前目标已重新出现。
- `CHANGED` 是 NFC 流程的终止状态；一旦进入，不再执行 retry、recalibration、reapproach 或 Attempt 2。
- Attempt 2 只有在 Attempt 1 失败、重新找到当前目标且 FPGA 明确得到 `flower != target_flower` 后才允许。
- NFC Retry 的目标重获最多 3 轮，不会循环到全局任务超时。

## 2. 当前坐标、目标和编号约定

- 场地为 `300 cm × 300 cm`，地图分辨率为 `5 cm`。
- Debug 地图左上角是世界坐标 `(0, 0)`；显示时世界 `x` 轴向下、世界 `y` 轴向右。
- 世界位置单位是厘米，yaw 单位是度；yaw `0°` 指向 `+X`，逆时针为正。
- 云台中心角是 `100°`；更大角度向左看，更小角度向右看。
- AprilTag ID、`screen_id` 和 NFC `worker_id` 必须完全相同。
- Screen 任务目标由完整建筑面的中心、cardinal 外法线和机器人横向补偿共同生成。
- 当前正式参数：`target_distance_cm=20.0`、`target_lateral_offset_cm=-1.0`、`target_final_forward_cm=10.0`。
- `target_xy`、`interaction_xy` 和 `task_target_xy` 指向同一个 20 cm 身体目标；Screen/Tag anchor 与机器人目标点不是同一个点。
- 当前配置排除 Screen：`2, 6, 9, 20, 28`。

## 3. 部署和运行环境

机器人代码目录：

```text
/home/pi/robot_tonypi
```

登录并检查目录：

```bash
ssh pi@192.168.31.220
ls /home/pi/robot_tonypi
ls /home/pi/robotall
ls /home/pi/TonyPi/ActionGroups
```

停止可能同时控制舵机的 TonyPi 默认服务：

```bash
sudo systemctl stop tonypi
```

确认依赖：

```bash
cd /home/pi
python3 -c "import cv2, numpy, requests, robotall; print(robotall.__file__)"
```

动作组实际从以下目录读取：

```text
/home/pi/TonyPi/ActionGroups/
```

仓库内 `action_groups/` 只是附带资源；缺少自定义转向动作组时，在确认来源后复制：

```bash
cp /home/pi/robot_tonypi/action_groups/*.d6a /home/pi/TonyPi/ActionGroups/
```

## 4. 当前运行配置

正式覆盖配置：

```text
/home/pi/robot_tonypi/config/competition_config.json
```

程序先深拷贝 `config.py` 的 `DEFAULT_CONFIG`，再递归覆盖 JSON。文档中的数字均按当前两者合并后的结果填写。

### 4.1 定位

| 配置 | 当前值 | 含义 |
|---|---:|---|
| `head_center_angle` | `100°` | 云台中心 |
| `scan_pan_angles` | `[100,135,65,155,45]` | 完整定位扫描顺序 |
| `startup_attempts` | `14` | 初始搜索动作预算 |
| `startup_search_actions` | `左转×4，后退×1` 循环 | 每次身体动作后重新完整扫描 |
| `min_tag_area_px` | `350` | 定位 Tag 最小面积 |
| `edge_margin_px` | `35` | 图像边缘过滤 |
| `no_tag_recovery_failures` | `2` | no-tag 恢复阈值 |
| `no_tag_recovery_cooldown_s` | `4.0` | 防止重复进入恢复 |

普通 `localize_scan()` 当前会按配置扫描多角度，并在第一个可接受视觉 Pose 处停止、回中。靠边界时可能过滤朝场外看的 pan。NFC 目标重获模式不同：即使其他 Tag 已成功定位，也继续扫描，直到当前目标 Tag 与 Screen 正确绑定。

### 4.2 导航和动作

| 配置 | 当前值 |
|---|---:|
| task target 到达半径 | `4 cm` |
| task target yaw 容差 | `10°` |
| 每目标最大导航步骤 | `80` |
| target-direct 距离 | `40 cm` |
| 普通导航最小 clearance | `25 cm` |
| 相同规划失败升级阈值 | `3` |
| `forward_fast` | `3.5 cm/周期` |
| `forward_micro` | `2.0 cm/周期` |
| `back_fast` | `-2.5 cm/周期` |
| 左/右平移 | `+4.0 / -3.0 cm/周期` |
| 大左/右转模型 | `+15 / -18°/逻辑动作` |

`navigate_to_screen()` 通过原子化 `TargetGoal` 导航到唯一 task target。它仍优先使用 target-direct、A*、approach/staging 和 start projection；但是当前 task-target 调用显式设置 `bypass_action_safety=True`，因此已选出的前进、后退、平移或转向不会再被 near-wall、corridor、footprint、center-free 等执行前安全门否决。普通非 task-target 导航仍保留这些安全与恢复逻辑。

自适应重定位使用动作数、运动不确定度、Pose 置信度和导航阶段：

- normal HIGH：最多 6 个动作，不确定度上限 6.0；
- staging HIGH：最多 5 个动作，上限 5.0；
- recovery HIGH：最多 4 个动作，上限 4.5；
- target-direct HIGH：最多 3 个动作，上限 3.5；
- LOW confidence、大转向、障碍紧张、进入 target-direct 前、ARRIVED 前会提前定位。

动作不确定度：直行 `0.6`、后退 `0.9`、平移 `1.0`、普通转向 `1.8`、大转向 `2.6` 每周期。只有接受新的视觉 Pose 才清零累计值。

### 4.3 视觉和交互

| 配置 | 当前值 |
|---|---:|
| Tag↔Screen 分类缓存 TTL | `15 s` |
| 同 Screen 分类最小间隔 | `1 s` |
| 最低分类置信度 | `0.2` |
| 目标确认新鲜帧上限 | `3` 次 |
| 目标可见性恢复 | `2` 轮 |
| final forward | `10 cm`，一次逻辑动作 |
| post-interaction retreat | `10 cm`，`back_fast` |
| NFC 单次物理尝试 deadline | `15 s` |
| NFC 物理尝试上限 | `2` 次 |
| NFC Retry 目标重获 | `3` 轮 |

导航/定位帧只在 `Tag ID == Screen ID`、crop 有效、置信度合格时写入最新绑定缓存。分类失败不会删除上一条成功缓存；缓存本身不会直接改变 Screen 状态。到达目标后仍必须实时看到当前目标 Tag，才能把同 ID 的缓存转换为 `VisualAuthorization`。

## 5. 推荐执行顺序

### 5.1 自动化测试和语法检查

```bash
cd /home/pi
python3 -m unittest discover -s robot_tonypi/tests -p 'test_*.py' -v
python3 -m compileall -q robot_tonypi
```

### 5.2 完全无硬件 dry-run

```bash
cd /home/pi
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --dry-run \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 60
```

`--dry-run` 不连接相机、动作组、FPGA 或 NFC。它验证启动和配置，不验证真实视觉与运动。

### 5.3 只定位

```bash
python3 -u -m robot_tonypi.main \
  --mode localize \
  --target-flower hehua \
  --debug --debug-host 0.0.0.0 --debug-port 8090
```

### 5.4 导航并识别一个目标

```bash
python3 -u -m robot_tonypi.main \
  --mode harvest \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --debug --debug-host 0.0.0.0 --debug-port 8090
```

`harvest` 会定位、选择目标、导航和确认分类，不执行 final forward 或 NFC。

### 5.5 真实视觉和导航、模拟换花

```bash
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --skip-change \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 570
```

`--skip-change` 保留真实相机、定位、导航和 FPGA 分类，但跳过专用 10 cm 动作、举手和 NFC；`--skip-api` 是同一参数的旧别名。

### 5.6 正式运行（直连 KV260 Worker）

```bash
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team YOUR_TEAM \
  --robot-id YOUR_ROBOT_ID \
  --robot-secret YOUR_SECRET \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 570
```

`direct` 是 `--classifier-mode` 的默认值。为保持旧部署兼容，上面的原命令无需增加任何参数：TonyPi 仍将 `crop_28x28` 编码为 JPEG，并只以 multipart `image` 字段直接提交给指定的 KV260 Worker。

### 5.7 正式运行（通过 Central Server）

运行前先确认 TonyPi 能访问 Central Server 的真实健康检查接口：

```bash
curl http://192.168.31.254:8000/health
```

还必须事先使用本次运行所需的 `student_id` 和密码上传状态为 ready 的 FPGA Artifact；否则 `/predict` 会返回 `404 student has no ready artifact`。密码通过环境变量传给 TonyPi，不要写入命令、README 或配置文件：

```bash
read -s STUDENT_PASSWORD
export STUDENT_PASSWORD

curl -X POST http://192.168.31.254:8000/fpga/artifacts \
  -F student_id=student01 \
  -F password="$STUDENT_PASSWORD" \
  -F bit=@design_1_wrapper.bit \
  -F hwh=@design_1_wrapper.hwh
```

上传成功后，在同一个终端中使用完全相同的 `student_id` 和 `STUDENT_PASSWORD` 启动 TonyPi：

```bash
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-mode central \
  --classifier-url http://192.168.31.254:8000/predict \
  --classifier-student-id student01 \
  --team YOUR_TEAM \
  --robot-id YOUR_ROBOT_ID \
  --robot-secret YOUR_SECRET \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 570
```

`192.168.31.254` 只是当前测试 Central Server 的 IP，实际部署时应替换为服务器真实的局域网 IP。`student01` 必须与上传 FPGA Artifact 时使用的 `student_id` 完全一致，`STUDENT_PASSWORD` 也必须与上传 Artifact 时设置的密码完全一致。也可通过 `--classifier-password` 显式传入密码，但正式运行推荐使用环境变量，避免密码出现在 shell 历史或进程命令行中。

```text
TonyPi
   │
   │ crop_28x28.jpg + student_id
   │ HTTP POST /predict
   │ Header: X-Student-Password
   ▼
Central Server :8000
   │
   │ 根据 student_id 查找 Artifact
   │ 调度可用 KV260
   ▼
KV260 Worker
   │
   ▼
PYNQ Overlay
   │
   ▼
AXI DMA → CNN FPGA IP → AXI DMA
   │
   ▼
分类结果
   │
   ▼
Central Server
   │
   ▼
TonyPi
```

TonyPi 不再硬编码具体 KV260 Worker。Central Server 先用 `student_id` 和 `X-Student-Password` 完成认证，再查找 Artifact 并调度 Worker；TonyPi 只知道 Central Server。若请求进入队列，TonyPi 会使用同一密码 Header，按 `request_id` 有间隔地查询 `/requests/{request_id}`，并在 180 秒 deadline 到达后返回可重试错误，由现有任务恢复逻辑决定是否再次分类。

运行结束后可清除当前 shell 中的密码变量：

```bash
unset STUDENT_PASSWORD
```

可用花名：

```text
bailianhua chuju hehua juhua lamei lanhua meiguihua
shuixianhua taohua yinghua yuanweihua zijinghua
```

## 6. 失败与恢复的实际语义

- 初始定位失败不会立即退出；主循环反复执行初始搜索，直到得到 Pose 或全局 timeout。
- `no_tag` 与“看见 Tag 但没有 Pose”分开计数；no-tag 恢复只在无 Tag 达阈值且 Pose 缺失、朝场外或边界受困时启动。
- 普通规划会依次考虑 exact/direct、start projection、reachable approach/staging、A*、动作空间规划；相同输入失败 3 次后升级到 interior recovery。
- near-wall 恢复顺序为后退、左右平移、小转向；普通动作全被拒绝时可进入 bounded forced escape。真实动作后会重新定位。
- 单个导航目标失败进入临时失败集合，不是永久黑名单；所有未完成目标都临时失败时执行全局恢复并释放它们。
- NFC GAVE_UP 是单独集合：该 Screen 在本次进程中不再立即选择，但 mission 继续处理其他目标。
- `MISSION_BLOCKED`、`NAVIGATION_BLOCKED` 是可观测状态，不是 `run_mission()` 的自动进程终点。
- 所有 Screen 已 `CHANGED` 或 `ALREADY_TARGET` 后进入 `MISSION_COMPLETE` 并保持程序/Dashboard 运行，直到全局 timeout。
- 当前主循环唯一正常自动终止是 `MISSION_TIMEOUT`；`Ctrl+C` 仍会安全清理并以 130 退出。

## 7. Debug Dashboard

启用：

```text
--debug --debug-host 0.0.0.0 --debug-port 8090
```

访问：

```text
http://192.168.31.220:8090
```

输出目录：

```text
/home/pi/TonyPi/debug_runs/<timestamp>/
```

Dashboard/文件包括：

- `latest_state.json`：MissionState、Pose、TargetGoal、缓存、NFC retry、恢复计数和最近事件；
- `latest_map.jpg`：Robot Pose、Screen anchor、task goal、当前导航 goal、recovery waypoint 和路径；
- `latest_annotated.jpg`：Tag 与 Screen 绑定标注；
- `events.jsonl` 与 `interaction_calls.jsonl`：状态和 NFC 审计。

## 8. 停止与常见检查

人工停止：

```text
Ctrl+C
```

检查 Dashboard：

```bash
ss -lntp | grep 8090
```

检查 FPGA：

```bash
curl -i http://192.168.31.81:8080/predict
```

返回 `405 Method Not Allowed` 表示服务可达且 `/predict` 正确要求 POST；`Connection refused` 表示服务未运行或网络不可达。

## 9. 现场检查清单

```text
[ ] TonyPi 默认服务已停止
[ ] robotall、cv2、numpy、requests 可导入
[ ] /home/pi/TonyPi/ActionGroups 中动作组完整
[ ] 相机画面、AprilTag 定位和云台方向正确
[ ] Debug 地图坐标轴与现场一致：左上 (0,0)，x 向下，y 向右
[ ] FPGA /predict 可访问
[ ] Tag ID == screen_id == worker_id
[ ] 20 cm / -1 cm task target 和 cardinal yaw 已现场验证
[ ] interaction_forward_10cm 与 10 cm retreat 已现场验证
[ ] --skip-change 全流程已通过
[ ] 正式命令已删除 --skip-change
[ ] 操作员可随时 Ctrl+C / emergency stop
```
