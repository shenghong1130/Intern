# TonyPi 比赛程序

本目录是 TonyPi 比赛主控。当前架构严格拆分：

```text
远距离发现 / classify / vote       近距离实体换花
            ↓                              ↓
只更新 Screen 花朵状态          15 cm + yaw + lateral + pose 安全门
                                           ↓
                              lift_left_hand → Worker → stand
```

远距离识别、定位顺带识别、opportunistic 和 passby 均不能直接换花。完整真实调用树见 `robot_decision_tree.html`。

## 部署

把整个包放到机器人 Python 可导入的位置，例如：

```text
/home/pi/TonyPi/robot_tonypi
```

启动前停止可能占用动作控制器的系统服务：

```bash
sudo systemctl stop tonypi
cd /home/pi/TonyPi
```

FPGA 分类服务仍由 `--classifier-url` 指定。

## 配置 screen → worker 映射

视觉 `screen_id` 与实体 `worker_id` 没有资料证明必然相同，因此程序不会自动套用相同编号。正式运行前必须编辑 `config/competition_config.json`：

```json
{
  "interaction": {
    "worker_mapping": {
      "2": 12
    }
  }
}
```

上例只是表达 screen 2 映射到 Worker 12 的格式，不是比赛完整映射。

## 安全 dry-run

`--dry-run` 不连接相机、动作硬件或 Worker；`--skip-change` 在真实导航/定位流程中模拟已经安全对准后本来会执行的举手和请求参数。

```bash
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red \
  --robot-id red-1 \
  --robot-secret 1234 \
  --skip-change \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 600
```

`--skip-api` 仅作为旧命令兼容别名，等价于 `--skip-change`；代码不再使用旧 HTTP change API。

## 正式运行

确认 Worker 映射、机器人注册信息和现场标定参数后，去掉 `--skip-change`：

```bash
python3 -u -m robot_tonypi.main \
  --mode mission \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team YOUR_TEAM \
  --robot-id YOUR_ROBOT_ID \
  --robot-secret YOUR_SECRET \
  --debug --debug-host 0.0.0.0 --debug-port 8090 \
  --time-limit-s 600
```

`--robot-name` 是兼容旧命令的 fallback，会同时填充缺失的 team 和 robot-id；正式比赛建议明确传入 `--team` 与 `--robot-id`。

## 主要参数

| 参数 | 说明 |
|---|---|
| `--mode` | `mission`、`localize` 或 `harvest` |
| `--target-flower` | 目标花拼音 API 名 |
| `--classifier-url` | FPGA 分类服务地址 |
| `--team` / `--robot-id` / `--robot-secret` | `robotall.send_request` 注册信息 |
| `--skip-change` | 完整运行至安全交互点，但模拟举手和 Worker 请求 |
| `--dry-run` | 完全无硬件模式 |
| `--debug` | 写日志并启动调试面板 |
| `--max-screens` | 测试用，达到实际换花成功数后停止 |

支持的花名：

```text
bailianhua chuju hehua juhua lamei lanhua meiguihua
shuixianhua taohua yinghua yuanweihua zijinghua
```

## 任务状态机

```text
LOCALIZE
↓
INITIAL DISCOVERY（可关闭，只识别）
↓
DISCOVER / IDENTIFY / VOTE
↓
UNKNOWN ──→ observation pose ──→ 继续识别
NEEDS_CHANGE ──→ interaction staging pose
                         ↓
                    FINAL ALIGN
                         ↓
       distance + body yaw + lateral + fresh pose gate
                         ↓
          stand → lift_left_hand(stand=False)
                         ↓
                  send_request / wait
                         ↓
                  finally: stand
                         ↓
                  CHANGED / RETRY
```

四个预设视野点的 scripted route 已删除。保留的一次 `initial_discovery_scan` 是独立的原地发现扫描，不进行实体换花。

## 交互几何与现场标定

`normal_xy` 从屏幕平面指向可见正面。站在正面看向屏幕时：

```text
screen_left = (normal.y, -normal.x)
reader_xy = center + screen_left * 5 cm
interaction_xy = center + normal * 15 cm
                      + screen_left * (5 cm - left_hand_body_offset_cm)
interaction_yaw = normal_yaw + 180°
```

请现场标定：

- `left_hand_body_offset_cm`：举左手后，手部读卡位置相对身体中心的横向偏移；默认 0 只是未知占位。
- `interaction_distance_cm`：默认 15 cm。
- `interaction_distance_tolerance_cm`：默认 ±4 cm。
- `interaction_yaw_tolerance_deg`：默认 ±10°。
- `interaction_lateral_tolerance_cm`：默认 ±4 cm。
- `sensor_left_offset_cm`：默认屏幕左侧 5 cm。

## Debug

打开 `http://<robot-ip>:8090` 可查看：

- robot pose、定位健康度、规划路线和恢复状态；
- Screen 的 observation / interaction / reader 点；
- 当前 interaction phase、ready、left hand 状态；
- 实际 distance、yaw error、lateral error 和阻断原因；
- worker_id、请求、响应及失败信息。

日志目录默认：

```text
/home/pi/TonyPi/debug_runs/<timestamp>/
  events.jsonl
  interaction_calls.jsonl
  latest_state.json
  latest_annotated.jpg
  latest_map.jpg
```

只有 `result['ok'] == True` 才会将 Screen 标记为 `CHANGED`。失败和异常会记录 pose 与全部误差，并回到 `NEEDS_CHANGE`。

## 无硬件测试

测试仅使用 Python 标准库：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

覆盖远距离观察、错误 yaw、错误 lateral、正确动作顺序、`ok=False`、异常收尾，以及 passby/localize 无旁路。
