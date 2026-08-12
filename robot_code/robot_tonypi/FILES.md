# robot_tonypi 文件说明

本文解释 `robot_tonypi` 目录内所有源文件、配置、测试和动作资源。使用步骤见 [README.md](README.md)，任务决策树见 [robot_decision_tree.html](robot_decision_tree.html)。

## 1. 总体调用关系

```text
main.py
└─ config.py：加载配置
└─ task_manager.py：组织完整任务
   ├─ hardware.py：相机、舵机和动作组硬件接口
   ├─ motion.py：动作选择、次数计算、推算位姿
   ├─ localizer.py + load_pos.py：AprilTag 世界定位
   ├─ vision.py：屏幕检测、左侧 Tag 绑定和裁剪
   ├─ classifier.py：调用 FPGA 花朵分类服务
   ├─ map_model.py：Screen 几何、障碍地图和 A*
   ├─ interaction_logic.py：纯交互几何和安全判断
   ├─ interaction_client.py：举左手和 Worker 原子事务
   ├─ debug.py：日志、图片、地图和 8090 Dashboard
   └─ models.py / utils.py：共享数据结构和工具函数
```

## 2. 顶层文档和包文件

### `README.md`

面向机器人使用者的操作手册。内容按执行顺序组织：环境检查、动作组检查、Worker 映射、测试、定位、识别、安全导航、正式换花、Dashboard 和故障处理。

### `robot_decision_tree.html`

面向调试和评审的可视化决策树。它描述程序从定位到识别、导航、最终对准、双重安全门、举手、Worker 响应和重试的真实分支。

### `FILES.md`

当前文件。用于解释目录中每个文件的职责以及模块间调用关系。

### `CLAUDE.md`

面向代码维护工具的仓库约束。它强调识别与实体换花必须分离、禁止恢复旧 HTTP ApiClient、禁止假设 `screen_id == worker_id`，并记录开发约定。

### `__init__.py`

把目录声明为 Python 包，并保存包版本。因为存在这个文件，推荐从 `/home/pi` 使用：

```bash
python3 -m robot_tonypi.main ...
```

## 3. 启动、配置和共享基础

### `main.py`

命令行入口，主要负责：

- 定义 `mission`、`localize`、`harvest` 三种模式；
- 读取目标花、FPGA 地址、机器人注册信息和 Debug 参数；
- 校验目标花名称；
- 加载配置并创建 `TaskManager`；
- 把任务返回值转换成进程退出码。

它不直接执行视觉、导航或换花业务。

### `config.py`

保存 Python `DEFAULT_CONFIG`，并把 JSON 配置覆盖到默认值上。配置包含：

- TonyPi SDK、动作组、相机标定和日志路径；
- 相机和头部舵机参数；
- 地图、导航和动作模型参数；
- AprilTag 定位和屏幕视觉阈值；
- interaction pose、安全容差和 Worker 映射；
- 任务时限、障碍检测和 Debug 设置。

`default_config_path()` 返回 `config/competition_config.json`。

### `config/competition_config.json`

现场优先调整的配置覆盖文件。它会覆盖 `config.py` 中同名字段。正式使用时主要在这里填写 Worker 映射和调节场地参数，而不是直接散改业务代码。

### `models.py`

所有模块共享的数据模型：

- `Confidence`：定位置信度；
- `ScreenStatus`：`UNKNOWN`、`NEEDS_CHANGE`、`INTERACTING`、`CHANGED` 等状态；
- `RobotPose`：机器人世界坐标、yaw、来源和时间；
- `Screen`：屏幕中心、normal、观察点、交互点、reader 点和 worker_id；
- `TagDetection`：AprilTag 检测结果；
- `ScreenCandidate`：视觉屏幕候选；
- `ClassificationResult`：FPGA 分类结果；
- `InteractionPoseCheck`：距离、yaw、lateral 和 pose 安全判断；
- `WorkerChangeResult`：Worker 请求结果；
- `ActionResult`：动作执行及推算位移。

### `utils.py`

小型通用工具，包括单调时间、角度归一化、角度差、二维距离、数值裁剪、目录创建、JSON 读写和配置递归合并。

## 4. 主状态机

### `task_manager.py`

中央调度器，也是项目最大的业务文件。主要职责：

1. 创建地图、相机、AprilTag、视觉、FPGA、动作、交互和 Debug 组件；
2. 初始定位和可选起始发现扫描；
3. 扫描屏幕、分类并投票；
4. 将非目标花标记为 `NEEDS_CHANGE`，但不在识别路径直接换花；
5. 根据 Screen 状态选择 observation 或 interaction 导航目标；
6. 执行 A* 导航、passby 观察、障碍和边界恢复；
7. 在交互前反复定位并修正距离、身体 yaw 和横向误差；
8. 通过最终安全门后调用 `RobotInteractionClient`；
9. 发布 Dashboard 状态并写交互审计日志；
10. 退出时关闭硬件、相机和日志。

真正的 `send_request` 不在这个文件中直接出现；它只能通过交互客户端调用。

## 5. 相机、定位和视觉

### `hardware.py`

提供两个硬件适配类：

- `RealtimeCamera`：通过 `hiwonder.Camera` 打开相机，使用后台线程持续保存最新帧，拍摄时丢弃转头后的旧帧；
- `TonyPiHardware`：初始化控制板、头部舵机、IMU 和 Hiwonder 动作组控制器。

它还负责：

- 将头部角度换算成 PWM；
- 检查 `.d6a` 动作组是否存在；
- 执行单一或组合动作组；
- 在左手交互事务期间禁止普通导航动作。

### `localizer.py`

包含 AprilTag 检测和机器人位姿估计：

- `AprilTagDetector` 支持可用的 AprilTag Python 后端；
- 读取相机内参和畸变参数；
- 通过屏幕 Tag 的世界角点和 OpenCV PnP 求机器人 pose；
- 按 Tag 面积、画面边缘和场地范围做质量过滤；
- 支持估算其他 Tag 在世界坐标中的位置，用于动态障碍。

### `load_pos.py`

保存比赛场地 AprilTag 的三维世界角点坐标。`MapModel` 和 `Localizer` 都依赖这些数据。若使用外部坐标文件，可通过 `--load-pos` 覆盖。

### `vision.py`

负责花朵屏幕视觉候选：

1. 灰度、模糊、边缘检测和轮廓提取；
2. 筛选凸四边形、面积、宽高比和边长比例；
3. 排除 AprilTag 落在屏幕四边形内部的错误候选；
4. 将候选与其左上附近、且位于屏幕图像中心左侧的 1～36 号 AprilTag 绑定；
5. 使用 tag_id 作为 screen_id，并保留地图有效性检查；
6. 透视变换为 28×28 分类图；
7. 绘制候选框和 Tag 标注。

屏幕左侧 Tag 绑定与实体换花的左侧读卡区是两个独立概念。

### `classifier.py`

FPGA 分类服务客户端。它把 28×28 屏幕裁剪编码为 JPEG，通过 HTTP POST 发给 `--classifier-url`，并把返回花名、中文名、置信度和类别编号封装成 `ClassificationResult`。

### `fpga_server_api_ready.py`

运行在 PYNQ/FPGA 端，而不是 TonyPi 树莓派主控端。它加载 FPGA Overlay，接收 28×28 图片，执行 DMA 推理，并以 HTTP JSON 返回分类结果。运行该文件需要 PYNQ、模型 bit/hwh 文件和 FPGA 环境。

## 6. 地图和运动

### `map_model.py`

构造 300×300 cm 场地模型并负责路径规划：

- 根据 Tag 世界坐标生成 Screen；
- 计算屏幕 normal、observation point、interaction point、reader point 和 interaction yaw；
- 建立建筑物障碍、软膨胀 cost 和动态机器人障碍；
- 提供网格 A*、路径平滑和直线可通行检查；
- 提供考虑机器人转向/横移宏动作的 action-level A*；
- 统计未完成 Screen 和实际换花成功数。

### `motion.py`

包含：

- `MotionController`：根据目标角度或距离选择左转、右转、前进、横移及动作次数；
- `RobotState`：动作完成后按配置的 `forward_cm`、`lateral_cm`、`yaw_deg` 更新推算位姿。

连续动作会让 pose 置信度从 HIGH 降到 MEDIUM/LOW，随后触发 AprilTag 重定位。

### `action_groups/*.d6a`

仓库附带的自定义左右小步转身动作组：

```text
turn_left_small_step_s70.d6a
turn_left_small_step_s75.d6a
turn_left_small_step_s80.d6a
turn_left_small_step_s85.d6a
turn_right_small_step_s70.d6a
turn_right_small_step_s75.d6a
turn_right_small_step_s80.d6a
turn_right_small_step_s85.d6a
```

不同后缀代表不同速度版本。当前比赛配置使用 `s80`。这些是二进制动作资源，不是 Python 源码。TonyPi SDK 默认从 `/home/pi/TonyPi/ActionGroups/` 加载，所以仓库内文件需要按部署要求安装到 SDK 动作组目录。

## 7. 交互和 Worker 请求

### `interaction_logic.py`

不依赖相机、NumPy、硬件或网络的纯逻辑层：

- 根据屏幕 center 和 normal 构造左侧 reader、interaction point 和目标 yaw；
- 视觉识别只更新 `ALREADY_TARGET` 或 `NEEDS_CHANGE`；
- 集中检查 pose 新鲜度和置信度、15 cm 距离、身体 yaw、横向误差、花朵状态和 Worker 映射；
- 根据 Worker 结果设置 `CHANGED` 或恢复为 `NEEDS_CHANGE`。

因为安全规则集中在纯函数中，可以在没有机器人硬件时测试。

### `interaction_client.py`

封装唯一的实体换花事务：

```text
安全门
→ robotall.act('stand')
→ robotall.act('lift_left_hand', stand=False)
→ 再次安全门
→ robotall.send_request(...)
→ finally robotall.act('stand')
```

支持 `--dry-run` 和 `--skip-change` 模拟。只有 `result['ok']` 为真才返回成功；异常和失败都执行站立收尾。

## 8. Debug

### `debug.py`

`DebugReporter` 负责：

- 输出结构化事件；
- 保存相机标注图和屏幕裁剪；
- 保存 `latest_state.json`；
- 绘制机器人、路线、observation、interaction 和 reader 的场地图；
- 启动内置 HTTP Server，默认端口 8090；
- 在网页显示 pose、投票、Screen 状态、安全门误差和 Worker 响应。

Debug 目录默认在 `/home/pi/TonyPi/debug_runs/<timestamp>/`。

## 9. 测试

### `tests/test_interaction_flow.py`

使用 Python `unittest` 测试：

- 远距离识别不能换花；
- 距离、yaw、lateral 不满足时安全门拒绝；
- 正确事务顺序和 notebook 参数；
- `ok=False` 和异常不能标记成功；
- `finally` 必须 stand；
- passby 和 localize_scan 没有换花旁路；
- from_flower 在发送前发生变化时拒绝请求。

### `tests/test_vision_tag_binding.py`

测试花朵屏幕左侧 AprilTag 绑定：

- 左侧合法 Tag 可以绑定；
- 屏幕右侧 Tag 被拒绝；
- 多个合法 Tag 选择距离左上角最近者；
- 超过距离阈值拒绝；
- 37+ Tag 不能作为 screen_id。

## 10. 部署辅助文件

### `deploy.py`

历史 Paramiko/SFTP 部署脚本。文件内仍带旧机器人 IP、旧密码和旧目录 `/home/pi/TonyPi/competition_tonypi`，与当前 `/home/pi/robot_tonypi` 部署方式不一致。

因此当前不要直接运行它。实际同步应显式确认机器人 IP、目标路径和是否删除远端文件，再使用受控的 SSH/rsync 流程。

## 11. 每个文件运行在哪里

| 文件类别 | 运行位置 |
|---|---|
| `main.py`、`task_manager.py`、相机/定位/导航/交互模块 | TonyPi 树莓派 |
| `config/competition_config.json` | TonyPi 树莓派读取 |
| `fpga_server_api_ready.py` | PYNQ FPGA 板 |
| `action_groups/*.d6a` | 安装到 TonyPi SDK 动作目录后由 SDK 执行 |
| `tests/*.py` | 开发电脑或 TonyPi，交互测试不触发真实硬件 |
| `robot_decision_tree.html` | 任意浏览器离线查看 |
| `README.md`、`FILES.md` | 使用者和维护者阅读 |
