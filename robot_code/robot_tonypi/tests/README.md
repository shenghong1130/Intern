# TonyPi 测试使用说明

本目录同时包含不接触真实硬件的自动化单元测试，以及需要 TonyPi、相机和 FPGA 的独立实机集成测试。运行命令时应位于包含 `robot_tonypi` 目录的上级目录：

```bash
cd /home/pi
```

开发机上的对应目录是：

```bash
cd /home/robot/robot_code/Intern/robot_code
```

## 1. 运行全部自动化测试

```bash
python3 -m unittest discover -s robot_tonypi/tests -p 'test_*.py' -v
```

这些自动化测试不会打开真实相机、执行 TonyPi 动作组、访问 FPGA 或发送 Worker 请求。`test_capture_fpga_change.py` 虽然符合 `test_*.py` 命名，但它没有 `unittest.TestCase`，因此 discover 只会导入它，不会自动启动实机流程。

## 2. `test_calibrate_motion.py`

验证人工动作标定工具的纯数据处理，包括：

- 前进、后退、左右横移和左右转向的符号；
- `--times` 测量值归一化；
- median 推荐值；
- large turn 的实际 sequence 描述保持不变；
- 只为已测动作生成推荐配置；
- `--write-config` 写回前创建备份，并只修改对应运动字段。

测试使用临时目录，不执行真实动作，也不修改项目正式配置。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_calibrate_motion -v
```

## 3. `test_interaction_flow.py`

验证交互纯逻辑和 `RobotInteractionClient` 的事务保护，包括：

- 四种 Tag 平面、四向法线、17 cm 唯一目标和身体横向补偿；
- `stand → lift_left_hand(stand=False) → send_request → finally stand` 顺序；
- 举手后的第二次安全门；
- Worker `ok=False` 或异常不能标记成功；
- `--dry-run`、`--skip-change` 不调用真实动作和 Worker；
- 定位扫描和途中绑定不能旁路调用分类或换花。

动作和 `send_request` 均由测试假函数代替，不访问真实硬件、网络或 FPGA。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_interaction_flow -v
```

## 4. `test_mission_scheduler.py`

验证任务调度和导航保护的局部逻辑，包括：

- 地图与 Tag 参考坐标未改变；
- 按 17 cm 最终目标选择最近 Screen，并按 ID 稳定破平局；
- 每处理一个目标后使用最新 pose 重新排序；
- 旧 34/15 cm 两阶段函数和状态不再存在；
- 导航直接使用唯一 17 cm 坐标、四向 yaw 和高代价终点参数；
- 分类器只能由到达当前锁定目标的入口调用；
- 视觉授权要求目标锁匹配且已经到达；
- 初始定位配置和 `--dry-run`/`--skip-change` 参数语义保持不变；
- 转向 watchdog 的进展、±180° wrap、stale pose、反方向转向和连续失败中止。

测试使用假地图、假 pose 和通过 `TaskManager.__new__` 创建的轻量对象，不初始化真实 `TaskManager` 硬件组件。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_mission_scheduler -v
```

## 5. `test_direct_17cm_flow.py`

验证当前直接任务流程：

- 所有别名字段都指向带横向补偿的唯一 17 cm 目标；
- 当前精确终点可作为高代价终点，但建筑实体和普通目标不获得例外；
- 正确目标 Tag 与绑定屏幕共同出现后才允许专用 3 cm 动作；
- 缺 Tag、缺屏幕或绑定错误时不前进；3 cm 后 FPGA 失败不生成视觉授权；
- 专用 3 cm 动作恰好一次，随后才分类；已是目标花时不调用 Worker；
- 3 cm 动作失败时不调用 Worker，旧授权不能跨目标复用。

测试使用假画面、假分类器、假动作和假 Worker，不访问真实硬件或网络。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_direct_17cm_flow -v
```

## 6. `test_target_direct_approach.py`

验证锁定目标 40 cm 范围内的窄通道直达、当前目标膨胀代价豁免、其他障碍保持生效、较短末步、直走优先及转向惩罚。只创建地图和轻量任务对象，不调用真实硬件。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_target_direct_approach -v
```

## 7. `test_vision_tag_binding.py`

验证花朵屏幕与左上方 AprilTag 的绑定规则：

- 合法左侧 Tag 可以绑定；
- 右侧 Tag 被拒绝；
- 多个合法 Tag 选择距离屏幕左上角最近者；
- 超过距离阈值的 Tag 被拒绝；
- 37 以上的 Tag 不能作为 `screen_id`。

测试只构造 NumPy 四边形与 `TagDetection` 数据，不读取相机，也不运行真实 AprilTag 检测器。

单独运行：

```bash
python3 -m unittest robot_tonypi.tests.test_vision_tag_binding -v
```

## 7. `test_capture_fpga_change.py`：独立实机集成测试

这个脚本不属于正式任务状态机。它假设操作者已经把机器人放在正确目标点并正对屏幕，不执行定位、导航或姿态调整，只测试：

```text
相机拍照
→ 检测屏幕并绑定左上 AprilTag
→ 取得指定 screen_id 的 28×28 裁剪
→ FPGA 分类
→ 已是目标花则结束
→ 否则按安全开关模拟或真实执行 Worker 换花事务
```

它会保存：

```text
capture_fpga_change_runs/<timestamp>/
├── raw.jpg
├── annotated.jpg
└── screen_<id>_crop_28x28.png
```

### 7.1 无硬件检查

只检查参数、配置加载和安全清理，不打开相机、不访问 FPGA、不举手、不请求 Worker：

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 \
  --target-flower hehua \
  --dry-run
```

### 7.2 相机和 FPGA 模拟换花测试

使用真实相机和 FPGA 完成识别，但不真实举手、不发送 NFC/Worker 请求：

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --skip-change
```

即使省略 `--skip-change`，只要没有 `--execute`，脚本也会强制模拟交互。显式写出 `--skip-change` 更便于现场确认当前是安全测试。

### 7.3 真实执行换花

执行前必须满足：

- 机器人已由操作者放在正确目标点并正对屏幕；
- FPGA 服务可访问；
- 已确认视觉绑定到正确的 `screen_id`；程序会把相同编号直接作为 `worker_id`；
- Team、Robot ID 和 Secret 正确；
- 操作者理解这里使用的是人工确认门，不是真实 AprilTag 定位安全门。

命令：

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 \
  --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red \
  --robot-id red-1 \
  --robot-secret 1234 \
  --execute
```

识别成功且需要换花时，程序会打印 `screen_id`、`worker_id`、`from_flower`、`to_flower` 和 confidence。只有操作者再次准确输入：

```text
EXECUTE 2
```

才会通过 `RobotInteractionClient` 真实执行：

```text
stand
→ lift_left_hand(stand=False)
→ send_request
→ finally stand
```

以下任一情况都会禁止真实交互：指定 Screen 未检测到、FPGA 失败、置信度不足、识别结果已经是目标花、未提供 `--execute` 或二次确认不匹配。

## 8. 编译检查

```bash
python3 -m compileall -q robot_tonypi
```

该命令只检查 Python 文件能否编译，不代替真实相机、FPGA、动作组或 Worker 集成测试。
