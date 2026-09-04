# robot_tonypi 文件说明

本文按当前源码说明目录职责和调用关系。运行方法见 [README.md](README.md)，完整决策树见 [robot_decision_tree.md](robot_decision_tree.md) 和 [robot_decision_tree.html](robot_decision_tree.html)。

## 1. 当前调用关系

```text
main.py
├─ config.py + config/competition_config.json
├─ task_manager.py
│  ├─ models.py / utils.py
│  ├─ load_pos.py → localizer.py
│  ├─ map_model.py
│  ├─ motion.py → hardware.py → robotall / TonyPi ActionGroups
│  ├─ vision.py → classifier.py → Kria /predict
│  ├─ interaction_logic.py
│  ├─ interaction_client.py → robotall.send_request
│  └─ debug.py
└─ finally: TaskManager.close()
```

## 2. 顶层文档

### `README.md`

面向操作员的当前运行手册。包含部署目录、真实参数、运行模式、定位/导航/交互流程、Debug 和现场检查清单。

### `robot_decision_tree.html`

单文件离线决策树。使用浅层卡片、表格和 ASCII 流程，避免深层 branch DOM 的连线错位。

### `robot_decision_tree.md`

长期维护的完整流程说明，覆盖定位事实、genuine NO-TAG、Turn Progress 三态、规划/动作 yaw 模型、FPGA/NFC 与所有 Recovery 去向。

### `FILES.md`

当前文件索引。

### `CLAUDE.md`

面向维护工具的当前约束，记录不能破坏的数据一致性、不变量和测试入口。

## 3. 启动与配置

### `main.py`

- 定义 `mission`、`localize`、`harvest` 三种模式；
- 检查目标花名；
- 读取 JSON 覆盖配置；
- 创建并运行 `TaskManager`；
- 正常返回码为 0，运行返回 False 时为 2，`Ctrl+C` 为 130。

### `config.py`

定义 `DEFAULT_CONFIG` 和递归覆盖加载。关键默认域：

- `paths`、`camera`、`map`、`localization`；
- `vision`、`navigation`、`motion`；
- `interaction`、`mission`、`obstacle`、`debug`。

### `config/competition_config.json`

正式现场覆盖。当前覆盖相机标定、障碍代价、动作模型、任务目标几何、NFC timeout、恢复和测试排除 Screen。最终运行值必须以 `load_config(default_config_path())` 的合并结果为准。

### `models.py`

共享状态和数据结构：

- `Confidence`：HIGH/MEDIUM/LOW/UNKNOWN；
- `ScreenStatus`：UNKNOWN、NEEDS_CHANGE、INTERACTING、CHANGED、ALREADY_TARGET、FAILED；
- `MissionState`：定位、目标选择、导航、确认、分类、NFC、retreat、complete/timeout/blocked 等可观测状态；
- `TargetGoal`：原子化 screen/tag/anchor/25 cm interaction target/yaw/generation；兼容字段 `navigation_staging_xy` 与 interaction target 相同；
- `NavigationPlan` / `PlannedNavigationAction`：Motion-Aware A* 的真实动作计划、预测起止 Pose、周期与成本；
- `TargetTagConfirmation`、`TargetVisualConfirmation`、`VisualAuthorization`；
- `RecentBoundFlowerObservation`：每个 Screen 最新有效 Tag↔Screen 分类证据；
- `WorkerChangeResult`、`InteractionAuthorizationCheck`、动作结果等。

`Screen.done()` 当前只把 `CHANGED` 和 `ALREADY_TARGET` 视为已处理；普通 `FAILED` 不是永久导航黑名单。NFC GAVE_UP 由 `TaskManager.nfc_gave_up_screen_ids` 单独排除。

### `utils.py`

角度归一化、距离、时间、JSON 和目录工具。

## 4. 主状态机

### `task_manager.py`

项目的业务编排中心，主要职责如下。

#### 目标生命周期

- `configure_cardinal_task_targets()`：从建筑面中心和外法线生成 `25 cm / -1 cm` interaction target，并仅对最终 yaw 增加 `+5°`；
- `resolve_target_goal()`、`lock_target_goal()`、`validate_target_goal()`：原子化目标身份与坐标，防止 stale screen/goal；
- `choose_nearest_screen()`：保留合法当前锁；否则先按 Pose 到 25 cm interaction target 的最近距离窗口筛选，再用 behind-turn + final-yaw 有界惩罚评分，最后按 ID 稳定破平局；
- `run_mission()`：临时失败轮换、全局恢复、交互后退、完成等待和 timeout。

#### 定位

- `initial_localize()`；
- `run_localization_search_sequence()`：启动与高级恢复共用“完整 pan → 身体搜索动作 → 完整 pan”；运行时 genuine NO-TAG 使用独立的有界位置恢复；
- `localize_scan()`：普通模式在任意有效视觉 Pose 后停止；指定 `required_target_screen_id` 时必须等到该目标 Tag↔Screen 绑定；
- `accept_visual_localization()`：只有接受视觉 Pose 才清零动作计数与运动不确定度；
- `record_localization_failure()`：区分 `no_tag`、`pose_unavailable_with_tags` 和 `capture_failed`。

#### NO-TAG 与转向证据

- `recover_from_no_tag_if_needed()`：仅连续 genuine `no_tag` 触发；墙边横移，否则 5 cm 后退 + 向内 45° 身体转向，中央复拍最多 3 轮；
- `evaluate_turn_progress()` / `monitor_turn_result()`：只让可靠视觉 Pose 产生 VERIFIED 结论；不可定位统一为 `PROGRESS_UNVERIFIED`；
- 正式位置 A* 使用可被 15° state 精确表示的 ±90° physical quarter-turn macro；action-level yaw lattice 也可细化到 1.5°。预测、dead reckoning 和 watchdog 始终使用配置物理动作角。

#### 途中视觉和目标确认

- `observe_transit_bindings()`、`process_bound_screen_candidate()`：从定位/导航帧提取合法绑定并写 15 秒缓存，不改变 ScreenStatus；
- `confirm_target_tag_now()`：最多看 `[100,130,70]`，只确认当前目标 Tag；
- `bounded_fresh_target_observation()`：当前目标新鲜分类最多 3 帧；
- `confirm_target_tag_and_screen()`：实时 Tag + 同 ID 绑定分类；
- `confirm_target_with_visibility_recovery()`：分类服务不可用时保持目标和 mission，目标不可见时最多 2 轮局部恢复。

#### 导航

- `plan_navigation_path()`：保留给 debug、fallback 和 Recovery 兼容的二维路径接口；
- `plan_motion_actions()`：正式 `POSITION_NAVIGATION` 的 Motion-Aware A*；yaw 保留在 state 中解释动作世界方向，但 Screen 最终 yaw 不参与位置阶段终点或 heuristic；
- `navigate_to_xy()`：自适应重定位、动作选择、到达前新鲜视觉 Pose、最终 yaw；
- `navigate_to_screen()`：一次建立 25 cm XY + desired yaw goal，并进入 `POSITION_NAVIGATION → FINAL_YAW_ALIGNMENT`；不再有中途 staging；
- `execute_motion_astar_action()`：直接执行 Planner 的 action key，可合并连续同动作，按 confidence/距离限制 batch，随后定位并重规划；
- `choose_translation_action()`：前进、短距离正后方倒退和平移；
- `adaptive_relocalization_decision()`：动作预算、置信度、不确定度、阶段和大转向触发；
- `register_plan_failure()`：相同输入 3 次失败后升级，不等到 80 步才处理。

#### 恢复

- `recover_from_near_wall()`：后退、左右平移、小转向；
- `execute_bounded_escape()`：普通恢复全被 veto 时，从不安全起点选择更安全的小动作；
- `recover_via_indoor_waypoint()`：在内缩区域选可达、安全、尽量保 yaw 的 waypoint；
- `perform_global_recovery()`：重新定位，必要时 near-wall 或 interior recovery；
- `register_temporary_target_failure()`、`release_temporary_target_failures()`：失败目标轮换和释放。

#### FPGA 与 NFC

- `latest_valid_bound_flower_observation()`：15 秒、同 ID、binding、置信度检查；
- `adopt_cached_target_observation()`：实时当前 Tag 存在后把缓存变成授权；
- `execute_final_forward()`：仅 NEEDS_CHANGE 时执行一次 `interaction_forward_final`（大步×4，约 20 cm），并设置 retreat pending；
- `process_screen_interaction()`：最多两次 NFC 物理尝试；
- `restore_nfc_physical_contact()`：Attempt1 失败后后退、定位、最多 3 轮重新寻找当前目标；
- `recalibrate_target_for_nfc_retry()`：只有当前目标重新分类仍不是 target 才重新导航/确认/final forward；
- `nfc_change_is_terminal()`：CHANGED 后禁止任何 retry；
- `give_up_nfc_change()`：两次失败或目标重获耗尽后结束该 Screen，mission 继续。

### `interaction_logic.py`

无硬件纯逻辑：

- 从 Tag 平面确定 WEST/EAST/SOUTH/NORTH；
- 从同一建筑 `face_center` 和 cardinal normal 生成 reader、25 cm interaction target 和 cardinal yaw；
- 保存分类但不执行交互；
- 只有 Worker `success=True` 才写 `CHANGED`。

## 5. 定位与地图

### `load_pos.py`

保存 AprilTag 世界四角坐标。1–36 是 Screen Tag，37 以上可用于定位/障碍语义。此文件是地图事实源，不应因文档或 Dashboard 显示需求改坐标。

### `localizer.py`

- AprilTag detector 适配；
- 面积、边缘、世界坐标存在性、PnP、向量/旋转合法性、场地范围质量检查；
- 同一帧逐个尝试 Tag，一个失败不会阻止后续 Tag；
- 输出结构化 rejection detail 和 frame summary。

### `map_model.py`

- 300×300 cm、5 cm 栅格；
- 根据 Screen 建筑矩形生成硬障碍、软 inflation 和 cost；
- `tag_front_xy` 固定为物理建筑面锚点；交互距离、横移、yaw/final-forward 只改变允许的 Screen 目标几何，不污染静态地图层；
- 动态障碍、footprint、clearance、直线/旋转 corridor；
- 普通 A* 和带 yaw/action 的动作空间 A*；
- 当前目标建筑的软 inflation 仅在受限 final approach 中可被忽略，其他建筑和硬占用仍生效。

Debug 显示由 `_map_pt(xy) -> (y, x)` 转换，因此左上为 `(0,0)`、x 向下、y 向右。

## 6. 视觉与分类

### `vision.py`

检测 Screen 四边形、做几何/白色比例过滤、把 Screen 与其左上附近的 1–36 Tag 绑定，并生成 `28×28` crop。绑定只接受 `candidate.screen_id == candidate.tag.tag_id`。

### `classifier.py`

把 crop 编码成 JPEG。`direct` 模式保持只以 multipart `image` POST 到 KV260 Worker `/predict`；`central` 模式额外提交 `student_id`，并在 POST `/predict` 和 queued GET `/requests/{request_id}` 中使用 `X-Student-Password`。连接异常、5xx、408、429 被标记为可恢复 service unavailable；401 等其他 HTTP 错误不可重试；无合法花名/JSON 属于 invalid response。

### `fpga_flower_server/fpga_server_api_ready.py`

运行在 Kria/PYNQ：加载 bit/hwh、DMA 和 12 类模型；串行处理 `/predict`，返回 API 花名、中文花名、类别编号和 confidence。服务说明见 [fpga_flower_server/README.md](fpga_flower_server/README.md)。

## 7. 动作和硬件

### `motion.py`

- `RobotState`：视觉 Pose、dead reckoning、动作计数和运动不确定度；
- `MotionController`：执行配置动作、按真实完成周期更新模型；
- 失败或部分动作不会虚报全部 requested cycles。

### `hardware.py`

相机后台读取、云台、ActionGroup 执行、动作序列、stop/close。动作完成前检查动作组是否存在；交互期间阻止普通动作并保持 stand 清理。

### `action_groups/*.d6a`

仓库附带的自定义转向动作组。实际运行目录仍是 `/home/pi/TonyPi/ActionGroups/`。

### `calibrate_motion.py`

人工标定现有动作模型。只运行指定动作并记录操作者测量；写配置前创建备份。

## 8. NFC

### `interaction_client.py`

严格顺序：

```text
授权检查
→ stand
→ lift_left_hand(stand=False)
→ 再次授权检查
→ 生成新 seq
→ send_request(retries=0, scan_timeout≤15s, overall_timeout≤15s)
→ finally stand
```

每次物理 Attempt 使用新 seq，底层继续校验 worker_id/seq；同一物理位置不会由底层自动重试。

## 9. Debug

### `debug.py`

事件 JSONL、latest_state、标注图、地图、crop 和 8090 Dashboard。地图同时显示 Screen anchor、TargetGoal、当前导航 goal、recovery waypoint 和 path。

## 10. 测试和辅助脚本

### 自动化测试

- `test_calibrate_motion.py`：动作标定纯逻辑；
- `test_interaction_flow.py`：几何、授权、NFC deadline/seq/异常；
- `test_mission_scheduler.py`：目标选择、状态机、near-wall/forced escape、timeout；
- `test_navigation_adaptive.py`：动作批次、自适应定位、no-tag、倒退/平移、task safety bypass；
- `test_navigation_path_fallback.py`：clearance、兼容 fallback、规划失败升级；
- `test_recovery_target_consistency.py`：TargetGoal 原子一致和 interior recovery；
- `test_target_direct_approach.py`：当前目标软 cost 例外和直接动作；
- `test_mission_refactor.py`：物理定位 gate、hard jump、两阶段目标评分、Motion-Aware A* goal/action、Position 免费安全转向、15 cm reverse 边界和 Planner→Executor 一致性；
- `test_target_standoff_flow.py`：目标确认、缓存、final forward、NFC 两次尝试和目标重获；
- `test_vision_tag_binding.py`：Tag↔Screen 绑定和 15 秒缓存。

详细命令见 [tests/README.md](tests/README.md)。

### 独立实机脚本

- `tests/test_capture_fpga_change.py`：人工放置后测试相机、FPGA 和可选 NFC；
- `tests/test_capture_15_frames.py`：交互式连续拍照；
- `deploy.py`：历史 Paramiko 脚本，仍含旧 IP、旧密码和旧路径 `/home/pi/TonyPi/competition_tonypi`，不适用于当前 `/home/pi/robot_tonypi` 部署。

## 11. 文件运行位置

| 内容 | 运行位置 |
|---|---|
| `robot_tonypi/*.py` | TonyPi Raspberry Pi |
| `fpga_flower_server/fpga_server_api_ready.py` | Kria/PYNQ |
| `action_groups/*.d6a` | 复制到 TonyPi ActionGroups 后由机器人执行 |
| 单元测试 | 开发机或 TonyPi，均不应触发硬件 |
| `test_capture_*` | 明确由操作者在真机手动启动 |
