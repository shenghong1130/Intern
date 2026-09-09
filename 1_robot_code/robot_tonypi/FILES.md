# robot_tonypi 文件说明

本文按当前源码说明目录职责和调用关系。运行方法见 [README.md](README.md)，完整决策树见 [robot_decision_tree.md](robot_decision_tree.md) 和 [robot_decision_tree.html](robot_decision_tree.html)。

## 1. 当前调用关系

```text
main.py
├─ config.py + config/competition_config.json
├─ task_manager.py
│  ├─ models.py / utils.py
│  ├─ map_model.py → load_pos.py（load_tag_positions）
│  ├─ localizer.py（共享 Tag 世界坐标）
│  ├─ motion.py → hardware.py → hiwonder / TonyPi ActionGroups
│  ├─ vision.py + classifier.py → Worker /predict 或 Central /predict、/requests/{id}
│  ├─ interaction_logic.py
│  ├─ interaction_client.py → robotall.send_request
│  └─ debug.py
└─ TaskManager.run() 的 finally → TaskManager.close()
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
- 检查目标花名及 central 模式的 student_id/password；直接脚本入口仍导入 `competition_tonypi`，包模式使用相对导入；
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
- `run_localization_search_sequence()`：启动与缺失 Pose 的高级恢复共用“完整 pan → 身体搜索动作 → 完整 pan”；运行时连续定位失败使用独立的有界位置恢复；
- `localize_scan()`：普通模式在任意有效视觉 Pose 后停止；指定 `required_target_screen_id` 时必须等到该目标 Tag↔Screen 绑定；
- `accept_visual_localization()`：只有接受视觉 Pose 才清零动作计数与运动不确定度；
- `record_localization_failure()`：所有失败累加 `consecutive_localize_failures`；看到 Tag 清零 `consecutive_no_tag_scans`，未看到则累加（包括 `capture_failed`）；记录 `no_tag`、`pose_unavailable_with_tags`、`suspect_visual_pose_rejected` 或 `capture_failed`，不直接降低旧 Pose confidence。

#### NO-TAG 与转向证据

- `recover_from_no_tag_if_needed()`：两项连续失败计数的最大值≥2、cooldown≥4 s 且 enabled、未 active/耗尽时触发；墙边选安全横移，否则检查后退 corridor 后退 5 cm，再检查新位置 rotation sweep 后向内转约 45°，中央复拍最多 3 轮；看到 Tag 无 Pose 或取帧失败仍继续有界循环。安全拒绝或轮数耗尽设置 `no_tag_recovery_exhausted` 并返回调用者，不递归 global recovery；成功清两项计数并设置 `pending_post_action_replan=True`；
- `evaluate_turn_progress()` / `monitor_turn_result()`：只让可靠视觉 Pose 产生 VERIFIED 结论；不可定位统一为 `PROGRESS_UNVERIFIED`；
- 正式位置 A* 使用可被 15° state 精确表示的 ±90° physical quarter-turn macro；action-level yaw lattice 也可细化到 1.5°。预测、dead reckoning 和 watchdog 始终使用配置物理动作角。

#### 途中视觉和目标确认

- `observe_transit_bindings()`、`process_bound_screen_candidate()`：从定位/导航帧提取合法绑定并写 15 秒缓存，不改变 ScreenStatus；
- `confirm_target_tag_now()`：最多看 `[100,130,70]`，只确认当前目标 Tag；
- `bounded_fresh_target_observation()`：当前目标新鲜分类最多 3 帧；
- `confirm_target_tag_and_screen()`：实时 Tag + 同 ID 绑定分类；
- `confirm_target_with_visibility_recovery()`：分类服务不可用/结果无效时保持目标，进入 WAIT/DEGRADED；目标确认最多 2 轮局部恢复仍失败时，以 `target_screen_confirmation_unresolved` 临时排除并清锁、重选。

#### 导航

- `plan_navigation_path()`：保留给 debug、fallback 和 Recovery 兼容的二维路径接口；
- `MapModel.plan_motion_actions()`：由 `navigate_motion_plan_to_target()` 调用的正式 `POSITION_NAVIGATION` Motion-Aware A*；yaw 保留在 state 中解释动作世界方向，但 Screen 最终 yaw 不参与位置阶段终点或 heuristic；
- `navigate_to_xy()`：兼容路径的自适应重定位、动作选择、到达前新鲜视觉 Pose、最终 yaw；与正式目标导航一样，连续定位失败达门槛且恢复不成功时返回 `localization_recovery_blocked`，自适应定位及恢复失败时返回 `localization_required`；
- `navigate_to_screen()`：一次建立 25 cm XY + desired yaw goal，并按真实距离进入 `POSITION A* (>10 cm) → NEAR_TARGET_ADJUSTMENT (5–10 cm) → FINAL_YAW_ALIGNMENT (≤5 cm)`；不再有中途 staging；
- `perform_near_target_adjustment()`：使用四个真实单周期 translation action 的连续坐标预测和既有 target-owned corridor safety 做有限次数精调；每次动作后强制视觉定位；
- `execute_motion_astar_action()`：直接执行 Planner 的 action key，可合并连续同动作，按 confidence/距离限制 batch；转向做视觉进展检查，平移自适应定位（可跳过），均请求重规划；平移不传播 post-action 定位的 False，由下一导航循环处理；
- `choose_translation_action()`：前进、短距离正后方倒退和平移；
- `adaptive_relocalization_decision()`：动作预算、Pose 原始置信度、不确定度、阶段和新大转向触发；`motion_sequence != last_relocalization_motion_sequence` 且动作数>0 才有大转向待定位请求，定位尝试即消费该序号。失败不直接把旧 Pose 改成 LOW；batch 使用的 `effective_localization_confidence()` 仍可因失败/陈旧而返回 LOW；
- `register_plan_failure()`：相同输入 3 次失败后升级，不等到 80 步才处理。

#### 恢复

- `recover_from_near_wall()`：后退、左右平移、小转向；
- `execute_bounded_escape()`：普通恢复全被 veto 时，从不安全起点选择更安全的小动作；
- `recover_via_indoor_waypoint()`：在内缩区域选可达、安全、尽量保 yaw 的 waypoint；
- `perform_global_recovery()`：重新定位，必要时 near-wall 或 interior recovery；
- `register_temporary_target_failure()`、`release_temporary_target_failures()`：导航失败/可见性耗尽直接临时排除；普通 `register_target_failure()` 达2次才升级。无可选目标时 global recovery 返回后释放普通临时失败（不要求恢复成功），NFC GAVE_UP 集合不释放。

#### FPGA 与 NFC

- `latest_valid_bound_flower_observation()`：15 秒、同 ID、binding、置信度检查；
- `adopt_cached_target_observation()`：实时当前 Tag 存在后把缓存变成授权；
- `execute_final_forward()`：主流程在非目标花且已分类授权、未 skip-change 时调用；函数自身检查同 ID 绑定确认和未执行标记，执行 `interaction_forward_final`（大步×4，约 20 cm），成功才设置 retreat pending；
- `process_screen_interaction()`：最多两次 NFC 物理尝试；
- `restore_nfc_physical_contact()`：Attempt1 失败后调用后退/定位，即使返回 False 仍进入最多3轮重获；仅接受 retry 开始后同 ID、有效 binding、confidence≥0.20 的新证据；已为目标花则直接 CHANGED，否则重新接近；pending 后退由 `complete_post_interaction_retreat()` 收尾，动作失败 blocked 不重发，动作成功但定位失败仅重试定位；
- `recalibrate_target_for_nfc_retry()`：只有当前目标重新分类仍不是 target 才重新导航/确认/final forward；
- `nfc_change_is_terminal()`：CHANGED 后禁止任何 retry；
- `give_up_nfc_change()`：两次失败或目标重获耗尽后结束该 Screen，mission 继续。

### `interaction_logic.py`

无硬件纯逻辑：

- 从 Tag 平面确定 WEST/EAST/SOUTH/NORTH；
- 从同一建筑 `face_center` 和 cardinal normal 生成 reader、25 cm interaction target 和 cardinal yaw；
- 保存分类但不执行交互；
- `apply_worker_change_result()` 在 Worker `success=True` 时写 `CHANGED`；另一路由 `TaskManager.restore_nfc_physical_contact()` 在 NFC 失败后凭同目标新分类证实已换，直接写 `CHANGED`。

## 5. 定位与地图

### `load_pos.py`

保存 AprilTag 世界四角坐标。1–36 是建筑/Screen Tag，37–61 是地面定位 Tag；动态障碍另由 `obstacle.tag_min_id=81` 控制，正式 JSON 关闭该功能。此文件是地图事实源，不应因文档或 Dashboard 显示需求改坐标。

### `localizer.py`

- AprilTag detector 适配；
- ID 1–61、面积、边缘、世界坐标存在性、PnP、向量/旋转合法性检查；场地/建筑实体及时间一致性由 TaskManager 检查；
- 先建筑 Tag 1–36，再地面 Tag 37–61；同类按面积降序、ID 升序。TaskManager 拒绝一个候选后继续后续候选，建筑全部不可用才回退地面；PnP 候选初始 HIGH，TaskManager 按帧内 Tag 数量/面积可降为 MEDIUM；
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

把 crop 编码成 JPEG。`direct` 模式保持只以 multipart `image` POST 到 KV260 Worker `/predict`；`central` 模式额外提交 `student_id`，并在 POST `/predict` 和 queued GET `/requests/{request_id}` 中使用 `X-Student-Password`。连接异常、5xx、408、429 被标记为可恢复 service unavailable；401 等其他 HTTP 错误不可重试；缺失花名、非法 confidence/JSON 属于 invalid response，解析器不校验花名是否在12类白名单内。

### `fpga_flower_server/fpga_server_api_ready.py`

运行在 Kria/PYNQ：加载 bit/hwh、DMA 和 12 类模型；串行处理 `/predict`，返回 API 花名、中文花名、类别编号和 confidence。服务说明见 [fpga_flower_server/README.md](fpga_flower_server/README.md)。

## 7. 动作和硬件

### `motion.py`

- `RobotState`：视觉 Pose、dead reckoning、动作计数、运动不确定度及 `motion_sequence`；有 Pose 且实际完成周期>0 时每个 action result 增加一次序号，`set_pose()` 不清该序号；
- `MotionController`：执行配置动作、按真实完成周期更新模型；
- 失败或部分动作不会虚报全部 requested cycles。

### `hardware.py`

相机后台读取、云台、ActionGroup 执行、动作序列、stop/close。动作执行前检查动作组是否存在；交互期间阻止普通动作并允许 stand 清理。

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
→ 稳定等待 0.5 s，再次授权检查
→ 生成新 seq
→ send_request(retries=0, clear_first=True, scan_timeout_s≤25s, overall_timeout_s≤25s)
→ finally stand
```

每次物理 Attempt 使用新 seq，底层继续校验 worker_id/seq；同一物理位置不会由底层自动重试。25 s 是正式 JSON 覆盖值（默认 15 s），任务管理器按调用前剩余任务时间截短，response timeout 为 1 s；授权检查核对身份、绑定、到达和花名，不重新取帧或检查证据年龄。

## 9. Debug

### `debug.py`

事件 JSONL、latest_state、标注图、地图、crop 和 8090 Dashboard。地图同时显示 Screen anchor、TargetGoal、当前导航 goal、recovery waypoint 和 path。

## 10. 测试和辅助脚本

### 自动化测试

- `test_calibrate_motion.py`：动作标定纯逻辑；
- `test_interaction_flow.py`：几何、授权、NFC deadline/seq/异常；
- `test_mission_scheduler.py`：目标选择、状态机、near-wall/forced escape、timeout；
- `test_navigation_adaptive.py`：动作批次、新大转向单次定位、连续定位失败恢复/安全 corridor/非递归耗尽、建筑与地面 Tag 优先级、倒退/平移、task safety bypass；
- `test_navigation_path_fallback.py`：clearance、兼容 fallback、规划失败升级；
- `test_recovery_target_consistency.py`：TargetGoal 原子一致和 interior recovery；
- `test_target_direct_approach.py`：当前目标软 cost 例外和直接动作；
- `test_mission_refactor.py`：物理定位 gate、hard jump、建筑候选拒绝后地面 fallback、两阶段目标评分、Motion-Aware A* goal/action、Position 轻量正代价转向、15 cm reverse 边界和 Planner→Executor 一致性；同目录 `test_near_target_adjustment.py`、`test_target_geometry.py` 核对单周期精调与目标几何；
- `test_target_standoff_flow.py`：目标确认、缓存、final forward、NFC 两次尝试和目标重获；
- `test_vision_tag_binding.py`：Tag↔Screen 绑定和 15 秒缓存；`test_classifier.py` 核对 direct/central 请求、鉴权和错误解析。

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
