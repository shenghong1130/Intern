# TonyPi 比赛决策树说明（按当前代码与配置核实）

本文面向实习报告，说明 TonyPi 在比赛中如何定位、选目标、规划、执行、识别和换花。事实来源为当前 `task_manager.py`、定位/地图/运动/视觉/交互模块、`config.py`、`config/competition_config.json` 与相关测试；规划器中的离散模型与机器人实际 ActionGroup 标定分别说明。

> 当前实现提示：`record_localization_failure()` 对每次失败累加 `consecutive_localize_failures`，并按是否看到 Tag 清零或累加 `consecutive_no_tag_scans`（`capture_failed` 也累加）。Recovery 取两项计数的最大值判定门槛，所以看到 Tag 但无可接受 Pose 也能触发；恢复内定位失败继续有限轮次，安全拒绝或耗尽后返回调用者，不递归 global recovery。定位尝试消费当前 motion request，大转向不会仅因历史 action 名持续要求定位。

# 0. 整体比赛主流程

```text
启动
  ↓
初始化硬件、地图、视觉、FPGA/Worker 客户端与 RobotState
  ↓
首次 AprilTag 定位 ──预算耗尽──→ 返回任务外层再次首次定位
  │ accepted pose                         │
  ↓                                       └─比赛时间到→安全停止
选择未完成且当前可选的 Screen
  ↓
真实距离 >10 cm：POSITION_NAVIGATION 使用 5 cm grid Motion A*
  ↓
真实距离 5<d≤10 cm：NEAR_TARGET_ADJUSTMENT 单周期平移并强制视觉定位
  ↓
真实距离 ≤5 cm：POSITION 完成
  ↓
FINAL_YAW_ALIGNMENT：fresh visual Pose 后原地校正 Screen-facing yaw
  ↓
实时确认 locked Tag 与对应 Screen，FPGA 分类
  ↓
花卉已经正确？
  ├─是→ ALREADY_TARGET → 当前目标完成 → 选择下一 Screen
  └─否→ final forward 20 cm → NFC（最多 2 次物理 Attempt）
                                  ├─成功/视觉证实已换→ CHANGED
                                  └─两次失败或重获耗尽→ GAVE_UP
                                                   ↓
                                  处理当前目标并选择下一 Screen
                                                   ↓
                      仍有可选目标？──是→回到“选择 Screen”
                           │否
                           ↓
             全部 Screen.done()→MISSION_COMPLETE 等待总时限
             尚有未完成（含 GAVE_UP）→global recovery/释放临时失败→重新选择
                           ↓
          比赛总时限 570 s 到达（任意阶段均适用）→安全停止
```

机器人采用“计划—执行一小批—必要时视觉定位—重新计划”的闭环，而不是一次执行完整 A* 路径。正常运行唯一自动终止条件是全局比赛时限；全部目标处理完后进入完成状态等待，最终仍由时限触发 `hardware.stop()`。

代码参考：`TaskManager.run_mission()`、`navigate_motion_plan_to_target()`、`process_screen_interaction()`。

# 1. 初始化和首次定位

## 1.1 流程图

```text
加载默认配置，再用 competition_config.json 深合并覆盖
  ↓
加载 Tag 世界坐标；建立 300×300 cm、5 cm 网格地图
  ↓
按 Tag 构造 Building/Screen（排除 2、6、9、20、28）
  ↓
建立 Camera、AprilTag detector、Localizer、ScreenDetector
  ↓
建立 FPGA ClassifierClient、NFC/Worker Client、Hardware、Motion、RobotState
  ↓
云台回中（pan=100）
  ↓
按完整 pan 序列 [100,135,65,155,45] 扫描（接受 Pose 即提前结束）
  ├─accepted pose→RobotState.set_pose()→转 3 目标选择
  └─无 accepted pose
       ↓
    依次循环执行 [左转,左转,左转,左转,后退] 中的一个身体动作
       ├─动作未执行/失败→消耗该搜索序号→尝试下一个搜索动作
       └─动作完成→再次完整 pan 扫描
                       ├─accepted pose→转 3
                       └─仍失败→下一搜索动作（最多 14 个）
  ↓
14 个身体搜索动作预算耗尽
  ↓
initial_localize() 返回失败→run_mission 外层等待 0.5 s 后重新开始首次定位
  ├─后来 accepted pose→转 3
  └─总时限到→MISSION_TIMEOUT→hardware.stop()→安全停止
```

## 1.2 流程说明

配置先采用 `config.py` 的完整默认值，再以 JSON 覆盖现场标定值。地图由 Tag 世界坐标和建筑几何建立，Screen 的 ID、AprilTag ID、Worker ID 必须一致。首次定位不是“扫描 14 次”：先做一次完整五角度扫描；失败后最多执行 14 个身体搜索动作，每个成功动作后再做一次完整扫描。启动阶段尚无可信 Pose，因此搜索动作以 `runtime_safety=False` 执行，这是专门用于打破初始不可见状态的有限预算。

## 1.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 场地 / 网格 | 300×300 cm / 5 cm | 世界地图与 A* 离散分辨率 | 统一定位和规划坐标 |
| 初始 pan 序列 | 100, 135, 65, 155, 45 | 中、左、右、更左、更右 | 扩大首次可见范围 |
| `startup_attempts` | 14 | 完整首扫之后允许的身体搜索动作数 | 搜索有限，避免无限盲动 |
| 身体搜索序列 | 左转×4、后退×1，循环 | 每次动作后重新完整扫描 | 改变视角和位置 |
| 总比赛时限 | 570 s | 所有阶段共享 | 最终安全停止边界 |

## 1.4 对应动作 / 代码

| 动作 | 实际 action key / ActionGroup | 模型效果 | 使用场景 |
|---|---|---:|---|
| 云台回中 | `center_head` | pan=100 | 首次扫描前 |
| 启动左转 | `turn_left_fast` / `turn_left_small_step_s80` | +7.5°/cycle | 连续改变观察方向 |
| 启动后退 | `back_fast` / `back_start→back→back_end` | -2.5 cm/cycle | 改变相机位置 |

代码参考：`TaskManager.__init__()`、`initial_localize()`、`run_localization_search_sequence()`。

# 2. AprilTag 定位

### 流程图

```text
按当前 pan 拍摄
  ├─所有 pan 均无有效帧→capture_failed
  │                         ↓
  │       两项连续计数均 +1；不直接修改旧 Pose confidence
  │                         ↓
  │       返回调用者的定位/导航循环；若 Recovery 门槛被满足→转 2.1
  └─获得帧
       ↓
    AprilTag detector→detected_tag_ids
       ├─完整扫描中始终 []→genuine no_tag→转 2.1
       └─至少看见一个 Tag
            ↓
     ID 1..61、面积≥350 px²、边缘≥35 px、世界坐标存在
            ↓
     solvePnP，检查 rvec/tvec 有限，生成候选 RobotPose
       ├─全部候选不可用→TAG_SEEN_POSE_UNAVAILABLE→转 2.2
       └─建筑 1..36 优先、同类面积降序/ID 升序→逐个通过 2.2 检查
           └─建筑候选全部拒绝→再试地面 37..61；接受一个即结束普通扫描
```

### 流程说明

检测到 Tag 不等于获得位置。Localizer 对每个候选做 ID、面积、边缘和地图坐标检查，再用 PnP 计算机器人位姿；先建筑 Tag，再地面 Tag，同类按面积降序、ID 升序。TaskManager 逐个检查场地/建筑物理合法性和与当前 Pose 的一致性，拒绝一个仍继续后续候选，建筑全部不可接受才回退地面。整次扫描失败才返回调用者：导航走第 5 节的定位失败/恢复返回分支；目标重获回第 7 节；初始定位回第 1 节搜索循环。

### 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 合法 Tag ID | 1..61 | 建筑 1..36 优先，地面 37..61 后备 | 地面 Tag 不授权 Screen/NFC |
| 最小面积 | 350 px² | 过小 Tag 不参与 PnP | 远小目标角点误差大 |
| 图像边缘留白 | 35 px | 过于贴边的 Tag 拒绝 | 避免裁切角点 |
| detector upscale | 1.5 | 检测前放大比例 | 提高小标签检测能力 |
| HIGH 门槛 | 帧内≥2 个 ID/面积合格 Tag，或最大面积≥700 px² | PnP 初始 HIGH，TaskManager 按帧质量可降 MEDIUM | 仍须物理/时间门控，非多 Tag 融合 |

### 对应动作 / 代码

本节没有身体动作；云台通常按 `[100,135,65,155,45]` 扫描，目标确认使用 `[100,130,70]`。

代码参考：`localize_scan()`、`Localizer.estimate_from_frame()`、`evaluate_and_accept_visual_pose()`。

## 2.1 Genuine NO-TAG

```text
完整扫描有帧但 detected_tag_ids 始终为空→genuine no_tag
  ↓
两项连续失败计数 +1（看到 Tag 无 Pose 只增加 localize failures）
  ↓
max(consecutive_no_tag_scans, consecutive_localize_failures)≥2 且 cooldown≥4 s？
还须 enabled、非 active、非 no_tag_recovery_exhausted
  ├─否→返回原定位调用者：初始搜索 / 第 5 节导航重定位 / 第 7 节重获
  └─是→NO-TAG Recovery（最多 3 cycle）
          ↓
       当前保留 Pose 靠墙或靠边界？
       ├─是→选择通过 corridor 检查的净空侧→横移 1 cycle（左4/右3 cm）
       │      └─无安全横移→设置耗尽标记→返回 False 给调用者
       └─否→有 Pose 时检查后退 escape corridor→后退约 5 cm
              └─后退后的 Pose 检查 10 cm rotation sweep、cost≤55→向内转约 45°
                 任一安全检查拒绝→设置耗尽标记→返回 False 给调用者
          ↓
       上述身体动作成功？
          ├─否→hardware_failure→返回当前调用者重定位/重规划；无法继续则目标临时轮换
          └─是
             ↓
       云台回中，只在 pan=100 重新扫描
          ├─accepted pose→清两项连续计数→pending_post_action_replan=True→保留目标/重规划
          ├─看到 Tag 但 Pose 不可用→清 no-tag、增加 localize failures→下一 cycle
          ├─capture_failed→两项计数增加→下一 cycle
          └─仍 genuine no_tag→两项计数增加→下一 cycle
                                  ├─未到 3→继续 Recovery
                                  └─3 cycle 耗尽→no_tag_recovery_exhausted=True
                                           ├─failure reason=no_tag_recovery_exhausted→返回 False
                                           ├─导航调用者可覆盖为 localization_recovery_blocked/required→5
                                           └─不在此调用 global recovery；后续主循环按目标可选情况处理
```

普通区域先后退再转向；有 Pose 时先检查实际后退终点的 escape corridor：物理走廊清晰、终点 footprint 自由且最大 cost 不升，并满足 cost 至少改善2、净空至少改善1 cm 或终点已 traversable 之一；再在后退后的 Pose 检查 10 cm rotation sweep（max cost 55）。靠墙/边界时选择安全横移。无 Pose 时跳过这两项依赖位置的检查，转向按奇偶轮交替左右。看到 Tag 但无可接受 Pose、零 Tag 或取帧失败都继续下一轮，每轮重新检查安全；硬件失败立即返回 `hardware_failure`，安全拒绝或三轮耗尽设置耗尽标记。该标记阻止后续同类 Recovery，直到接受视觉 Pose 清除；函数自身不进入 global recovery。

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 触发计数 / cooldown | max(连续 no-tag, 连续定位失败)≥2 / 4 s | enabled 且非 active/耗尽 | 抑制单帧抖动和频繁恢复 |
| 普通区动作 | 后退 5 cm + 向内转 45° | 扩大可见视角 | 打破遮挡/背向状态 |
| 墙边动作 | 选向步长4 cm；实执行左4/右3 cm | 单周期横移 | 降低扫墙风险 |
| 最大 cycle | 3 | 单次 Recovery 上限 | 耗尽返回调用者并阻止递归恢复 |

动作：后退使用 `back_fast`（-2.5 cm/cycle，计算为 2 cycle）；转向使用 `turn_left_fast` 或 `turn_right_fast`（±7.5°，计算为 6 cycle）；横移使用左右 `*_fast` 1 cycle。

代码参考：`record_localization_failure()`、`no_tag_recovery_needed()`、`recover_from_no_tag_if_needed()`。

## 2.2 看见 Tag 但定位失败及其他定位情况

```text
至少看见一个 Tag
  ↓
经过质量门与 PnP 后有候选 Pose？
  ├─否→TAG_SEEN_POSE_UNAVAILABLE
  │       ↓
  │   清 genuine no-tag 计数，普通 localization failure +1
  │       ↓
  │   扫描失败返回调用者；连续定位失败≥2 也可触发 2.1；NFC 重获→7
  └─是→候选 Pose
          ↓
       在 300×300 cm 场内且不在建筑实体内？
       ├─否→physical rejection→保留旧 Pose→继续后续候选；全轮失败才返回调用者
       └─是→与旧 Pose 比较
               ├─距离>40 cm 或 yaw>60°→hard jump 立即拒绝
               │                            ↓
               │                  保留旧 Pose→尝试下一候选/地面 Tag；全失败才回调用者
               ├─距离≤15 cm 且 yaw≤25°→accepted pose
               └─普通冲突→再取 1 帧确认
                            ├─确认帧与 suspect 在 10 cm/15°内且自身合法→接受确认 Pose
                            ├─确认帧回到旧 Pose 一侧→接受确认 Pose
                            └─不支持/无 Pose/仍 hard jump→suspect rejected
                                                          ↓
                                               保留旧 Pose→后续候选；整轮失败计数→调用者/2.1

accepted pose
  ↓
RobotState.set_pose()；actions_since_localize=0；motion_uncertainty=0
  ↓
正常扫描成功清两项连续计数；清耗尽标记、更新 last_localize_success_s
  ↓
回发起定位的任务；导航时转第 5 节重新规划
```

hard jump 不做“两个远帧一致就接受”，因为它可能把机器人瞬间安装到完全错误的建筑一侧；普通冲突才允许一次二次确认。其他 Tag 可建立机器人 Pose，但在 NFC 重获模式中，只有 locked target 的 `Tag ID == Screen ID` 且绑定 crop 有效才算目标重获。

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 普通冲突 | 15 cm / 25° | 与旧 Pose 的可疑差异 | 防止单帧跳变 |
| hard jump | 40 cm / 60° | 无条件拒绝阈值 | 防止灾难性位置跳跃 |
| 确认一致 | 10 cm / 15° | 二次帧支持范围 | 要求独立证据 |
| 确认次数 | 1 | suspect 的追加帧数 | 有限延迟 |

代码参考：`assess_visual_localization()`、`evaluate_and_accept_visual_pose()`、`evaluate_and_accept_visual_candidates()`、`accept_visual_localization()`。

# 3. 目标 Screen 选择

## 3.1 流程图

```text
全部配置 Screen
  ↓
排除 CHANGED / ALREADY_TARGET、临时失败集合、nfc_gave_up 集合
  ↓
已有仍合法的 locked target？
  ├─是→保持同一 Screen 与原子 TargetGoal→转 4
  └─否→取每个候选的建筑面外法向 25 cm、左向偏移 -1 cm（即右移1 cm）interaction target
          ↓
       计算机器人到 interaction target 的直线距离
          ↓
       保留“最近距离 +25 cm”窗口内候选
          ↓
       score = distance + min(25, behind_turn×0.20 + final_yaw_error×0.03)
       behind_turn = max(0, |目标方位与当前 yaw 差| - 90°)
          ↓
       按 score、distance、Screen ID 依次打破平局
          ↓
       原子锁定 Screen ID = Tag ID = Worker ID + XY + desired yaw + generation
          ↓
       → 4. Motion A*

无候选
  ├─全部已 CHANGED/ALREADY_TARGET→MISSION_COMPLETE→等待 570 s 时限→安全停止
  └─仍有未完成目标→global recovery→释放普通临时失败→回本节
       （NFC GAVE_UP 不释放；若只剩它们则持续恢复/等待到总时限）
```

## 3.2 流程说明

目标不是建筑墙面中心，而是对应建筑面的交互站位：沿 outward normal 离墙 25 cm，再沿机器人左侧切向偏移 -1 cm。canonical Screen-facing yaw 只在 `build_interaction_geometry()` 中由 `atan2(-normal_y,-normal_x)` 生成，再加 5°。先用距离形成 25 cm 近邻窗口，再轻度惩罚位于身后和最终朝向差大的候选，使机器人仍以“近”为主，但减少大转身。`TargetGoal` 将身份和坐标一起锁定，防止导航的是 A 屏坐标、视觉/NFC 却操作 B 屏。

## 3.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| standoff / lateral | 25 cm / -1 cm | 正式交互目标 | 留出 final forward 与左手 NFC 几何 |
| desired yaw offset | +5° | 最终面向修正 | 匹配现场标定 |
| 最近距离窗口 | 25 cm | 可参与朝向比较的候选 | 避免为省转向绕远路 |
| behind / final-yaw 惩罚 | 0.20 / 0.03 cm/° | 朝向代价 | 更偏好易接近目标 |
| 惩罚上限 | 25 cm | 朝向总代价上限 | 保持距离主导 |

## 3.4 对应动作 / 代码

本节只计算并锁定目标，不执行实体动作。

代码参考：`configure_cardinal_task_targets()`、`build_interaction_geometry()`、`choose_nearest_screen()`、`lock_target_goal()`。

# 4. Motion A* 路径规划

## 4.1 流程图

```text
输入 accepted/dead-reckoning Pose + 原子 TargetGoal
  ↓
量化 XY 为 5 cm 网格；以当前真实 yaw 为原点建立 15° yaw state
  ↓
以 forward / reverse / strafe / ±90° quarter turn 扩展 Position A*
  ↓
每条平移边检查机器人走廊、建筑/场界、非目标障碍代价和净空
每条转向边检查 10 cm rotation sweep
  ↓
累计平移距离、软障碍、净空不足、动作切换/反转、远离目标和轻量 Position turn 代价
  ↓
真实 Pose 到目标仍 >10 cm？（最终 Screen yaw 不参与 goal test / heuristic）
  ├─是→重建可执行 PlannedNavigationAction 列表→转 5
  ├─5~10 cm→转 4.5 做连续坐标单周期精调
  └─≤5 cm→Position 完成→Final Yaw
open set 耗尽或扩展到 45000→A* 无路径
       ↓
    同签名失败次数 <3？
       ├─是→重新 AprilTag 定位→回本节重规划
       └─否→选择室内安全 waypoint 做 Recovery
                ├─成功→清失败 watchdog→保留目标→回本节
                └─失败→navigation_blocked→该目标记为临时失败→回 3 选其他目标
总时限到或严重硬件故障→安全停止/交回任务主循环处理
```

## 4.2 流程说明

普通 XY A* 会假设机器人能沿任意方向连续移动，而 TonyPi 只能执行有限 ActionGroup，所以正式规划状态仍包含朝向。平移边按当前 yaw 解释 forward/reverse/左右横移，转向边改变 yaw state；但终点条件和启发式只看 interaction XY。正式 Position 模式保留 ±90° quarter turn，并为其加入轻量固定与角度代价；完全同代价时再用较少 turn 次数破平局。每个 turn 仍必须通过 rotation sweep safety。最终 Screen yaw 继续只在到达 XY 后处理。

正式位置搜索只扩展对称的 ±90° quarter-turn macro，因此能被 15° state 精确表示，又不会探索大量细碎 yaw 组合。macro 由 `turn_left/right_fast` 的 7.5°/cycle 执行；安全的 HIGH/MEDIUM 90° objective 可连续执行到接近目标角度再统一定位。action-level 精确 yaw 模式会把 15°请求细化成当前 1.5° lattice；动作预测 Pose 始终直接按物理 action model 重建，因此 -18° 不再被预测为 -30°或污染后续平移方向。

## 4.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 网格 / position yaw state | 5 cm / 15°，只扩展 ±90° | A* 状态离散 | 保留方向语义并限制搜索规模 |
| position heuristic 权重 | 1.0 | 可采纳 XY 距离 | 保证较短安全平移路线优先展开 |
| 规划步长 | forward 28、fine 7、strafe 12、reverse 5 cm | 扩展模型 | 对应可批量执行动作 |
| 位置阶段到达容差 | 5 cm | 真实 Pose 到 interaction XY 的连续距离 | 避免 grid center 误差阻止到达 |
| 走廊半宽 / 转动扫掠 | 8 cm / 10 cm | 碰撞检查 | 不只检查质点 |
| segment 最大代价 / 目标净空 | 55 / 25 cm | 软障碍约束 | 远离墙体与非目标建筑 |
| 最大扩展 / 同签名失败 | 45000 / 3 | 规划预算与升级门槛 | 避免无限重算同一失败 |
| Position 主要代价 | 平移距离 + obstacle/clearance/away/switch + 轻量 turn | 动作偏好 | 近似等长时少转弯；明显缩短或绕障时允许转弯 |
| Position quarter-turn | fixed 4 cm + 0.08 cm/deg | 防止 turn cost≈0 | 不压制有明显收益的转向 |
| 横移转换惩罚 | 18 cm | `90° turn→forward→反向 90° turn` | 优先直接 strafe 的近似等长方案 |

## 4.4 对应动作 / 代码

| Planner 边 | 实际 action key | 规划模型 | 备注 |
|---|---|---:|---|
| forward / fine | `forward_fast` | 28 / 7 cm | 最终换算为 3.5 cm/cycle |
| reverse | `back_fast` | 5 cm | 只在目标位于后方且最终距离≤15 cm，并通过角度/横差/缩距/corridor gate 时扩展 |
| strafe left/right | `strafe_*_fast` | ±12 cm | yaw 保持不变 |
| position quarter turn | `turn_left/right_fast` | 规划 ±90°；物理 ±7.5°/cycle | 安全 objective 可连续执行，风险条件退回短批次 |
| final yaw large turn | `turn_left/right_large` | 物理 +15° / -18° | 只在近点闭环校正中按真实角执行 |

代码参考：`MapModel.plan_motion_actions()`、`action_planner_transition()`、`navigate_motion_plan_to_target()`。

## 4.5 接近目标后，用真实单周期动作精确靠近目标

小字：Near Target Adjustment

```text
                 目标 Screen
                     │
                     ▼
            interaction XY
                 约25cm
                     │
                     ▼
             当前真实 RobotPose
                     │
                     ▼
              计算真实距离
                     │
        ┌────────────┼─────────────┐
        │            │             │
     >10cm        5~10cm          <=5cm
        │            │             │
        ▼            ▼             ▼
    Motion A*    Near Target    POSITION完成
        │        Adjustment          │
        │            │               ▼
   5cm网格规划        │           Final Yaw
        │            │               │
        │      预测单周期动作         ▼
        │            │             ARRIVED
        │    ┌───────┼────────┐
        │    │       │        │
        │ forward   back    strafe
        │  3.5cm    2.5cm   3~4cm
        │    │       │        │
        │    └───────┼────────┘
        │            │
        │       安全且距离下降？
        │         /        \
        │       YES         NO
        │        │           │
        │    选最优动作      全部不合格→定位；成功有限重试，失败返回导航调用者
        │        │
        │    执行1 cycle；硬件失败→导航 False→临时排除/3
        │        │
        │    重新视觉定位；失败→near_target_visual_localization_required→导航 False/3
        │        │
        └────────┼───────────
                 ▼
            重新计算距离
                 │
            distance<=5？
             /          \
           YES           NO
            │             │
            ▼             └──再次判断距离区间
     POSITION完成
            │
            ▼
        Final Yaw
            │
            ▼
         ARRIVED
```

全局 Motion A* 的 5 cm grid 继续负责大范围路径、障碍绕行和方向规划。真实动作 `forward_fast≈3.5 cm`、`back_fast≈2.5 cm`、`strafe_left_fast≈4 cm`、`strafe_right_fast≈3 cm` 小于或接近一个 grid；执行后可能仍映射到同一 cell。若把网格改为 2.5 cm，XY state 数约增至原来的四倍，再乘 yaw state；若把所有小动作塞进全局 A*，分支数和近点重复状态也会增加。因此 `5 < real distance ≤10 cm` 不进入 A*，只在真实连续坐标中试算四个单周期动作。

每个候选使用带符号的 motion config：`predicted_xy = pose_xy + forward_cm·heading + lateral_cm·left_heading`，按 `predicted_distance` 最小、候选顺序破平局。候选必须严格缩短距离，并通过 field bounds、hard occupancy、robot footprint、无关建筑、动态障碍和 translation corridor 检查；当前目标建筑的 soft inflation 沿用既有豁免。back 另要求 reverse enabled、目标在后方、距离≤15 cm、后向角≤30°、横差≤8 cm、有效定位置信度非 LOW。选中后只执行 1 cycle，并立即定位；失败直接返回导航 False。四个候选均不合格时先定位，成功才有限重试（失败分支未另设 reason，外层使用现有 reason 或 `navigation_failed`）。最多尝试 4 次，仍在近点区间则调用 `recover_via_indoor_waypoint()`；成功清精调计数再循环，失败返回。距离重新>10 cm 时也清精调计数。

| 状态 / 事件 | 中文说明 | 硬件故障 | 是否立即放弃目标 | 下一步 |
|---|---|---|---|---|
| `NEAR_TARGET_ADJUSTMENT` | 接近目标后的单周期位置精调 | 否 | 否 | 预测并执行一个安全缩距动作，然后重新定位 |
| `near_target_adjustment_stalled` | 当前四个单周期动作都无法安全地继续靠近 | 否 | 定位失败则返回并临时排除 | 定位成功后在有限次数内重试 |
| `near_target_adjustment_exhausted` | 连续精调达到上限后仍未进入 5 cm 范围 | 否 | 室内恢复失败则临时排除 | 室内 waypoint 成功后清计数继续 |

代码参考：`perform_near_target_adjustment()`、`navigate_motion_plan_to_target()`。

# 5. 动作执行、批次、重新定位与重规划

## 5.1 流程图

```text
A* 动作序列
  ↓
只取开头连续同 action_key 的动作，汇总 requested cycles
  ↓
按 HIGH/MEDIUM/LOW、动作类型、距目标、模式限制 batch
  ↓
执行前再做走廊/转动扫掠检查
  ├─安全拒绝→近墙 Recovery
  │              ├─恢复可继续→回 4 重规划
  │              └─硬件失败/耗尽→当前目标临时失败→回 3
  └─通过→执行 ActionGroup
          ├─动作失败→hardware_failure→当前目标临时失败→回 3
          └─动作完成→dead reckoning 更新 Pose、实际周期数、uncertainty、motion_sequence
                         ↓
                  是转向？
                  ├─是→Turn Progress Watchdog（下图）
                  └─否→是否强制重新定位？
                         ├─是→AprilTag 定位
                         │      ├─accepted→回 4
                         │      └─失败→保留旧 Pose；正式 A* 平移动作仍返回 True→下一导航循环
                         └─否→保留 dead reckoning→仍回 4 重规划
下一导航循环（正式目标导航 / navigate_to_xy）
  ├─无 Pose→扫描/尝试 2.1 后仍无 Pose→localization_required→返回 False/临时排除/3
  ├─连续定位失败≥2→尝试 2.1
  │                  ├─成功→continue→重新计算 Pose/计划
  │                  └─未触发或失败（含 cooldown/耗尽）→localization_recovery_blocked→False/3
  ├─自适应要求定位→扫描失败→尝试 2.1
  │                              ├─成功→continue→重规划
  │                              └─未触发或失败→localization_required→False/3
  └─无需定位→按当前 Pose 距离进入 Position / Near Target / Final Yaw
```

```text
完成当前 TURN batch / objective
  ↓
统一 post-turn visual localization
  ├─无可信 before/after Pose→PROGRESS_UNVERIFIED
  │                            ↓
  │             不增加“没转动”计数→回 4 重规划/后续定位
  └─有可信视觉前后 Pose
       ↓
    |expected yaw|≥5° 且 |实际 yaw 变化|<max(2°, 25%×|expected yaw|)？
       ├─否→VERIFIED_PROGRESS→保留已接受视觉 Pose→清计数→回 4
       └─是→VERIFIED_NO_PROGRESS（方向冲突仅诊断），计数 +1
                    ├─<2→回 4
                    └─≥2→强制再定位确认
                           ├─确认有进展→清计数→回 4
                           ├─仍无可靠 Pose→PROGRESS_UNVERIFIED→回 4/定位
                           └─确认仍无进展→RECOVERY_NO_PROGRESS
                                                 ↓
                              当前导航失败→目标临时失败→回 3；有时间可再选
```

## 5.2 流程说明

A* 选择 forward/strafe 时按安全平移距离和路径附加代价比较；Position quarter-turn 使用轻量正代价，且 `turn→forward→反向 turn` 增加横移转换 penalty，因此近似等长时优先少转弯，路线明显更短、需要绕障或平移不安全时仍会转向。reverse 仅用于目标在后方、横向误差≤8 cm、后向角≤30°、最终目标距离≤15 cm且动作确实缩短距离、后方 corridor 安全的情况，并禁止先转向/其他平移再 reverse。Position 只扩展 ±90°；`require_goal_yaw=True` 和兼容路径评分继续使用完整 turn cost。无论从哪条路径选中 turn，执行安全检查和视觉复核均保持不变。

不能一次跑完整路径，因为动作误差会累计，且到障碍/目标附近容错更小。普通动作按 HIGH/MEDIUM/LOW、距离和 uncertainty 限批。约 45°/90° 的 turn 会形成 objective；HIGH/MEDIUM、rotation sweep 安全、净空不紧且无 VERIFIED_NO_PROGRESS 时可连续多个 cycle，再视觉定位并重新 A*；风险条件下退回短批次，最后保留 micro turn 微调。`motion_sequence` 在有 Pose 且实际执行周期>0 时每个 action result 增加一次，`set_pose()` 不重置它。大转向待定位要求 action 名含 large 或每周期模型 yaw≥35°、动作数>0、序号不同于 `last_relocalization_motion_sequence`；`localize_scan()`、实际 post-turn 扫描及 post-action 定位尝试记录当前序号，失败也消费该请求。历史 large action 不单独反复触发，但 LOW、动作预算和不确定度等独立条件仍能要求定位。定位失败不直接改旧 Pose confidence；`effective_localization_confidence()` 可因连续失败返回 LOW 用于 batch，真实动作失败/误差累计仍可在 `RobotState` 内降级。正式 A* 平移忽略 post-action 定位的布尔返回，下一轮再按上图门控；Turn Progress 未验证也先返回 True，由下一轮处理定位失败。

## 5.3 真实动作表

| Planner 动作 | action key | ActionGroup / sequence | 单 cycle 模型效果 | 常规最大 batch H/M/L |
|---|---|---|---:|---:|
| FORWARD | `forward_fast` | `go_forward_fast` | +3.5 cm | 8 / 4 / 1 |
| REVERSE | `back_fast` | `back_start→back(repeat)→back_end` | -2.5 cm | 6 / 3 / 1 |
| STRAFE_LEFT | `strafe_left_fast` | `left_move_fast` | +4 cm | 4 / 2 / 1 |
| STRAFE_RIGHT | `strafe_right_fast` | `right_move_fast` | -3 cm | 4 / 2 / 1 |
| TURN_LEFT small | `turn_left_fast` | `turn_left_small_step_s80` | +7.5° | 2 / 1 / 1 |
| TURN_RIGHT small | `turn_right_fast` | `turn_right_small_step_s80` | -7.5° | 2 / 1 / 1 |
| TURN_LEFT large | `turn_left_large` | 左小步×4 | +15° | 2 / 1 / 1 |
| TURN_RIGHT large | `turn_right_large` | 右小步×4 | -18° | 2 / 1 / 1 |

## 5.4 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| normal 动作预算 H/M/L | 6 / 4 / 1 cycles | 到此强制定位 | 限制里程推算漂移 |
| target-direct 预算 H/M/L | 3 / 2 / 1 | 近目标更严格 | 保护交互精度 |
| normal uncertainty limit | 6.0 | 自适应定位门槛 | 不同动作误差可加权 |
| uncertainty/cycle | forward .6；strafe 1；reverse .9；turn 1.8；large 2.6 | 误差累计 | 横移/转向更不稳定 |
| Near Target Adjustment | `5 < distance ≤10 cm`，固定 1 cycle | 连续坐标闭环精调 | 避免最后几厘米进入 grid A* |
| 程序化 large-turn 门槛 | yaw 差≥35° | `turn_toward()` 改用 large action；A* 本身按代价选择 | 减少恢复/校正中的小步次数 |
| turn objective 门槛 / H-M 最大周期 | ≥45° / 12 | 安全时连续完成大角度目标 | 避免每 7.5°/15° 都定位 |
| watchdog 无进展 | 2 次可靠确认 | 导航中止门槛 | 不把取帧失败误判成没转 |
| `collision_recovery_enabled` | false | 通用碰撞停滞恢复当前关闭 | 现行主流程依靠规划安全门和 near-wall recovery |

代码参考：`execute_motion_astar_action()`、`select_adaptive_action_batch()`、`adaptive_relocalization_decision()`、`monitor_turn_result()`。

# 6. 到达目标后的 Screen / Tag / 花卉识别与换花决策

## 6.1 流程图

```text
POSITION_NAVIGATION / NEAR_TARGET_ADJUSTMENT 接近 interaction target
  ↓
真实 XY 距离≤5 cm？
  ├─否→>10 cm 回 Motion A*；5~10 cm 做单周期精调（不优化最终 yaw）
  └─是→FINAL_YAW_ALIGNMENT：非 LOW、非 DEAD_RECKONING/UNKNOWN、动作数=0、视觉年龄≤3 s？
          ├─否→重新 AprilTag 定位
          │      ├─accepted 且仍在 5 cm 内→检查 yaw
          │      └─失败→记 final_yaw_visual_localization_required 并继续导航循环/5；位置改变→重判区间
          └─是→desired yaw 误差≤10°？
                 ├─否→target rotation sweep 安全？拒绝→final_yaw_turn_safety_rejected→False/3
                 │      └─通过→原地转向/视觉定位→重判 XY/yaw；硬件/确认无进展失败→False/3
                 └─是→ARRIVED
                 ↓
          实时扫描 [100,130,70]，必须看见 locked target Tag
                 ├─未见/绑定无效→最多 2 cycle 可见性恢复
                 │                 ├─恢复后确认→继续
                 │                 └─耗尽→target_screen_confirmation_unresolved→临时排除、清锁
                 │                         ↓
                 │                  SELECT_NEAREST_TARGET→回 3
                 └─实时 Tag ID=locked Screen ID（可不重新提取 crop）
                                      ↓
                     使用≤15 s 同屏绑定分类缓存；无缓存才取绑定 crop，最多 3 帧 fresh 分类
                                      ├─服务不可用→WAIT；结果错误→DEGRADED→保留目标、等1 s→主循环导航/再确认
                                      ├─置信度<0.20/绑定缺失→有限 fresh 重试；仍失败走可见性恢复/临时排除
                                      └─可信 flower
                                           ↓
                                  flower == target？
                                  ├─是→ALREADY_TARGET
                                  │      不 final forward、不举手、不 NFC→转 8
                                  └─否→NEEDS_CHANGE + fresh visual authorization
                                                  ↓
                                      `interaction_forward_final`×1
                                                  ├─失败→目标失败计数；必要时临时轮换→3
                                                  └─成功（模型 +20 cm）→转 7
```

## 6.2 流程说明

“到达”最终仍同时要求交互目标 XY、最终 yaw 和新鲜视觉 Pose，但顺序明确分为位置导航和最终朝向校正。其他 Tag 可用于定位机器人，却不能授权当前 Screen：业务证据必须满足 `Tag ID == Screen ID == locked target ID`，并由该 Tag 的几何关系截出正确屏幕 crop。

普通确认的 15 s cache 只复用最近的同 ID、binding 有效且 confidence≥0.20 的分类结果；还须实时看到 locked Tag，缓存路径不要求当前帧重新生成 Screen crop。途中帧也会提取绑定 crop、调用 FPGA 并更新缓存，但不修改 ScreenStatus；采用证据后才写花名和状态。分类服务不可用或结果错误分别保留目标进入 WAIT/DEGRADED，等 1 s 后回主循环重新导航/确认；低置信度未必属于服务错误，有限重试后可走可见性恢复，耗尽临时排除。NFC 的 `visual_authorization_check()` 只核对锁、确认/授权的同 ID 与 binding、到达标记及花名一致性，不重新拍摄，也不检查时间戳/置信度/Pose；不能把它描述成每次发包前重新验证新鲜度。

机器人先停在墙外约 25 cm，是为了以精确姿态确认屏幕；确认确实需要换花后，才用一个专用序列前进模型 20 cm，接近 NFC 工作距离。

## 6.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| ARRIVED | 5 cm / 10° / 视觉年龄≤3 s | Pose 非 LOW、非推算/UNKNOWN、动作数=0 | 防止 grid center 或陈旧 XY 误判到达 |
| 目标实时 pan | 100, 130, 70 | locked Tag 搜索 | 小范围找回目标 |
| cache TTL / 分类间隔 | 15 s / 1 s | 同屏分类缓存 | 降低重复 FPGA 请求 |
| fresh 分类重试 | 最多 3 帧，间隔 0.5 s | 无可用缓存时 | 有限等待视觉结果 |
| 可见性恢复 | 最多 2 cycle | 重新定位/小修正 | 轮内保持目标，耗尽临时排除并重选 |
| 分类置信度 | ≥0.20 | 可信花卉门槛 | 低置信度不授权 NFC |
| final forward | 20 cm，一次 | 25 cm 站位后的接近 | 到达 NFC 距离 |

## 6.4 对应动作 / 代码

| 动作 | action key / ActionGroup | 模型效果 | 使用场景 |
|---|---|---:|---|
| 可见性后退 | `back_fast` | -2.5 cm/cycle | 离墙过近且后方安全 |
| 可见性横移 | `strafe_left/right_fast` | +4 / -3 cm | 修正屏幕横向位置 |
| 最终接近 | `interaction_forward_final` / `go_forward_one_step`×4 | +20 cm | 仅 NEEDS_CHANGE 且已授权 |

代码参考：`navigate_motion_plan_to_target()`、`confirm_target_with_visibility_recovery()`、`bounded_fresh_target_observation()`、`execute_final_forward()`。

# 7. NFC 换花通信流程

## 7.1 流程图

```text
final forward 完成
  ↓
已有 visual authorization 的身份/绑定/到达/花名检查（不检查年龄或重新拍摄）
  ├─失败→不举手、不发 NFC→登记一次普通目标失败
  │                              ↓
  │              完成近位后退/重定位→清当前锁→回 3（未达 2 次时可再选本屏）
  └─通过→Client 举手前再检查→stand→举左手→稳定等待 0.5 s→再次 authorization 检查
          ├─失败→作为 Attempt 1 失败处理→finally 恢复 stand/左手→进入下方重获
          └─通过→生成新的 uint8 seq，清旧响应→NFC Attempt 1
                    ├─25 s 内收到匹配 seq 且 ok→CHANGED→恢复左手→转 8
                    └─timeout/无效响应/异常→恢复左手
                              ↓
                    后退约 10 cm并重新 AprilTag 定位
                              ├─后退/定位未完成→保留 pending；本函数仍进入同目标重获循环
                              └─完成或尚 pending→最多 3 cycle 重获同一 locked target
                                      ↓
                            其他 Tag→只能定位，不能替代目标授权
                                      ↓
                            同 ID Tag+Screen crop+本次 retry 开始后采集的有效 FPGA 分类？
                              ├─否且未耗尽→下一重获 cycle
                              ├─3 cycle 耗尽→GAVE_UP→转 8
                              └─是→fresh flower == target？
                                      ├─是→第一次很可能已换成功→CHANGED→转 8
                                      └─否→重新导航至同一 25 cm TargetGoal
                                               ↓
                                           再确认 locked Tag
                                               ↓
                                           final forward 20 cm
                                               ├─任一步失败→继续有限重获；耗尽→GAVE_UP→8
                                               └─成功→Attempt 2（新 seq）
                                                         ├─success→CHANGED→8
                                                         └─failure→GAVE_UP→8

循环检查总时限到→MISSION_TIMEOUT→恢复左手/停止→安全停止；本次流程绝无 Attempt 3
```

## 7.2 流程说明

通信链路是 `TonyPi ─NFC→ 对应 Worker → 换花执行`。TaskManager 每轮先检查已有授权，失败直接返回，登记普通失败并处理 pending retreat 后回第 3 节；未达两次普通失败门槛时仍可再次选择。Client 在举手前、举手后发送前再次检查，Client 返回的失败按本次 Attempt 失败处理（Attempt 1 进入同目标重获，Attempt 2 则 GAVE_UP）。这些门控不采集新帧、不检查证据年龄，检查的是身份、绑定、到达与花名。正式配置的单次 request/scan 上限为 25 s，并按调用前剩余任务时间截短；举手等待、阻塞硬件调用和部分扫描并非可被总时限瞬时抢占。

每次 Attempt 使用递增并按 uint8 回绕的新 `seq`，发送前清旧邮箱响应，再只接受当前请求对应的回复，避免把旧 ACK 当作成功。无论成功、失败还是异常，`finally` 都恢复站立/左手状态。

Attempt 1 失败后先调用后退/重定位，再进入最多三轮同目标重获；该调用返回 False 并不会阻止重获循环。重获只采用 `captured_s≥retry_visual_after_s`、同 ID、binding 有效且 confidence≥0.20 的证据（不再套普通缓存 TTL）；它可以来自后退定位帧，也可来自后续 required-target 扫描。该扫描以目标绑定为成功条件，即使新 Pose 不可接受也可返回 True；其他 Tag 安装 Pose 但未绑定目标时仍返回 False。若新证据显示目标花，则直接记 CHANGED；否则导航同一 TargetGoal、再次确认 Tag、final forward 成功后才进入 Attempt 2。三轮无法取得有效分类或重新接近则 GAVE_UP；期间总时限到返回 mission_timeout。成功后不再 retry，pending retreat 由主循环收尾。

## 7.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 最大物理 Attempt | 2 | NFC 实际请求上限 | 防止无限重复操作 |
| Attempt / scan timeout | 25 s / 25 s | 单次交互上限 | 受剩余总时间进一步截断 |
| response timeout | 1 s | 单次邮箱等待粒度 | 周期性检查总时限 |
| 左手稳定等待 | 0.5 s | 举手后等待 | 稳定 NFC 耦合位置 |
| retry retreat | 10 cm | Attempt 1 后退出 | 重新建立视觉与接近过程 |
| target reacquire | 最多 3 cycle | 同一目标重获预算 | 禁止无限寻找 |

## 7.4 对应动作 / 代码

| 动作 | 实际 action / ActionGroup | 模型效果 | 使用场景 |
|---|---|---:|---|
| 站立 | `stand` / `stand` | 0 | 举手前和 finally 恢复 |
| 举左手 | Hardware 左手动作 | 姿态动作 | NFC 耦合 |
| 重获后退 | `back_fast` | 请求约 -10 cm | 离开近距离位置 |
| 再接近 | `interaction_forward_final` | +20 cm | fresh 分类仍需换花 |

代码参考：`RobotInteractionClient.change_flower()`、`process_screen_interaction()`、`restore_nfc_physical_contact()`、`recalibrate_target_for_nfc_retry()`。

# 8. 当前目标结束与下一个目标选择

## 8.1 流程图

```text
当前目标业务结果
  ├─ALREADY_TARGET
  │    ↓
  │ 标记已处理；没有 final forward，因此无需交互后退
  │    ↓
  │ 清 TargetGoal→回 3
  ├─CHANGED
  │    ↓
  │ 若 final forward 后仍在近位→后退约 10 cm（只执行一次）
  │    ├─stand/后退失败→MISSION_BLOCKED，pending+blocked→主循环等待，不再重发动作
  │    └─动作完成→重新 AprilTag 定位
  │                   ├─失败→MISSION_BLOCKED，保留 pending retreat→重试定位
  │                   └─成功→清 TargetGoal→回 3
  └─GAVE_UP
       ↓
    Screen.status=FAILED，加入 nfc_gave_up_screen_ids，本次任务不再选择
       ↓
    若 final forward 后仍在近位，同样完成后退/重定位
       ↓
    清当前目标→回 3 选择其他 Screen

第 3 节还有可选目标？
  ├─是→继续比赛
  └─否
      ├─全部为 CHANGED/ALREADY_TARGET→MISSION_COMPLETE→等待总时限→安全停止
      └─仍有未完成目标→global recovery→释放普通临时失败并重选
          （NFC GAVE_UP 不释放；只剩它们时循环恢复，最终 570 s 到→安全停止）
```

## 8.2 流程说明

`ALREADY_TARGET` 与 `CHANGED` 都是 `Screen.done()`，不会再次参加选择；只有 `CHANGED` 计入 `completed_count()`（包括 NFC 失败后视觉证实已换和模拟成功）。导航返回 False 或可见性两轮耗尽会直接临时排除并清锁，不等待两次普通失败；`register_target_failure()` 处理的 final-forward/普通交互等失败才按 `max_target_attempts=2` 升级临时排除。无可选目标但仍有未完成项时，调用 global recovery 后无论其成功与否都释放普通临时失败、清相应未完成目标的 attempts，再选目标。分类服务 WAIT/DEGRADED 保留锁，不消耗该普通失败次数。

`GAVE_UP` 不同：NFC 两次物理 Attempt 失败，或 Attempt 1 后 3 轮同目标重获耗尽，会把 Screen 设为兼容状态 `FAILED` 并加入 `nfc_gave_up_screen_ids`。`FAILED` 本身不是永久 terminal，但该集合使它在本次任务内不再入选。机器人随后处理其他目标；若没有其他可选目标，当前实现并不立即宣布成功，而是继续恢复/等待，直到全局时限安全停止。

## 8.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 普通目标失败门槛 | 2 | 普通失败门槛；导航/可见性耗尽直接轮换 | 避免单一目标长期占用 |
| post-interaction retreat | 10 cm | 从 NFC 近位退出 | 为下一次定位/导航留空间 |
| retreat retry interval | 1 s | pending 等待周期；仅定位失败会重新定位 | 动作失败 blocked 后不重发 |
| `continue_after_target_count` | true | 不按成功数提前结束 | 尽量处理全部 Screen |
| 总时限 | 570 s | 最终终止条件 | 无候选/恢复也不会无限运行 |

## 8.4 对应动作 / 代码

| 结果 | Screen 状态 / 集合 | 后续动作 | 下一步 |
|---|---|---|---|
| ALREADY_TARGET | `ALREADY_TARGET` | 无近位后退 | 回 3 |
| CHANGED | `CHANGED` | 必要时 `back_fast` 约 10 cm并定位 | 回 3 |
| 普通失败达 2 次 / 导航失败 / 可见性耗尽 | 状态保持可重试 + temporary 集合 | 无候选时 global recovery 后释放 | 回 3 |
| GAVE_UP | `FAILED` + `nfc_gave_up_screen_ids` | 必要时后退/定位 | 跳过本屏，回 3 |
| TIMEOUT | `MISSION_TIMEOUT` | `hardware.stop()` | 安全停止 |

代码参考：`Screen.done()`、`register_temporary_target_failure()`、`give_up_nfc_change()`、`complete_post_interaction_retreat()`、`finish_mission_without_available_targets()`。

---

## 闭环检查结论

- `capture_failed`：记录失败后回当前定位调用者；当前代码还会影响 no-tag 计数，达到门槛可能进入 2.1。
- `no_tag`：按两项连续计数最大值、cooldown 和状态门触发 2.1；安全拒绝/耗尽设置标记并返回调用者，不递归 global recovery。
- `pose_unavailable_with_tags`、质量/PnP/physical/hard jump/suspect rejected：候选/扫描预算内继续尝试；整轮失败保留旧 Pose、增加连续定位失败，达到门槛同样可进入 2.1；Recovery 内继续有界轮次，NFC 内回同目标重获。
- A* `no path`：先重新定位重算；同签名 3 次后走室内 waypoint；失败则当前目标临时轮换。
- 动作/转向无进展：证据不足为 PROGRESS_UNVERIFIED，后续按定位失败门控；VERIFIED_NO_PROGRESS 累计≥2 后强制定位仍确认无进展才返回 RECOVERY_NO_PROGRESS 并轮换目标。
- Screen/FPGA 失败：服务不可用/结果错误保留 locked target 并等 1 s；可见性恢复两轮耗尽则临时排除、清锁重选；不会用其他 Tag 授权 NFC。
- NFC 失败：Attempt 1 后退并重获同目标；仅允许 Attempt 2；之后 CHANGED 或 GAVE_UP。
- 恢复/主循环最终在总时限检查处进入 `MISSION_TIMEOUT → hardware.stop()`；默认 570 s，CLI 可覆盖，阻塞动作/扫描不会被即时抢占。全部 done 后等待时限；测试 `--max-screens` 或关闭 continue-after 后的成功数门槛也可提前宣布 COMPLETE，但仍等待时限。
