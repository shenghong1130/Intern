# TonyPi 比赛决策树说明（按当前代码与配置核实）

本文面向实习报告，说明 TonyPi 在比赛中如何定位、选目标、规划、执行、识别和换花。事实来源为当前 `task_manager.py`、定位/地图/运动/视觉/交互模块、`config.py`、`config/competition_config.json` 与相关测试；规划器中的离散模型与机器人实际 ActionGroup 标定分别说明。

> 当前实现提示：`record_localization_failure()` 以“本轮是否看到 Tag”更新 `consecutive_no_tag_scans`。因此全程无法取帧的 `capture_failed` 目前也会累加该计数；这与变量名以及“只有完整扫描零 Tag 才算 genuine NO-TAG”的设计意图不完全一致。下文如实描述：`no_tag` 才是可确认的 genuine NO-TAG，但当前计数器也会受 `capture_failed` 影响。看到 Tag 但无 Pose 会清零该计数。

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
POSITION_NAVIGATION：Motion A* 保留 yaw state，只规划到 25 cm interaction XY
  ↓
执行同类动作的有限 batch
  ↓
按置信度/动作数/uncertainty 决定是否重新定位
  ↓
无论是否定位都重新规划，直到到达 25 cm interaction XY
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
             全部 Screen 已处理→MISSION_COMPLETE 等待总时限
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
完整 pan 扫描 [100,135,65,155,45]
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
  │       记录普通定位失败；当前实现也把 no-tag 计数 +1
  │                         ↓
  │       返回调用者的定位/导航循环；若 Recovery 门槛被满足→转 2.1
  └─获得帧
       ↓
    AprilTag detector→detected_tag_ids
       ├─完整扫描中始终 []→genuine no_tag→转 2.1
       └─至少看见一个 Tag
            ↓
     ID 1..36、面积≥350 px²、边缘≥35 px、世界坐标存在
            ↓
     solvePnP，检查 rvec/tvec 有限，生成候选 RobotPose
       ├─无可信 Pose→TAG_SEEN_POSE_UNAVAILABLE→转 2.2
       └─候选 Pose→物理地图与时间一致性检查→转 2.2
```

### 流程说明

检测到 Tag 不等于获得位置。Localizer 对每个候选做 ID、面积、边缘和地图坐标检查，再用 PnP 计算机器人位姿；可用候选中以 Tag 图像面积优先。候选随后还要通过场地/建筑物理合法性和与当前 Pose 的一致性门控。任一失败都会回到发起本次定位的明确调用者：导航中回第 5 节重定位/重规划；目标重获中回第 7 节；初始定位中回第 1 节搜索循环。

### 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 合法 Tag ID | 1..36 | 可用于比赛定位的标签 | 排除非地图标签 |
| 最小面积 | 350 px² | 过小 Tag 不参与 PnP | 远小目标角点误差大 |
| 图像边缘留白 | 35 px | 过于贴边的 Tag 拒绝 | 避免裁切角点 |
| detector upscale | 1.5 | 检测前放大比例 | 提高小标签检测能力 |
| HIGH 门槛 | ≥2 个有效 Tag，或最佳面积≥700 px² | 视觉置信度 | 控制后续 batch 大小 |

### 对应动作 / 代码

本节没有身体动作；云台通常按 `[100,135,65,155,45]` 扫描，目标确认使用 `[100,130,70]`。

代码参考：`localize_scan()`、`Localizer.estimate_from_frame()`、`evaluate_and_accept_visual_pose()`。

## 2.1 Genuine NO-TAG

```text
完整扫描有帧但 detected_tag_ids 始终为空→genuine no_tag
  ↓
consecutive_no_tag_scans +1
  ↓
计数≥2 且距上次 Recovery≥4 s？
  ├─否→返回原定位调用者：初始搜索 / 第 5 节导航重定位 / 第 7 节重获
  └─是→NO-TAG Recovery（最多 3 cycle）
          ↓
       当前可信 Pose 靠墙或靠边界？
       ├─是→选择净空更大的横向→横移 4 cm
       │      └─无安全横移→本次 Recovery 记为耗尽→global recovery
       └─否→后退约 5 cm→朝场地中心方向转约 45°
          ↓
       上述身体动作成功？
          ├─否→hardware_failure→返回当前调用者重定位/重规划；无法继续则目标临时轮换
          └─是
             ↓
       云台回中，只在 pan=100 重新扫描
          ├─accepted pose→清 no-tag 计数→保留 locked target→回原任务/第 5 节重规划
          ├─看到 Tag 但 Pose 不可用→停止继续盲动→转 2.2→回原任务恢复逻辑
          ├─capture_failed→停止本轮盲动→回原任务恢复逻辑
          └─仍 genuine no_tag→下一 cycle
                                  ├─未到 3→继续 Recovery
                                  └─3 cycle 耗尽→global recovery
                                           ├─恢复成功→保留任务上下文→第 5 节重规划
                                           ├─仍失败但有时间→主循环后续再次恢复/重试
                                           └─总时限到→安全停止
```

普通区域先后退再转向，是为了同时改变相机位置和朝向；靠墙时优先横移净空侧，避免 10 cm 旋转扫掠半径撞到建筑。Recovery 中一旦看到 Tag 但不能接受 Pose，就不再把它当作“什么都没看到”继续盲动。

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 触发计数 / cooldown | 2 / 4 s | Recovery 门槛 | 抑制单帧抖动和频繁恢复 |
| 普通区动作 | 后退 5 cm + 向内转 45° | 扩大可见视角 | 打破遮挡/背向状态 |
| 墙边动作 | 横移 4 cm/cycle | 增大侧向净空 | 降低扫墙风险 |
| 最大 cycle | 3 | 单次 Recovery 上限 | 耗尽后升级全局恢复 |

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
  │   返回当前调用者：初始定位→1；导航→5；NFC 重获→7
  └─是→候选 Pose
          ↓
       在 300×300 cm 场内且不在建筑实体内？
       ├─否→physical rejection→保留旧 Pose→返回当前调用者
       └─是→与旧 Pose 比较
               ├─距离>40 cm 或 yaw>60°→hard jump 立即拒绝
               │                            ↓
               │                  保留旧 Pose→回定位/第 5 节重规划
               ├─距离≤15 cm 且 yaw≤25°→accepted pose
               └─普通冲突→再取 1 帧确认
                            ├─确认帧与 suspect 在 10 cm/15°内且自身合法→接受确认 Pose
                            ├─确认帧回到旧 Pose 一侧→接受确认 Pose
                            └─不支持/无 Pose/仍 hard jump→suspect rejected
                                                          ↓
                                               保留旧 Pose→回当前定位/重规划循环

accepted pose
  ↓
RobotState.set_pose()；actions_since_localize=0；motion_uncertainty=0
  ↓
consecutive_no_tag_scans=0；更新 last_localize_success_s
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

代码参考：`assess_visual_localization()`、`confirm_suspect_visual_pose()`、`accept_visual_localization()`。

# 3. 目标 Screen 选择

## 3.1 流程图

```text
全部配置 Screen
  ↓
排除 CHANGED / ALREADY_TARGET、临时失败集合、nfc_gave_up 集合
  ↓
已有仍合法的 locked target？
  ├─是→保持同一 Screen 与原子 TargetGoal→转 4
  └─否→为每个候选计算建筑面外法向 25 cm、左偏 -1 cm 的 interaction target
          ↓
       计算机器人到 interaction target 的直线距离
          ↓
       保留“最近距离 +25 cm”窗口内候选
          ↓
       score = distance + min(25, behind_turn×0.20 + final_yaw_error×0.03)
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
以 forward / reverse / strafe / small turn / large turn 扩展 A*
  ↓
每条平移边检查机器人走廊、建筑/场界、非目标障碍代价和净空
每条转向边检查 10 cm rotation sweep
  ↓
累计动作代价、软障碍、净空不足、连续/反向转弯、动作反转、远离目标惩罚
  ↓
到达 XY≤4 cm？（最终 Screen yaw 不参与 goal test / heuristic）
  ├─是→重建可执行 PlannedNavigationAction 列表→转 5
  └─open set 耗尽或扩展到 45000→A* 无路径
       ↓
    同签名失败次数 <3？
       ├─是→重新 AprilTag 定位→回本节重规划
       └─否→选择室内安全 waypoint 做 Recovery
                ├─成功→清失败 watchdog→保留目标→回本节
                └─失败→navigation_blocked→该目标记为临时失败→回 3 选其他目标
总时限到或严重硬件故障→安全停止/交回任务主循环处理
```

## 4.2 流程说明

普通 XY A* 会假设机器人能沿任意方向连续移动，而 TonyPi 只能执行有限 ActionGroup，所以正式规划状态仍包含朝向。平移边按当前 yaw 解释 forward/reverse/左右横移，转向边按配置的物理角改变 yaw state；但终点条件和启发式只看 interaction XY。转向仍可为了让后续平移更自然、更安全而发生，不会从远处为了最终 Screen yaw 提前横移或绕路。

正式位置搜索只扩展对称的 ±90° quarter-turn macro，因此能被 15° state 精确表示，又不会探索大量细碎 yaw 组合。macro 由 `turn_left/right_fast` 的 7.5°/cycle 执行，Executor 按 batch 上限分批并重规划。action-level 精确 yaw 模式会把 15°请求细化成当前 1.5° lattice；动作预测 Pose 始终直接按物理 action model 重建，因此 -18° 不再被预测为 -30°或污染后续平移方向。

## 4.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 网格 / position yaw state | 5 cm / 15°，只扩展 ±90° | A* 状态离散 | 保留方向语义并限制搜索规模 |
| position heuristic 权重 | 3.0 | 加权 XY 距离 | 优先自然接近目标位置 |
| 规划步长 | forward 28、fine 7、strafe 12、reverse 5 cm | 扩展模型 | 对应可批量执行动作 |
| 位置阶段到达容差 | 4 cm | Motion A* 目标条件 | 先可靠到达 interaction XY |
| 走廊半宽 / 转动扫掠 | 8 cm / 10 cm | 碰撞检查 | 不只检查质点 |
| segment 最大代价 / 目标净空 | 55 / 25 cm | 软障碍约束 | 远离墙体与非目标建筑 |
| 最大扩展 / 同签名失败 | 45000 / 3 | 规划预算与升级门槛 | 避免无限重算同一失败 |
| 主要代价 | turn 20+0.8/°；large +12；strafe×1.05；reverse×1.08 | 动作偏好 | 优先少转、少反复且可执行的路线 |

## 4.4 对应动作 / 代码

| Planner 边 | 实际 action key | 规划模型 | 备注 |
|---|---|---:|---|
| forward / fine | `forward_fast` | 28 / 7 cm | 最终换算为 3.5 cm/cycle |
| reverse | `back_fast` | 5 cm | 只在目标位于后方且≤5 cm 时扩展 |
| strafe left/right | `strafe_*_fast` | ±12 cm | yaw 保持不变 |
| position quarter turn | `turn_left/right_fast` | 规划 ±90°；物理 ±7.5°/cycle | Executor 限批并重规划 |
| final yaw large turn | `turn_left/right_large` | 物理 +15° / -18° | 只在近点闭环校正中按真实角执行 |

代码参考：`MapModel.plan_motion_actions()`、`action_planner_transition()`、`navigate_motion_plan_to_target()`。

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
          └─动作完成→dead reckoning 更新 Pose、动作数、uncertainty
                         ↓
                  是转向？
                  ├─是→Turn Progress Watchdog（下图）
                  └─否→是否强制重新定位？
                         ├─是→AprilTag 定位
                         │      ├─accepted→回 4
                         │      └─失败→按 2/2.1 Recovery→再回 4 或临时放弃目标
                         └─否→保留 dead reckoning→仍回 4 重规划
```

```text
执行 TURN 成功
  ↓
立即 post-turn visual localization
  ├─无可信 before/after Pose→PROGRESS_UNVERIFIED
  │                            ↓
  │             不增加“没转动”计数→回 4 重规划/后续定位
  └─有可信视觉前后 Pose
       ↓
    实际 yaw 有正确、足够的变化？
       ├─是→VERIFIED_PROGRESS→安装视觉 Pose→清计数→回 4
       └─否→VERIFIED_NO_PROGRESS，计数 +1
                    ├─<2→回 4
                    └─≥2→强制再定位确认
                           ├─确认有进展→清计数→回 4
                           ├─仍无可靠 Pose→PROGRESS_UNVERIFIED→回 4/定位
                           └─确认仍无进展→RECOVERY_NO_PROGRESS
                                                 ↓
                              当前导航失败→目标临时失败→回 3；有时间可再选
```

## 5.2 流程说明

A* 选择 forward，是因为按当前朝向前进的边在安全走廊内且综合代价更低；选择 strafe，是因为可在不改变最终朝向的情况下消除足够大的横向分量；reverse 仅用于目标在后方、横向误差≤8 cm、后向角≤30°、目标距离≤5 cm且动作确实缩短距离。正式 Motion A* 会同时扩展 small/large turn，再由总代价选择；`MotionController.turn_toward()` 等程序化转向才以 35°作为 large turn 门槛。无论从哪条路径选中 large turn，执行后都强制视觉复核。

不能一次跑完整路径，因为舵机动作和标定角存在误差，dead reckoning 会累计，且到障碍/目标附近容错更小。HIGH 可执行较长 batch；MEDIUM 收缩；LOW 只执行 1 cycle。距离目标 `<15 cm` 时所有 batch 最多 1 cycle。每个 batch 后总会重新 A*；只有达到动作预算、uncertainty 门槛、LOW、large turn、障碍紧或显式强制条件时才立即 AprilTag 定位。

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
| near target | `<15 cm`，最多 1 cycle | 单步逼近 | 防止越过目标 |
| 程序化 large-turn 门槛 | yaw 差≥35° | `turn_toward()` 改用 large action；A* 本身按代价选择 | 减少恢复/校正中的小步次数 |
| watchdog 无进展 | 2 次可靠确认 | 导航中止门槛 | 不把取帧失败误判成没转 |
| `collision_recovery_enabled` | false | 通用碰撞停滞恢复当前关闭 | 现行主流程依靠规划安全门和 near-wall recovery |

代码参考：`execute_motion_astar_action()`、`select_adaptive_action_batch()`、`adaptive_relocalization_decision()`、`monitor_turn_result()`。

# 6. 到达目标后的 Screen / Tag / 花卉识别与换花决策

## 6.1 流程图

```text
POSITION_NAVIGATION 接近 interaction target
  ↓
XY≤4 cm？
  ├─否→回 4/5 重新规划与执行（不优化最终 yaw）
  └─是→FINAL_YAW_ALIGNMENT：accepted visual Pose 年龄≤3 s？
          ├─否→重新 AprilTag 定位
          │      ├─accepted 且仍在 4 cm 内→检查 yaw
          │      └─失败/位置改变→回 4/5
          └─是→desired yaw 误差≤10°？
                 ├─否→安全原地转向→视觉定位→重新检查 XY/yaw
                 └─是→ARRIVED
                 ↓
          实时扫描 [100,130,70]，必须看见 locked target Tag
                 ├─未见/绑定无效→最多 2 cycle 可见性恢复
                 │                 ├─恢复后确认→继续
                 │                 └─耗尽→保留目标并 MISSION_BLOCKED
                 │                         ↓
                 │                  主循环等待 1 s 后重试本节
                 └─Tag ID=locked Screen ID 且几何 crop 有效
                                      ↓
                     使用≤15 s 的同屏绑定缓存，或最多 3 帧 fresh 分类
                                      ├─FPGA/服务不可用→保留目标→等待 1 s→重试本节
                                      ├─结果无效/置信度<0.20→保留目标→重试/可见性恢复
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

15 s cache 只复用最近的“已绑定屏幕分类结果”；它不能单独授权 NFC。每次使用缓存仍必须在当前时刻重新看见 locked Tag，并把当前 Tag 时间与同屏证据配对。分类阈值为 0.20。分类服务失败不会把 Screen 当作花卉错误，也不会换目标，而是保持锁定目标等待服务恢复。

机器人先停在墙外约 25 cm，是为了以精确姿态确认屏幕；确认确实需要换花后，才用一个专用序列前进模型 20 cm，接近 NFC 工作距离。

## 6.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| ARRIVED | 4 cm / 10° / Pose≤3 s | 到达三重门 | 防止只靠陈旧 XY 判到达 |
| 目标实时 pan | 100, 130, 70 | locked Tag 搜索 | 小范围找回目标 |
| cache TTL / 分类间隔 | 15 s / 1 s | 同屏分类缓存 | 降低重复 FPGA 请求 |
| fresh 分类重试 | 最多 3 帧，间隔 0.5 s | 无可用缓存时 | 有限等待视觉结果 |
| 可见性恢复 | 最多 2 cycle | 重新定位/小修正 | 保持同一目标 |
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
fresh visual authorization 检查
  ├─失败→不举手、不发 NFC→登记一次普通目标失败
  │                              ↓
  │              完成近位后退/重定位→清当前锁→回 3（未达 2 次时可再选本屏）
  └─通过→stand→举左手→稳定等待 0.5 s→再次 authorization 检查
          ├─失败→作为 Attempt 1 失败处理→finally 恢复 stand/左手→进入下方重获
          └─通过→生成新的 uint8 seq，清旧响应→NFC Attempt 1
                    ├─25 s 内收到匹配 seq 且 ok→CHANGED→恢复左手→转 8
                    └─timeout/无效响应/异常→恢复左手
                              ↓
                    后退约 10 cm并重新 AprilTag 定位
                              ├─后退/定位未完成→保持 pending retreat，主循环重试
                              └─完成→最多 3 cycle 重获同一 locked target
                                      ↓
                            其他 Tag→只能定位，不能替代目标授权
                                      ↓
                            同 ID Tag+Screen crop+fresh FPGA 分类？
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

任意等待中总时限到→MISSION_TIMEOUT→恢复左手/停止→安全停止；绝无 Attempt 3
```

## 7.2 流程说明

通信链路是 `TonyPi ─NFC→ 对应 Worker → 换花执行`。每次物理尝试前后各做一次视觉安全门，举手后证据失效就不发送。若第一次门控在举手前已经失败，调用会直接返回，任务管理器登记普通目标失败，完成已安排的近位后退/重定位后回第 3 节；该屏未达两次普通失败门槛时仍可再次选择。若举手后的第二次门控失败，则按 Attempt 失败进入同目标重获流程。

每次 Attempt 使用递增并按 uint8 回绕的新 `seq`，发送前清旧邮箱响应，再只接受当前请求对应的回复，避免把旧 ACK 当作成功。无论成功、失败还是异常，`finally` 都恢复站立/左手状态。

Attempt 1 失败后不能立即原地重发：机器人先退回、重定位并找回同一 locked target，再取得 fresh FPGA 分类。若花已是目标，按视觉证据记为 CHANGED；仍需更换时才重新接近并允许 Attempt 2。

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
  │    ├─动作失败→MISSION_BLOCKED，保留 pending retreat→主循环稍后重试
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
      └─仍有普通临时失败→global recovery→释放并重选
          （NFC GAVE_UP 不释放；只剩它们时循环恢复，最终 570 s 到→安全停止）
```

## 8.2 流程说明

`ALREADY_TARGET` 与 `CHANGED` 都是 `Screen.done()`，不会再次参加选择；只有 `CHANGED` 计入“实际换花成功数”。普通导航、可见性或分类流程中的失败由 `max_target_attempts=2` 控制，达到门槛后只进入临时失败集合；全局恢复后尝试次数清零，可以再次选择。

`GAVE_UP` 不同：NFC 两次物理 Attempt 失败，或 Attempt 1 后 3 轮同目标重获耗尽，会把 Screen 设为兼容状态 `FAILED` 并加入 `nfc_gave_up_screen_ids`。`FAILED` 本身不是永久 terminal，但该集合使它在本次任务内不再入选。机器人随后处理其他目标；若没有其他可选目标，当前实现并不立即宣布成功，而是继续恢复/等待，直到全局时限安全停止。

## 8.3 关键参数

| 参数 | 当前值 | 含义 | 为什么需要 |
|---|---:|---|---|
| 普通目标失败门槛 | 2 | 达到后临时轮换 | 避免单一目标长期占用 |
| post-interaction retreat | 10 cm | 从 NFC 近位退出 | 为下一次定位/导航留空间 |
| retreat retry interval | 1 s | blocked 后重试周期 | 避免重复快速发动作 |
| `continue_after_target_count` | true | 不按成功数提前结束 | 尽量处理全部 Screen |
| 总时限 | 570 s | 最终终止条件 | 无候选/恢复也不会无限运行 |

## 8.4 对应动作 / 代码

| 结果 | Screen 状态 / 集合 | 后续动作 | 下一步 |
|---|---|---|---|
| ALREADY_TARGET | `ALREADY_TARGET` | 无近位后退 | 回 3 |
| CHANGED | `CHANGED` | 必要时 `back_fast` 约 10 cm并定位 | 回 3 |
| 普通失败达 2 次 | 状态保持可重试 + temporary 集合 | global recovery 后释放 | 回 3 |
| GAVE_UP | `FAILED` + `nfc_gave_up_screen_ids` | 必要时后退/定位 | 跳过本屏，回 3 |
| TIMEOUT | `MISSION_TIMEOUT` | `hardware.stop()` | 安全停止 |

代码参考：`Screen.done()`、`register_temporary_target_failure()`、`give_up_nfc_change()`、`complete_post_interaction_retreat()`、`finish_mission_without_available_targets()`。

---

## 闭环检查结论

- `capture_failed`：记录失败后回当前定位调用者；当前代码还会影响 no-tag 计数，达到门槛可能进入 2.1。
- `no_tag`：未达门槛回扫描；达门槛进入 2.1；耗尽升级 global recovery。
- `pose_unavailable_with_tags`、质量/PnP/physical/hard jump/suspect rejected：保留旧 Pose，回初始搜索、导航重规划或 NFC 同目标重获，不进入下一轮盲动。
- A* `no path`：先重新定位重算；同签名 3 次后走室内 waypoint；失败则当前目标临时轮换。
- 动作/转向无进展：可靠证据不足时继续定位/重规划；可靠无进展 2 次后中止本次导航并轮换目标。
- Screen/FPGA 失败：保持 locked target，按有限帧、可见性恢复和 1 s 服务重试闭环；不会用其他 Tag 授权 NFC。
- NFC 失败：Attempt 1 后退并重获同目标；仅允许 Attempt 2；之后 CHANGED 或 GAVE_UP。
- 任意恢复最终都受 570 s 总时限约束，并进入 `MISSION_TIMEOUT → hardware.stop()`。
