# TonyPi 机器人决策流程

本文描述机器人“正在做什么”和“为什么这样做”。代码名只作为小字参考。世界距离单位为 cm，yaw 单位为度；`0°` 朝 `+X`，逆时针为正。

## 0. 整体比赛主流程

```text
启动、站稳、云台回中
        │
        ▼
拍摄 AprilTag，确认机器人位置
        │
        ▼
选择最近的未完成 Screen，并锁定完整目标
        │
        ▼
规划到距建筑面 25 cm 的位置与目标朝向
        │
        ▼
按 Motion A* 动作执行 → 重新定位 → 重规划
        │
        ▼
确认已到目标位置和朝向
        │
        ▼
现场确认当前 Tag 与 Screen → FPGA 识别花朵
        │
   ┌────┴────┐
   │已是目标  │不是目标
   ▼          ▼
ALREADY_   前进约 20 cm → NFC 换花
TARGET        │
   │       CHANGED / 有界重试后 GAVE_UP
   └────┬─────┘
        ▼
必要时后退 10 cm 并重新定位
        │
        ▼
完成当前目标 → 选择下一个目标
```

全局超时在所有循环之上；超时立即安全停止。`ALREADY_TARGET` 和 `CHANGED` 都是当前 Screen 的终态。

## 1. 初始化和首次定位

```text
初始化配置、地图、相机、动作、FPGA/NFC
                    │
                    ▼
              云台回到 100°
                    │
                    ▼
             按完整 pan 顺序拍摄
                    │
          ┌─────────┴─────────┐
          │获得可信视觉 Pose   │没有获得
          ▼                   ▼
       进入选目标       执行一个配置的身体搜索动作
                              │
                              └──→ 再做完整 pan（总预算 14）
```

代码参考：`initial_localize()`、`run_localization_search_sequence()`。首次定位的身体搜索仍使用既有配置；运行中的 genuine NO-TAG 使用第 3 节的专用恢复，不再复用它。

## 2. AprilTag 定位

```text
拍摄一帧 capture
      │
      ├─ 失败 ───────────────→ capture_failed
      ▼
检测 AprilTag
      │
      ├─ detected_tag_ids=[] ─→ no_tag
      │                         只有这里增加 consecutive_no_tag_scans
      ▼
至少看到一个 Tag
      │
      ▼
ID / 面积 / 边缘 / quality gate
      │
      ▼
solvePnP + 多 Tag 一致性
      │
      ├─ 无 Pose ─────────────→ pose_unavailable_with_tags
      ▼
候选 RobotPose
      │
      ├─ 场外或建筑实体内 ───→ physical rejection
      ├─ 跳变 >40 cm 或 >60° ─→ hard jump，立即拒绝
      ├─ 普通冲突 >15 cm/25° ─→ suspect confirmation
      │                           ├─ 二次支持 → accepted pose
      │                           └─ 不支持   → suspect pose rejected
      ▼
accepted pose
      │
      ▼
更新 RobotState；清零动作累计、运动不确定度和 no-tag counter
```

统一视觉事实是“检测器返回的 `detected_tag_ids` 是否为空”。`too_small`、edge margin、ID filter、quality gate、solvePnP failure、pose conflict、physical rejection、hard jump 都属于“看见 Tag 但 Pose 不可用”，会增加一般定位失败计数，但会把 genuine no-tag 连续计数清零。

代码参考：`localize_scan()`、`record_localization_failure()`、`evaluate_and_accept_visual_pose()`。

## 3. genuine NO-TAG Recovery

触发条件只有一个：完整定位扫描的 `detected_tag_ids == []`，并且连续达到 `no_tag_recovery_failures=2`。普通位置和墙边使用同一入口。

```text
                 当前扫描完全没有 Tag
                           │
              consecutive_no_tag_scans += 1
                           │
                     连续达到 2？
                    /            \
                  否              是
                  │               │
                继续              ▼
                         进入 NO-TAG Recovery
                           （最多 3 cycle）
                                 │
                      最近 Pose 靠墙/边界？
                       /                 \
                     是                   否
                     │                    │
          选择增加 clearance 的方向      后退约 5 cm
                     │                    │
             左或右安全横移约 4 cm       朝场地中心方向
                     │              身体旋转约 45°
                     └─────────┬──────────┘
                               ▼
                         云台回到 100°
                               │
                         等待相机稳定并
                         只拍中央视角
                               │
                    ┌──────────┼──────────┐
                    │可信 Pose │看到 Tag  │仍无任何 Tag
                    ▼          │但无 Pose ▼
                 接受定位      ▼         cycle += 1
                 清零计数   PROGRESS/      │
                 退出恢复   LOCALIZATION   ├─ <3：重复
                            UNAVAILABLE     └─ =3：高级恢复
                            停止盲动
```

- 普通位置：`back_fast` 按其 `-2.5 cm/cycle` 模型计算 2 cycle；随后读取 `turn_left_fast.yaw_deg` / `turn_right_fast.yaw_deg`，计算约 45° 所需 cycle。该专用转向不受 LOW confidence 自适应上限限制。
- 墙边/边界：调用 clearance 选择，只执行一次 `strafe_left_fast` 或 `strafe_right_fast`，不优先大幅原地转身。
- 每次动作后只先看中央视角，不做无限左右 pan。
- 第 3 次仍 genuine no-tag 时记录 `no_tag_recovery_exhausted`，保留当前目标并进入 `perform_global_recovery()`。
- 中途一旦看到任何 Tag，即使 Pose 不可信，也立即停止后续盲退/盲转。

代码参考：`recover_from_no_tag_if_needed()`、`no_tag_recovery_turn()`、`choose_near_wall_lateral_direction()`。

## 4. 目标 Screen 选择

机器人先保留仍合法的已锁目标；否则排除已完成、NFC 已放弃和临时失败目标。距离统一计算到 25 cm `interaction_target_xy`。最近距离窗口内再加入“目标在身后”和最终 yaw 的有界惩罚，最后用 Screen ID 稳定破平局。

```text
候选 Screen → 计算到 25 cm 目标距离 → 最近距离窗口
                                      │
                                      ▼
                         朝向惩罚（有上限）+ ID 破平局
                                      │
                                      ▼
                         原子锁定 TargetGoal
```

## 5. Motion A* 规划

```text
当前 Pose + 25 cm 目标 XY + desired yaw
                    │
                    ▼
       状态量化为 (x_grid,y_grid,yaw_bin)
                    │
                    ▼
展开前进 / 后退 / 横移 / 左右转动作
                    │
          corridor 与 rotation sweep 安全？
             /                         \
           否                           是
           │                            │
        丢弃边                    累计动作与障碍代价
                                          │
                                XY 与 yaw 都满足？
                                   /          \
                                 否            是
                                 │             ▼
                              继续搜索     NavigationPlan
```

动作的物理模型只来自 `motion.actions.*`。A* 的 yaw 是离散状态：15° bin 下，配置为 `-18°` 的 `turn_right_large` 必须跨 2 个 bin，所以内部 transition 是 `-30°`。日志明确区分：

- `configured_yaw_deg=-18°`：硬件、dead reckoning、turn watchdog 使用的物理期望；
- `planner_yaw_delta_deg=-30°`：仅 A* 离散状态转移。

## 6. 动作执行、批次、定位和重规划

```text
读取 NavigationPlan.actions
          │
          ▼
按置信度、距目标距离和安全通道决定 batch
          │
          ▼
执行 FORWARD / REVERSE / STRAFE / TURN
          │
          ▼
更新 dead reckoning 与 uncertainty
          │
          ▼
需要 post-action localization？
       /                    \
     是                      否
     │                       │
视觉 Pose 成功则更新          保留里程计 Pose
     └──────────┬────────────┘
                ▼
              REPLAN
```

距目标不超过 15 cm 时每批最多 1 cycle。普通导航 REVERSE 只允许在最终目标 5 cm 内、目标确在后方、能够缩短距离且后方通道安全时展开；Recovery 和交互 retreat 不受该规则影响。

## 7. Turn Progress Watchdog

```text
转向动作硬件成功
       │
       ▼
中央视觉获得可信 post-turn Pose？
       /                         \
     是                           否
     │                            │
比较 before/after yaw       看到 Tag/Screen？
     │                       /            \
     │                     是              否
     │                     │               │
     │             PROGRESS_UNVERIFIED  定位失败
     │              不增加 verified      同样不宣称没动
     │              no-progress count
     ▼
实际 yaw 变化足够？
   /             \
 是               否
 │                │
VERIFIED_      VERIFIED_NO_PROGRESS
PROGRESS             │
清零 counter          ▼
              连续可靠证据达到 2？
                 /            \
               否              是
               │               │
             重规划       再进行一次可信定位
                              │
                    ┌─────────┼─────────┐
                    │确认已转 │无法验证 │再次确认没转
                    ▼         ▼         ▼
                 清零继续  不报没动   RECOVERY_NO_PROGRESS
```

- `VERIFIED_PROGRESS`：可靠视觉 Pose 证明 yaw 有合理变化；方向相反也说明硬件发生了运动，安装真实 Pose 后重新规划，不计作“没动”。
- `PROGRESS_UNVERIFIED`：没有可靠 post-action Pose；包括 Tag 太小、quality gate、pose conflict、hard jump、物理拒绝或暂时看不到 Tag。绝不增加 verified counter。
- `VERIFIED_NO_PROGRESS`：可靠 before/after Pose 明确显示 yaw 变化不足。
- `RECOVERY_NO_PROGRESS`：连续达到阈值，且强制复核仍以可靠 Pose 证明没有转动。强制复核失败只会回到 `PROGRESS_UNVERIFIED`。

代码参考：`evaluate_turn_progress()`、`scan_after_turn()`、`monitor_turn_result()`。

## 8. 到达目标判断

必须同时满足：距 locked `interaction_target_xy` 不超过 4 cm、与 `desired_yaw_deg` 的误差不超过 10°，并具有足够新的可信视觉 Pose。只满足 XY 不算到达。

## 9. 当前目标 Tag + Screen 最终确认

到达后必须实时看到当前目标 Tag；Tag ID、Screen ID、locked goal ID 必须一致且几何绑定有效。途中 15 秒缓存可以提供同 ID 分类，但不能单独授权 NFC。

## 10. FPGA 花卉识别

分类器失败是可恢复的服务不可用，不代表目标消失。有效结果必须来自当前目标 Tag↔Screen 的绑定 crop，置信度不低于 0.2。

## 11. flower == target

进入 `ALREADY_TARGET`。不执行 20 cm final forward，不举手，不发送 NFC。

## 12. flower != target

进入 `NEEDS_CHANGE`，保存 fresh visual authorization，随后只执行一次 final forward。

## 13. final forward ≈ 20 cm

实体序列为 `go_forward_one_step × 4`，按当前约 5 cm 标定得到约 20 cm；模型 `interaction_forward_final.forward_cm` 和业务参数 `target_final_forward_cm` 同为 20.0。25 cm interaction target 不变。

## 14–21. NFC Attempt 1、重获与 Attempt 2

```text
20 cm final forward
       │
       ▼
授权复核 → stand → 举左手 → 再复核 → 新 seq → Attempt 1
       │
   ┌───┴───┐
   │成功    │失败
   ▼        ▼
CHANGED   后退 10 cm（一次）
            │
          重新定位
            │
      最多 3 cycle 重获“当前目标”
            │
       fresh FPGA
        /       \
 已是目标         仍非目标
    │              │
 CHANGED      重新靠近并 Attempt 2
                   │
             成功 CHANGED / 失败 GAVE_UP
```

其他 Tag 可以帮助定位，但不能结束当前目标重获。Attempt 2 只在 Attempt 1 失败、当前目标重获且 fresh FPGA 明确仍非目标时允许；不存在 Attempt 3。

## 22–24. 终态与交互后撤

- `ALREADY_TARGET`：无需物理换花，当前目标完成。
- `CHANGED`：换花流程终态，不再 retry/reapproach。
- 若进入过近距离交互位，按既有规则后退 10 cm 一次并重新定位；本次没有改变该行为。

## 25. 靠墙恢复

| 项目 | 说明 |
|---|---|
| 触发 | 当前 Pose 靠墙、通道受阻或局部恢复明确进入 near-wall 路径 |
| 实际动作 | 安全后退 → 选择提高清障距离的横移 → 最后才小转向；每次动作后定位 |
| 成功 | 新可信 Pose 已离墙，或 clearance 明显改善可用新 Pose 重规划 |
| 失败 | 动作被安全规划拒绝，或可靠 Pose 连续证明实体动作无进展 |
| 最大次数 | backoff 3、lateral 2、总动作 12（当前配置） |
| 之后去哪 | bounded escape / 返回重规划 / 上升为导航失败 |
| 当前目标 | 保留，不因 near-wall 恢复本身放弃 |

## 26. 全局 / 高级恢复

| 项目 | 说明 |
|---|---|
| 触发 | NO-TAG 3 cycle 耗尽，或所有未完成目标均临时失败 |
| 实际动作 | 重新完整定位；有 Pose 时 near-wall 或室内安全 waypoint；无 Pose 时有界搜索 |
| 成功 | 取得可信 Pose 或完成安全脱困 |
| 失败 | 本轮有界动作仍不能恢复；任务只要未超时仍可继续调度 |
| 最大次数 | 每次调用一轮；受全局 timeout 和各子恢复上限约束 |
| 之后去哪 | 释放临时失败列表，重新选目标 |
| 当前目标 | NO-TAG 升级时保留；全目标轮换时按调度规则释放 |

## 27–28. 临时目标失败、轮换和重试

重复同类规划失败达到 3 次时，不等待 80 步上限，当前目标进入临时失败并选择其他目标。当全部未完成目标都临时失败时执行全局恢复，然后释放临时失败列表重新选择。普通 `FAILED` 不是永久黑名单。

## 29–30. 比赛完成、超时和安全停止

所有目标处理完后进入 `MISSION_COMPLETE`。全局 deadline 耗尽进入 `MISSION_TIMEOUT`，停止继续规划、恢复和 NFC；Ctrl+C / emergency stop 仍是人工终止路径。

## 状态说明表

| 状态 | 中文含义与触发 | 代表硬件真的没动？ | 放弃当前目标？ | 下一步 |
|---|---|---:|---:|---|
| `no_tag` | 成功拍到帧，但 `detected_tag_ids=[]` | 否 | 否 | 连续 2 次才进入 NO-TAG Recovery |
| `pose_unavailable_with_tags` | 已看到 Tag，但质量、PnP 或 gate 无法产出可信 Pose | 否 | 否 | 保持 no-tag counter 为 0，继续定位/重规划 |
| `TAG_SEEN_POSE_UNAVAILABLE` | 上一项的统一可观测状态/事件 | 否 | 否 | 停止 NO-TAG 盲动，等待可信定位 |
| `LOCALIZATION_UNCERTAIN` | 当前视觉证据不足或 Pose 冲突待确认 | 否 | 否 | 降低动作批次并再次定位 |
| `PROGRESS_UNVERIFIED` | 转向后没有可信视觉 Pose，无法判断动作效果 | 否 | 否 | 不增加 verified counter；重新定位/重规划 |
| `VERIFIED_NO_PROGRESS` | 可靠视觉 before/after 明确显示 yaw 变化不足 | 是，本次证据如此 | 否 | verified counter +1；达到阈值复核 |
| `RECOVERY_NO_PROGRESS` | 连续可靠无进展且强制复核仍确认没转 | 是 | 临时失败路径可能轮换 | 停止当前导航动作并进入恢复/调度 |
| `near_wall_recovery_exhausted` | 靠墙恢复的有界动作耗尽或可靠证据持续无改善 | 仅物理无进展子路径是 | 默认否 | bounded/global recovery 或新 Pose 重规划 |
| `navigation_failed` | 规划、动作、安全或恢复无法完成本轮导航 | 不一定 | 临时 | 记录原因，轮换目标 |
| `target temporarily failed` | 当前目标的可恢复失败达到本轮上限 | 不一定 | 暂时 | 选择其他目标；全体失败后全局恢复并释放 |

## Recovery 参数分组

- NO-TAG：`no_tag_recovery_enabled/failures/cooldown_s/back_cm/turn_deg/strafe_cm/max_cycles`。
- Near-wall：`near_wall_backoff_*`、`near_wall_lateral_*`、`near_wall_recovery_*`、`forced_escape_*`。
- Turn watchdog：`turn_progress_verified_no_progress_threshold`，物理期望直接读取 `motion.actions.<action>.yaw_deg`。
- 自适应定位：`adaptive_relocalization_*`、`relocalize_action_budget_*`、`relocalize_uncertainty_limit_*`。
- 交互：`target_distance_cm=25`、`target_final_forward_cm=20`、retreat 与 NFC retry 参数。
