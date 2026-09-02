# TonyPi 测试使用说明

本目录包含 11 个离线 `unittest` 模块和 2 个需要人工启动的实机脚本。命令应从包含 `robot_tonypi` 包的 `robot_code` 目录执行。

机器人：

```bash
cd /home/pi
```

本仓库 Windows 工作区：

```powershell
cd D:\我\大学\Intern\robot_code
```

## 1. 运行全部离线测试

```bash
python3 -m unittest discover -s robot_tonypi/tests -p 'test_*.py' -v
```

`test_capture_fpga_change.py` 和 `test_capture_15_frames.py` 虽然匹配文件名，但没有 `unittest.TestCase`，discover 只导入它们，不会自动驱动硬件。

## 2. `test_calibrate_motion.py`（7 项）

检查动作标定工具的符号、times 归一化、median、large-turn sequence、推荐字段和写配置备份。使用临时目录，不修改正式配置。

```bash
python3 -m unittest robot_tonypi.tests.test_calibrate_motion -v
```

## 3. `test_interaction_flow.py`（15 项）

检查：

- 四向建筑面和 task target 几何；
- `stand → lift_left_hand → send_request → finally stand`；
- 举手前后两次授权门；
- 每次 NFC Attempt 使用新 seq；
- 成功不重试、旧 seq 响应不能匹配新请求；
- 单次物理 Attempt 的 15 秒 hard deadline 和 `retries=0`；
- Worker `ok=False`、异常和 skip/dry-run 语义；
- 途中绑定可分类，但不能触发交互。

```bash
python3 -m unittest robot_tonypi.tests.test_interaction_flow -v
```

## 4. `test_mission_scheduler.py`（38 项）

检查：

- Tag/Screen/Worker 同 ID、地图参考坐标和 TargetGoal；
- 25 cm interaction target 的最近距离窗口 + 朝向惩罚选择、同分 ID 破平局、完成后按新 Pose 重排；
- classifier/authorization 必须锁定且已到达；
- 已是目标花不执行物理交互；
- turn progress、scan-after-turn 和 stale pose；
- near-wall 的 backoff/lateral/small-turn 顺序；
- planner veto 与真实 no-progress 分离；
- forced escape 可从高 cost 起点选择更安全 endpoint；
- 导航失败临时轮换、全部临时失败后的 global recovery/release；
- 全局 timeout 是自动终止状态。

```bash
python3 -m unittest robot_tonypi.tests.test_mission_scheduler -v
```

## 5. `test_mission_refactor.py`（15 项）

集中检查本次主链路重构：严格场界/真实建筑 Pose gate、soft inflation 合法 Pose、moderate suspect 与 hard jump、失败分类计数、25 cm 两阶段目标评分、locked/exclusion 规则、XY+yaw action-space goal、早转/末转两类规划、turn cost 对真实序列的影响、5 cm normal reverse，以及 Planner action key 被 Executor 原样执行。

```bash
python3 -m unittest robot_tonypi.tests.test_mission_refactor -v
```

## 6. `test_navigation_adaptive.py`（72 项）

当前覆盖最广的导航单元测试：

- 实际/请求动作周期、partial failure 和 dead reckoning；
- 不同动作的不确定度、自适应批次和 phase-specific relocalization budget；
- 大转向只触发一次定位，新的大转向可再次触发；
- 普通 pan 在首个 Pose 成功时停止；required-target 模式忽略错误 Tag 并继续；
- no-tag 和 `pose_unavailable_with_tags` 计数、full pan、startup-equivalent body recovery；
- 失败定位保留运动累计，接受视觉 Pose 才清零；
- 多 Tag 中前一个失败不能阻止后一个成功；
- 5 cm 内正后方 reverse、平移与 corridor safety；
- 执行层重复 veto 的 decision-stall 防线。

```bash
python3 -m unittest robot_tonypi.tests.test_navigation_adaptive -v
```

## 7. `test_navigation_path_fallback.py`（19 项）

检查普通导航 `25 cm` clearance、Screen 4/17/18/35 复现场景、target-owned soft cost 例外、无关障碍、blocked anchor、兼容二维 fallback、start/map/goal failure signature、interior recovery 和 ARRIVED 前新鲜视觉 Pose。

```bash
python3 -m unittest robot_tonypi.tests.test_navigation_path_fallback -v
```

## 8. `test_recovery_target_consistency.py`（6 项）

检查所有配置 Screen 的 TargetGoal 单一正式目标原子一致性、Screen26 的 goal/anchor 区别、stale goal 拒绝、边缘 interior waypoint，以及保 yaw 的 strafe/reverse recovery。

```bash
python3 -m unittest robot_tonypi.tests.test_recovery_target_consistency -v
```

## 9. `test_target_direct_approach.py`（10 项）

检查 target-owned corridor 只忽略当前目标建筑软 inflation、硬障碍/其他建筑仍生效、短末步、forward 优先、5 cm 内短后方 reverse 和转向成本。该模块保留兼容测试，正式 mission 使用 Motion-Aware A*。

```bash
python3 -m unittest robot_tonypi.tests.test_target_direct_approach -v
```

## 10. `test_target_standoff_flow.py`（46 项）

检查完整到点交互状态机：

- 同一 `face_center + outward normal` 生成 `25 cm / -1 cm` interaction target，兼容 staging 字段与其相同；
- 正式导航一次规划到 25 cm + desired yaw，不存在中途 staging stop 或 staging relocalize；
- `20→25 cm` 只沿 cardinal normal 移动 task-target XY，`0→+5°` 只改变目标 yaw；
- 所有 interaction 几何/final-forward 参数都不改变 building bounds、grid、inflation 或 cost；
- 当前 Tag 与同 ID Screen 绑定后才授权；
- classifier offline 保留 live Tag 和 mission；
- pan 找到目标后先消费帧再回中；
- 目标确认和可见性恢复有界；
- `ALREADY_TARGET` 跳过 final forward；
- NEEDS_CHANGE 只执行一次 `interaction_forward_final`；mock AGC 验证 ×1 为大步×3+小步×1，×2 为大步×6+小步×2，模型同步为 17/34 cm；
- NFC Attempt1 成功、失败后 retreat/relocalize/reacquire；
- 其他 Screen 分类不能用于当前目标；
- 其他 Tag 定位成功不代表当前目标重获；
- 当前目标重新 FPGA 为 target 时直接 CHANGED，不进入 Attempt2；
- 明确仍非 target 时才 recalibrate/reapproach/Attempt2；
- 两次失败或 3 轮目标重获耗尽后 GAVE_UP，绝无 Attempt3；
- interaction retreat 只后退一次，定位失败不会重复物理后退。

```bash
python3 -m unittest robot_tonypi.tests.test_target_standoff_flow -v
```

## 11. `test_vision_tag_binding.py`（12 项）

检查左上 Tag↔Screen 绑定、错误侧/过远/非 1–36 ID 拒绝、最新有效缓存、失败不覆盖成功、1 秒分类 rate limit、15 秒 TTL、错误 Screen 拒绝，以及到点必须实时看到当前 Tag 才能采用缓存。

```bash
python3 -m unittest robot_tonypi.tests.test_vision_tag_binding -v
```

## 12. `test_classifier.py`（9 项）

检查 FPGA classifier 请求、响应解析、错误处理与既有 HTTP 通信语义。

```bash
python3 -m unittest robot_tonypi.tests.test_classifier -v
```

## 13. `test_capture_fpga_change.py`：人工实机集成测试

这个脚本不属于正式任务状态机。操作者必须先把机器人放在正确目标点并正对 Screen。它不做定位和导航，只做：

```text
相机 → Tag/Screen 绑定 → 28×28 crop → FPGA
→ 已是目标则结束
→ 否则按开关模拟或真实执行 NFC
```

### 13.1 无硬件

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 --target-flower hehua --dry-run
```

### 13.2 真实相机和 FPGA，模拟 NFC

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --skip-change
```

### 13.3 真实 NFC

```bash
python3 -u -m robot_tonypi.tests.test_capture_fpga_change \
  --screen-id 2 --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --execute
```

只有视觉和分类通过后，操作者再次输入 `EXECUTE 2` 才会发起真实事务。

## 14. `test_capture_15_frames.py`：人工相机采集

```bash
python3 -u -m robot_tonypi.tests.test_capture_15_frames
```

第 1 张立即拍摄；之后每次按 Enter 拍下一张，输入 `q` 提前结束。默认写入 `/home/pi/capture_15_frames_runs/<timestamp>/`。

## 15. 语法检查与注意事项

```bash
python3 -m compileall -q robot_tonypi
```

自动化测试不会证明以下真机参数正确：动作实际位移、相机内参、Tag 世界坐标、NFC 耦合距离、FPGA 网络稳定性、25 cm interaction target、5° yaw offset 和 17 cm final forward。正式运行前仍需现场验证。
