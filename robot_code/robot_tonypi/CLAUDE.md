# Repository guidance

TonyPi competition controller for a 300 × 300 cm field. Before changing task logic, read `README.md`, `FILES.md`, `robot_decision_tree.html`, the resolved `config.py` + `config/competition_config.json`, and the relevant tests.

## Current coordinate and identity contracts

- World distance is centimeters; yaw is degrees; yaw 0° points toward +X and positive yaw is counterclockwise.
- The Debug map keeps world `(0, 0)` at the top-left; world x is rendered downward and world y to the right (`_map_pt(xy) -> (y, x)`).
- Head pan 100° is center, larger angles look left, smaller angles look right.
- AprilTag ID == `screen_id` == NFC `worker_id`.
- Tag positions in `load_pos.py` and camera calibration are immutable unless a dedicated calibration task explicitly changes them.
- The locked `TargetGoal` atomically owns screen ID, tag ID, anchor, goal, desired yaw, source and generation. Never update those fields independently.

## Current target geometry

- `target_distance_cm = 25.0`.
- `target_lateral_offset_cm = -1.0` in the robot-left frame.
- `target_yaw_offset_deg = 5.0`; it affects only desired yaw, never target XY.
- `target_final_forward_cm = 17.0`.
- `target_xy`, `interaction_xy` and `task_target_xy` are the same configured body target; the Screen/Tag anchor is separate.
- Cardinal normals are exactly `(-1,0)`, `(1,0)`, `(0,-1)`, `(0,1)`. Desired yaw is the screen-facing cardinal yaw plus `5°`, normalized to `[-180, 180)`.

## Vision and authorization boundary

Transit/localization frames may classify a valid `Tag ID == Screen ID` crop and cache only the newest successful evidence for that Screen. The current TTL is 15 seconds. A failed classifier call must not destroy a previous valid cache item.

Cached evidence alone cannot authorize interaction. At the locked task target the current target Tag must be seen live. The adopted observation must match the locked screen/tag and binding before it creates `TargetVisualConfirmation` and `VisualAuthorization`.

Classifier service failure is recoverable and does not mean the target disappeared. Keep target identity, live Tag confirmation and classification availability as separate concepts.

## NFC invariants

Physical change is bounded and target-specific:

```text
live current Tag + same-ID bound classification != target
→ interaction_forward_10cm exactly once
→ visual authorization check
→ stand → lift_left_hand(stand=False)
→ authorization recheck
→ new seq → send_request(retries=0, physical deadline <= 15 s)
→ finally stand
```

After Attempt 1 failure:

```text
retreat 10 cm once
→ relocalize
→ reacquire the current target for at most 3 cycles
→ classify only the current target
```

The following invariants are mandatory:

1. Localization success is not current-target reacquisition. Other Tags may update Pose but cannot end the target search.
2. `CHANGED` is terminal for the current NFC flow. It never enters retry, recalibration, reapproach or Attempt 2.
3. Attempt 2 is allowed only after Attempt 1 failed, the current target was reacquired, and its fresh FPGA result explicitly differs from `target_flower`.
4. NFC has at most two physical Attempts. Target reacquisition has at most `nfc_retry_target_reacquire_max_cycles=3` cycles.
5. A fresh observation must match both `screen_id` and `tag_id`; a different Screen's target-looking flower is irrelevant.
6. Attempt 2 failure or target-reacquisition exhaustion calls the bounded GAVE_UP path and continues the mission.

## Navigation and recovery contracts

- Ordinary planners keep hard occupancy, footprint, soft cost, clearance, A*, action-space planning, near-wall recovery and interior recovery.
- Current task-target navigation calls `navigate_to_xy(..., allow_goal_high_cost=True, bypass_action_safety=True)`. It may still use the planners, but an already selected TURN/FORWARD/BACK/LATERAL action is not vetoed again by near-wall/corridor/footprint/center-free gates.
- The current target building's soft inflation may be ignored only by the bounded target-owned direct/approach rules. Hard occupancy, boundaries, unrelated buildings and dynamic obstacles remain hard constraints outside task-target bypass semantics.
- Repeated identical planning failures escalate at 3; they must not wait for the 80-step target limit.
- Navigation failures rotate a target temporarily. When all unfinished targets are temporary failures, run global recovery, release them and select again.
- `MISSION_FAILED` remains an enum value but is not the normal runtime terminal path. Global mission timeout is the automatic terminal event; Ctrl+C/emergency stop remain valid manual exits.

## Localization contracts

- Normal `localize_scan()` stops at the first accepted visual Pose and recenters.
- Required-target mode stops only when the requested Target Tag and Screen are bound; other successful localization Tags do not stop it.
- Only `accept_visual_localization()` resets `actions_since_localize` and `motion_uncertainty`.
- `no_tag`, `pose_unavailable_with_tags` and `capture_failed` are distinct failure results.
- Startup and configured recovery reuse `run_localization_search_sequence()`: full pan, configured body action, full pan.
- A large turn triggers one relocalization only while actions remain since the previous accepted visual Pose.

## Run and test

```bash
cd /home/pi
python3 -m unittest discover -s robot_tonypi/tests -p 'test_*.py' -v
python3 -m compileall -q robot_tonypi

python3 -u -m robot_tonypi.main \
  --mode mission --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --skip-change --debug --debug-host 0.0.0.0 --debug-port 8090
```

`--dry-run` disables hardware. `--skip-change` keeps real localization/navigation/classification but skips final forward, arm motion and NFC. `--skip-api` is the same deprecated alias.

## Change discipline

- Python defaults and `competition_config.json` must stay semantically aligned.
- Do not invent ActionGroup names.
- Do not silently change target geometry, map axes, Tag positions, camera calibration, NFC protocol or FPGA response format.
- Preserve debug events for target identity, target goal generation, classification source, NFC attempt/seq, target reacquisition and recovery decisions.
- Update `README.md`, `FILES.md`, `tests/README.md` and `robot_decision_tree.html` whenever state-machine behavior or a documented runtime parameter changes.
