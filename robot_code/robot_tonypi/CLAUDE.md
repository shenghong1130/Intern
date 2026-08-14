# Repository guidance

TonyPi competition controller for a 300 × 300 cm field. Read `robot_decision_tree.html` before changing task logic.

## Non-negotiable task boundary

Transit vision is geometry-only: `observe_transit_bindings()` may detect a screen quad and bind its left-upper Tag with `extract_crops=False`, but it cannot classify, vote, update flower state, move 3 cm, lift the hand, or call a Worker.

After initial localization, the task locks the nearest unfinished screen by distance to its single 17 cm body target. The target is built from the complete building face center, its quantized outward normal, and the existing reader/left-hand tangent compensation. `target_xy`, `interaction_xy`, and `task_target_xy` must refer to that same coordinate. Do not restore a 34 cm approach point or a separate 15 cm alignment stage.

Classification is allowed only after direct navigation reaches the locked 17 cm coordinate and its cardinal yaw. One live frame must contain the same 1–36 Tag and a screen quad bound to that Tag. A stable non-target FPGA result creates a target-specific `VisualAuthorization`.

Physical change is exclusively:

```text
NEEDS_CHANGE + locked visual authorization
→ interaction_forward_3cm exactly once
→ stand → lift_left_hand(stand=False)
→ recheck the same locked authorization
→ robotall.send_request → finally stand
```

There is no localization, capture, body alignment, turning, strafing, backing, or second forward action after the 3 cm motion. Selecting another target clears the authorization. Competition numbering is identical: AprilTag ID == `screen_id` == NFC `worker_id`.

## Planning rules

Keep static/dynamic obstacles, inflation, obstacle costs, A*, action-level A*, collision recovery, and near-wall recovery. The exact locked 17 cm goal cell may be accepted as a high-cost terminal, but it must remain physically free. Only samples in that terminal grid cell receive the exception; costs are not cleared and buildings remain impassable. Do not relocate the task goal with `nearest_free_xy()` or `nearest_traversable_xy()`.

Tag face normals must be exactly one of `(-1,0)`, `(1,0)`, `(0,-1)`, `(0,1)`. Final yaw must be `0°`, `-180°`, `+90°`, or `-90°`. Viewer-left is `(normal.y, -normal.x)`.

## Run and test

```bash
python3 -u -m robot_tonypi.main \
  --mode mission --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --skip-change --debug --debug-host 0.0.0.0 --debug-port 8090

python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

`--dry-run` means no hardware. `--skip-change` performs real navigation and classification but skips the dedicated 3 cm action, arm movement, and Worker request. `--skip-api` is its deprecated alias.

## Conventions

- World position is centimeters; yaw is degrees; yaw 0° points +X and positive is counterclockwise.
- Head pan 100 is center, greater is left, less is right.
- `(0, 0)` is the bottom-left of the field map.
- Python defaults and `config/competition_config.json` must remain aligned.
- Keep `max_screen_area_ratio` at its current configured value unless a separate task explicitly changes it.
- `left_hand_body_offset_cm` and the real displacement of `interaction_forward_3cm` require field calibration.
