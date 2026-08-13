# Repository guidance

TonyPi competition controller for a 300 × 300 cm field. Read `robot_decision_tree.html` before changing task logic.

## Non-negotiable interaction boundary

Transit vision is geometry-only: `observe_transit_bindings()` may detect a screen quad and bind its left-upper Tag, but it must use `extract_crops=False` and cannot classify, vote, update flower state, lift the hand, or call a Worker. `classify_arrived_target()` is the only task-level classifier entry and requires the locked target, `ARRIVED_AT_TARGET`, and the 15 cm cardinal arrival geometry gate.

Physical flower change is exclusively:

```text
NEEDS_CHANGE → recheck/near-target realignment when needed
→ centralized full interaction gate → stand → lift_left_hand(stand=False)
→ second pose gate → robotall.send_request → finally stand
```

Do not restore the deleted HTTP `ApiClient` or the deleted four-waypoint scripted route. Competition numbering is identical: AprilTag ID == `screen_id` == NFC `worker_id`; use `worker_id = screen_id` and do not restore manual mapping configuration.

## Run

From the directory containing this package:

```bash
python3 -u -m robot_tonypi.main \
  --mode mission --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --skip-change --debug --debug-host 0.0.0.0 --debug-port 8090
```

Remove `--skip-change` only after screen binding and physical calibration are confirmed. `--dry-run` means no hardware. `--skip-api` is only a deprecated alias for `--skip-change`.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
```

Robot-only localization, motion, camera and FPGA integration checks remain hardware tests and are not replaced by the mock suite.

## Architecture

```text
main.py → TaskManager
TaskManager → MapModel, Localizer, ScreenDetector, ClassifierClient
            → MotionController, TonyPiHardware, DebugReporter
            → RobotInteractionClient
interaction_logic.py → pure observation/state and pose-gate rules
RobotInteractionClient → robotall.act + robotall.send_request
```

Task arrival and the map-safe approach point are deliberately distinct:

- `target_xy`: legacy/map-safe ~34 cm approach point used only as an internal A* waypoint. It cannot set `ARRIVED_AT_TARGET` or open classification.
- `face_center_xy`: center of the Tag's complete rectangular building face, derived from the building bounds; the Tag only selects the face.
- `tag_front_xy`: face center plus 15 cm along the quantized outward normal, before hand/body lateral compensation.
- `task_target_xy` / `interaction_xy`: the unique 15 cm body target after the existing reader/left-hand tangent compensation.

Mission target selection is recomputed after every processed screen using Euclidean distance from the latest pose to `task_target_xy`, with `screen_id` (the bound Tag ID) as the stable tie-breaker. Do not restore pass-by/opportunistic classification, discovery scans, task-level observation stops, or fixed routes.

The task layer derives each Tag face from its four immutable world corners: fixed X means west/east and fixed Y means south/north. The outward `normal_xy` must be exactly one of `(-1,0)`, `(1,0)`, `(0,-1)`, `(0,1)`; final yaw must be `0°`, `-180°`, `+90°`, or `-90°`. Viewer-left is `(normal.y, -normal.x)`.

The pre-classification arrival geometry gate checks only pose confidence/freshness, 15 cm normal distance, cardinal body yaw, and tangent alignment. The full interaction gate reuses those checks and additionally requires a non-target flower and stable `from_flower`; Worker ID is the already-bound screen ID.

## Conventions

- World position is centimeters; yaw is degrees.
- World yaw 0° points +X, positive is counterclockwise.
- Head pan 100 is center, greater is left, less is right.
- `(0, 0)` is the bottom-left of the field map.
- Python defaults in `config.py` and overrides in `config/competition_config.json` must remain aligned.
- `left_hand_body_offset_cm` is unknown mechanical geometry and must be field-calibrated; do not replace it with an invented precise value.
