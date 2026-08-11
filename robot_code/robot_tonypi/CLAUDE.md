# Repository guidance

TonyPi competition controller for a 300 × 300 cm field. Read `robot_decision_tree.html` before changing task logic.

## Non-negotiable interaction boundary

Visual detection/classification/voting only updates `Screen` observations. `harvest_visible`, opportunistic/passby scans, `scan_after_turn`, and `harvest_during_localize` must never call `send_request`.

Physical flower change is exclusively:

```text
NEEDS_CHANGE → interaction staging → final AprilTag alignment
→ centralized pose gate → stand → lift_left_hand(stand=False)
→ second pose gate → robotall.send_request → finally stand
```

Do not restore the deleted HTTP `ApiClient` or the deleted four-waypoint scripted route. `screen_id` must map explicitly to `worker_id` in `interaction.worker_mapping`; do not assume equality.

## Run

From the directory containing this package:

```bash
python3 -u -m robot_tonypi.main \
  --mode mission --target-flower hehua \
  --classifier-url http://192.168.31.81:8080/predict \
  --team red --robot-id red-1 --robot-secret 1234 \
  --skip-change --debug --debug-host 0.0.0.0 --debug-port 8090
```

Remove `--skip-change` only after explicit Worker mapping and physical calibration are confirmed. `--dry-run` means no hardware. `--skip-api` is only a deprecated alias for `--skip-change`.

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

The two navigation targets are deliberately distinct:

- `observation_xy`: remote camera observation.
- `interaction_staging_xy` then `interaction_xy`: physical Worker interaction.

`normal_xy` points from the screen into its visible/front side. Viewer-left is `(normal.y, -normal.x)` and a robot facing the screen uses `normal_yaw_deg + 180°`.

## Conventions

- World position is centimeters; yaw is degrees.
- World yaw 0° points +X, positive is counterclockwise.
- Head pan 100 is center, greater is left, less is right.
- `(0, 0)` is the bottom-left of the field map.
- Python defaults in `config.py` and overrides in `config/competition_config.json` must remain aligned.
- `left_hand_body_offset_cm` is unknown mechanical geometry and must be field-calibrated; do not replace it with an invented precise value.
