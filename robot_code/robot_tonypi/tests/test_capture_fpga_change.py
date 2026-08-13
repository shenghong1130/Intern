#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual-position camera -> FPGA -> optional Worker integration test.

This standalone tool deliberately does not localize, navigate, or align the
robot.  Its safety gate means only that the operator has manually confirmed
that the robot is already at the intended interaction pose; it is not a
replacement for the mission controller's visual interaction-pose gate.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robot_tonypi.classifier import ClassifierClient
    from robot_tonypi.config import default_config_path, load_config
    from robot_tonypi.hardware import RealtimeCamera, TonyPiHardware
    from robot_tonypi.interaction_client import RobotInteractionClient
    from robot_tonypi.localizer import AprilTagDetector
    from robot_tonypi.main import VALID_FLOWERS
    from robot_tonypi.models import InteractionPoseCheck
    from robot_tonypi.vision import ScreenDetector
else:
    from ..classifier import ClassifierClient
    from ..config import default_config_path, load_config
    from ..hardware import RealtimeCamera, TonyPiHardware
    from ..interaction_client import RobotInteractionClient
    from ..localizer import AprilTagDetector
    from ..main import VALID_FLOWERS
    from ..models import InteractionPoseCheck
    from ..vision import ScreenDetector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Manually positioned TonyPi camera/FPGA/Worker integration test"
    )
    parser.add_argument("--screen-id", type=int, required=True)
    parser.add_argument("--target-flower", required=True)
    parser.add_argument("--classifier-url", default="http://192.168.31.81:8080/predict")
    parser.add_argument("--team", default=None)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--robot-secret", default=None)
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="initialize no real hardware; validate configuration and exit safely",
    )
    parser.add_argument(
        "--skip-change",
        action="store_true",
        help="force the Worker transaction to remain simulated",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow a real hand/NFC transaction after interactive confirmation",
    )
    return parser.parse_args(argv)


def worker_id_for_screen(config: dict, screen_id: int) -> Optional[int]:
    """Resolve the explicit mapping without assuming screen_id == worker_id."""
    mapping = config["interaction"].get("worker_mapping", {})
    value = mapping.get(str(screen_id), mapping.get(screen_id))
    return None if value is None else int(value)


def manual_operator_gate(confirmed: bool) -> InteractionPoseCheck:
    """Return the test-only manual-position gate, never a visual pose claim."""
    return InteractionPoseCheck(
        ready=bool(confirmed),
        pose_valid=False,
        reasons=[] if confirmed else ["manual_position_not_confirmed"],
    )


def make_output_dir() -> Path:
    root = Path.cwd() / "capture_fpga_change_runs"
    path = root / time.strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_image(path: Path, image) -> None:
    import cv2

    if image is None or not cv2.imwrite(str(path), image):
        raise RuntimeError("failed to save image: {}".format(path))


def print_event(name: str, **data) -> None:
    print("[{}] {}".format(name, json.dumps(data, ensure_ascii=False, default=str)))


def validate_args(args) -> None:
    if not 1 <= int(args.screen_id) <= 36:
        raise SystemExit("--screen-id must be in 1..36")
    if args.target_flower not in VALID_FLOWERS:
        raise SystemExit(
            "unknown --target-flower '{}'; valid: {}".format(
                args.target_flower, ", ".join(sorted(VALID_FLOWERS))
            )
        )
    if args.execute and args.dry_run:
        raise SystemExit("--execute cannot be combined with --dry-run")
    if args.execute and args.skip_change:
        raise SystemExit("--execute cannot be combined with --skip-change")
    if args.execute and not (args.team and args.robot_id and args.robot_secret):
        raise SystemExit("--execute requires --team, --robot-id and --robot-secret")


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config = load_config(args.config)
    worker_id = worker_id_for_screen(config, args.screen_id)
    real_change_enabled = bool(args.execute and not args.skip_change and not args.dry_run)

    print("=== Capture / FPGA / Flower Change Integration Test ===")
    print("screen_id={}".format(args.screen_id))
    print("target_flower={}".format(args.target_flower))
    print("classifier_url={}".format(args.classifier_url))
    print("worker_id={}".format(worker_id if worker_id is not None else "MISSING"))
    print("real_change_enabled={}".format(real_change_enabled))
    print(
        "SAFETY: this script performs no localization/navigation/alignment. "
        "Its gate is only the operator's manual confirmation that TonyPi is already positioned correctly."
    )

    hardware = None
    camera = None
    output_dir = None
    try:
        hardware = TonyPiHardware(config, dry_run=args.dry_run)
        camera = RealtimeCamera(config, dry_run=args.dry_run)
        hardware.center_head()

        if args.dry_run:
            print("[dry-run] configuration and no-hardware lifecycle check passed")
            print("[dry-run] no camera frame, FPGA request, hand action, or Worker request was performed")
            return 0

        output_dir = make_output_dir()
        frame = camera.capture_settled()
        if frame is None:
            print("ERROR: camera did not return a settled frame")
            return 2
        raw_path = output_dir / "raw.jpg"
        save_image(raw_path, frame)

        import cv2

        tag_detector = AprilTagDetector(
            config["localization"]["tag_family"],
            config["localization"].get("detector_upscale", 1.0),
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = tag_detector.detect(gray)
        screen_detector = ScreenDetector(config, map_model=None)
        candidates = screen_detector.detect(frame, tags, pose=None, extract_crops=True)
        annotated = screen_detector.annotate(frame, candidates, tags)
        annotated_path = output_dir / "annotated.jpg"
        save_image(annotated_path, annotated)

        matches = [item for item in candidates if int(item.screen_id) == int(args.screen_id)]
        if not matches:
            print(
                "ERROR: specified screen_id={} was not detected/bound; detected={}".format(
                    args.screen_id, [item.screen_id for item in candidates]
                )
            )
            print("images={}".format(output_dir))
            return 3

        candidate = matches[0]
        if candidate.crop_28x28 is None:
            print("ERROR: target candidate has no 28x28 crop")
            return 3
        crop_path = output_dir / "screen_{}_crop_28x28.png".format(args.screen_id)
        save_image(crop_path, candidate.crop_28x28)

        classifier = ClassifierClient(
            args.classifier_url,
            timeout_s=float(config.get("classifier", {}).get("timeout_s", 4.0)),
        )
        classification = classifier.classify_crop(candidate.crop_28x28)
        if not classification.ok:
            print("ERROR: FPGA classification failed: {}".format(classification.error))
            print("images={}".format(output_dir))
            return 4

        from_flower = classification.flower_api
        confidence = float(classification.confidence)
        min_confidence = float(config["vision"]["min_confidence"])
        print("\n=== Recognition Result ===")
        print("screen_id={}".format(args.screen_id))
        print("worker_id={}".format(worker_id if worker_id is not None else "MISSING"))
        print("from_flower={}".format(from_flower))
        print("to_flower={}".format(args.target_flower))
        print("confidence={:.4f} (minimum={:.4f})".format(confidence, min_confidence))
        print("raw={}".format(json.dumps(classification.raw, ensure_ascii=False, default=str)))
        print("images={}".format(output_dir))

        if confidence < min_confidence:
            print("STOP: confidence is below the configured minimum; no hand or Worker action")
            return 5
        if from_flower == args.target_flower:
            print("DONE: screen already shows the target flower; no hand or Worker action")
            return 0
        if worker_id is None:
            print("STOP: explicit screen_id -> worker_id mapping is missing; no hand or Worker action")
            return 6

        if not real_change_enabled:
            print("SIMULATION: recognition completed; --execute was not supplied, so no real hand/NFC action is allowed")

        operator_confirmed = False
        if real_change_enabled:
            print("\nWARNING: the next step will physically lift the left hand and send an NFC/Worker request.")
            print("Confirm manually: robot is already at the correct target point and facing the screen.")
            expected = "EXECUTE {}".format(args.screen_id)
            answer = input("Type '{}' to continue: ".format(expected)).strip()
            if answer != expected:
                print("STOP: operator confirmation did not match; no hand or Worker action")
                return 7
            operator_confirmed = True
        else:
            # This only opens RobotInteractionClient's simulated branch.  It
            # must never be described as a measured/localized pose check.
            operator_confirmed = True

        def on_phase(phase: str, **data) -> None:
            if phase == "transaction_start":
                hardware.set_interaction_active(True)
            elif phase == "transaction_end":
                hardware.set_interaction_active(False)
            print_event("phase:{}".format(phase), **data)

        interaction = RobotInteractionClient(
            args.team,
            args.robot_secret,
            args.robot_id,
            config,
            dry_run=False,
            skip_change=not real_change_enabled,
            event_callback=print_event,
            phase_callback=on_phase,
        )
        result = interaction.change_flower(
            screen_id=args.screen_id,
            worker_id=worker_id,
            from_flower=from_flower,
            to_flower=args.target_flower,
            safety_gate=lambda: manual_operator_gate(operator_confirmed),
        )
        print("\n=== Interaction Result ===")
        print(
            json.dumps(
                {
                    "success": result.success,
                    "simulated": result.simulated,
                    "worker_id": result.worker_id,
                    "response": result.response,
                    "error": result.error,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0 if result.success else 8
    except KeyboardInterrupt:
        print("\nInterrupted; entering safe cleanup")
        return 130
    except Exception as exc:
        print("ERROR: {}: {}".format(type(exc).__name__, exc))
        return 9
    finally:
        if hardware is not None:
            hardware.set_interaction_active(False)
            try:
                hardware.run_action("stand", times_override=1)
            except Exception as exc:
                print("[cleanup] stand failed: {}".format(exc))
            try:
                hardware.center_head()
            except Exception as exc:
                print("[cleanup] center_head failed: {}".format(exc))
        if camera is not None:
            try:
                camera.release()
            except Exception as exc:
                print("[cleanup] camera.release failed: {}".format(exc))
        if hardware is not None:
            try:
                hardware.close()
            except Exception as exc:
                print("[cleanup] hardware.close failed: {}".format(exc))
        if output_dir is not None:
            print("saved_artifacts={}".format(output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
