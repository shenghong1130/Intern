#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line entry point for the TonyPi competition controller."""

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from competition_tonypi.config import default_config_path, load_config
    from competition_tonypi.task_manager import TaskManager
else:
    from .config import default_config_path, load_config
    from .task_manager import TaskManager


VALID_FLOWERS = {
    "bailianhua",
    "chuju",
    "hehua",
    "juhua",
    "lamei",
    "lanhua",
    "meiguihua",
    "shuixianhua",
    "taohua",
    "yinghua",
    "yuanweihua",
    "zijinghua",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TonyPi Exp4 competition mission controller")
    parser.add_argument("--mode", choices=["mission", "localize", "harvest"], default="mission")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--load-pos", default=None, help="optional external load_pos.py or tag JSON")
    parser.add_argument("--classifier-url", default="http://192.168.31.81:8080/predict")
    parser.add_argument(
        "--classifier-mode",
        choices=["direct", "central"],
        default="direct",
        help="send crops directly to a KV260 Worker or through Central Server",
    )
    parser.add_argument(
        "--classifier-student-id",
        default=None,
        help="Central Server student_id used to select the FPGA Artifact",
    )
    parser.add_argument(
        "--classifier-password",
        default=os.environ.get("STUDENT_PASSWORD"),
        help="Central Server student password; defaults to STUDENT_PASSWORD environment variable",
    )
    parser.add_argument("--team", default=None, help="registered contest team name used by robotall.send_request")
    parser.add_argument("--robot-id", default=None, help="registered robot ID used by robotall.send_request")
    parser.add_argument("--robot-name", default=None, help="deprecated fallback for both --team and --robot-id")
    parser.add_argument("--robot-secret", default=None)
    parser.add_argument("--target-flower", required=True, help="API flower name, e.g. hehua")
    parser.add_argument("--max-screens", type=int, default=None, help="test-only success cap; omit for official runs")
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-port", type=int, default=None)
    parser.add_argument("--debug-host", default=None, help="debug dashboard bind host, e.g. 0.0.0.0 for LAN access")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-change", "--skip-api", dest="skip_change", action="store_true", help="classify at the configured target standoff but skip the 17 cm action, arm motion, and Worker request")
    parser.add_argument("--start-x", type=float, default=None, help="test-only manual start x")
    parser.add_argument("--start-y", type=float, default=None, help="test-only manual start y")
    parser.add_argument("--start-yaw", type=float, default=None, help="test-only manual start yaw")
    args = parser.parse_args(argv)
    if args.classifier_mode == "central" and not str(args.classifier_student_id or "").strip():
        parser.error("--classifier-student-id is required when --classifier-mode central")
    if args.classifier_mode == "central" and not str(args.classifier_password or "").strip():
        parser.error(
            "Central classifier password is required. "
            "Use --classifier-password or set STUDENT_PASSWORD."
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.target_flower not in VALID_FLOWERS:
        raise SystemExit("Unknown target flower '{}'. Valid: {}".format(args.target_flower, ", ".join(sorted(VALID_FLOWERS))))
    config = load_config(args.config)
    if args.debug:
        config["debug"]["enabled"] = True
    manager = TaskManager(args, config)
    ok = manager.run()
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
