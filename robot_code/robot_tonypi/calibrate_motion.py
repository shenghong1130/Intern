#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive manual measurement tool for TonyPi motion actions.

This module deliberately does not initialize the camera, AprilTag detector,
localizer, map, or classifier.  It runs the exact configured mission action
through ``MotionController`` and records measurements entered by an operator.
"""

import argparse
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import default_config_path, load_config
from .hardware import TonyPiHardware
from .motion import MotionController, RobotState
from .utils import save_json


ACTION_ORDER = (
    "forward_fast",
    "forward_micro",
    "back_fast",
    "strafe_left_fast",
    "strafe_right_fast",
    "turn_left_fast",
    "turn_right_fast",
    "turn_left_large",
    "turn_right_large",
)

ACTION_MEASUREMENTS = {
    "forward_fast": ("forward_cm", 1.0, "实际前进距离", "cm"),
    "forward_micro": ("forward_cm", 1.0, "实际前进距离", "cm"),
    "back_fast": ("forward_cm", -1.0, "实际后退距离", "cm"),
    "strafe_left_fast": ("lateral_cm", 1.0, "实际左移距离", "cm"),
    "strafe_right_fast": ("lateral_cm", -1.0, "实际右移距离", "cm"),
    "turn_left_fast": ("yaw_deg", 1.0, "实际左转角度", "deg"),
    "turn_right_fast": ("yaw_deg", -1.0, "实际右转角度", "deg"),
    "turn_left_large": ("yaw_deg", 1.0, "实际左转角度", "deg"),
    "turn_right_large": ("yaw_deg", -1.0, "实际右转角度", "deg"),
}


class CalibrationError(RuntimeError):
    """An unsafe or unusable calibration condition."""


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot summarize an empty sequence")
    median = float(statistics.median(data))
    return {
        "mean": float(statistics.mean(data)),
        "median": median,
        "std": float(statistics.pstdev(data)),
        "min": min(data),
        "max": max(data),
        "mad": float(statistics.median(abs(value - median) for value in data)),
    }


def normalize_manual_measurement(action: str, entered_value: float, times: int = 1) -> Tuple[float, float]:
    """Return signed total and signed per-action values from an absolute input."""
    if action not in ACTION_MEASUREMENTS:
        raise ValueError("unsupported action: {}".format(action))
    if int(times) <= 0:
        raise ValueError("times must be greater than zero")
    sign = ACTION_MEASUREMENTS[action][1]
    signed_total = abs(float(entered_value)) * sign
    return signed_total, signed_total / int(times)


def physical_action_spec(action_spec: Dict[str, object], times: int) -> Dict[str, object]:
    """Describe the groups exactly as TonyPiHardware.run_action will execute."""
    steps = action_spec.get("sequence")
    if steps:
        sequence = []
        for step in steps:
            step_times = int(step.get("times", 1))
            if bool(step.get("repeat", False)):
                step_times *= int(times)
            sequence.append(
                {
                    "group": step["group"],
                    "times": step_times,
                    "with_stand": bool(step.get("with_stand", False)),
                    "repeat": bool(step.get("repeat", False)),
                }
            )
        return {"sequence": sequence}
    return {
        "group": action_spec["group"],
        "times": int(times),
        "with_stand": bool(action_spec.get("with_stand", False)),
    }


def format_physical_action(physical: Dict[str, object]) -> str:
    if "sequence" in physical:
        return " + ".join(
            "{} x{}".format(step["group"], step["times"])
            for step in physical["sequence"]
        )
    return "{} x{}".format(physical["group"], physical["times"])


def build_recommendations(action_reports: Iterable[Dict[str, object]]) -> Dict[str, object]:
    actions = {}
    for result in action_reports:
        if result.get("recommended_value") is None:
            continue
        actions[result["action"]] = {
            result["metric"]: float(result["recommended_value"])
        }
    return {"motion": {"actions": actions}}


class ManualMotionCalibrator:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.hardware: Optional[TonyPiHardware] = None
        self.motion: Optional[MotionController] = None
        self.cleaned_up = False
        self.quit_requested = False
        self.report = {
            "schema_version": 2,
            "measurement_method": "MANUAL_OPERATOR_INPUT",
            "created_at": datetime.now().astimezone().isoformat(),
            "requested_actions": list(args.actions),
            "times_override": args.times,
            "trials_per_action": args.trials,
            "status": "INITIALIZING",
            "actions": [],
            "recommended_config": {"motion": {"actions": {}}},
        }

    def initialize(self) -> None:
        self.hardware = TonyPiHardware(self.config, dry_run=False)
        state = RobotState(self.config)
        self.motion = MotionController(self.hardware, state)

    def stand(self) -> None:
        if self.motion is None:
            raise CalibrationError("motion controller is not initialized")
        result = self.motion.run("stand")
        if not result.ok:
            raise CalibrationError("stand action failed: {}".format(result.error))

    def cleanup(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            if self.motion is not None:
                self.motion.run("stand")
            elif self.hardware is not None:
                self.hardware.run_action("stand")
        except Exception as exc:
            print("[cleanup-warning] stand failed: {}".format(exc), file=sys.stderr)
        try:
            if self.hardware is not None:
                self.hardware.close()
        except Exception as exc:
            print("[cleanup-warning] hardware.close failed: {}".format(exc), file=sys.stderr)

    @staticmethod
    def _command(prompt: str, allowed: Tuple[str, ...]) -> str:
        while True:
            command = input(prompt).strip().lower()
            if command in allowed:
                return command
            print("无效输入，请输入 {}。".format(" / ".join("ENTER" if item == "" else item for item in allowed)))

    def _read_measurement(self, label: str, unit: str) -> Tuple[str, Optional[float]]:
        while True:
            raw = input("请输入{}({})，r 重测，s 跳过，q 退出: ".format(label, unit)).strip().lower()
            if raw in ("r", "s", "q"):
                return raw, None
            try:
                return "value", float(raw)
            except ValueError:
                print("请输入数字，或输入 r / s / q。")

    def run(self) -> Dict[str, object]:
        self.report["status"] = "RUNNING"
        print("\n=== Motion Calibration / 动作标定 ===")
        print("测量时请输入距离或角度的绝对值；程序会自动应用方向符号。")
        for index, action in enumerate(self.args.actions, start=1):
            action_report = self.run_action(index, len(self.args.actions), action)
            self.report["actions"].append(action_report)
            if self.quit_requested:
                self.report["status"] = "QUIT"
                break

        if not self.quit_requested:
            skipped = any(item["status"] == "SKIPPED" for item in self.report["actions"])
            self.report["status"] = "COMPLETE_WITH_SKIPS" if skipped else "COMPLETE"
        self.report["recommended_config"] = build_recommendations(self.report["actions"])
        self.print_summary()
        return self.report

    def run_action(self, index: int, total: int, action: str) -> Dict[str, object]:
        spec = self.config["motion"]["actions"][action]
        metric, _, input_label, unit = ACTION_MEASUREMENTS[action]
        configured = float(spec.get(metric, 0.0))
        effective_times = int(
            self.args.times if self.args.times is not None else spec.get("times", 1)
        )
        physical = physical_action_spec(spec, effective_times)
        result = {
            "action": action,
            "metric": metric,
            "unit": unit,
            "physical": physical,
            "configured_value": configured,
            "times": effective_times,
            "trials_requested": self.args.trials,
            "trials": [],
            "measurements": [],
            "mean": None,
            "median": None,
            "recommended_value": None,
            "status": "RUNNING",
        }
        print("\n[{}/{}] {}".format(index, total, action))
        print("配置值：{} = {} {}".format(metric, configured, unit))

        trial_number = 1
        while trial_number <= self.args.trials:
            print("\n  Trial {}/{}".format(trial_number, self.args.trials))
            print("  即将执行：{}".format(format_physical_action(physical)))
            command = self._command("  请将机器人放回测量起点。按 ENTER 执行动作，s 跳过当前动作，q 退出: ", ("", "s", "q"))
            if command == "s":
                result["status"] = "SKIPPED"
                break
            if command == "q":
                result["status"] = "QUIT"
                self.quit_requested = True
                break

            self.stand()
            try:
                action_result = self.motion.run(action, times_override=effective_times)
                if not action_result.ok:
                    raise CalibrationError("action '{}' failed: {}".format(action, action_result.error))
            finally:
                self.stand()

            response, entered = self._read_measurement(input_label, unit)
            if response == "r":
                print("  本次不记录，将重新执行当前 trial。")
                continue
            if response == "s":
                result["status"] = "SKIPPED"
                break
            if response == "q":
                result["status"] = "QUIT"
                self.quit_requested = True
                break

            signed_total, signed_per_action = normalize_manual_measurement(
                action, entered, effective_times
            )
            trial = {
                "trial": trial_number,
                "entered_absolute_value": abs(float(entered)),
                "measured_total": signed_total,
                "measured_per_action": signed_per_action,
            }
            result["trials"].append(trial)
            result["measurements"].append(signed_per_action)
            print(
                "  记录：configured = {} {}, measured = {} {}{}".format(
                    configured,
                    unit,
                    signed_per_action,
                    unit,
                    " / action" if effective_times > 1 else "",
                )
            )
            command = self._command("  按 ENTER 继续，r 重测本次，q 安全退出: ", ("", "r", "q"))
            if command == "r":
                result["trials"].pop()
                result["measurements"].pop()
                print("  已撤销本次记录，将重新执行当前 trial。")
                continue
            if command == "q":
                result["status"] = "QUIT"
                self.quit_requested = True
                break
            trial_number += 1

        if result["measurements"]:
            stats = summarize(result["measurements"])
            result.update(stats)
            result["recommended_value"] = stats["median"]
            if result["status"] == "RUNNING":
                result["status"] = "COMPLETE"
        elif result["status"] == "RUNNING":
            result["status"] = "SKIPPED"
        return result

    def print_summary(self) -> None:
        print("\n=== Calibration Summary / 标定汇总 ===")
        for item in self.report["actions"]:
            recommended = item.get("recommended_value")
            if recommended is None:
                print("{:<20} {:<12} {:>8} -> {:>8}".format(
                    item["action"], item["metric"], item["configured_value"], item["status"]
                ))
            else:
                print("{:<20} {:<12} {:>8g} -> {:>8g}".format(
                    item["action"], item["metric"], item["configured_value"], recommended
                ))
                print("  measurements = {}  mean={:.3f} median={:.3f}".format(
                    item["measurements"], item["mean"], item["median"]
                ))


def write_recommended_config(config_path: Path, report: Dict[str, object], timestamp: str) -> Path:
    if report.get("status") not in ("COMPLETE", "COMPLETE_WITH_SKIPS"):
        raise CalibrationError("refusing to update config because calibration did not finish")
    recommendations = report["recommended_config"]["motion"]["actions"]
    if not recommendations:
        raise CalibrationError("refusing to update config because there are no measurements")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    actions = raw.setdefault("motion", {}).setdefault("actions", {})
    for action, fields in recommendations.items():
        # Some actions (currently forward_micro) may be inherited entirely
        # from DEFAULT_CONFIG and therefore absent from the override JSON.  A
        # metric-only override is sufficient and preserves that default group.
        actions.setdefault(action, {}).update(fields)
    backup = config_path.with_name("{}.backup_{}".format(config_path.name, timestamp))
    shutil.copy2(str(config_path), str(backup))
    save_json(config_path, raw)
    return backup


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run TonyPi mission actions and manually enter measured distances/angles"
    )
    parser.add_argument(
        "--action",
        action="append",
        choices=ACTION_ORDER,
        help="calibrate only this action; repeat the option for multiple actions",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=None,
        help="override repetitions in each physical run; default uses each action's config",
    )
    parser.add_argument("--trials", type=int, default=1, help="manual measurements per action")
    parser.add_argument("--write-config", action="store_true", help="back up and update competition_config.json")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    args.actions = tuple(args.action or ACTION_ORDER)
    if args.times is not None and args.times <= 0:
        parser.error("--times must be greater than zero")
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    missing = [action for action in args.actions if action not in config["motion"]["actions"]]
    if missing:
        raise CalibrationError("actions missing from loaded config: {}".format(", ".join(missing)))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "motion_calibration_{}.json".format(timestamp)
    calibrator = ManualMotionCalibrator(args, config)
    exit_code = 0
    try:
        calibrator.initialize()
        calibrator.run()
        if calibrator.report["status"] == "QUIT":
            exit_code = 130
    except KeyboardInterrupt:
        calibrator.report["status"] = "INTERRUPTED"
        calibrator.report["error"] = "KeyboardInterrupt"
        exit_code = 130
        print("\n标定已中断。", file=sys.stderr)
    except Exception as exc:
        calibrator.report["status"] = "FAILED"
        calibrator.report["error"] = "{}: {}".format(type(exc).__name__, exc)
        exit_code = 2
        print("标定失败：{}".format(exc), file=sys.stderr)
    finally:
        calibrator.cleanup()
        calibrator.report["recommended_config"] = build_recommendations(
            calibrator.report["actions"]
        )
        save_json(output_path, calibrator.report)
        print("标定报告：{}".format(output_path))

    if args.write_config:
        if calibrator.report["status"] in ("COMPLETE", "COMPLETE_WITH_SKIPS"):
            try:
                backup = write_recommended_config(config_path, calibrator.report, timestamp)
                print("配置备份：{}".format(backup))
                print("配置已更新：{}".format(config_path))
            except Exception as exc:
                print("配置未更新：{}".format(exc), file=sys.stderr)
                return 2
        else:
            print("配置未更新：标定状态为 {}。".format(calibrator.report["status"]))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
