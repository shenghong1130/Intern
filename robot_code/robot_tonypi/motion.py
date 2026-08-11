#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot state and dead-reckoning motion updates."""

import math
from typing import Optional

from .models import Confidence, RobotPose
from .utils import normalize_angle_deg, now_s


class RobotState:
    def __init__(self, config):
        self.config = config
        self.pose: Optional[RobotPose] = None
        self.actions_since_localize = 0

    def set_pose(self, pose: RobotPose) -> None:
        self.pose = pose
        self.actions_since_localize = 0

    def set_manual_pose(self, x_cm: float, y_cm: float, yaw_deg: float, source: str = "MANUAL") -> None:
        self.set_pose(
            RobotPose(
                x_cm=x_cm,
                y_cm=y_cm,
                yaw_deg=normalize_angle_deg(yaw_deg),
                confidence=Confidence.HIGH,
                source=source,
                last_update_s=now_s(),
            )
        )

    def apply_action_result(self, result) -> None:
        if self.pose is None:
            return
        yaw_rad = math.radians(self.pose.yaw_deg)
        left_rad = yaw_rad + math.pi / 2.0
        self.pose.x_cm += result.model_forward_cm * math.cos(yaw_rad)
        self.pose.y_cm += result.model_forward_cm * math.sin(yaw_rad)
        self.pose.x_cm += result.model_lateral_cm * math.cos(left_rad)
        self.pose.y_cm += result.model_lateral_cm * math.sin(left_rad)
        self.pose.yaw_deg = normalize_angle_deg(self.pose.yaw_deg + result.model_yaw_deg)
        self.pose.source = "DEAD_RECKONING"
        self.pose.last_update_s = now_s()
        self.actions_since_localize += 1
        if self.actions_since_localize >= self.config["navigation"]["relocalize_after_actions"]:
            self.pose.confidence = Confidence.LOW
        elif self.pose.confidence == Confidence.HIGH and self.actions_since_localize >= 3:
            self.pose.confidence = Confidence.MEDIUM

    def needs_relocalize(self) -> bool:
        if self.pose is None:
            return True
        if self.pose.confidence == Confidence.LOW:
            return True
        return self.actions_since_localize >= int(self.config["navigation"]["relocalize_after_actions"])

    def as_dict(self):
        return {
            "pose": None if self.pose is None else self.pose.as_dict(),
            "actions_since_localize": self.actions_since_localize,
        }


class MotionController:
    def __init__(self, hardware, state: RobotState, debug=None):
        self.hardware = hardware
        self.state = state
        self.debug = debug

    def run(self, key: str, times_override: Optional[int] = None):
        result = self.hardware.run_action(key, times_override=times_override)
        self.state.apply_action_result(result)
        if self.debug:
            self.debug.event(
                "action",
                key=result.key,
                group=result.group,
                times=result.times,
                elapsed_s=round(result.elapsed_s, 3),
                model_forward_cm=result.model_forward_cm,
                model_lateral_cm=result.model_lateral_cm,
                model_yaw_deg=result.model_yaw_deg,
                ok=result.ok,
                error=result.error,
            )
        return result

    def turn_toward(self, diff_yaw_deg: float):
        tolerance = float(self.state.config["navigation"]["turn_tolerance_deg"])
        if abs(diff_yaw_deg) <= tolerance:
            return None
        large_threshold = float(self.state.config["navigation"].get("large_turn_threshold_deg", 35.0))
        if abs(diff_yaw_deg) >= large_threshold:
            key = "turn_left_large" if diff_yaw_deg > 0 else "turn_right_large"
            actions = self.state.config["motion"]["actions"]
            if key in actions:
                step = abs(float(actions[key].get("yaw_deg", 45.0)))
                cycles = max(1, int(math.floor((abs(diff_yaw_deg) + tolerance) / max(1.0, step))))
                max_cycles = int(self.state.config["navigation"].get("max_large_turn_cycles_per_step", 2))
                cycles = max(1, min(max_cycles, cycles))
                return self.run(key, times_override=cycles)
        if diff_yaw_deg > 0:
            key = "turn_left_fast" if abs(diff_yaw_deg) >= 12.0 else "turn_left_micro"
        else:
            key = "turn_right_fast" if abs(diff_yaw_deg) >= 12.0 else "turn_right_micro"
        step = abs(float(self.state.config["motion"]["actions"][key].get("yaw_deg", 7.5)))
        excess = max(0.0, abs(diff_yaw_deg) - tolerance)
        cycles = max(1, int(math.ceil(excess / max(1.0, step))))
        max_cycles = int(self.state.config["navigation"].get("max_turn_cycles_per_step", 4))
        cycles = max(1, min(max_cycles, cycles))
        return self.run(key, times_override=cycles)

    def forward_cycles_for_distance(self, distance_cm: float) -> int:
        pose = self.state.pose
        if pose is None:
            return 1
        reserve = float(self.state.config["navigation"]["reserve_stop_distance_cm"])
        usable = max(0.0, distance_cm - reserve)
        step = abs(float(self.state.config["motion"]["actions"]["forward_fast"].get("forward_cm", 4.0)))
        cycles = max(1, int(usable / max(1.0, step)))
        if pose.confidence == Confidence.HIGH:
            max_cycles = int(self.state.config["navigation"]["max_forward_cycles_high"])
        elif pose.confidence == Confidence.MEDIUM:
            max_cycles = int(self.state.config["navigation"]["max_forward_cycles_medium"])
        else:
            max_cycles = int(self.state.config["navigation"]["max_forward_cycles_low"])
        return max(1, min(max_cycles, cycles))

    def move_forward(self, distance_cm: float):
        cycles = self.forward_cycles_for_distance(distance_cm)
        return self.run("forward_fast", times_override=cycles)

    def lateral_cycles_for_distance(self, distance_cm: float) -> int:
        pose = self.state.pose
        if pose is None:
            return 1
        step = abs(float(self.state.config["motion"]["actions"]["strafe_left_fast"].get("lateral_cm", 4.0)))
        cycles = max(1, int(abs(distance_cm) / max(1.0, step)))
        if pose.confidence == Confidence.HIGH:
            max_cycles = int(self.state.config["navigation"].get("max_strafe_cycles_high", 6))
        elif pose.confidence == Confidence.MEDIUM:
            max_cycles = int(self.state.config["navigation"].get("max_strafe_cycles_medium", 4))
        else:
            max_cycles = int(self.state.config["navigation"].get("max_strafe_cycles_low", 2))
        return max(1, min(max_cycles, cycles))

    def move_lateral(self, distance_cm: float):
        cycles = self.lateral_cycles_for_distance(distance_cm)
        key = "strafe_left_fast" if distance_cm > 0 else "strafe_right_fast"
        return self.run(key, times_override=cycles)
