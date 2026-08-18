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
        self.motion_uncertainty = 0.0

    def set_pose(self, pose: RobotPose) -> None:
        self.pose = pose
        self.actions_since_localize = 0
        self.motion_uncertainty = 0.0

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
        actual_cycles = getattr(result, "executed_times", None)
        if actual_cycles is None:
            actual_cycles = int(getattr(result, "times", 0)) if bool(getattr(result, "ok", False)) else 0
        actual_cycles = max(0, int(actual_cycles))
        requested_cycles = max(1, int(getattr(result, "times", actual_cycles or 1)))
        incomplete = not bool(getattr(result, "ok", False)) or actual_cycles < requested_cycles
        if actual_cycles <= 0:
            if incomplete:
                self.pose.confidence = Confidence.LOW
            return
        executed_fraction = min(1.0, actual_cycles / float(requested_cycles))
        yaw_rad = math.radians(self.pose.yaw_deg)
        left_rad = yaw_rad + math.pi / 2.0
        self.pose.x_cm += result.model_forward_cm * executed_fraction * math.cos(yaw_rad)
        self.pose.y_cm += result.model_forward_cm * executed_fraction * math.sin(yaw_rad)
        self.pose.x_cm += result.model_lateral_cm * executed_fraction * math.cos(left_rad)
        self.pose.y_cm += result.model_lateral_cm * executed_fraction * math.sin(left_rad)
        self.pose.yaw_deg = normalize_angle_deg(self.pose.yaw_deg + result.model_yaw_deg * executed_fraction)
        self.pose.source = "DEAD_RECKONING"
        self.pose.last_update_s = now_s()
        self.actions_since_localize += actual_cycles
        nav = self.config["navigation"]
        per_cycle_forward = abs(float(result.model_forward_cm)) / actual_cycles
        per_cycle_lateral = abs(float(result.model_lateral_cm)) / actual_cycles
        per_cycle_yaw = abs(float(result.model_yaw_deg)) / actual_cycles
        if per_cycle_yaw > 1e-6:
            large_turn_threshold = float(nav.get("large_turn_threshold_deg", 35.0))
            uncertainty = float(nav.get(
                "large_turn_uncertainty_per_cycle"
                if per_cycle_yaw >= large_turn_threshold
                else "turn_uncertainty_per_cycle",
                2.6 if per_cycle_yaw >= large_turn_threshold else 2.0,
            ))
        elif per_cycle_lateral > 1e-6:
            uncertainty = float(nav.get("strafe_uncertainty_per_cycle", 1.5))
        elif float(result.model_forward_cm) < -1e-6:
            uncertainty = float(nav.get("reverse_uncertainty_per_cycle", 1.0))
        else:
            uncertainty = float(nav.get("forward_uncertainty_per_cycle", 1.0))
        self.motion_uncertainty += actual_cycles * uncertainty
        threshold = float(nav.get("relocalize_uncertainty_threshold", 6.0))
        action_limit = self.relocalize_action_limit()
        if incomplete or self.motion_uncertainty >= threshold or self.actions_since_localize >= action_limit:
            self.pose.confidence = Confidence.LOW
        elif self.pose.confidence == Confidence.HIGH and self.motion_uncertainty >= threshold * 0.5:
            self.pose.confidence = Confidence.MEDIUM

    def needs_relocalize(self) -> bool:
        if self.pose is None:
            return True
        if self.pose.confidence == Confidence.LOW:
            return True
        if self.motion_uncertainty >= float(
            self.config["navigation"].get("relocalize_uncertainty_threshold", 6.0)
        ):
            return True
        return self.actions_since_localize >= self.relocalize_action_limit()

    def relocalize_action_limit(self) -> int:
        nav = self.config["navigation"]
        if self.pose is None or self.pose.confidence == Confidence.LOW:
            suffix = "low"
        elif self.pose.confidence == Confidence.MEDIUM:
            suffix = "medium"
        else:
            suffix = "high"
        return max(1, int(nav.get(
            "relocalize_after_actions_{}".format(suffix),
            nav.get("relocalize_after_actions", 1),
        )))

    def as_dict(self):
        return {
            "pose": None if self.pose is None else self.pose.as_dict(),
            "actions_since_localize": self.actions_since_localize,
            "motion_uncertainty": round(self.motion_uncertainty, 3),
        }


class MotionController:
    def __init__(self, hardware, state: RobotState, debug=None):
        self.hardware = hardware
        self.state = state
        self.debug = debug

    def run(self, key: str, times_override: Optional[int] = None):
        requested_cycles = int(times_override if times_override is not None else self.state.config["motion"]["actions"][key].get("times", 1))
        result = self.hardware.run_action(key, times_override=times_override)
        self.state.apply_action_result(result)
        actual_cycles = getattr(result, "executed_times", None)
        if actual_cycles is None:
            actual_cycles = int(result.times) if result.ok else 0
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
                requested_action_cycles=requested_cycles,
                actual_action_cycles=int(actual_cycles),
                actions_since_localize=self.state.actions_since_localize,
                motion_uncertainty=round(self.state.motion_uncertainty, 3),
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
                if self.state.pose.confidence == Confidence.HIGH:
                    adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_high", 2))
                elif self.state.pose.confidence == Confidence.MEDIUM:
                    adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_medium", 1))
                else:
                    adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_low", 1))
                max_cycles = min(max_cycles, max(1, adaptive_max))
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
        if self.state.pose.confidence == Confidence.HIGH:
            adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_high", 2))
        elif self.state.pose.confidence == Confidence.MEDIUM:
            adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_medium", 1))
        else:
            adaptive_max = int(self.state.config["navigation"].get("max_turn_cycles_low", 1))
        max_cycles = min(max_cycles, max(1, adaptive_max))
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

    def reverse_cycles_for_distance(self, distance_cm: float) -> int:
        pose = self.state.pose
        if pose is None:
            return 1
        step = abs(float(self.state.config["motion"]["actions"]["back_fast"].get("forward_cm", -2.5)))
        cycles = max(1, int(math.floor(abs(float(distance_cm)) / max(0.1, step))))
        if pose.confidence == Confidence.HIGH:
            max_cycles = int(self.state.config["navigation"].get("max_reverse_cycles_high", 6))
        elif pose.confidence == Confidence.MEDIUM:
            max_cycles = int(self.state.config["navigation"].get("max_reverse_cycles_medium", 3))
        else:
            max_cycles = int(self.state.config["navigation"].get("max_reverse_cycles_low", 1))
        return max(1, min(max_cycles, cycles))

    def move_reverse(self, distance_cm: float):
        cycles = self.reverse_cycles_for_distance(distance_cm)
        return self.run("back_fast", times_override=cycles)

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
