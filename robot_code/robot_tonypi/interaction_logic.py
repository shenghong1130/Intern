#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure flower-observation and interaction-pose rules.

This module intentionally has no camera, numpy, hardware, or network dependency,
so the safety rules can be tested on a development computer.
"""

import math
from typing import Optional, Tuple

from .models import InteractionPoseCheck, RobotPose, Screen, ScreenStatus, WorkerChangeResult
from .utils import angle_diff_deg, distance_xy, now_s


def build_interaction_geometry(center_xy: Tuple[float, float], normal_xy: Tuple[float, float], cfg: dict) -> dict:
    """Build reader, body target and facing yaw in the screen-local frame."""
    norm = math.hypot(float(normal_xy[0]), float(normal_xy[1]))
    if norm <= 1e-9:
        raise ValueError("screen normal must be non-zero")
    normal = (float(normal_xy[0]) / norm, float(normal_xy[1]) / norm)
    screen_left = (normal[1], -normal[0])
    sensor_left = float(cfg["sensor_left_offset_cm"])
    body_lateral = sensor_left - float(cfg["left_hand_body_offset_cm"])
    interaction_distance = float(cfg["interaction_distance_cm"])
    staging_distance = float(cfg["interaction_staging_distance_cm"])
    reader = (
        float(center_xy[0]) + screen_left[0] * sensor_left,
        float(center_xy[1]) + screen_left[1] * sensor_left,
    )
    interaction = (
        float(center_xy[0]) + normal[0] * interaction_distance + screen_left[0] * body_lateral,
        float(center_xy[1]) + normal[1] * interaction_distance + screen_left[1] * body_lateral,
    )
    staging = (
        float(center_xy[0]) + normal[0] * staging_distance + screen_left[0] * body_lateral,
        float(center_xy[1]) + normal[1] * staging_distance + screen_left[1] * body_lateral,
    )
    normal_yaw = math.degrees(math.atan2(normal[1], normal[0]))
    return {
        "normal_xy": normal,
        "normal_yaw_deg": ((normal_yaw + 180.0) % 360.0) - 180.0,
        "screen_left_tangent_xy": screen_left,
        "reader_xy": reader,
        "interaction_xy": interaction,
        "interaction_staging_xy": staging,
        "interaction_yaw_deg": ((normal_yaw + 360.0) % 360.0) - 180.0,
    }


def store_flower_observation(screen: Screen, target_flower: str, flower: str, confidence: float) -> str:
    """Store a visual result only; never perform or authorize an interaction."""
    screen.last_seen_s = now_s()
    screen.last_classification = flower
    screen.last_confidence = float(confidence)
    if flower == target_flower:
        if screen.status != ScreenStatus.CHANGED:
            screen.status = ScreenStatus.ALREADY_TARGET
        return "changed_verified" if screen.status == ScreenStatus.CHANGED else "already_target_observed"
    screen.status = ScreenStatus.NEEDS_CHANGE
    return "needs_physical_interaction"


def evaluate_interaction_pose(
    screen: Screen,
    pose: Optional[RobotPose],
    target_flower: str,
    cfg: dict,
    worker_id: Optional[int],
    expected_from_flower: Optional[str] = None,
    check_time_s: Optional[float] = None,
) -> InteractionPoseCheck:
    """Central safety gate for distance, body yaw, lateral reader alignment and pose."""
    if pose is None:
        return InteractionPoseCheck(ready=False, pose_valid=False, reasons=["pose_missing"])

    reasons = []
    pose_valid = True
    required_confidence = str(cfg.get("interaction_pose_min_confidence", "HIGH")).upper()
    allowed = {"HIGH"} if required_confidence == "HIGH" else {"HIGH", "MEDIUM"}
    if pose.confidence.value not in allowed:
        pose_valid = False
        reasons.append("pose_confidence_{}".format(pose.confidence.value.lower()))
    check_time = now_s() if check_time_s is None else float(check_time_s)
    max_age = float(cfg.get("interaction_pose_max_age_s", 3.0))
    if pose.last_update_s <= 0.0 or check_time - pose.last_update_s > max_age:
        pose_valid = False
        reasons.append("pose_stale")

    rel_x = pose.x_cm - screen.center_xy[0]
    rel_y = pose.y_cm - screen.center_xy[1]
    distance_cm = rel_x * screen.normal_xy[0] + rel_y * screen.normal_xy[1]
    distance_error = distance_cm - float(cfg["interaction_distance_cm"])
    desired_lateral = float(cfg["sensor_left_offset_cm"]) - float(cfg["left_hand_body_offset_cm"])
    actual_lateral = rel_x * screen.screen_left_tangent_xy[0] + rel_y * screen.screen_left_tangent_xy[1]
    lateral_error = actual_lateral - desired_lateral
    yaw_error = angle_diff_deg(screen.interaction_yaw_deg, pose.yaw_deg)
    target_error = distance_xy(pose.xy(), screen.interaction_xy)

    if abs(distance_error) > float(cfg["interaction_distance_tolerance_cm"]):
        reasons.append("distance")
    if abs(yaw_error) > float(cfg["interaction_yaw_tolerance_deg"]):
        reasons.append("yaw")
    if abs(lateral_error) > float(cfg["interaction_lateral_tolerance_cm"]):
        reasons.append("lateral")
    if screen.last_classification is None:
        reasons.append("flower_unknown")
    elif screen.last_classification == target_flower:
        reasons.append("already_target")
    elif expected_from_flower is not None and screen.last_classification != expected_from_flower:
        reasons.append("flower_changed_since_alignment")
    if worker_id is None:
        reasons.append("worker_mapping_missing")

    return InteractionPoseCheck(
        ready=pose_valid and not reasons,
        pose_valid=pose_valid,
        distance_cm=round(distance_cm, 3),
        distance_error_cm=round(distance_error, 3),
        yaw_error_deg=round(yaw_error, 3),
        lateral_error_cm=round(lateral_error, 3),
        target_error_cm=round(target_error, 3),
        reasons=reasons,
    )


def apply_worker_change_result(screen: Screen, result: WorkerChangeResult) -> bool:
    """Apply a Worker result without ever treating ok=False/exception as changed."""
    screen.attempts += 1
    if result.success:
        screen.status = ScreenStatus.CHANGED
        screen.notes.append("worker_change_success")
        return True
    screen.status = ScreenStatus.NEEDS_CHANGE
    screen.notes.append("worker_change_failed:{}".format(result.error or "ok_false"))
    return False
