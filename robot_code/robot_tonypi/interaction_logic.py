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


def cardinal_surface_from_tag(tag_corners_3d, building_center_xy: Tuple[float, float], plane_epsilon_cm: float = 0.5) -> dict:
    """Quantize a vertical Tag plane to one of the four building faces.

    A fixed-X Tag is on a west/east face; a fixed-Y Tag is on a south/north
    face.  The sign is determined against the center of its four-Tag building.
    """
    points = [tuple(float(value) for value in point[:2]) for point in tag_corners_3d[:4]]
    if len(points) != 4:
        raise ValueError("a screen Tag must provide four world corners")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    center = (sum(xs) / 4.0, sum(ys) / 4.0)
    fixed_x = spread_x <= float(plane_epsilon_cm)
    fixed_y = spread_y <= float(plane_epsilon_cm)
    if fixed_x and (not fixed_y or spread_x <= spread_y):
        if center[0] < float(building_center_xy[0]):
            face, normal, yaw = "WEST", (-1.0, 0.0), 0.0
        else:
            face, normal, yaw = "EAST", (1.0, 0.0), -180.0
    elif fixed_y:
        if center[1] < float(building_center_xy[1]):
            face, normal, yaw = "SOUTH", (0.0, -1.0), 90.0
        else:
            face, normal, yaw = "NORTH", (0.0, 1.0), -90.0
    else:
        raise ValueError(
            "Tag plane is not axis-aligned: spread_x={:.3f}, spread_y={:.3f}".format(
                spread_x, spread_y
            )
        )
    return {
        "face": face,
        "center_xy": center,
        "normal_xy": normal,
        "target_yaw_deg": yaw,
        "spread_x_cm": spread_x,
        "spread_y_cm": spread_y,
    }


def building_centers_from_tags(tag_poses: dict) -> dict:
    """Return four-Tag building centers without changing the stored map data."""
    grouped = {}
    for raw_id, corners in tag_poses.items():
        tag_id = int(raw_id)
        if not 1 <= tag_id <= 36:
            continue
        pts = [tuple(float(value) for value in point[:2]) for point in corners[:4]]
        grouped.setdefault((tag_id - 1) // 4, []).extend(pts)
    return {
        group_id: (
            (min(point[0] for point in points) + max(point[0] for point in points)) / 2.0,
            (min(point[1] for point in points) + max(point[1] for point in points)) / 2.0,
        )
        for group_id, points in grouped.items()
    }


def building_bounds_from_tags(tag_poses: dict) -> dict:
    """Return each four-Tag building's immutable XY boundary."""
    grouped = {}
    for raw_id, corners in tag_poses.items():
        tag_id = int(raw_id)
        if not 1 <= tag_id <= 36:
            continue
        points = [tuple(float(value) for value in point[:2]) for point in corners[:4]]
        grouped.setdefault((tag_id - 1) // 4, []).extend(points)
    return {
        group_id: {
            "x_min": min(point[0] for point in points),
            "x_max": max(point[0] for point in points),
            "y_min": min(point[1] for point in points),
            "y_max": max(point[1] for point in points),
        }
        for group_id, points in grouped.items()
    }


def face_center_from_bounds(bounds: dict, face: str) -> Tuple[float, float]:
    """Return the center of one cardinal face of a rectangular building."""
    center_x = (float(bounds["x_min"]) + float(bounds["x_max"])) / 2.0
    center_y = (float(bounds["y_min"]) + float(bounds["y_max"])) / 2.0
    if face == "WEST":
        return float(bounds["x_min"]), center_y
    if face == "EAST":
        return float(bounds["x_max"]), center_y
    if face == "SOUTH":
        return center_x, float(bounds["y_min"])
    if face == "NORTH":
        return center_x, float(bounds["y_max"])
    raise ValueError("unknown building face: {}".format(face))


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


def evaluate_arrival_geometry(
    screen: Screen,
    pose: Optional[RobotPose],
    cfg: dict,
    check_time_s: Optional[float] = None,
) -> InteractionPoseCheck:
    """Geometry-only 15 cm arrival gate used before flower classification."""
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

    geometry_center = screen.face_center_xy or screen.center_xy
    rel_x = pose.x_cm - geometry_center[0]
    rel_y = pose.y_cm - geometry_center[1]
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
    geometry = evaluate_arrival_geometry(screen, pose, cfg, check_time_s=check_time_s)
    reasons = list(geometry.reasons)
    if screen.last_classification is None:
        reasons.append("flower_unknown")
    elif screen.last_classification == target_flower:
        reasons.append("already_target")
    elif expected_from_flower is not None and screen.last_classification != expected_from_flower:
        reasons.append("flower_changed_since_alignment")
    if worker_id is None:
        reasons.append("worker_id_missing")

    return InteractionPoseCheck(
        ready=geometry.pose_valid and not reasons,
        pose_valid=geometry.pose_valid,
        distance_cm=geometry.distance_cm,
        distance_error_cm=geometry.distance_error_cm,
        yaw_error_deg=geometry.yaw_error_deg,
        lateral_error_cm=geometry.lateral_error_cm,
        target_error_cm=geometry.target_error_cm,
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
