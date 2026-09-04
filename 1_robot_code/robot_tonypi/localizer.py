#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AprilTag detection and pose estimation."""

import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .models import Confidence, RobotPose, TagDetection
from .utils import normalize_angle_deg, now_s


def _cv2():
    import cv2

    return cv2


class AprilTagDetector:
    def __init__(self, family: str = "tag36h11", detect_upscale: float = 1.0):
        self.family = family
        self.detect_upscale = max(1.0, float(detect_upscale))
        self.detector = self._create_detector(family)

    def _create_detector(self, family):
        try:
            import apriltag

            if hasattr(apriltag, "Detector"):
                return apriltag.Detector(apriltag.DetectorOptions(families=family))
            if hasattr(apriltag, "apriltag"):
                return apriltag.apriltag(family)
        except Exception as exc:
            raise RuntimeError("Cannot initialize apriltag detector: {}".format(exc)) from exc
        raise RuntimeError("Unsupported apriltag Python binding")

    def detect(self, gray) -> List[TagDetection]:
        if self.detect_upscale > 1.01:
            cv = _cv2()
            scaled = cv.resize(gray, None, fx=self.detect_upscale, fy=self.detect_upscale, interpolation=cv.INTER_LINEAR)
            raw_tags = self.detector.detect(scaled)
            scale = self.detect_upscale
        else:
            raw_tags = self.detector.detect(gray)
            scale = 1.0
        out = []
        for raw in raw_tags:
            parsed = self._parse_raw(raw, scale=scale)
            if parsed is not None:
                out.append(parsed)
        return out

    def _parse_raw(self, raw, scale: float = 1.0) -> Optional[TagDetection]:
        tag_id = center = corners = None
        if isinstance(raw, dict):
            tag_id = raw.get("id", raw.get("tag_id"))
            center = raw.get("center", raw.get("c"))
            corners = raw.get("corners", raw.get("p"))
            if corners is None and "lb-rb-rt-lt" in raw:
                corners = np.array(raw["lb-rb-rt-lt"], dtype=np.float64)[[3, 2, 1, 0]]
        else:
            tag_id = getattr(raw, "tag_id", getattr(raw, "id", None))
            center = getattr(raw, "center", getattr(raw, "c", None))
            corners = getattr(raw, "corners", getattr(raw, "p", None))
        if tag_id is None or center is None or corners is None:
            try:
                tag_id = raw[1]
                center = raw[6]
                corners = raw[7]
            except Exception:
                return None
        center_arr = np.array(center, dtype=np.float64) / float(scale)
        corners_arr = np.array(corners, dtype=np.float64) / float(scale)
        return TagDetection(int(tag_id), center_arr, corners_arr)


def load_camera_calibration(config) -> Tuple[np.ndarray, np.ndarray]:
    path = config["paths"]["camera_calibration"]
    if path and os.path.exists(path):
        data = np.load(path)
        return np.array(data["mtx_array"], dtype=np.float64), np.array(data["dist_array"], dtype=np.float64).reshape(-1)
    return (
        np.array(config["camera"]["default_matrix"], dtype=np.float64),
        np.array(config["camera"]["default_dist_coeff"], dtype=np.float64).reshape(-1),
    )


class Localizer:
    def __init__(self, tag_poses, config):
        self.tag_poses = tag_poses
        self.cfg = config
        self.cam_matrix, self.dist_coeff = load_camera_calibration(config)
        self.camera_forward_offset_cm = float(config["camera"]["forward_offset_cm"])
        self.min_area = float(config["localization"]["min_tag_area_px"])
        self.edge_margin = float(config["localization"]["edge_margin_px"])
        self.min_id = int(config["localization"]["allowed_min_id"])
        self.max_id = int(config["localization"]["allowed_max_id"])
        self.last_estimation_diagnostics = {}

    def estimate_from_frame(
        self,
        frame,
        tags: Iterable[TagDetection],
        head_pan_angle: float = 100.0,
        annotate: bool = True,
    ) -> Tuple[Optional[RobotPose], object]:
        annotated = frame.copy() if annotate else frame
        best = None
        best_area = -1.0
        detected_ids = []
        candidate_ids = []
        rejected = []
        accepted_tag_id = None
        accepted_tag_area_px = None
        accepted_tag_center_px = None
        for tag in tags:
            tag_id = int(tag.tag_id)
            detected_ids.append(tag_id)
            if not (self.min_id <= int(tag.tag_id) <= self.max_id):
                rejected.append(self.tag_rejection_detail(
                    tag, "id_filter", "id_out_of_range"
                ))
                continue
            candidate_ids.append(tag_id)
            area = self.tag_area(tag)
            if area < best_area:
                rejected.append(self.tag_rejection_detail(
                    tag, "candidate_selection", "lower_area_than_selected"
                ))
                continue
            pose, stage, rejection_reason = self._solve_tag_pose_detailed(
                tag, annotated
            )
            if pose is not None:
                pose.yaw_deg = normalize_angle_deg(pose.yaw_deg - (float(head_pan_angle) - 100.0))
                best = pose
                best_area = area
                accepted_tag_id = tag_id
                accepted_tag_area_px = round(float(area), 1)
                accepted_tag_center_px = [
                    round(float(tag.center[0]), 1),
                    round(float(tag.center[1]), 1),
                ]
            else:
                rejected.append(self.tag_rejection_detail(
                    tag, stage, rejection_reason
                ))
        self.last_estimation_diagnostics = {
            "detected_tag_ids": detected_ids,
            "candidate_localization_tag_ids": candidate_ids,
            "rejected_tags": rejected,
            "accepted_tag_id": accepted_tag_id,
            "accepted_tag_area_px": accepted_tag_area_px,
            "accepted_tag_center_px": accepted_tag_center_px,
            "result": (
                "accepted_visual_pose"
                if best is not None
                else "pose_unavailable_with_tags"
                if detected_ids
                else "no_tag"
            ),
        }
        return best, annotated

    def _solve_tag_pose(self, tag: TagDetection, frame) -> Optional[RobotPose]:
        pose, _, _ = self._solve_tag_pose_detailed(tag, frame)
        return pose

    def tag_rejection_detail(
        self,
        tag: TagDetection,
        stage: str,
        reason: str,
    ) -> dict:
        try:
            area = round(float(self.tag_area(tag)), 1)
        except Exception:
            area = None
        try:
            center = [round(float(tag.center[0]), 1), round(float(tag.center[1]), 1)]
        except Exception:
            center = None
        return {
            "tag_id": int(tag.tag_id),
            "tag_area_px": area,
            "tag_center_px": center,
            "stage": str(stage),
            "reason": str(reason),
        }

    def _solve_tag_pose_detailed(
        self,
        tag: TagDetection,
        frame,
    ) -> Tuple[Optional[RobotPose], str, str]:
        """Run the existing pose math while retaining its real rejection stage."""
        cv = _cv2()
        try:
            ok, reason = self.quality_gate(tag, frame.shape)
        except Exception as exc:
            return None, "corner_geometry", "invalid_corner_geometry:{}".format(
                type(exc).__name__
            )
        color = (0, 220, 0) if ok else (0, 0, 255)
        cv.polylines(frame, [np.int32(tag.corners)], True, color, 2)
        cv.putText(
            frame,
            "ID:{} {}".format(tag.tag_id, reason),
            (int(tag.center[0] - 30), int(tag.center[1])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        if not ok:
            rejection = "edge_margin" if reason == "EDGE" else "too_small"
            return None, "quality_gate", rejection

        key = str(int(tag.tag_id))
        if key not in self.tag_poses:
            return None, "world_position_lookup", "world_position_missing"
        try:
            obj_pts = np.array(self.tag_poses[key][:4], dtype=np.float64)
            img_pts = np.array(tag.corners, dtype=np.float64)
            if obj_pts.shape != (4, 3) or img_pts.shape != (4, 2):
                return None, "corner_geometry", "invalid_corner_geometry"
            success, rvec, tvec = cv.solvePnP(
                obj_pts, img_pts, self.cam_matrix, self.dist_coeff
            )
        except Exception as exc:
            return None, "solve_pnp", "pnp_exception:{}".format(type(exc).__name__)
        if not success:
            return None, "solve_pnp", "pnp_failed"
        if not np.all(np.isfinite(tvec)):
            return None, "pose_vector", "invalid_tvec"
        if not np.all(np.isfinite(rvec)):
            return None, "pose_vector", "invalid_rvec"
        try:
            rmat, _ = cv.Rodrigues(rvec)
        except Exception as exc:
            return None, "rotation", "rodrigues_exception:{}".format(type(exc).__name__)
        if not np.all(np.isfinite(rmat)):
            return None, "rotation", "invalid_rotation"
        cam_pos = -np.dot(rmat.T, tvec)
        heading = np.dot(rmat.T, np.array([[0], [0], [1]], dtype=np.float64))
        yaw = math.degrees(math.atan2(heading[1][0], heading[0][0]))
        camera_xy = np.array([cam_pos[0][0], cam_pos[1][0]], dtype=np.float64)
        forward = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))], dtype=np.float64)
        robot_xy = camera_xy - self.camera_forward_offset_cm * forward
        # Physical field/building acceptance belongs to TaskManager, where the
        # immutable MapModel is available.  Keeping a permissive filter here
        # would otherwise hide out-of-field candidates before that gate.
        return RobotPose(
            x_cm=float(robot_xy[0]),
            y_cm=float(robot_xy[1]),
            yaw_deg=normalize_angle_deg(yaw),
            confidence=Confidence.HIGH,
            source="VISION_TAG_{}".format(tag.tag_id),
            last_update_s=now_s(),
        ), "accepted", "accepted_visual_pose"

    def estimate_tag_world_xy(
        self,
        tag: TagDetection,
        robot_pose: RobotPose,
        head_pan_angle: float = 100.0,
        tag_size_cm: float = 5.0,
    ) -> Optional[Tuple[float, float]]:
        """Estimate world XY of an arbitrary tag using solvePnP with a known tag size.

        Unlike _solve_tag_pose, this does NOT require the tag to be in tag_poses.
        It uses a simple square model centered at origin for obj_pts.
        Returns the tag center position in field coordinates, or None on failure.
        """
        cv = _cv2()
        s = tag_size_cm
        # Square tag centered at origin in the tag's local frame (matches expand_2 layout)
        obj_pts = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], dtype=np.float64)
        img_pts = np.array(tag.corners, dtype=np.float64)
        success, rvec, tvec = cv.solvePnP(obj_pts, img_pts, self.cam_matrix, self.dist_coeff)
        if not success:
            return None
        rmat, _ = cv.Rodrigues(rvec)
        # tvec = position of tag origin in camera frame
        tag_in_cam = tvec.reshape(3)
        # tag center offset from origin: (s/2, s/2, 0)
        tag_center_in_cam = tag_in_cam + rmat @ np.array([s / 2.0, s / 2.0, 0.0])

        # Camera frame: x=right, y=down, z=forward
        # The tag is in front of the camera, so tag_center_in_cam[2] is the forward distance
        forward_cm = float(tag_center_in_cam[2])
        right_cm = float(tag_center_in_cam[0])

        # Convert to robot-relative field offset
        # head_pan_angle=100 is center; >100 = look left, <100 = look right
        camera_yaw_deg = normalize_angle_deg(robot_pose.yaw_deg + (float(head_pan_angle) - 100.0))
        camera_yaw_rad = math.radians(camera_yaw_deg)

        # right_cm is camera-right; in field frame:
        #   camera forward = (cos(camera_yaw), sin(camera_yaw))
        #   camera right = (sin(camera_yaw), -cos(camera_yaw))  ... actually:
        # For yaw measured from +X axis: forward=(cos, sin), left=(-sin, cos)
        # camera right = -left = (sin, -cos)
        dx = forward_cm * math.cos(camera_yaw_rad) + right_cm * math.sin(camera_yaw_rad)
        dy = forward_cm * math.sin(camera_yaw_rad) - right_cm * math.cos(camera_yaw_rad)

        world_x = robot_pose.x_cm + dx
        world_y = robot_pose.y_cm + dy
        return (float(world_x), float(world_y))

    def _pose_in_bounds(self, xy) -> bool:
        return -20.0 <= xy[0] <= self.cfg["map"]["width_cm"] + 20.0 and -20.0 <= xy[1] <= self.cfg["map"]["height_cm"] + 20.0

    def quality_gate(self, tag: TagDetection, frame_shape) -> Tuple[bool, str]:
        h, w = frame_shape[:2]
        cx, cy = tag.center
        if cx < self.edge_margin or cx > w - self.edge_margin or cy < self.edge_margin or cy > h - self.edge_margin:
            return False, "EDGE"
        area = self.tag_area(tag)
        if area < self.min_area:
            return False, "SMALL:{:.0f}".format(area)
        return True, "OK"

    def tag_area(self, tag: TagDetection) -> float:
        return float(abs(_cv2().contourArea(np.array(tag.corners, dtype=np.float32))))
