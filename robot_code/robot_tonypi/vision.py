#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen candidate detection and crop extraction."""

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .models import RobotPose, ScreenCandidate, TagDetection
from .utils import angle_diff_deg


def _cv2():
    import cv2

    return cv2


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def point_inside_quad(point, quad) -> bool:
    cv = _cv2()
    pts_int = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    return cv.pointPolygonTest(pts_int, (float(point[0]), float(point[1])), False) >= 0


def warp_to_28x28(frame, src_pts):
    cv = _cv2()
    dst_pts = np.array([[0, 0], [27, 0], [27, 27], [0, 27]], dtype=np.float32)
    mat = cv.getPerspectiveTransform(np.array(src_pts, dtype=np.float32), dst_pts)
    return cv.warpPerspective(frame, mat, (28, 28))


class ScreenDetector:
    def __init__(self, config, map_model=None):
        self.cfg = config
        self.map = map_model

    def detect(
        self,
        frame,
        tags: Iterable[TagDetection],
        pose: Optional[RobotPose] = None,
        *,
        extract_crops: bool = True,
    ):
        """Detect screen geometry and bind its left-upper Tag.

        ``extract_crops=False`` is the navigation/localization boundary: it
        returns geometry-only candidates and never creates classifier input.
        Flower crops are produced only by the arrived-target path.
        """
        tags = list(tags)
        candidates = []
        for quad, area, aspect in self._detect_quads(frame):
            reject = self._reject_quad(frame, quad, area, aspect, tags)
            if reject:
                continue
            tag = self._bind_left_upper_tag(quad, tags)
            if tag is None:
                continue
            if not (1 <= int(tag.tag_id) <= 36):
                continue
            if self.map is not None and int(tag.tag_id) not in self.map.screens:
                continue
            if self._center_white_ratio(frame, quad) > float(self.cfg["vision"].get("max_tagged_center_white_ratio", 0.96)):
                continue
            map_score = self._map_score(int(tag.tag_id), pose)
            crop = warp_to_28x28(frame, quad) if extract_crops else None
            candidates.append(
                ScreenCandidate(
                    screen_id=int(tag.tag_id),
                    quad=quad,
                    area=float(area),
                    aspect_ratio=float(aspect),
                    tag=tag,
                    crop_28x28=crop,
                    geometric_score=float(area),
                    map_score=map_score,
                )
            )
        candidates.sort(key=lambda item: (item.map_score, item.geometric_score), reverse=True)
        # Deduplicate by screen_id: each screen has a unique id, keep only the best match.
        seen_ids = set()
        deduped = []
        for cand in candidates:
            if cand.screen_id in seen_ids:
                continue
            seen_ids.add(cand.screen_id)
            deduped.append(cand)
        return deduped[: int(self.cfg["vision"]["max_candidates_per_frame"])]

    def _detect_quads(self, frame):
        cv = _cv2()
        h, w = frame.shape[:2]
        frame_area = h * w
        min_area = float(self.cfg["vision"]["min_screen_area_px"])
        max_area = frame_area * float(self.cfg["vision"]["max_screen_area_ratio"])
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 30, 100)
        edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        out = []
        for cnt in contours:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv.isContourConvex(approx):
                continue
            area = abs(cv.contourArea(approx))
            if not (min_area <= area <= max_area):
                continue
            quad = order_points(approx.reshape(4, 2))
            aspect, side_ratio, width_height_ratio = self._aspect_and_side_ratio(quad)
            if aspect is None:
                continue
            if not (float(self.cfg["vision"]["min_aspect_ratio"]) <= aspect <= float(self.cfg["vision"]["max_aspect_ratio"])):
                continue
            if width_height_ratio > float(self.cfg["vision"].get("max_screen_width_height_ratio", 1.0)):
                continue
            if side_ratio > float(self.cfg["vision"]["max_side_ratio"]):
                continue
            out.append((quad, area, aspect))
        return out

    def _aspect_and_side_ratio(self, quad):
        top_w = np.linalg.norm(quad[1] - quad[0])
        bottom_w = np.linalg.norm(quad[2] - quad[3])
        left_h = np.linalg.norm(quad[3] - quad[0])
        right_h = np.linalg.norm(quad[2] - quad[1])
        sides = [top_w, bottom_w, left_h, right_h]
        if min(sides) < 4.0:
            return None, None, None
        avg_w = (top_w + bottom_w) / 2.0
        avg_h = (left_h + right_h) / 2.0
        aspect = max(avg_w, avg_h) / max(1.0, min(avg_w, avg_h))
        width_height_ratio = avg_w / max(1.0, avg_h)
        side_ratio = max(sides) / max(1.0, min(sides))
        return aspect, side_ratio, width_height_ratio

    def _reject_quad(self, frame, quad, area, aspect, tags) -> str:
        for tag in tags:
            if point_inside_quad(tag.center, quad):
                return "tag_center_inside"
            for corner in tag.corners:
                if point_inside_quad(corner, quad):
                    return "tag_corner_inside"
        return ""

    def _center_white_ratio(self, frame, quad) -> float:
        cv = _cv2()
        dst_pts = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.float32)
        mat = cv.getPerspectiveTransform(np.array(quad, dtype=np.float32), dst_pts)
        warped = cv.warpPerspective(frame, mat, (100, 100))
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        crop = gray[25:75, 25:75]
        white = np.sum(crop >= int(self.cfg["vision"]["white_threshold"]))
        return float(white) / float(max(1, crop.size))

    def _bind_left_upper_tag(self, quad, tags) -> Optional[TagDetection]:
        tl = np.array(quad[0], dtype=np.float64)
        center = np.mean(np.array(quad, dtype=np.float64), axis=0)
        diag = np.linalg.norm(np.array(quad[2], dtype=np.float64) - np.array(quad[0], dtype=np.float64))
        max_px = max(float(self.cfg["vision"]["tag_bind_max_px"]), diag * float(self.cfg["vision"]["tag_bind_diag_ratio"]))
        best = None
        best_dist = float("inf")
        for tag in tags:
            if not (1 <= int(tag.tag_id) <= 36):
                continue
            tc = np.array(tag.center, dtype=np.float64)
            dist = np.linalg.norm(tc - tl)
            if dist > max_px:
                continue
            # Tag must be to the left of the quad center (field physical constraint).
            if tc[0] > center[0]:
                continue
            if dist < best_dist:
                best = tag
                best_dist = dist
        return best

    def _map_score(self, screen_id: int, pose: Optional[RobotPose]) -> float:
        if self.map is None or pose is None or screen_id not in self.map.screens:
            return 0.0
        screen = self.map.screens[screen_id]
        dx = screen.center_xy[0] - pose.x_cm
        dy = screen.center_xy[1] - pose.y_cm
        target_yaw = np.degrees(np.arctan2(dy, dx))
        view_penalty = abs(angle_diff_deg(target_yaw, pose.yaw_deg))
        return max(0.0, 180.0 - view_penalty)

    def annotate(self, frame, candidates: List[ScreenCandidate], tags: Iterable[TagDetection]):
        cv = _cv2()
        out = frame.copy()
        for tag in tags:
            color = (0, 220, 255) if 1 <= int(tag.tag_id) <= 36 else (80, 80, 255)
            cv.polylines(out, [np.int32(tag.corners)], True, color, 2)
            cv.putText(out, str(tag.tag_id), tuple(np.int32(tag.center)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for cand in candidates:
            cv.polylines(out, [np.int32(cand.quad)], True, (0, 255, 0), 2)
            p = np.int32(cand.quad[0])
            cv.putText(
                out,
                "screen {}".format(cand.screen_id),
                (int(p[0]), int(p[1]) - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        return out
