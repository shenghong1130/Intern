#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared dataclasses used by the TonyPi competition controller."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ScreenStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    NEEDS_CHANGE = "NEEDS_CHANGE"
    INTERACTING = "INTERACTING"
    CHANGED = "CHANGED"
    ALREADY_TARGET = "ALREADY_TARGET"
    FAILED = "FAILED"


class MissionState(str, Enum):
    IDLE = "IDLE"
    LOCALIZE = "LOCALIZE"
    SELECT_NEAREST_TARGET = "SELECT_NEAREST_TARGET"
    BUILD_CARDINAL_TARGET_POSE = "BUILD_CARDINAL_TARGET_POSE"
    NAVIGATE_TO_TARGET = "NAVIGATE_TO_TARGET"
    TARGET_DIRECT_APPROACH = "TARGET_DIRECT_APPROACH"
    ARRIVED_AT_TARGET = "ARRIVED_AT_TARGET"
    CONFIRM_TARGET_SCREEN = "CONFIRM_TARGET_SCREEN"
    TARGET_TAG_SCREEN_CONFIRMED = "TARGET_TAG_SCREEN_CONFIRMED"
    FORWARD_3CM = "FORWARD_3CM"
    CAPTURE_TARGET_SCREEN = "CAPTURE_TARGET_SCREEN"
    CLASSIFY_TARGET_FLOWER = "CLASSIFY_TARGET_FLOWER"
    TARGET_ALREADY_CORRECT = "TARGET_ALREADY_CORRECT"
    NEEDS_CHANGE = "NEEDS_CHANGE"
    EXECUTE_CHANGE = "EXECUTE_CHANGE"
    MARK_TARGET_COMPLETE = "MARK_TARGET_COMPLETE"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    MISSION_FAILED = "MISSION_FAILED"


@dataclass
class RobotPose:
    x_cm: float
    y_cm: float
    yaw_deg: float
    confidence: Confidence = Confidence.UNKNOWN
    source: str = "UNKNOWN"
    last_update_s: float = 0.0

    def xy(self) -> Tuple[float, float]:
        return self.x_cm, self.y_cm

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x_cm": self.x_cm,
            "y_cm": self.y_cm,
            "yaw_deg": self.yaw_deg,
            "confidence": self.confidence.value,
            "source": self.source,
            "last_update_s": self.last_update_s,
        }


@dataclass
class Screen:
    screen_id: int
    tag_corners_3d: Any
    center_xy: Tuple[float, float]
    normal_xy: Tuple[float, float]
    normal_yaw_deg: float
    target_xy: Tuple[float, float]
    interaction_xy: Tuple[float, float]
    interaction_yaw_deg: float
    reader_xy: Tuple[float, float]
    screen_left_tangent_xy: Tuple[float, float]
    surface_face: str = "UNKNOWN"
    cardinal_normal_xy: Tuple[float, float] = (0.0, 0.0)
    face_center_xy: Optional[Tuple[float, float]] = None
    tag_front_xy: Optional[Tuple[float, float]] = None
    task_target_xy: Optional[Tuple[float, float]] = None
    task_target_yaw_deg: Optional[float] = None
    worker_id: Optional[int] = None
    status: ScreenStatus = ScreenStatus.UNKNOWN
    attempts: int = 0
    last_seen_s: float = 0.0
    last_classification: Optional[str] = None
    last_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    transit_visible: bool = False
    last_binding_s: float = 0.0

    def done(self) -> bool:
        # ALREADY_TARGET needs no physical transaction, so it is complete for
        # target selection even though completed_count() tracks actual changes.
        return self.status in (ScreenStatus.CHANGED, ScreenStatus.ALREADY_TARGET)

    def terminal(self) -> bool:
        """Return whether this target must no longer be selected."""
        return self.done() or self.status == ScreenStatus.FAILED

    def successful(self) -> bool:
        return self.status == ScreenStatus.CHANGED

    def needs_interaction(self) -> bool:
        return self.status == ScreenStatus.NEEDS_CHANGE and bool(self.last_classification)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "center_xy": list(self.center_xy),
            "normal_xy": list(self.normal_xy),
            "normal_yaw_deg": self.normal_yaw_deg,
            "target_xy": list(self.target_xy),
            "interaction_xy": list(self.interaction_xy),
            "interaction_yaw_deg": self.interaction_yaw_deg,
            "reader_xy": list(self.reader_xy),
            "screen_left_tangent_xy": list(self.screen_left_tangent_xy),
            "surface_face": self.surface_face,
            "cardinal_normal_xy": list(self.cardinal_normal_xy),
            "face_center_xy": None if self.face_center_xy is None else list(self.face_center_xy),
            "tag_front_xy": None if self.tag_front_xy is None else list(self.tag_front_xy),
            "task_target_xy": None if self.task_target_xy is None else list(self.task_target_xy),
            "task_target_yaw_deg": self.task_target_yaw_deg,
            "worker_id": self.worker_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_seen_s": self.last_seen_s,
            "last_classification": self.last_classification,
            "last_confidence": self.last_confidence,
            "transit_visible": self.transit_visible,
            "last_binding_s": self.last_binding_s,
            "notes": self.notes[-5:],
        }


@dataclass
class TagDetection:
    tag_id: int
    center: Any
    corners: Any


@dataclass
class ScreenCandidate:
    screen_id: int
    quad: Any
    area: float
    aspect_ratio: float
    tag: TagDetection
    crop_28x28: Any
    geometric_score: float = 0.0
    map_score: float = 0.0
    reject_reason: str = ""


@dataclass
class ClassificationResult:
    ok: bool
    flower_api: Optional[str] = None
    flower_cn: Optional[str] = None
    confidence: float = 0.0
    class_index: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class TargetVisualConfirmation:
    """One live 17 cm observation binding the locked Tag to its screen."""
    screen_id: int
    tag_id: int
    binding_ok: bool
    captured_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "tag_id": self.tag_id,
            "binding_ok": self.binding_ok,
            "captured_s": self.captured_s,
        }


@dataclass
class VisualAuthorization:
    """Locked arrived-target evidence authorizing one change transaction."""
    screen_id: int
    tag_id: int
    binding_ok: bool
    flower: str
    confidence: float
    captured_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "tag_id": self.tag_id,
            "binding_ok": self.binding_ok,
            "flower": self.flower,
            "confidence": self.confidence,
            "captured_s": self.captured_s,
        }


@dataclass
class InteractionAuthorizationCheck:
    ready: bool
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "reasons": list(self.reasons),
        }


@dataclass
class WorkerChangeResult:
    success: bool
    simulated: bool = False
    worker_id: Optional[int] = None
    response: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class ActionResult:
    key: str
    group: str
    times: int
    elapsed_s: float
    model_forward_cm: float = 0.0
    model_lateral_cm: float = 0.0
    model_yaw_deg: float = 0.0
    imu_yaw_delta_deg: Optional[float] = None
    ok: bool = True
    error: str = ""
