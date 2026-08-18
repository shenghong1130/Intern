#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main competition decision tree."""

import collections
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .classifier import ClassifierClient
from .debug import DebugReporter
from .hardware import RealtimeCamera, TonyPiHardware
from .interaction_client import RobotInteractionClient
from .interaction_logic import (
    apply_worker_change_result,
    build_interaction_geometry,
    building_bounds_from_tags,
    building_centers_from_tags,
    cardinal_surface_from_tag,
    face_center_from_bounds,
    store_flower_observation,
)
from .localizer import AprilTagDetector, Localizer
from .map_model import MapModel, load_tag_positions
from .models import (
    Confidence,
    InteractionAuthorizationCheck,
    MissionState,
    NearWallRecoveryResult,
    RecentBoundFlowerObservation,
    RobotPose,
    Screen,
    ScreenStatus,
    TargetVisualConfirmation,
    VisualAuthorization,
)
from .motion import MotionController, RobotState
from .utils import angle_diff_deg, distance_xy, ensure_dir, normalize_angle_deg, now_s
from .vision import ScreenDetector


def evaluate_turn_progress(
    before_pose: RobotPose,
    visual_pose: RobotPose,
    expected_delta: float,
    target_yaw: Optional[float] = None,
) -> dict:
    """Evaluate a post-turn visual pose without trusting it prematurely."""
    actual_delta = angle_diff_deg(visual_pose.yaw_deg, before_pose.yaw_deg)
    position_delta = distance_xy(before_pose.xy(), visual_pose.xy())
    minimum_progress = max(2.0, abs(float(expected_delta)) * 0.25)
    stale_pose = abs(actual_delta) < 1.0 and position_delta < 1.0
    direction_conflict = (
        abs(float(expected_delta)) >= 5.0
        and abs(actual_delta) >= 2.0
        and float(expected_delta) * actual_delta < 0.0
    )
    no_progress = (
        abs(float(expected_delta)) >= 5.0
        and abs(actual_delta) < minimum_progress
    )
    diff_before = None if target_yaw is None else angle_diff_deg(target_yaw, before_pose.yaw_deg)
    diff_after = None if target_yaw is None else angle_diff_deg(target_yaw, visual_pose.yaw_deg)
    target_improvement = None
    if diff_before is not None and diff_after is not None:
        target_improvement = abs(diff_before) - abs(diff_after)
    reject_visual_pose = stale_pose or direction_conflict or no_progress
    return {
        "before_yaw": float(before_pose.yaw_deg),
        "after_yaw": float(visual_pose.yaw_deg),
        "expected_delta": float(expected_delta),
        "actual_delta": float(actual_delta),
        "position_delta_cm": float(position_delta),
        "minimum_progress_deg": float(minimum_progress),
        "target_yaw": None if target_yaw is None else float(target_yaw),
        "diff_before": diff_before,
        "diff_after": diff_after,
        "target_improvement_deg": target_improvement,
        "suspect_stale_pose": stale_pose,
        "direction_conflict": direction_conflict,
        "turn_no_progress": no_progress,
        "reject_visual_pose": reject_visual_pose,
    }


class TaskManager:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.start_time = time.monotonic()
        self.target_flower = args.target_flower
        self.tag_poses = load_tag_positions(args.load_pos)
        self.map = MapModel(self.tag_poses, config)
        self.configure_cardinal_task_targets()
        self.debug = DebugReporter(config, enabled=args.debug, port=args.debug_port, host=args.debug_host)
        self.hardware = TonyPiHardware(config, dry_run=args.dry_run)
        self.camera = RealtimeCamera(config, dry_run=args.dry_run)
        self.detector = (
            AprilTagDetector(
                config["localization"]["tag_family"],
                detect_upscale=float(config["localization"].get("detector_upscale", 1.0)),
            )
            if not args.dry_run
            else None
        )
        self.localizer = Localizer(self.tag_poses, config)
        self.screen_detector = ScreenDetector(config, self.map)
        self.classifier = ClassifierClient(args.classifier_url, dry_run=args.dry_run)
        team = getattr(args, "team", None) or getattr(args, "robot_name", None)
        robot_id = getattr(args, "robot_id", None) or getattr(args, "robot_name", None)
        self.interaction = RobotInteractionClient(
            team,
            args.robot_secret,
            robot_id,
            config,
            dry_run=args.dry_run,
            skip_change=args.skip_change,
            event_callback=self.debug.event,
            phase_callback=self.on_interaction_phase,
        )
        self.state = RobotState(config)
        self.motion = MotionController(self.hardware, self.state, self.debug)
        self.last_vote_summary = {}
        self.latest_interaction_result = None
        self.recent_interaction_results = []
        self.last_interaction_check = None
        self.interaction_phase = "idle"
        self.left_hand_lifted = False
        self.last_target_plan = {}
        self.mission_state = MissionState.IDLE
        self.current_target_screen_id = None
        self.arrived_at_target = False
        self.classifier_allowed = False
        self.target_visual_confirmation: Optional[TargetVisualConfirmation] = None
        self.visual_authorization: Optional[VisualAuthorization] = None
        self.final_forward_executed = False
        self.target_confirmation_retry_count = 0
        self.target_confirmation_recovery_cycle = 0
        self.last_target_confirmation_diagnostics = {}
        self.transit_bindings = {}
        self.recent_bound_flower_observations: Dict[int, RecentBoundFlowerObservation] = {}
        self.bound_classification_last_attempt_s: Dict[int, float] = {}
        self.last_scan_after_turn_s = 0.0
        self.last_any_tag_seen_s = now_s()
        self.last_localize_success_s = 0.0
        self.last_localization_tag_count = 0
        self.last_localization_quality = "NONE"
        self.last_localization_pose_conflict = False
        self.consecutive_no_tag_scans = 0
        self.consecutive_localize_failures = 0
        self.last_no_tag_recovery_s = 0.0
        self.recovery_count = 0
        self.last_recovery = {}
        self.pending_progress_check = None
        self.no_progress_count = 0
        self.visual_no_progress_count = 0
        self.collision_recovery_pending = False
        self.forward_map_block_count = 0
        self.navigation_noop_count = 0
        self.turn_no_progress_count = 0
        self.turn_navigation_abort = False
        self.turn_progress_failure_start_diff = None
        self.near_wall_recovery_no_progress_count = 0
        self.near_wall_recovery_actions = 0
        self.navigation_stall_signature = None
        self.navigation_stall_count = 0
        self.local_replan_failures = 0
        self.localization_failures = 0
        self.fatal_target_failures = 0
        self.pending_post_action_replan = False
        self.last_navigation_failure_reason = ""
        self.interaction_audit_file = None
        self.interaction_audit_path = ""
        self.open_interaction_audit_log()
        if args.start_x is not None and args.start_y is not None and args.start_yaw is not None:
            self.state.set_manual_pose(args.start_x, args.start_y, args.start_yaw, source="START_ARG")

    def configure_cardinal_task_targets(self) -> None:
        """Build the task's configured stand-off pose from immutable Tag coordinates."""
        building_centers = building_centers_from_tags(self.tag_poses)
        building_bounds = building_bounds_from_tags(self.tag_poses)
        for screen in self.map.screens.values():
            group_id = (int(screen.screen_id) - 1) // 4
            if group_id not in building_centers or group_id not in building_bounds:
                raise ValueError("missing building geometry for screen {}".format(screen.screen_id))
            surface = cardinal_surface_from_tag(
                screen.tag_corners_3d,
                building_centers[group_id],
            )
            face_center = face_center_from_bounds(
                building_bounds[group_id],
                surface["face"],
            )
            geometry = build_interaction_geometry(
                face_center,
                surface["normal_xy"],
                self.config["interaction"],
            )
            screen.surface_face = surface["face"]
            screen.cardinal_normal_xy = surface["normal_xy"]
            screen.face_center_xy = face_center
            screen.normal_xy = geometry["normal_xy"]
            screen.normal_yaw_deg = geometry["normal_yaw_deg"]
            screen.screen_left_tangent_xy = geometry["screen_left_tangent_xy"]
            screen.reader_xy = geometry["reader_xy"]
            screen.target_xy = geometry["target_xy"]
            screen.interaction_xy = geometry["target_xy"]
            screen.interaction_yaw_deg = geometry["interaction_yaw_deg"]
            distance = float(self.config["interaction"]["target_distance_cm"])
            screen.tag_front_xy = (
                face_center[0] + screen.normal_xy[0] * distance,
                face_center[1] + screen.normal_xy[1] * distance,
            )
            screen.task_target_xy = geometry["target_xy"]
            screen.task_target_yaw_deg = screen.interaction_yaw_deg

    def open_interaction_audit_log(self) -> None:
        if self.debug.root is not None:
            root = self.debug.root
            path = root / "interaction_calls.jsonl"
        else:
            tonypi_root = Path(self.config["paths"].get("tonypi_root", "."))
            root = tonypi_root / "competition_logs" if tonypi_root.exists() else Path.cwd() / "competition_logs"
            path = ensure_dir(root) / time.strftime("interaction_calls_%Y%m%d_%H%M%S.jsonl")
        self.interaction_audit_path = str(path)
        try:
            self.interaction_audit_file = path.open("a", encoding="utf-8")
            print("[interaction-log] {}".format(self.interaction_audit_path))
        except Exception as exc:
            self.interaction_audit_file = None
            print("[interaction-log] disabled: {}".format(exc))

    def write_interaction_audit(self, record) -> None:
        if self.interaction_audit_file is None:
            return
        self.interaction_audit_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.interaction_audit_file.flush()

    def on_interaction_phase(self, phase: str, **data) -> None:
        self.interaction_phase = phase
        if phase == "transaction_start":
            self.hardware.set_interaction_active(True)
        elif phase == "left_hand_lifted":
            self.left_hand_lifted = True
        elif phase in ("stand", "transaction_end") and data.get("final", phase == "transaction_end"):
            self.left_hand_lifted = False
        if phase == "transaction_end":
            self.hardware.set_interaction_active(False)
            self.interaction_phase = "idle"
        self.debug.event(phase, left_hand_lifted=self.left_hand_lifted, **data)

    def run(self):
        try:
            self.hardware.center_head()
            if self.args.mode == "localize":
                ok = self.initial_localize()
                self.debug.event("localize_only_done", ok=ok)
                return ok
            if self.args.mode == "harvest":
                if self.state.pose is None and not self.initial_localize():
                    return False
                return self.run_harvest_mode()
            return self.run_mission()
        finally:
            self.close()

    def run_mission(self) -> bool:
        self.set_mission_state(MissionState.LOCALIZE)
        if self.state.pose is None:
            if not self.initial_localize():
                self.debug.event("mission_abort", reason="initial_localize_failed")
                return False
        loops = 0
        while loops < int(self.config["mission"]["max_main_loops"]):
            loops += 1
            if self.time_left_s() <= 0:
                self.debug.event("mission_stop", reason="time_limit")
                break
            if self.target_reached():
                self.set_mission_state(MissionState.MISSION_COMPLETE)
                self.debug.event("mission_success", completed=self.map.completed_count())
                break
            self.set_mission_state(MissionState.SELECT_NEAREST_TARGET)
            target = self.choose_nearest_screen()
            if target is None:
                self.finish_mission_without_available_targets()
                break
            new_target = self.current_target_screen_id != target.screen_id
            self.current_target_screen_id = target.screen_id
            self.arrived_at_target = False
            self.classifier_allowed = False
            self.target_visual_confirmation = None
            self.visual_authorization = None
            self.final_forward_executed = False
            if new_target:
                self.target_confirmation_retry_count = 0
                self.target_confirmation_recovery_cycle = 0
                self.last_target_confirmation_diagnostics = {}
            self.set_mission_state(MissionState.BUILD_CARDINAL_TARGET_POSE)
            self.debug.event(
                "target_selected",
                tag_id=target.screen_id,
                screen_id=target.screen_id,
                surface_face=target.surface_face,
                cardinal_normal_xy=target.cardinal_normal_xy,
                tag_front_xy=target.tag_front_xy,
                task_target_xy=target.task_target_xy,
                task_target_yaw_deg=target.task_target_yaw_deg,
                target_distance_cm=float(self.config["interaction"]["target_distance_cm"]),
                target_final_forward_cm=float(self.config["interaction"]["target_final_forward_cm"]),
                actual_target_offset_cm=self.target_surface_offset_cm(target),
                plan=self.last_target_plan,
            )
            if self.args.dry_run:
                self.debug.event(
                    "dry_run_target_flow_planned",
                    screen_id=target.screen_id,
                    target_xy=target.task_target_xy,
                    target_yaw_deg=target.task_target_yaw_deg,
                    visual_confirmation="target_tag_and_bound_screen_required",
                    final_forward_action="interaction_forward_10cm x1",
                )
            self.set_mission_state(MissionState.NAVIGATE_TO_TARGET)
            ok = self.navigate_to_screen(target)
            if not ok:
                failure_reason = self.last_navigation_failure_reason or "navigation_failed"
                if self.is_retryable_target_failure(failure_reason):
                    self.preserve_current_target(target, failure_reason)
                    self.localize_scan()
                    continue
                self.register_target_failure(target, failure_reason, relocalize=True)
                self.fatal_target_failures += 1
                self.current_target_screen_id = None
                self.arrived_at_target = False
                self.classifier_allowed = False
                self.target_visual_confirmation = None
                self.visual_authorization = None
                continue
            self.arrived_at_target = True
            self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
            if not self.confirm_target_with_visibility_recovery(target):
                return False
            classified = bool(target.last_classification and self.visual_authorization is not None)
            if target.status == ScreenStatus.ALREADY_TARGET:
                classified = True
            elif classified and not self.args.skip_change and not self.execute_final_forward(target):
                self.register_target_failure(target, "target_final_forward_failed")
                self.set_mission_state(MissionState.MARK_TARGET_COMPLETE)
                self.publish_state(target)
                self.current_target_screen_id = None
                self.arrived_at_target = False
                self.classifier_allowed = False
                self.target_visual_confirmation = None
                self.visual_authorization = None
                continue
            elif not classified:
                # Compatibility fallback for dry-run/custom integrations. The
                # normal path classifies at the stand-off before any final move.
                classified = bool(self.classify_after_final_forward(
                    target,
                    allow_without_forward=bool(self.args.skip_change),
                ))
            if not classified:
                self.register_target_failure(target, "post_forward_classification_failed")
            elif target.needs_interaction():
                self.set_mission_state(MissionState.NEEDS_CHANGE)
                attempts_before = target.attempts
                changed = self.process_screen_interaction(target)
                if not changed and target.status == ScreenStatus.NEEDS_CHANGE:
                    if target.attempts == attempts_before:
                        self.register_target_failure(target, "interaction_failed")
                    elif target.attempts >= int(self.config["mission"].get("max_target_attempts", 2)):
                        self.mark_target_terminal_failed(target, "interaction_retry_limit")
                    else:
                        self.debug.event(
                            "target_retry",
                            screen_id=target.screen_id,
                            reason="interaction_failed",
                            attempts=target.attempts,
                            max_attempts=int(self.config["mission"].get("max_target_attempts", 2)),
                        )
            self.set_mission_state(MissionState.MARK_TARGET_COMPLETE)
            self.publish_state(target)
            self.current_target_screen_id = None
            self.arrived_at_target = False
            self.classifier_allowed = False
            self.target_visual_confirmation = None
            self.visual_authorization = None
            self.final_forward_executed = False
        return self.mission_state == MissionState.MISSION_COMPLETE

    def mark_target_terminal_failed(self, target: Screen, reason: str) -> None:
        target.status = ScreenStatus.FAILED
        target.notes.append(reason)
        self.debug.event(
            "target_terminal_failed",
            screen_id=target.screen_id,
            reason=reason,
            attempts=target.attempts,
            max_attempts=int(self.config["mission"].get("max_target_attempts", 2)),
        )

    def register_target_failure(self, target: Screen, reason: str, relocalize: bool = False) -> bool:
        """Record a retryable target failure; return True only when terminal."""
        target.attempts += 1
        target.notes.append(reason)
        max_attempts = max(1, int(self.config["mission"].get("max_target_attempts", 2)))
        if target.attempts >= max_attempts:
            self.mark_target_terminal_failed(target, reason)
            return True
        self.debug.event(
            "target_retry",
            screen_id=target.screen_id,
            reason=reason,
            attempts=target.attempts,
            max_attempts=max_attempts,
        )
        if relocalize:
            self.localize_scan()
        return False

    @staticmethod
    def is_retryable_target_failure(reason: str) -> bool:
        transient = {
            "near_wall_recovery_exhausted",
            "RECOVERY_NO_PROGRESS",
            "no_safe_path_to_exact_target",
            "navigation_step_limit",
            "target_tag_screen_confirmation_failed",
            "target_direct_action_failed",
            "localization_required",
            "localization_failed",
            "visual_progress_temporarily_failed",
        }
        return str(reason) in transient

    def preserve_current_target(self, target: Screen, reason: str) -> None:
        self.current_target_screen_id = int(target.screen_id)
        self.debug.event(
            "current_target_preserved",
            current_target_screen_id=target.screen_id,
            current_target_xy=target.task_target_xy or target.target_xy,
            current_target_yaw=target.task_target_yaw_deg,
            target_preserved=True,
            reason=reason,
            target_attempts=target.attempts,
        )

    def set_mission_state(self, state: MissionState) -> None:
        self.mission_state = state
        self.debug.event("mission_state", state=state.value)

    def finish_mission_without_available_targets(self) -> MissionState:
        """Distinguish successful completion from exhausted failed targets."""
        failed_ids = [
            item.screen_id
            for item in self.map.screens.values()
            if item.status == ScreenStatus.FAILED
        ]
        if failed_ids:
            self.set_mission_state(MissionState.MISSION_FAILED)
            self.debug.event(
                "mission_failed",
                successful=self.map.processed_count(),
                changed=self.map.completed_count(),
                failed=len(failed_ids),
                failed_ids=failed_ids,
            )
        else:
            self.set_mission_state(MissionState.MISSION_COMPLETE)
            self.debug.event(
                "mission_complete",
                processed=self.map.processed_count(),
                changed=self.map.completed_count(),
            )
        return self.mission_state

    def run_harvest_mode(self) -> bool:
        """Navigate directly to one nearest configured target and classify it."""
        self.set_mission_state(MissionState.SELECT_NEAREST_TARGET)
        target = self.choose_nearest_screen()
        if target is None:
            return True
        self.current_target_screen_id = target.screen_id
        self.target_visual_confirmation = None
        self.visual_authorization = None
        self.final_forward_executed = False
        self.target_confirmation_retry_count = 0
        self.target_confirmation_recovery_cycle = 0
        self.last_target_confirmation_diagnostics = {}
        self.set_mission_state(MissionState.BUILD_CARDINAL_TARGET_POSE)
        self.set_mission_state(MissionState.NAVIGATE_TO_TARGET)
        if not self.navigate_to_screen(target):
            return False
        self.arrived_at_target = True
        self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
        if not self.confirm_target_with_visibility_recovery(target):
            return False
        return bool(target.last_classification and self.visual_authorization is not None)

    def time_left_s(self) -> float:
        return float(self.args.time_limit_s or self.config["mission"]["time_limit_s"]) - (time.monotonic() - self.start_time)

    def target_reached(self) -> bool:
        if self.args.max_screens is not None:
            goal = int(self.args.max_screens)
            if self.map.completed_count() < goal:
                return False
            return True
        if bool(self.config["mission"].get("continue_after_target_count", True)):
            return False
        goal = int(self.config["mission"].get("target_success_count", 0))
        if goal <= 0:
            return False
        if self.map.completed_count() < goal:
            return False
        return True

    def initial_localize(self) -> bool:
        if self.args.dry_run and self.state.pose is None:
            self.state.set_manual_pose(150.0, 150.0, 0.0, source="DRY_RUN_DEFAULT")
            return True
        attempts = int(self.config["localization"]["startup_attempts"])
        search_actions = list(self.config["localization"]["startup_search_actions"])
        for attempt in range(1, attempts + 1):
            self.debug.event("initial_localize_attempt", attempt=attempt, max_attempts=attempts)
            if self.localize_scan(reason="initial_localize", allow_pan_search=True):
                return True
            action = search_actions[(attempt - 1) % len(search_actions)]
            result = self.motion.run(action)
            if str(action).startswith("turn_"):
                self.scan_after_turn("initial_localize_search", str(action), result)
                if self.state.pose is not None:
                    return True
        return False

    def assess_visual_localization(self, pose: RobotPose, tags, prior_pose: Optional[RobotPose]) -> dict:
        """Fold Tag quantity/quality and odometry agreement into pose confidence."""
        valid_areas = []
        min_id = int(self.config["localization"].get("allowed_min_id", 1))
        max_id = int(self.config["localization"].get("allowed_max_id", 36))
        min_area = float(self.config["localization"].get("min_tag_area_px", 350.0))
        for tag in tags:
            if not (min_id <= int(tag.tag_id) <= max_id):
                continue
            try:
                area = float(self.localizer.tag_area(tag))
            except Exception:
                area = 0.0
            if area >= min_area:
                valid_areas.append(area)
        best_area = max(valid_areas) if valid_areas else 0.0
        high_area = min_area * float(
            self.config["localization"].get("high_confidence_tag_area_scale", 2.0)
        )
        quality = "HIGH" if len(valid_areas) >= 2 or best_area >= high_area else "MEDIUM"
        if quality == "MEDIUM" and pose.confidence == Confidence.HIGH:
            pose.confidence = Confidence.MEDIUM
        position_conflict_cm = 0.0
        yaw_conflict_deg = 0.0
        conflict = False
        if prior_pose is not None and prior_pose.source == "DEAD_RECKONING":
            position_conflict_cm = distance_xy(prior_pose.xy(), pose.xy())
            yaw_conflict_deg = abs(angle_diff_deg(pose.yaw_deg, prior_pose.yaw_deg))
            conflict = (
                position_conflict_cm > float(self.config["navigation"].get("localization_pose_conflict_distance_cm", 15.0))
                or yaw_conflict_deg > float(self.config["navigation"].get("localization_pose_conflict_yaw_deg", 25.0))
            )
            if conflict:
                pose.confidence = Confidence.LOW
                quality = "CONFLICT"
        self.last_localization_tag_count = len(valid_areas)
        self.last_localization_quality = quality
        self.last_localization_pose_conflict = conflict
        return {
            "localization_tag_count": len(valid_areas),
            "localization_best_tag_area_px": round(best_area, 1),
            "localization_quality": quality,
            "visual_odometry_position_delta_cm": round(position_conflict_cm, 2),
            "visual_odometry_yaw_delta_deg": round(yaw_conflict_deg, 2),
            "visual_odometry_conflict": conflict,
        }

    def localize_scan(
        self,
        reset_turn_watchdog: bool = True,
        *,
        reason: str = "routine",
        allow_pan_search: bool = False,
        pan_angles: Optional[List[float]] = None,
        allow_failure_escalation: bool = True,
    ) -> bool:
        """Localize center-first; routine calls never sweep the head eagerly."""
        saw_any_tag = False
        last_scan_pan = None
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        failure_threshold = max(
            1,
            int(self.config["localization"].get("center_failures_before_pan_search", 2)),
        )
        routine_center_only = bool(
            self.config["localization"].get("routine_center_only", True)
        )
        automatic_escalation = (
            not allow_pan_search
            and allow_failure_escalation
            and routine_center_only
            and int(getattr(self, "consecutive_localize_failures", 0)) + 1 >= failure_threshold
        )
        search_enabled = bool(
            allow_pan_search or automatic_escalation or not routine_center_only
        )
        requested_pans = list(
            pan_angles
            if pan_angles is not None
            else self.config["localization"].get("scan_pan_angles", [center])
        )
        scan_pans = self.unique_pan_angles(
            [center] + (requested_pans if search_enabled else [])
        )
        self.debug.event(
            "pan_search_escalated"
            if allow_pan_search or not routine_center_only
            else "routine_center_localize",
            reason=reason,
            automatic=automatic_escalation,
            pan_angles=scan_pans,
            consecutive_center_failures=int(getattr(self, "consecutive_localize_failures", 0)),
        )
        try:
            for pan_index, pan in enumerate(scan_pans):
                if pan_index == 1 and automatic_escalation:
                    self.debug.event(
                        "pan_search_escalated",
                        reason=reason,
                        automatic=True,
                        pan_angles=scan_pans,
                        consecutive_center_failures=failure_threshold,
                    )
                last_scan_pan = pan
                frame, tags = self.capture_with_tags(pan)
                if frame is None:
                    continue
                if tags:
                    saw_any_tag = True
                    self.update_dynamic_obstacles(tags, pan=pan)
                pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=pan, annotate=True)
                if pose is not None:
                    prior_pose = None if self.state.pose is None else self.copy_pose(self.state.pose)
                    localization_detail = self.assess_visual_localization(pose, tags, prior_pose)
                    self.state.set_pose(pose)
                    annotated = self.observe_transit_bindings(frame, tags, annotated, pan, reason)
                    if reset_turn_watchdog:
                        self.clear_turn_progress_watchdog("normal_relocalize")
                    self.last_localize_success_s = now_s()
                    self.consecutive_localize_failures = 0
                    self.consecutive_no_tag_scans = 0
                    self.evaluate_pending_progress(pose)
                    self.debug.event("pose_update", **pose.as_dict(), head_pan_angle=pan, **localization_detail)
                    self.debug.save_image("latest_annotated.jpg", annotated, force=True)
                    if search_enabled:
                        self.debug.event(
                            "pan_search_stopped_on_success",
                            reason=reason,
                            successful_pan=float(pan),
                            visited_through=float(pan),
                        )
                    self.publish_state()
                    return True
                # Even when this frame cannot produce a new pose, the old
                # dead-reckoning pose may still safely support visual evidence.
                annotated = self.observe_transit_bindings(frame, tags, annotated, pan, reason)
                self.debug.save_image("latest_annotated.jpg", annotated, force=True)
            self.consecutive_localize_failures += 1
            self.localization_failures = getattr(self, "localization_failures", 0) + 1
            if saw_any_tag:
                self.consecutive_no_tag_scans = 0
            else:
                self.consecutive_no_tag_scans += 1
            self.debug.event(
                "localize_failed",
                saw_any_tag=saw_any_tag,
                no_tag_scans=self.consecutive_no_tag_scans,
                failures=self.consecutive_localize_failures,
            )
            return False
        finally:
            self.center_head_after_scan(reason, last_scan_pan)

    def capture_with_tags(self, pan_angle):
        if self.args.dry_run:
            return None, []
        self.hardware.set_head_pan_angle(float(pan_angle))
        frame = self.camera.capture_settled()
        if frame is None:
            return None, []
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)
        if tags:
            self.last_any_tag_seen_s = now_s()
        return frame, tags

    def center_head_after_scan(self, reason: str, last_pan: Optional[float]) -> None:
        if self.args.dry_run or last_pan is None:
            return
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        if abs(float(last_pan) - center) <= 0.5:
            return
        self.debug.event(
            "head_recenter_after_scan",
            reason=reason,
            previous_pan=round(float(last_pan), 1),
            center_pan=round(center, 1),
        )
        try:
            self.hardware.center_head()
        except Exception as exc:
            self.debug.event("head_recenter_failed", reason=reason, error=str(exc))

    def unique_pan_angles(self, pan_angles: List[float]) -> List[float]:
        out = []
        seen = set()
        left = float(self.config["camera"].get("head_left_angle", 145.0))
        right = float(self.config["camera"].get("head_right_angle", 55.0))
        lo, hi = min(left, right), max(left, right)
        for item in pan_angles:
            pan = max(lo, min(hi, float(item)))
            key = int(round(pan))
            if key in seen:
                continue
            seen.add(key)
            out.append(pan)
        return out

    def boundary_safe_pan_angles(
        self,
        pan_angles: List[float],
        reason: str = "",
        pose: Optional[RobotPose] = None,
        emit_event: bool = True,
    ) -> List[float]:
        pans = self.unique_pan_angles(pan_angles)
        pose = self.state.pose if pose is None else pose
        if pose is None or not bool(self.config["navigation"].get("boundary_safe_turn_enabled", True)):
            return pans
        margin = float(self.config["navigation"].get("boundary_trapped_margin_cm", 45.0))
        if self.distance_to_nearest_boundary(pose) > margin:
            return pans
        threshold = float(self.config["navigation"].get("boundary_outward_look_exit_cm", 75.0))
        safe = []
        rejected = []
        for pan in pans:
            camera_yaw = normalize_angle_deg(pose.yaw_deg + (float(pan) - 100.0))
            exit_dist = self.distance_to_field_exit_for_yaw(pose, camera_yaw)
            if exit_dist <= threshold:
                rejected.append({"pan": pan, "camera_yaw": round(camera_yaw, 1), "exit_cm": round(exit_dist, 1)})
            else:
                safe.append(pan)
        if rejected and emit_event:
            self.debug.event(
                "boundary_pan_filtered",
                reason=reason,
                pose=pose.as_dict(),
                kept=safe,
                rejected=rejected,
            )
        return safe

    def pan_angles_for_screen(self, screen_id: int, fallback: Optional[List[float]] = None) -> List[float]:
        fallback = list(self.config["vision"]["harvest_pan_angles"] if fallback is None else fallback)
        pose = self.state.pose
        screen = self.map.screens.get(int(screen_id))
        if pose is None or screen is None:
            return self.unique_pan_angles(fallback)
        dx = screen.center_xy[0] - pose.x_cm
        dy = screen.center_xy[1] - pose.y_cm
        desired_yaw = math.degrees(math.atan2(dy, dx))
        pan = 100.0 + angle_diff_deg(desired_yaw, pose.yaw_deg)
        return self.unique_pan_angles([pan] + fallback)

    def screen_visibility_score(self, screen: Screen, pose: RobotPose, pan_angle: float = 100.0) -> float:
        dx = screen.center_xy[0] - pose.x_cm
        dy = screen.center_xy[1] - pose.y_cm
        dist = math.hypot(dx, dy)
        vision_cfg = self.config["vision"]
        if dist < float(vision_cfg["map_visible_min_distance_cm"]):
            return 0.0
        if dist > float(vision_cfg["map_visible_max_distance_cm"]):
            return 0.0
        target_yaw = math.degrees(math.atan2(dy, dx))
        camera_yaw = normalize_angle_deg(pose.yaw_deg + (float(pan_angle) - 100.0))
        view_angle = abs(angle_diff_deg(target_yaw, camera_yaw))
        half_fov = float(vision_cfg["map_visible_half_fov_deg"])
        if view_angle > half_fov:
            return 0.0
        screen_to_robot_yaw = math.degrees(math.atan2(pose.y_cm - screen.center_xy[1], pose.x_cm - screen.center_xy[0]))
        face_angle = abs(angle_diff_deg(screen_to_robot_yaw, screen.normal_yaw_deg))
        if face_angle > float(vision_cfg["screen_front_max_angle_deg"]):
            return 0.0
        distance_score = max(0.0, 1.0 - dist / max(1.0, float(vision_cfg["map_visible_max_distance_cm"])))
        view_score = max(0.0, 1.0 - view_angle / max(1.0, half_fov))
        face_score = max(0.0, 1.0 - face_angle / max(1.0, float(vision_cfg["screen_front_max_angle_deg"])))
        return 0.45 * view_score + 0.35 * distance_score + 0.20 * face_score

    def predict_visible_screens(
        self,
        pose: Optional[RobotPose] = None,
        pan_angles: Optional[List[float]] = None,
        include_done: bool = False,
    ) -> List[Tuple[int, float, float]]:
        pose = self.state.pose if pose is None else pose
        if pose is None:
            return []
        pan_angles = self.boundary_safe_pan_angles(
            list(self.config["vision"]["harvest_pan_angles"] if pan_angles is None else pan_angles),
            reason="predict_visible_screens",
            pose=pose,
            emit_event=False,
        )
        if not pan_angles:
            return []
        scored = []
        for screen in self.map.screens.values():
            if not include_done and screen.done():
                continue
            best_score = 0.0
            best_pan = pan_angles[0] if pan_angles else 100.0
            for pan in pan_angles:
                score = self.screen_visibility_score(screen, pose, pan)
                if score > best_score:
                    best_score = score
                    best_pan = pan
            if best_score > 0.0:
                scored.append((screen.screen_id, float(best_score), float(best_pan)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def observe_transit_bindings(self, frame, tags, annotated, pan: float, reason: str):
        """Enrich an existing navigation frame with bound-screen classifications."""
        try:
            candidates = self.screen_detector.detect(frame, tags, self.state.pose, extract_crops=True)
            annotated = self.screen_detector.annotate(annotated, candidates, tags)
        except Exception as exc:
            self.debug.event("transit_binding_failed", reason=reason, pan=pan, error=str(exc))
            return annotated
        seen = set()
        timestamp = now_s()
        for cand in candidates:
            screen = self.map.screens.get(cand.screen_id)
            if screen is None:
                continue
            if int(cand.screen_id) != int(cand.tag.tag_id) or not (
                int(self.config["localization"].get("allowed_min_id", 1))
                <= int(cand.screen_id)
                <= int(self.config["localization"].get("allowed_max_id", 36))
            ):
                continue
            seen.add(cand.screen_id)
            screen.transit_visible = True
            screen.last_binding_s = timestamp
            self.transit_bindings[str(cand.screen_id)] = {
                "tag_id": int(cand.tag.tag_id),
                "screen_id": cand.screen_id,
                "pan": float(pan),
                "reason": reason,
                "last_seen_s": round(timestamp, 3),
                "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in cand.quad],
            }
            self.process_bound_screen_candidate(cand, pan=pan, reason=reason, captured_s=timestamp)
        for screen in self.map.screens.values():
            if screen.screen_id not in seen:
                screen.transit_visible = False
        self.debug.event(
            "transit_bindings_updated",
            reason=reason,
            pan=pan,
            bindings=[self.transit_bindings[str(c.screen_id)] for c in candidates if str(c.screen_id) in self.transit_bindings],
            classifier_called=True,
        )
        return annotated

    def process_bound_screen_candidate(
        self,
        candidate,
        *,
        pan: float,
        reason: str,
        captured_s: Optional[float] = None,
    ) -> Optional[RecentBoundFlowerObservation]:
        """Classify one valid Tag==Screen crop without mutating ScreenStatus."""
        screen_id = int(candidate.screen_id)
        tag_id = int(candidate.tag.tag_id)
        timestamp = now_s() if captured_s is None else float(captured_s)
        if screen_id != tag_id or candidate.crop_28x28 is None:
            self.debug.event(
                "bound_flower_observation_failed",
                screen_id=screen_id,
                tag_id=tag_id,
                reason="binding_or_crop_invalid",
            )
            return None
        last_attempts = getattr(self, "bound_classification_last_attempt_s", None)
        if last_attempts is None:
            self.bound_classification_last_attempt_s = {}
            last_attempts = self.bound_classification_last_attempt_s
        min_interval = max(
            0.0,
            float(self.config["vision"].get("bound_classification_min_interval_s", 1.0)),
        )
        last_attempt = float(last_attempts.get(screen_id, -float("inf")))
        if timestamp - last_attempt < min_interval:
            self.debug.event(
                "bound_flower_observation_skipped_rate_limit",
                screen_id=screen_id,
                tag_id=tag_id,
                reason=reason,
                seconds_since_attempt=round(timestamp - last_attempt, 3),
            )
            return None
        last_attempts[screen_id] = timestamp
        try:
            result = self.classifier.classify_crop(candidate.crop_28x28)
        except Exception as exc:
            self.debug.event(
                "bound_flower_observation_failed",
                screen_id=screen_id,
                tag_id=tag_id,
                reason=reason,
                error=str(exc),
            )
            return None
        confidence = float(result.confidence) if result.ok else 0.0
        if (
            not result.ok
            or not result.flower_api
            or confidence < float(self.config["vision"].get("min_confidence", 0.2))
        ):
            self.debug.event(
                "bound_flower_observation_failed",
                screen_id=screen_id,
                tag_id=tag_id,
                reason=reason,
                error=result.error or "low_confidence",
                confidence=round(confidence, 4),
            )
            return None
        observation = RecentBoundFlowerObservation(
            screen_id=screen_id,
            tag_id=tag_id,
            binding_ok=True,
            flower=str(result.flower_api),
            confidence=confidence,
            captured_s=timestamp,
            pan=float(pan),
            reason=reason,
        )
        cache = getattr(self, "recent_bound_flower_observations", None)
        if cache is None:
            self.recent_bound_flower_observations = {}
            cache = self.recent_bound_flower_observations
        previous = cache.get(screen_id)
        if previous is None or observation.captured_s >= previous.captured_s:
            cache[screen_id] = observation
        save_crop = getattr(self.debug, "save_crop", None)
        if callable(save_crop):
            save_crop(screen_id, candidate.crop_28x28, "bound_navigation")
        self.debug.event("bound_flower_observation_cached", **observation.as_dict())
        return observation

    def latest_valid_bound_flower_observation(
        self,
        screen_id: int,
        *,
        current_s: Optional[float] = None,
    ) -> Optional[RecentBoundFlowerObservation]:
        cache = getattr(self, "recent_bound_flower_observations", {})
        observation = cache.get(int(screen_id))
        if observation is None:
            self.debug.event("target_cached_observation_missing", screen_id=int(screen_id))
            return None
        timestamp = now_s() if current_s is None else float(current_s)
        age = timestamp - float(observation.captured_s)
        ttl = float(self.config["vision"].get("bound_classification_cache_ttl_s", 15.0))
        valid = (
            0.0 <= age <= ttl
            and observation.binding_ok
            and int(observation.screen_id) == int(screen_id)
            and int(observation.tag_id) == int(screen_id)
            and observation.confidence >= float(self.config["vision"].get("min_confidence", 0.2))
        )
        if not valid:
            self.debug.event(
                "target_cached_observation_expired",
                screen_id=int(screen_id),
                cache_age_s=round(age, 3),
                ttl_s=ttl,
            )
            return None
        self.debug.event(
            "target_cached_observation_selected",
            screen_id=int(screen_id),
            cache_age_s=round(age, 3),
            classification_captured_s=observation.captured_s,
        )
        return observation

    def adopt_cached_target_observation(
        self,
        screen: Screen,
        observation: RecentBoundFlowerObservation,
        *,
        current_tag_seen_s: float,
        source: str = "recent_bound_cache",
    ) -> bool:
        """Convert independent cache evidence only after a live target Tag check."""
        if not (
            self.current_target_screen_id == screen.screen_id
            and int(screen.worker_id or screen.screen_id) == int(screen.screen_id)
            and observation.screen_id == screen.screen_id
            and observation.tag_id == screen.screen_id
            and observation.binding_ok
        ):
            return False
        age = max(0.0, float(current_tag_seen_s) - float(observation.captured_s))
        self.target_visual_confirmation = TargetVisualConfirmation(
            screen_id=screen.screen_id,
            tag_id=screen.screen_id,
            binding_ok=True,
            captured_s=float(current_tag_seen_s),
            source=source,
            classification_captured_s=float(observation.captured_s),
            current_tag_seen_s=float(current_tag_seen_s),
            cache_age_s=age,
        )
        self.visual_authorization = VisualAuthorization(
            screen_id=screen.screen_id,
            tag_id=screen.screen_id,
            binding_ok=True,
            flower=observation.flower,
            confidence=observation.confidence,
            captured_s=observation.captured_s,
            source=source,
            cache_age_s=age,
            current_tag_seen_s=float(current_tag_seen_s),
        )
        self.record_flower_observation(screen, observation.flower, observation.confidence)
        self.debug.event(
            "target_cached_authorization_created",
            **self.visual_authorization.as_dict(),
            classification_captured_s=observation.captured_s,
        )
        return True

    def record_flower_observation(self, screen: Screen, flower: str, confidence: float, vote_entry=None) -> None:
        """Persist a reliable visual result without performing physical interaction."""
        decision = store_flower_observation(screen, self.target_flower, flower, confidence)
        if flower == self.target_flower:
            self.debug.event("already_target", screen_id=screen.screen_id, flower=flower, status=screen.status.value)
        else:
            self.debug.event(
                "screen_needs_change",
                screen_id=screen.screen_id,
                from_flower=flower,
                to_flower=self.target_flower,
                confidence=round(float(confidence), 4),
                interaction_xy=screen.interaction_xy,
                interaction_yaw_deg=screen.interaction_yaw_deg,
                worker_id=screen.worker_id,
            )
        if vote_entry is not None:
            vote_entry["decision"] = decision
            vote_entry["interaction_requested"] = False

    def classifier_gate_open(self, screen: Screen) -> bool:
        return bool(
            self.arrived_at_target
            and self.current_target_screen_id == screen.screen_id
            and self.mission_state in (
                MissionState.ARRIVED_AT_TARGET,
                MissionState.CONFIRM_TARGET_SCREEN,
                MissionState.TARGET_VISIBILITY_RECOVERY,
                MissionState.CAPTURE_TARGET_SCREEN,
                MissionState.CLASSIFY_TARGET_FLOWER,
            )
        )

    def capture_locked_target_candidate(self, screen: Screen, *, extract_crops: bool, reason: str):
        """Capture one centered frame and return only the locked Tag-screen pair."""
        pan = float(self.config["camera"].get("head_center_angle", 100.0))
        frame, tags = self.capture_with_tags(pan)
        detected_tag_ids = [int(tag.tag_id) for tag in tags if 1 <= int(tag.tag_id) <= 36]
        if frame is None:
            self.last_target_confirmation_diagnostics = {
                "screen_id": screen.screen_id,
                "target_tag_detected": False,
                "detected_tag_ids": detected_tag_ids,
                "screen_candidate_count": 0,
                "matched_screen_count": 0,
                "failure_reason": "capture_failed",
            }
            self.debug.event("target_tag_and_screen_confirmed", confirmed=False, **self.last_target_confirmation_diagnostics)
            return None
        target_tags = [
            tag
            for tag in tags
            if 1 <= int(tag.tag_id) <= 36 and int(tag.tag_id) == int(screen.screen_id)
        ]
        if not target_tags:
            self.last_target_confirmation_diagnostics = {
                "screen_id": screen.screen_id,
                "target_tag_detected": False,
                "detected_tag_ids": detected_tag_ids,
                "screen_candidate_count": 0,
                "matched_screen_count": 0,
                "failure_reason": "target_tag_missing",
            }
            self.debug.event("target_tag_and_screen_confirmed", confirmed=False, **self.last_target_confirmation_diagnostics)
            return None
        candidates = self.screen_detector.detect(
            frame,
            tags,
            self.state.pose,
            extract_crops=extract_crops,
        )
        annotated = self.screen_detector.annotate(frame, candidates, tags)
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        matches = [
            candidate
            for candidate in candidates
            if int(candidate.screen_id) == int(screen.screen_id)
            and int(candidate.tag.tag_id) == int(screen.screen_id)
        ]
        if not matches:
            self.last_target_confirmation_diagnostics = {
                "screen_id": screen.screen_id,
                "target_tag_detected": True,
                "detected_tag_ids": detected_tag_ids,
                "screen_candidate_count": len(candidates),
                "matched_screen_count": 0,
                "failure_reason": "target_screen_binding_missing",
            }
            self.debug.event("target_tag_and_screen_confirmed", confirmed=False, **self.last_target_confirmation_diagnostics)
            return None
        self.last_target_confirmation_diagnostics = {
            "screen_id": screen.screen_id,
            "target_tag_detected": True,
            "detected_tag_ids": detected_tag_ids,
            "screen_candidate_count": len(candidates),
            "matched_screen_count": len(matches),
            "failure_reason": "",
        }
        self.debug.event(
            "target_tag_and_screen_confirmed",
            tag_id=int(matches[0].tag.tag_id),
            confirmed=True,
            reason=reason,
            **self.last_target_confirmation_diagnostics,
        )
        return matches[0]

    def confirm_target_tag_now(self, screen: Screen) -> bool:
        """Confirm the locked Tag live with a strict center/left/right budget."""
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        pans = self.unique_pan_angles([center, 130.0, 70.0])
        self._last_target_live_frame = None
        self._last_target_live_tags = []
        self._last_target_live_pan = center
        last_pan = None
        try:
            for pan in pans:
                last_pan = pan
                frame, tags = self.capture_with_tags(pan)
                if frame is None:
                    continue
                detected = [int(tag.tag_id) for tag in tags]
                if int(screen.screen_id) in detected:
                    seen_s = now_s()
                    self._last_target_live_frame = frame
                    self._last_target_live_tags = tags
                    self._last_target_live_pan = float(pan)
                    self._last_target_tag_seen_s = seen_s
                    self.debug.event(
                        "target_tag_live_confirmed",
                        screen_id=screen.screen_id,
                        tag_id=screen.screen_id,
                        pan=float(pan),
                        current_tag_seen_s=seen_s,
                    )
                    return True
            self.debug.event(
                "target_tag_live_missing",
                screen_id=screen.screen_id,
                pan_angles=pans,
            )
            return False
        finally:
            self.center_head_after_scan("confirm_target_tag_now", last_pan)

    def bounded_fresh_target_observation(
        self,
        screen: Screen,
    ) -> Optional[RecentBoundFlowerObservation]:
        """Use at most the configured finite target frames to create fresh evidence."""
        self.debug.event("target_fresh_fallback_started", screen_id=screen.screen_id)
        max_retries = max(1, min(3, int(
            self.config["interaction"].get("target_confirmation_max_retries", 3)
        )))
        retry_interval = max(0.0, float(
            self.config["interaction"].get("target_confirmation_retry_interval_s", 0.5)
        ))
        frames = []
        first_frame = getattr(self, "_last_target_live_frame", None)
        if first_frame is not None:
            frames.append((
                first_frame,
                list(getattr(self, "_last_target_live_tags", [])),
                float(getattr(self, "_last_target_live_pan", 100.0)),
            ))
        for attempt in range(max_retries):
            if attempt >= len(frames):
                frame, tags = self.capture_with_tags(
                    float(self.config["camera"].get("head_center_angle", 100.0))
                )
                frames.append((frame, tags, float(self.config["camera"].get("head_center_angle", 100.0))))
            frame, tags, pan = frames[attempt]
            if frame is not None and any(int(tag.tag_id) == screen.screen_id for tag in tags):
                try:
                    candidates = self.screen_detector.detect(
                        frame, tags, self.state.pose, extract_crops=True
                    )
                    matches = [
                        candidate for candidate in candidates
                        if int(candidate.screen_id) == screen.screen_id
                        and int(candidate.tag.tag_id) == screen.screen_id
                        and candidate.crop_28x28 is not None
                    ]
                    if matches:
                        observation = self.process_bound_screen_candidate(
                            matches[0],
                            pan=pan,
                            reason="target_fresh_fallback",
                            captured_s=now_s(),
                        )
                        if observation is not None:
                            self.target_confirmation_retry_count = 0
                            return observation
                except Exception as exc:
                    self.debug.event(
                        "bound_flower_observation_failed",
                        screen_id=screen.screen_id,
                        reason="target_fresh_fallback",
                        error=str(exc),
                    )
            if attempt + 1 < max_retries and retry_interval > 0.0:
                time.sleep(retry_interval)
            self.target_confirmation_retry_count = attempt + 1
        self.debug.event("target_fresh_fallback_failed", screen_id=screen.screen_id)
        return None

    def confirm_target_tag_and_screen(self, screen: Screen) -> bool:
        """Confirm live identity, then adopt recent or finitely refreshed evidence."""
        if not self.classifier_gate_open(screen):
            return False
        self.set_mission_state(MissionState.CONFIRM_TARGET_SCREEN)
        if self.args.dry_run:
            self.target_visual_confirmation = TargetVisualConfirmation(
                screen_id=screen.screen_id,
                tag_id=screen.screen_id,
                binding_ok=True,
                captured_s=now_s(),
                source="dry_run",
            )
            self.debug.event(
                "dry_run_target_tag_screen_confirmation",
                **self.target_visual_confirmation.as_dict(),
            )
            self.target_confirmation_retry_count = 0
        else:
            if not self.confirm_target_tag_now(screen):
                return False
            seen_s = float(getattr(self, "_last_target_tag_seen_s", now_s()))
            observation = self.latest_valid_bound_flower_observation(
                screen.screen_id,
                current_s=seen_s,
            )
            if observation is None:
                observation = self.bounded_fresh_target_observation(screen)
                source = "fresh_target_observation"
                if observation is not None:
                    seen_s = max(seen_s, float(observation.captured_s))
            else:
                source = "recent_bound_cache"
            if observation is None or not self.adopt_cached_target_observation(
                screen,
                observation,
                current_tag_seen_s=seen_s,
                source=source,
            ):
                return False
            self.target_confirmation_retry_count = 0
        self.set_mission_state(MissionState.TARGET_TAG_SCREEN_CONFIRMED)
        self.publish_state(screen)
        return True

    def target_surface_offset_cm(self, screen: Screen) -> Optional[float]:
        if screen.face_center_xy is None or screen.task_target_xy is None:
            return None
        normal = screen.cardinal_normal_xy
        if math.hypot(normal[0], normal[1]) < 0.5:
            normal = screen.normal_xy
        return round(
            (screen.task_target_xy[0] - screen.face_center_xy[0]) * normal[0]
            + (screen.task_target_xy[1] - screen.face_center_xy[1]) * normal[1],
            3,
        )

    def recover_target_visibility(self, screen: Screen, cycle: int) -> bool:
        """Perform one bounded local correction; never re-enter full navigation."""
        self.set_mission_state(MissionState.TARGET_VISIBILITY_RECOVERY)
        self.preserve_current_target(screen, "target_visibility_recovery")
        self.debug.event(
            "target_visibility_recovery_started",
            screen_id=screen.screen_id,
            target_confirmation_recovery_cycle=cycle,
            target_preserved=True,
        )
        if not self.localize_scan(
            reason="target_visibility_recovery",
            allow_failure_escalation=False,
        ) or self.state.pose is None:
            self.debug.event(
                "target_visibility_recovery_action",
                screen_id=screen.screen_id,
                action="relocalize_only",
                ok=False,
            )
            return False
        pose = self.state.pose
        normal = screen.cardinal_normal_xy
        if math.hypot(normal[0], normal[1]) < 0.5:
            normal = screen.normal_xy
        desired = float(self.config["interaction"]["target_distance_cm"])
        current_offset = (
            (pose.x_cm - screen.face_center_xy[0]) * normal[0]
            + (pose.y_cm - screen.face_center_xy[1]) * normal[1]
        ) if screen.face_center_xy is not None else desired
        tolerance = float(self.config["navigation"].get("target_arrival_radius_cm", 4.0))
        action = "relocalize_only"
        result = None
        if current_offset < desired - tolerance and self.recovery_translation_clear(pose, forward_cm=-5.0):
            action = "back_fast"
            result = self.motion.run(action, times_override=1)
        else:
            _, lateral_cm = self.local_vector_to(pose, screen.task_target_xy or screen.target_xy)
            if abs(lateral_cm) > tolerance:
                action = "strafe_left_fast" if lateral_cm > 0.0 else "strafe_right_fast"
                model_lateral = float(self.config["motion"]["actions"][action].get("lateral_cm", 0.0))
                if self.recovery_translation_clear(pose, lateral_cm=model_lateral):
                    result = self.motion.run(action, times_override=1)
                else:
                    action = "relocalize_only"
        self.debug.event(
            "target_visibility_recovery_action",
            screen_id=screen.screen_id,
            target_confirmation_recovery_cycle=cycle,
            action=action,
            actual_offset_cm=round(current_offset, 2),
            ok=True if result is None else bool(result.ok),
            target_preserved=True,
        )
        if result is not None and not result.ok:
            return False
        if result is not None:
            self.hardware.center_head()
            if not self.localize_scan(
                reason="target_visibility_recovery_after_action",
                allow_failure_escalation=False,
            ):
                return False
        self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
        return True

    def confirm_target_with_visibility_recovery(self, screen: Screen) -> bool:
        """Confirm the locked target without returning to target selection on failure."""
        max_cycles = max(0, int(self.config["interaction"].get("target_confirmation_recovery_max_cycles", 2)))
        self.target_confirmation_recovery_cycle = 0
        self.arrived_at_target = True
        self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
        if self.confirm_target_tag_and_screen(screen):
            return True
        for cycle in range(1, max_cycles + 1):
            self.target_confirmation_recovery_cycle = cycle
            recovered = self.recover_target_visibility(screen, cycle)
            self.arrived_at_target = bool(recovered)
            if not recovered:
                continue
            self.arrived_at_target = True
            self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
            if self.confirm_target_tag_and_screen(screen):
                return True
        screen.attempts += 1
        self.last_navigation_failure_reason = "target_screen_confirmation_unresolved"
        self.classifier_allowed = False
        self.target_visual_confirmation = None
        self.visual_authorization = None
        self.preserve_current_target(screen, self.last_navigation_failure_reason)
        self.set_mission_state(MissionState.MISSION_FAILED)
        hardware = getattr(self, "hardware", None)
        stop = getattr(hardware, "stop", None)
        if callable(stop):
            stop()
        self.debug.event(
            "target_screen_confirmation_unresolved",
            screen_id=screen.screen_id,
            target_confirmation_retry_count=self.target_confirmation_retry_count,
            target_confirmation_max_retries=int(self.config["interaction"].get("target_confirmation_max_retries", 3)),
            target_confirmation_recovery_cycle=self.target_confirmation_recovery_cycle,
            final_forward_executed=self.final_forward_executed,
            target_preserved=True,
        )
        self.publish_state(screen)
        return False

    def classify_after_final_forward(self, screen: Screen, *, allow_without_forward: bool = False) -> int:
        """Capture once and classify after the dedicated final 10 cm motion."""
        confirmation = self.target_visual_confirmation
        if (
            confirmation is None
            or confirmation.screen_id != int(screen.screen_id)
            or confirmation.tag_id != int(screen.screen_id)
            or not confirmation.binding_ok
        ):
            self.debug.event("classifier_gate_blocked", screen_id=screen.screen_id, reason="target_confirmation_missing")
            return 0
        if not self.final_forward_executed and not allow_without_forward:
            self.debug.event("classifier_gate_blocked", screen_id=screen.screen_id, reason="final_forward_not_completed")
            return 0
        if self.args.dry_run:
            self.debug.event(
                "dry_run_post_forward_classification_planned",
                screen_id=screen.screen_id,
                classifier_called=False,
            )
            return 0
        self.classifier_allowed = True
        self.set_mission_state(MissionState.CAPTURE_TARGET_SCREEN)
        summary = {
            "started_s": round(now_s(), 3),
            "reason": "single_frame_after_final_forward",
            "target_screen_id": screen.screen_id,
            "target_tag_id": screen.screen_id,
            "vote_frames": 1,
            "min_votes": 1,
            "min_confidence": float(self.config["vision"]["min_confidence"]),
            "screens": {},
        }
        entry = self._vote_entry(summary, screen.screen_id)
        try:
            candidate = self.capture_locked_target_candidate(
                screen,
                extract_crops=True,
                reason="after_final_forward_10cm",
            )
            if candidate is None or candidate.crop_28x28 is None:
                entry["decision"] = "post_forward_capture_failed"
                return 0
            self.set_mission_state(MissionState.CLASSIFY_TARGET_FLOWER)
            result = self.classifier.classify_crop(candidate.crop_28x28)
            self.debug.save_crop(screen.screen_id, candidate.crop_28x28, "post_forward_10cm")
            confidence = float(result.confidence) if result.ok else 0.0
            entry["observations"].append(
                {
                    "ok": bool(result.ok),
                    "flower": result.flower_api if result.ok else None,
                    "confidence": round(confidence, 4),
                    "error": result.error if not result.ok else "",
                    "tag_id": int(candidate.tag.tag_id),
                    "screen_id": int(candidate.screen_id),
                    "binding_ok": True,
                }
            )
            if not result.ok or confidence < float(self.config["vision"]["min_confidence"]):
                entry["decision"] = "classification_failed_or_low_confidence"
                self.debug.event(
                    "target_classification_failed",
                    screen_id=screen.screen_id,
                    reason=result.error or "low_confidence",
                    confidence=round(confidence, 4),
                )
                return 0
            flower = str(result.flower_api)
            entry["votes"][flower] = {"count": 1, "confidences": [round(confidence, 4)]}
            entry["best"] = {"flower": flower, "count": 1, "avg_confidence": round(confidence, 4)}
            self.visual_authorization = VisualAuthorization(
                screen_id=int(candidate.screen_id),
                tag_id=int(candidate.tag.tag_id),
                binding_ok=True,
                flower=flower,
                confidence=confidence,
                captured_s=now_s(),
            )
            self.debug.event("target_visual_authorized", **self.visual_authorization.as_dict())
            self.record_flower_observation(screen, flower, confidence, entry)
            self.set_mission_state(
                MissionState.TARGET_ALREADY_CORRECT
                if flower == self.target_flower
                else MissionState.NEEDS_CHANGE
            )
            return 1
        finally:
            self.classifier_allowed = False
            self.last_vote_summary = summary
            self.publish_state(screen)

    def scan_after_turn(
        self,
        reason: str,
        action_key: str = "",
        action_result=None,
        before_pose: Optional[RobotPose] = None,
        target_yaw: Optional[float] = None,
    ) -> dict:
        outcome = {
            "localized": False,
            "accepted": False,
            "turn_no_progress": False,
            "direction_conflict": False,
            "suspect_stale_pose": False,
        }
        if self.args.dry_run or not bool(self.config["vision"].get("scan_after_turn_enabled", True)):
            return outcome
        if self.time_left_s() <= 0:
            return outcome
        t = now_s()
        min_interval = float(self.config["vision"].get("scan_after_turn_min_interval_s", 1.0))
        watchdog_scan = action_result is not None and before_pose is not None
        if not watchdog_scan and t - self.last_scan_after_turn_s < min_interval:
            return outcome
        self.last_scan_after_turn_s = t
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        frame, tags = self.capture_with_tags(center)
        if frame is None:
            self.debug.event("scan_after_turn_failed", reason=reason, action_key=action_key, error="capture_failed")
            return outcome
        pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=center, annotate=True)
        localized = pose is not None
        outcome["localized"] = localized
        if pose is not None:
            progress = None
            if watchdog_scan:
                progress = evaluate_turn_progress(
                    before_pose,
                    pose,
                    float(action_result.model_yaw_deg),
                    target_yaw,
                )
                outcome.update(progress)
                if progress["suspect_stale_pose"]:
                    self.debug.event("suspect_stale_pose_after_turn", reason=reason, action_key=action_key, **progress)
                if progress["direction_conflict"]:
                    self.debug.event("turn_direction_conflict", reason=reason, action_key=action_key, **progress)
                if progress["turn_no_progress"]:
                    self.debug.event("turn_no_progress", reason=reason, action_key=action_key, **progress)
            if progress is not None and progress["reject_visual_pose"]:
                # MotionController has already applied dead reckoning.  Keep it
                # instead of replacing it with a stale/contradictory frame.
                if self.state.pose is not None:
                    self.state.pose.confidence = Confidence.LOW
                self.debug.event(
                    "scan_after_turn_pose_rejected",
                    reason=reason,
                    action_key=action_key,
                    kept_pose=None if self.state.pose is None else self.state.pose.as_dict(),
                    visual_pose=pose.as_dict(),
                    **progress
                )
            else:
                localization_detail = self.assess_visual_localization(pose, tags, None)
                self.state.set_pose(pose)
                outcome["accepted"] = True
                self.last_localize_success_s = now_s()
                self.consecutive_localize_failures = 0
                self.consecutive_no_tag_scans = 0
                self.evaluate_pending_progress(pose)
                self.debug.event(
                    "pose_update",
                    **pose.as_dict(),
                    head_pan_angle=center,
                    reason="scan_after_turn",
                    **localization_detail
                )
        annotated = self.observe_transit_bindings(frame, tags, annotated, center, "scan_after_turn:" + reason)
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        self.debug.event(
            "scan_after_turn_done",
            reason=reason,
            action_key=action_key,
            localized=localized,
            tag_count=len(tags),
            bindings=len(self.transit_bindings),
            classifier_called=True,
        )
        self.publish_state()
        outcome["bindings"] = len(self.transit_bindings)
        return outcome

    def _vote_entry(self, vote_summary, screen_id: int):
        key = str(int(screen_id))
        if key not in vote_summary["screens"]:
            vote_summary["screens"][key] = {
                "screen_id": int(screen_id),
                "observations": [],
                "votes": {},
                "best": None,
                "decision": "collecting",
            }
        return vote_summary["screens"][key]

    def worker_id_for_screen(self, screen: Screen) -> int:
        """Competition numbering is identical: Tag ID == screen ID == Worker ID."""
        return int(screen.screen_id)

    def visual_authorization_check(
        self,
        screen: Screen,
        expected_from_flower: Optional[str] = None,
    ) -> InteractionAuthorizationCheck:
        """Validate the locked target visual evidence without reading pose/camera."""
        reasons = []
        confirmation = getattr(self, "target_visual_confirmation", None)
        if confirmation is None:
            reasons.append("target_confirmation_missing")
        else:
            if confirmation.screen_id != int(screen.screen_id):
                reasons.append("confirmation_screen_mismatch")
            if confirmation.tag_id != int(screen.screen_id):
                reasons.append("confirmation_tag_mismatch")
            if not confirmation.binding_ok:
                reasons.append("confirmation_binding_missing")
        authorization = self.visual_authorization
        if authorization is None:
            reasons.append("visual_authorization_missing")
        else:
            if authorization.screen_id != int(screen.screen_id):
                reasons.append("authorization_screen_mismatch")
            if authorization.tag_id != int(screen.screen_id):
                reasons.append("authorization_tag_mismatch")
            if not authorization.binding_ok:
                reasons.append("authorization_binding_missing")
            if expected_from_flower is not None and authorization.flower != expected_from_flower:
                reasons.append("authorization_flower_mismatch")
        if self.current_target_screen_id != screen.screen_id:
            reasons.append("target_lock_mismatch")
        if not self.arrived_at_target:
            reasons.append("target_not_arrived")
        if not screen.last_classification:
            reasons.append("flower_unknown")
        elif screen.last_classification == self.target_flower:
            reasons.append("already_target")
        elif expected_from_flower is not None and screen.last_classification != expected_from_flower:
            reasons.append("flower_changed_since_capture")
        return InteractionAuthorizationCheck(
            ready=not reasons,
            reasons=reasons,
        )

    def navigate_directly_to_target(self, screen: Screen) -> bool:
        """Navigate once to the configured endpoint and finish its cardinal yaw."""
        target_xy = screen.task_target_xy or screen.target_xy
        return self.navigate_to_xy(
            target_xy,
            reason="task_target",
            arrival_radius_cm=float(self.config["navigation"]["target_arrival_radius_cm"]),
            max_steps=int(self.config["navigation"]["max_steps_per_target"]),
            target_yaw_deg=screen.task_target_yaw_deg,
            target_yaw_tolerance_deg=float(
                self.config["navigation"]["target_arrival_yaw_tolerance_deg"]
            ),
            allow_goal_high_cost=True,
            target_screen=screen,
        )

    def execute_final_forward(self, screen: Screen) -> bool:
        """Execute the dedicated 10 cm action exactly once before classification."""
        if self.final_forward_executed:
            self.debug.event("target_final_forward_failed", screen_id=screen.screen_id, reason="already_executed")
            return False
        confirmation = self.target_visual_confirmation
        if (
            confirmation is None
            or confirmation.screen_id != int(screen.screen_id)
            or confirmation.tag_id != int(screen.screen_id)
            or not confirmation.binding_ok
        ):
            self.debug.event("target_final_forward_failed", screen_id=screen.screen_id, reason="target_confirmation_missing")
            return False
        distance = float(self.config["interaction"]["target_final_forward_cm"])
        self.set_mission_state(MissionState.FORWARD_10CM)
        self.debug.event(
            "target_final_forward_started",
            screen_id=screen.screen_id,
            action="interaction_forward_10cm",
            final_forward_cm=distance,
            target_distance_cm=float(self.config["interaction"]["target_distance_cm"]),
            final_forward_executed=False,
            dry_run=self.args.dry_run,
        )
        result = self.motion.run("interaction_forward_10cm", times_override=1)
        self.final_forward_executed = True
        self.debug.event(
            "target_final_forward_done" if result.ok else "target_final_forward_failed",
            screen_id=screen.screen_id,
            action=result.key,
            times=result.times,
            model_forward_cm=result.model_forward_cm,
            final_forward_cm=distance,
            final_forward_executed=self.final_forward_executed,
            ok=result.ok,
            error=result.error,
        )
        return bool(result.ok)

    def process_screen_interaction(self, screen: Screen) -> bool:
        worker_id = self.worker_id_for_screen(screen)
        if not screen.last_classification or screen.last_classification == self.target_flower:
            self.debug.event("interaction_skipped", screen_id=screen.screen_id, reason="flower_not_changeable")
            return False
        from_flower = screen.last_classification
        authorization_check = self.visual_authorization_check(screen, expected_from_flower=from_flower)
        self.last_interaction_check = authorization_check.as_dict()
        if not authorization_check.ready:
            self.debug.event(
                "interaction_safety_gate_blocked",
                screen_id=screen.screen_id,
                stage="visual_authorization",
                check=authorization_check.as_dict(),
            )
            return False

        pose_snapshot = None if self.state.pose is None else self.state.pose.as_dict()
        screen.status = ScreenStatus.INTERACTING
        self.set_mission_state(MissionState.EXECUTE_CHANGE)
        result = self.interaction.change_flower(
            screen_id=screen.screen_id,
            worker_id=worker_id,
            from_flower=from_flower,
            to_flower=self.target_flower,
            safety_gate=lambda: self.visual_authorization_check(
                screen,
                expected_from_flower=from_flower,
            ),
        )
        record = {
            "t": round(time.time(), 3),
            "screen_id": screen.screen_id,
            "worker_id": worker_id,
            "from_flower": from_flower,
            "to_flower": self.target_flower,
            "success": result.success,
            "simulated": result.simulated,
            "error": result.error,
            "response": result.response,
            "pose": pose_snapshot,
            "visual_authorization": None if self.visual_authorization is None else self.visual_authorization.as_dict(),
            "target_visual_confirmation": None if self.target_visual_confirmation is None else self.target_visual_confirmation.as_dict(),
            "target_distance_cm": float(self.config["interaction"]["target_distance_cm"]),
            "target_final_forward_cm": float(self.config["interaction"]["target_final_forward_cm"]),
            "target_confirmation_retry_count": getattr(self, "target_confirmation_retry_count", 0),
            "target_confirmation_max_retries": int(self.config["interaction"].get("target_confirmation_max_retries", 3)),
            "target_confirmation_recovery_cycle": getattr(self, "target_confirmation_recovery_cycle", 0),
            "target_confirmation_diagnostics": getattr(self, "last_target_confirmation_diagnostics", {}),
            "final_forward_executed": self.final_forward_executed,
            "interaction_check": authorization_check.as_dict(),
        }
        self.write_interaction_audit(record)
        self.latest_interaction_result = record
        self.recent_interaction_results.append(record)
        self.recent_interaction_results = self.recent_interaction_results[-5:]
        changed = apply_worker_change_result(screen, result)
        if changed:
            self.debug.event("interaction_changed", **record)
            return True
        self.debug.event("interaction_not_changed", **record)
        return False

    def path_length_cm(self, path: List[Tuple[float, float]], fallback_start=None, fallback_goal=None) -> float:
        if len(path) >= 2:
            return sum(distance_xy(path[i - 1], path[i]) for i in range(1, len(path)))
        if fallback_start is not None and fallback_goal is not None:
            return distance_xy(fallback_start, fallback_goal)
        return 0.0

    def compact_path_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for pt in points:
            clean = (float(pt[0]), float(pt[1]))
            if not out or distance_xy(out[-1], clean) > 1.0:
                out.append(clean)
        return out

    def movement_corridor_metrics(
        self,
        start_xy,
        end_xy,
        allow_goal_high_cost: bool = False,
    ) -> dict:
        nav = self.config["navigation"]
        half_width = float(nav.get("translation_corridor_half_width_cm", 0.0))
        max_cost = min(
            float(nav.get("normal_navigation_max_cost", 55.0)),
            float(self.config["map"].get("obstacle_cost_max", 80.0)),
        )
        if hasattr(self.map, "translation_corridor_metrics"):
            return self.map.translation_corridor_metrics(
                start_xy,
                end_xy,
                half_width,
                max_cost,
                allow_goal_high_cost=allow_goal_high_cost,
            )
        clear = self.map.line_clear(
            start_xy,
            end_xy,
            max_cost=max_cost,
            allow_goal_high_cost=allow_goal_high_cost,
        )
        return {
            "clear": clear,
            "path_length_cm": distance_xy(start_xy, end_xy),
            "path_obstacle_cost": 0.0,
            "maximum_obstacle_cost": 0.0,
            "minimum_wall_clearance_cm": float("inf"),
        }

    def movement_corridor_clear(
        self,
        start_xy,
        end_xy,
        allow_goal_high_cost: bool = False,
    ) -> bool:
        return bool(
            self.movement_corridor_metrics(
                start_xy,
                end_xy,
                allow_goal_high_cost=allow_goal_high_cost,
            )["clear"]
        )

    def normal_path_metrics(
        self,
        pose: RobotPose,
        path: List[Tuple[float, float]],
        *,
        allow_goal_high_cost: bool = False,
        translation_only: bool = False,
    ) -> dict:
        nav = self.config["navigation"]
        if len(path) < 2:
            return {"total_cost": float("inf"), "clear": False}
        length = 0.0
        obstacle_integral = 0.0
        minimum_clearance = float("inf")
        wall_penalty = 0.0
        clear = True
        wall_target = float(nav.get("normal_wall_clearance_target_cm", 25.0))
        wall_scale = float(nav.get("normal_wall_clearance_penalty_scale", 4.0))
        headings = []
        for index, (start, end) in enumerate(zip(path, path[1:]), start=1):
            is_goal = allow_goal_high_cost and index == len(path) - 1
            metrics = self.movement_corridor_metrics(start, end, allow_goal_high_cost=is_goal)
            segment_length = float(metrics["path_length_cm"])
            length += segment_length
            obstacle_integral += float(metrics["path_obstacle_cost"]) * segment_length / 10.0
            clearance = float(metrics["minimum_wall_clearance_cm"])
            minimum_clearance = min(minimum_clearance, clearance)
            wall_penalty += max(0.0, wall_target - clearance) * wall_scale * segment_length / 10.0
            clear = clear and bool(metrics["clear"])
            headings.append(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))
        turn_cost = 0.0
        switches = max(0, len(path) - 2)
        if not translation_only and headings:
            previous = pose.yaw_deg
            for heading in headings:
                delta = abs(angle_diff_deg(heading, previous))
                if delta > float(nav.get("turn_tolerance_deg", 20.0)):
                    turn_cost += float(nav.get("action_planner_turn_fixed_cost_cm", 20.0))
                    turn_cost += delta * float(nav.get("action_planner_turn_cost_cm_per_deg", 0.8))
                previous = heading
        switch_penalty = switches * float(nav.get("normal_path_action_switch_penalty_cm", 10.0))
        obstacle_component = obstacle_integral * float(nav.get("normal_path_obstacle_cost_scale", 0.35))
        return {
            "clear": clear,
            "total_cost": length + obstacle_component + wall_penalty + turn_cost + switch_penalty,
            "path_length_cm": length,
            "path_obstacle_cost": obstacle_integral,
            "minimum_wall_clearance_cm": minimum_clearance,
            "wall_clearance_penalty": wall_penalty,
            "turn_cost": turn_cost,
            "action_switch_penalty": switch_penalty,
        }

    def path_segments_clear(
        self,
        points: List[Tuple[float, float]],
        allow_goal_high_cost: bool = False,
    ) -> bool:
        if len(points) < 2:
            return False
        for index, pt in enumerate(points[1:], start=1):
            is_goal = allow_goal_high_cost and index == len(points) - 1
            if is_goal:
                if not self.map.is_free_xy(pt):
                    return False
            elif not self.map.is_traversable_xy(
                pt,
                max_cost=float(self.config["navigation"].get("normal_navigation_max_cost", 55.0)),
            ):
                return False
        for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
            is_goal = allow_goal_high_cost and index == len(points) - 1
            if not self.movement_corridor_clear(start, end, allow_goal_high_cost=is_goal):
                return False
        return True

    def body_translation_candidate_paths(
        self,
        pose: RobotPose,
        goal_xy: Tuple[float, float],
        allow_goal_high_cost: bool = False,
    ) -> List[List[Tuple[float, float]]]:
        nav_cfg = self.config["navigation"]
        if not bool(nav_cfg.get("translation_path_prefer_enabled", True)):
            return []
        forward, lateral = self.local_vector_to(pose, goal_xy)
        max_backward = float(nav_cfg.get("translation_max_backward_cm", 8.0))
        if forward < -max_backward:
            return []
        min_component = float(nav_cfg.get("translation_path_min_component_cm", 8.0))
        yaw = math.radians(pose.yaw_deg)
        forward_xy = (
            pose.x_cm + forward * math.cos(yaw),
            pose.y_cm + forward * math.sin(yaw),
        )
        left_rad = yaw + math.pi / 2.0
        lateral_xy = (
            pose.x_cm + lateral * math.cos(left_rad),
            pose.y_cm + lateral * math.sin(left_rad),
        )
        start_xy = pose.xy()
        candidates: List[List[Tuple[float, float]]] = []
        if forward >= min_component and abs(lateral) >= min_component:
            candidates.append(self.compact_path_points([start_xy, forward_xy, goal_xy]))
            candidates.append(self.compact_path_points([start_xy, lateral_xy, goal_xy]))
        elif forward >= min_component or abs(lateral) >= min_component:
            candidates.append(self.compact_path_points([start_xy, goal_xy]))
        return [
            path
            for path in candidates
            if self.path_segments_clear(path, allow_goal_high_cost=allow_goal_high_cost)
        ]

    def target_direct_approach_path(
        self,
        pose: RobotPose,
        screen: Optional[Screen],
        goal_xy: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Prefer a narrow, target-cost-exempt path inside the final range."""
        if screen is None or self.current_target_screen_id != int(screen.screen_id):
            return []
        nav = self.config["navigation"]
        distance = distance_xy(pose.xy(), goal_xy)
        limit = float(nav.get("target_direct_approach_distance_cm", 40.0))
        if distance > limit:
            return []
        half_width = float(nav.get("target_direct_corridor_half_width_cm", 6.0))
        max_cost = float(nav.get("target_direct_non_target_max_cost", 60.0))

        def segment_clear(start, end):
            return self.map.target_direct_corridor_clear(
                start,
                end,
                screen.screen_id,
                half_width,
                max_cost,
            )

        start = pose.xy()
        if segment_clear(start, goal_xy):
            return [start, goal_xy]
        forward, lateral = self.local_vector_to(pose, goal_xy)
        candidates = []
        if forward > 0.0 and abs(lateral) > 0.0:
            forward_point = self.translated_pose_xy(pose, forward_cm=forward)
            lateral_point = self.translated_pose_xy(pose, lateral_cm=lateral)
            candidates.extend(([start, forward_point, goal_xy], [start, lateral_point, goal_xy]))
        for path in candidates:
            if all(segment_clear(a, b) for a, b in zip(path, path[1:])):
                return self.compact_path_points(path)
        return []

    def plan_navigation_path(
        self,
        pose: RobotPose,
        goal_xy: Tuple[float, float],
        allow_goal_high_cost: bool = False,
        target_screen: Optional[Screen] = None,
    ) -> List[Tuple[float, float]]:
        direct = self.target_direct_approach_path(pose, target_screen, goal_xy)
        if direct:
            self.debug.event(
                "target_direct_approach",
                navigation_mode="target_direct_approach",
                current_target_screen_id=None if target_screen is None else target_screen.screen_id,
                target_direct_corridor_clear=True,
                target_direct_cost_exemption=True,
                target_xy=goal_xy,
            )
            return direct
        normal_goal = goal_xy
        normal_allow_goal_high_cost = allow_goal_high_cost
        direct_limit = float(
            self.config["navigation"].get("target_direct_approach_distance_cm", 40.0)
        )
        goal_distance = distance_xy(pose.xy(), goal_xy)
        if (
            target_screen is not None
            and self.current_target_screen_id == int(target_screen.screen_id)
            and goal_distance > direct_limit
        ):
            scale = max(0.0, direct_limit - 2.0) / max(1e-6, goal_distance)
            normal_goal = (
                float(goal_xy[0]) + (float(pose.x_cm) - float(goal_xy[0])) * scale,
                float(goal_xy[1]) + (float(pose.y_cm) - float(goal_xy[1])) * scale,
            )
            if not self.map.is_traversable_xy(
                normal_goal,
                max_cost=float(
                    self.config["navigation"].get("normal_navigation_max_cost", 55.0)
                ),
            ):
                normal_goal = self.map.nearest_traversable_xy(normal_goal)
            normal_allow_goal_high_cost = False
            self.debug.event(
                "normal_navigation_staging_target",
                navigation_mode="normal",
                target_xy=goal_xy,
                staging_xy=normal_goal,
                target_direct_cost_exemption=False,
            )
        translation_paths = self.body_translation_candidate_paths(
            pose,
            normal_goal,
            allow_goal_high_cost=normal_allow_goal_high_cost,
        )
        candidates = []
        if bool(self.config["navigation"].get("action_planner_enabled", True)):
            action_path = self.map.plan_action_path(
                pose,
                normal_goal,
                self.config["navigation"],
                self.config["motion"],
                allow_goal_high_cost=normal_allow_goal_high_cost,
            )
            if action_path:
                metrics = self.normal_path_metrics(
                    pose, action_path, allow_goal_high_cost=normal_allow_goal_high_cost
                )
                planner_metrics = getattr(self.map, "last_action_plan_metrics", {})
                if planner_metrics.get("total_cost") is not None:
                    metrics["total_cost"] = float(planner_metrics["total_cost"])
                    metrics["turn_cost"] = float(planner_metrics.get("turn_cost", 0.0))
                    metrics["selected_actions"] = list(
                        planner_metrics.get("selected_actions", [])
                    )
                candidates.append(("action_planner", action_path, metrics))
        for translation_path in translation_paths:
            metrics = self.normal_path_metrics(
                pose,
                translation_path,
                allow_goal_high_cost=normal_allow_goal_high_cost,
                translation_only=True,
            )
            if metrics.get("clear"):
                candidates.append(("body_translation", translation_path, metrics))
        astar_path = self.map.plan(
            pose.xy(),
            normal_goal,
            allow_goal_high_cost=normal_allow_goal_high_cost,
        )
        if astar_path:
            astar_path = self.compact_path_points([pose.xy()] + list(astar_path))
            metrics = self.normal_path_metrics(
                pose, astar_path, allow_goal_high_cost=normal_allow_goal_high_cost
            )
            if metrics.get("clear"):
                candidates.append(("astar", astar_path, metrics))
        if not candidates:
            return []
        priority = {"body_translation": 0, "action_planner": 1, "astar": 2}
        selected_name, selected_path, selected_metrics = min(
            candidates,
            key=lambda item: (
                float(item[2].get("total_cost", float("inf"))),
                priority.get(item[0], 9),
            ),
        )
        self.debug.event(
            "navigation_path_selected",
            navigation_mode="normal",
            selected_path_type=selected_name,
            path_length_cm=round(float(selected_metrics.get("path_length_cm", 0.0)), 2),
            path_obstacle_cost=round(float(selected_metrics.get("path_obstacle_cost", 0.0)), 2),
            minimum_wall_clearance_cm=round(
                float(selected_metrics.get("minimum_wall_clearance_cm", 0.0)), 2
            ),
            turn_cost=round(float(selected_metrics.get("turn_cost", 0.0)), 2),
            wall_clearance_penalty=round(
                float(selected_metrics.get("wall_clearance_penalty", 0.0)), 2
            ),
            total_cost=round(float(selected_metrics.get("total_cost", 0.0)), 2),
            target_direct_cost_exemption=False,
            candidate_costs={
                name: round(float(metrics.get("total_cost", 0.0)), 2)
                for name, _, metrics in candidates
            },
            planned_actions=selected_metrics.get("selected_actions", []),
        )
        return selected_path

    def choose_target_direct_action(
        self,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        screen: Screen,
    ) -> Optional[dict]:
        """Choose forward first, then lateral, with a shortened final step."""
        nav = self.config["navigation"]
        forward, lateral = self.local_vector_to(pose, waypoint)
        minimum = float(nav.get("target_direct_min_component_cm", 2.0))
        radius = float(nav.get("target_arrival_radius_cm", 4.0))
        half_width = float(nav.get("target_direct_corridor_half_width_cm", 6.0))
        max_cost = float(nav.get("target_direct_non_target_max_cost", 60.0))

        def corridor_for(forward_cm=0.0, lateral_cm=0.0):
            end = self.translated_pose_xy(pose, forward_cm=forward_cm, lateral_cm=lateral_cm)
            return self.map.target_direct_corridor_clear(
                pose.xy(), end, screen.screen_id, half_width, max_cost
            )

        if forward < 0.0:
            rear_angle_error = math.degrees(
                math.atan2(abs(float(lateral)), max(1e-6, -float(forward)))
            )
            rear_tolerance = float(nav.get("reverse_prefer_rear_angle_tolerance_deg", 30.0))
            max_lateral = float(nav.get("reverse_prefer_max_lateral_cm", 8.0))
            step = abs(float(self.config["motion"]["actions"]["back_fast"].get("forward_cm", -2.5)))
            next_xy = self.translated_pose_xy(pose, forward_cm=-step)
            next_distance = distance_xy(next_xy, waypoint)
            current_distance = distance_xy(pose.xy(), waypoint)
            if (
                rear_angle_error <= rear_tolerance
                and abs(lateral) <= max_lateral
                and next_distance < current_distance
                and corridor_for(forward_cm=-step)
            ):
                self.debug.event(
                    "reverse_short_target_selected",
                    navigation_mode="target_direct_approach",
                    target_local_forward_cm=round(float(forward), 2),
                    target_local_lateral_cm=round(float(lateral), 2),
                    action_cycles=1,
                    next_distance_cm=round(next_distance, 2),
                )
                return {"kind": "reverse", "key": "back_fast", "times": 1, "planned_cm": -step}

        if forward >= minimum:
            fast_min = float(nav.get("target_direct_forward_fast_min_cm", 10.0))
            if forward >= fast_min:
                step = abs(float(self.config["motion"]["actions"]["forward_fast"].get("forward_cm", 3.5)))
                times = max(1, int(math.floor(max(0.0, forward - radius) / max(1.0, step))))
                times, _ = self.select_adaptive_action_batch(
                    "forward", times, step, forward, distance_xy(pose.xy(), waypoint),
                    navigation_mode="target_direct_approach",
                )
                travel = times * step
                key = "forward_fast"
            else:
                key = "forward_micro"
                times = 1
                travel = abs(float(self.config["motion"]["actions"][key].get("forward_cm", 2.0)))
            if travel <= forward + radius and corridor_for(forward_cm=travel):
                return {"kind": "forward", "key": key, "times": times, "planned_cm": travel}

        if abs(lateral) >= minimum:
            key = "strafe_left_fast" if lateral > 0.0 else "strafe_right_fast"
            step = abs(float(self.config["motion"]["actions"][key].get("lateral_cm", 4.0)))
            times = max(1, int(math.floor(max(0.0, abs(lateral) - radius) / max(1.0, step))))
            times, _ = self.select_adaptive_action_batch(
                "strafe", times, step, abs(lateral), distance_xy(pose.xy(), waypoint),
                navigation_mode="target_direct_approach",
            )
            travel = math.copysign(times * step, lateral)
            if abs(travel) <= abs(lateral) + radius and corridor_for(lateral_cm=travel):
                return {"kind": "strafe", "key": key, "times": times, "planned_cm": travel}
        return None

    def execute_target_direct_action(self, action: dict, screen: Screen, target_xy) -> bool:
        pose_before = self.copy_pose(self.state.pose)
        self.debug.event(
            "target_direct_approach_action",
            navigation_mode="target_direct_approach",
            screen_id=screen.screen_id,
            action=action["key"],
            times=action["times"],
            planned_cm=round(float(action["planned_cm"]), 2),
            target_xy=target_xy,
        )
        result = self.motion.run(action["key"], times_override=int(action["times"]))
        if result.ok:
            self.clear_turn_progress_watchdog("target_direct_translation")
            self.post_action_relocalize("target_direct_approach", pose_before, result, target_xy)
        return bool(result.ok)

    def waypoint_has_navigation_action(self, pose: RobotPose, waypoint: Tuple[float, float]) -> bool:
        if distance_xy(pose.xy(), waypoint) < 1.0:
            return False
        if self.choose_translation_action(pose, waypoint) is not None:
            return True
        desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
        diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
        return abs(diff) > float(self.config["navigation"].get("turn_tolerance_deg", 20.0))

    def max_forward_cycles_for_pose(self, pose: RobotPose) -> int:
        nav_cfg = self.config["navigation"]
        if pose.confidence == Confidence.HIGH:
            key = "max_forward_cycles_high"
        elif pose.confidence == Confidence.MEDIUM:
            key = "max_forward_cycles_medium"
        else:
            key = "max_forward_cycles_low"
        return max(1, int(nav_cfg.get(key, 1)))

    def effective_localization_confidence(self, pose: RobotPose) -> Confidence:
        """Downgrade stale or motion-diverged poses before selecting a batch."""
        nav = self.config["navigation"]
        confidence = pose.confidence
        failures = int(getattr(self, "consecutive_localize_failures", 0))
        uncertainty = float(getattr(self.state, "motion_uncertainty", 0.0))
        last_success = float(getattr(self, "last_localize_success_s", 0.0))
        if last_success <= 0.0 and pose.source not in ("DEAD_RECKONING", "UNKNOWN"):
            last_success = float(pose.last_update_s)
        age = float("inf") if last_success <= 0.0 else max(0.0, now_s() - last_success)
        if failures > 0 or uncertainty >= float(nav.get("relocalize_uncertainty_threshold", 6.0)):
            return Confidence.LOW
        if confidence == Confidence.HIGH and age > float(nav.get("localization_fresh_high_s", 4.0)):
            confidence = Confidence.MEDIUM
        if age > float(nav.get("localization_fresh_medium_s", 10.0)):
            confidence = Confidence.LOW
        return confidence

    def select_adaptive_action_batch(
        self,
        action_kind: str,
        requested_cycles: int,
        step_cm: float,
        remaining_cm: float,
        goal_distance_cm: float,
        *,
        navigation_mode: str = "normal",
        near_wall: bool = False,
        recovery: bool = False,
    ) -> Tuple[int, str]:
        nav = self.config["navigation"]
        pose = self.state.pose
        requested = max(1, int(requested_cycles))
        if pose is None or not bool(nav.get("adaptive_action_batch_enabled", True)):
            return 1 if pose is None else requested, "adaptive_disabled_or_pose_missing"
        confidence = self.effective_localization_confidence(pose)
        if action_kind in ("forward", "reverse", "strafe", "turn"):
            prefix = action_kind
        else:
            prefix = "turn"
        cap = max(1, int(nav.get("max_{}_cycles_{}".format(prefix, confidence.value.lower()), 1)))
        reasons = ["confidence_{}".format(confidence.value.lower())]
        if action_kind in ("reverse", "strafe", "turn"):
            reasons.append("higher_uncertainty_action")
        if near_wall:
            cap = min(cap, int(nav.get("near_wall_max_action_cycles", 1)))
            reasons.append("near_wall")
        if recovery:
            cap = min(cap, int(nav.get("recovery_max_action_cycles", 1)))
            reasons.append("recovery")
        near_target = goal_distance_cm < float(nav.get("near_target_distance_cm", 40.0))
        if near_target:
            cap = min(cap, int(nav.get("near_target_max_action_cycles", 1)))
            reasons.append("near_target")
        if navigation_mode == "target_direct_approach":
            cap = min(cap, int(nav.get("near_target_max_action_cycles", 1)))
            reasons.append("target_direct_approach")
        # Never cross the active waypoint/goal, and reserve uncertainty budget.
        if step_cm > 0.0:
            distance_cap = max(1, int(math.floor(max(0.0, min(remaining_cm, goal_distance_cm)) / step_cm)))
            cap = min(cap, distance_cap)
            uncertainty_per_cycle = float(nav.get(
                "forward_uncertainty_per_cycle" if action_kind == "forward" else
                "reverse_uncertainty_per_cycle" if action_kind == "reverse" else
                "strafe_uncertainty_per_cycle" if action_kind == "strafe" else
                "turn_uncertainty_per_cycle",
                1.0,
            ))
            budget = max(0.0, float(nav.get("relocalize_uncertainty_threshold", 6.0)) - float(getattr(self.state, "motion_uncertainty", 0.0)))
            cap = min(cap, max(1, int(math.floor(budget / max(0.1, uncertainty_per_cycle)))))
        selected = max(1, min(requested, cap))
        reason = ",".join(reasons)
        self.debug.event(
            "adaptive_action_batch_selected",
            action_kind=action_kind,
            localization_confidence=confidence.value,
            actions_since_localize=int(getattr(self.state, "actions_since_localize", 0)),
            motion_uncertainty=round(float(getattr(self.state, "motion_uncertainty", 0.0)), 3),
            requested_action_cycles=requested,
            selected_action_cycles=selected,
            remaining_cm=round(float(remaining_cm), 2),
            goal_distance_cm=round(float(goal_distance_cm), 2),
            adaptive_batch_reason=reason,
            navigation_mode=navigation_mode,
        )
        return selected, reason

    def post_action_relocalize(self, reason: str, pose_before: RobotPose, result, target_xy) -> bool:
        """End a motion batch with one fresh pose; the caller replans next loop."""
        actual = getattr(result, "executed_times", None)
        if actual is None:
            actual = int(getattr(result, "times", 0)) if bool(getattr(result, "ok", False)) else 0
        self.hardware.center_head()
        dry_run = bool(getattr(getattr(self, "args", None), "dry_run", False))
        if dry_run:
            localized = True
            if self.state.pose is not None and hasattr(self.state, "set_pose"):
                self.state.set_pose(self.copy_pose(self.state.pose))
        else:
            localized = bool(self.localize_scan(
                reason=reason,
                allow_failure_escalation=False,
            ))
        if not localized and self.state.pose is not None:
            self.state.pose.confidence = Confidence.LOW
        self.debug.event(
            "post_action_relocalize",
            reason=reason,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
            target_preserved=getattr(self, "current_target_screen_id", None) is not None,
            pose_before_action=pose_before.as_dict(),
            pose_after_action=None if self.state.pose is None else self.state.pose.as_dict(),
            requested_action_cycles=int(getattr(result, "times", 0)),
            actual_action_cycles=int(actual),
            localization_confidence=None if self.state.pose is None else self.state.pose.confidence.value,
            actions_since_localize=int(getattr(self.state, "actions_since_localize", 0)),
            motion_uncertainty=round(float(getattr(self.state, "motion_uncertainty", 0.0)), 3),
            post_action_relocalized=localized,
        )
        self.debug.event(
            "post_action_replan",
            target_xy=target_xy,
            post_action_replanned=False,
            replan_requested=True,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
        )
        self.pending_post_action_replan = True
        return localized

    def navigation_waypoint_max_lookahead_cm(self, pose: RobotPose, lookahead_cm: float) -> float:
        nav_cfg = self.config["navigation"]
        explicit = float(nav_cfg.get("navigation_waypoint_max_lookahead_cm", 0.0))
        if explicit > 0.0:
            return max(float(lookahead_cm), explicit)
        forward_step = abs(float(self.config["motion"]["actions"]["forward_fast"].get("forward_cm", 3.5)))
        reserve = max(0.0, float(nav_cfg.get("reserve_stop_distance_cm", 0.0)))
        buffer_cm = max(0.0, float(nav_cfg.get("navigation_waypoint_forward_buffer_cm", 16.0)))
        cycles = self.max_forward_cycles_for_pose(pose)
        return max(float(lookahead_cm), reserve + forward_step * cycles + buffer_cm)

    def select_navigation_waypoint(
        self,
        pose: RobotPose,
        path: List[Tuple[float, float]],
        target_xy: Tuple[float, float],
        allow_goal_high_cost: bool = False,
    ) -> Tuple[float, float]:
        points = list(path or [])
        if not points:
            return target_xy
        lookahead = max(1.0, float(self.config["navigation"].get("navigation_waypoint_lookahead_cm", 14.0)))
        max_lookahead = self.navigation_waypoint_max_lookahead_cm(pose, lookahead)
        if distance_xy(pose.xy(), target_xy) <= lookahead:
            return target_xy
        candidates = points[1:] if len(points) >= 2 else points
        if not candidates or distance_xy(candidates[-1], target_xy) > 1.0:
            candidates = list(candidates) + [target_xy]
        fallback = candidates[0] if candidates else target_xy
        last_clear = None
        last_actionable = None
        far_actionable = None
        for point in candidates:
            dist = distance_xy(pose.xy(), point)
            if dist <= 1.0:
                continue
            if dist > max_lookahead and far_actionable is not None:
                break
            is_target = distance_xy(point, target_xy) <= 0.1
            if not self.map.line_clear(
                pose.xy(),
                point,
                allow_goal_high_cost=allow_goal_high_cost and is_target,
            ):
                continue
            last_clear = point
            if self.waypoint_has_navigation_action(pose, point):
                last_actionable = point
                if dist >= lookahead:
                    far_actionable = point
            if dist >= max_lookahead:
                break
        if self.map.line_clear(
            pose.xy(),
            target_xy,
            allow_goal_high_cost=allow_goal_high_cost,
        ):
            if self.waypoint_has_navigation_action(pose, target_xy):
                target_dist = distance_xy(pose.xy(), target_xy)
                if target_dist <= max_lookahead:
                    return target_xy
                if far_actionable is None and target_dist >= lookahead:
                    far_actionable = target_xy
            last_clear = target_xy
        if far_actionable is not None:
            return far_actionable
        if last_actionable is not None:
            return last_actionable
        return last_clear or fallback

    def choose_nearest_screen(self) -> Optional[Screen]:
        if self.state.pose is None:
            return None
        pose = self.state.pose
        locked = self.map.screens.get(getattr(self, "current_target_screen_id", None))
        if locked is not None and not locked.terminal():
            self.last_target_plan = {
                "selection_rule": "preserve_locked_target",
                "screen_id": locked.screen_id,
                "task_target_xy": list(locked.task_target_xy or locked.interaction_xy),
            }
            return locked
        ranked = sorted(
            (screen for screen in self.map.unfinished_screens() if not screen.terminal()),
            key=lambda screen: (
                distance_xy(pose.xy(), screen.task_target_xy or screen.interaction_xy),
                int(screen.screen_id),
            ),
        )
        if not ranked:
            self.last_target_plan = {}
            return None
        best = ranked[0]
        task_target = best.task_target_xy or best.interaction_xy
        distance = distance_xy(pose.xy(), task_target)
        sorted_candidates = [
            {
                "screen_id": screen.screen_id,
                "tag_id": screen.screen_id,
                "surface_face": screen.surface_face,
                "cardinal_normal_xy": list(screen.cardinal_normal_xy),
                "task_target_xy": list(screen.task_target_xy or screen.interaction_xy),
                "task_target_yaw_deg": screen.task_target_yaw_deg,
                "distance_cm": round(
                    distance_xy(pose.xy(), screen.task_target_xy or screen.interaction_xy), 2
                ),
            }
            for screen in ranked
        ]
        self.last_target_plan = {
            "selection_rule": "euclidean_current_pose_to_task_target_then_tag_id",
            "tag_id": best.screen_id,
            "screen_id": best.screen_id,
            "distance_cm": round(distance, 2),
            "surface_face": best.surface_face,
            "cardinal_normal_xy": list(best.cardinal_normal_xy),
            "tag_front_xy": list(best.tag_front_xy or best.interaction_xy),
            "task_target_xy": [round(float(task_target[0]), 2), round(float(task_target[1]), 2)],
            "task_target_yaw_deg": best.task_target_yaw_deg,
            "remaining_ids": [screen.screen_id for screen in ranked],
            "sorted_candidates": sorted_candidates,
        }
        return best

    def local_vector_to(self, pose: RobotPose, xy: Tuple[float, float]) -> Tuple[float, float]:
        dx = float(xy[0]) - pose.x_cm
        dy = float(xy[1]) - pose.y_cm
        yaw = math.radians(pose.yaw_deg)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return forward, lateral

    def strafe_target_clear(self, pose: RobotPose, lateral_cm: float) -> bool:
        step = abs(float(self.config["motion"]["actions"]["strafe_left_fast"].get("lateral_cm", 4.0)))
        travel = math.copysign(min(abs(lateral_cm), step * max(1, int(self.config["navigation"].get("max_strafe_cycles_high", 6)))), lateral_cm)
        yaw = math.radians(pose.yaw_deg)
        left_rad = yaw + math.pi / 2.0
        target_xy = (
            pose.x_cm + travel * math.cos(left_rad),
            pose.y_cm + travel * math.sin(left_rad),
        )
        return self.map.line_clear(pose.xy(), target_xy)

    def translated_pose_xy(self, pose: RobotPose, forward_cm: float = 0.0, lateral_cm: float = 0.0) -> Tuple[float, float]:
        yaw = math.radians(pose.yaw_deg)
        left_rad = yaw + math.pi / 2.0
        return (
            pose.x_cm + forward_cm * math.cos(yaw) + lateral_cm * math.cos(left_rad),
            pose.y_cm + forward_cm * math.sin(yaw) + lateral_cm * math.sin(left_rad),
        )

    def planned_lateral_step_cm(self, lateral_cm: float) -> float:
        step = abs(float(self.config["motion"]["actions"]["strafe_left_fast"].get("lateral_cm", 4.0)))
        cycles = self.motion.lateral_cycles_for_distance(lateral_cm)
        return math.copysign(cycles * step, lateral_cm)

    def choose_translation_action(
        self,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        allow_goal_high_cost: bool = False,
    ) -> Optional[dict]:
        nav_cfg = self.config["navigation"]
        if not bool(nav_cfg.get("translation_prefer_enabled", True)):
            return None
        forward, lateral = self.local_vector_to(pose, waypoint)
        current_dist = distance_xy(pose.xy(), waypoint)
        if current_dist < 1.0:
            return None
        min_progress = float(nav_cfg.get("translation_min_progress_cm", 2.0))
        options = []

        reverse_rejected_reason = "disabled"
        rear_angle_error = math.degrees(
            math.atan2(abs(float(lateral)), max(1e-6, -float(forward)))
        ) if forward < 0.0 else 180.0
        if bool(nav_cfg.get("reverse_prefer_enabled", True)) and "back_fast" in self.config["motion"]["actions"]:
            reverse_rejected_reason = "target_not_behind"
            rear_distance = -float(forward)
            if forward < 0.0:
                reverse_rejected_reason = "rear_angle_exceeds_tolerance"
                if rear_angle_error <= float(
                    nav_cfg.get("reverse_prefer_rear_angle_tolerance_deg", 30.0)
                ):
                    reverse_rejected_reason = "lateral_error_too_large"
                    if abs(float(lateral)) <= float(
                        nav_cfg.get("reverse_prefer_max_lateral_cm", 8.0)
                    ):
                        reverse_rejected_reason = "localization_confidence_low"
                        if self.effective_localization_confidence(pose) != Confidence.LOW:
                            arrival_tolerance = float(nav_cfg.get("target_arrival_radius_cm", 4.0))
                            back_step = abs(float(
                                self.config["motion"]["actions"]["back_fast"].get("forward_cm", -2.5)
                            ))
                            requested_cycles = self.motion.reverse_cycles_for_distance(rear_distance)
                            cycle_candidates = list(range(1, max(1, requested_cycles) + 1))
                            selected_cycles = None
                            selected_next_dist = None
                            # The first cycle that enters the arrival radius is
                            # the safest final move. Otherwise take the largest
                            # simulated batch that still improves distance.
                            improving = []
                            for cycles in cycle_candidates:
                                travel = cycles * back_step
                                next_xy = self.translated_pose_xy(pose, forward_cm=-travel)
                                next_dist = distance_xy(next_xy, waypoint)
                                if next_dist < current_dist:
                                    improving.append((cycles, next_dist))
                                    if next_dist <= arrival_tolerance:
                                        selected_cycles, selected_next_dist = cycles, next_dist
                                        break
                            if selected_cycles is None and improving:
                                selected_cycles, selected_next_dist = improving[-1]
                            reverse_rejected_reason = "would_not_reduce_goal_distance"
                            if selected_cycles is not None:
                                travel = selected_cycles * back_step
                                planned = -travel
                                next_xy = self.translated_pose_xy(pose, forward_cm=planned)
                                metrics = self.movement_corridor_metrics(pose.xy(), next_xy)
                                reverse_rejected_reason = "rear_corridor_blocked"
                                if metrics["clear"]:
                                    progress = current_dist - float(selected_next_dist)
                                    reverse_rejected_reason = "insufficient_progress"
                                    if progress >= min_progress or selected_next_dist <= arrival_tolerance:
                                        reverse_rejected_reason = ""
                                        options.append({
                                            "kind": "reverse",
                                            "distance_cm": travel,
                                            "planned_cm": planned,
                                            "progress_cm": progress,
                                            "forward_cm": forward,
                                            "lateral_cm": lateral,
                                            "corridor_metrics": metrics,
                                            "next_distance_cm": selected_next_dist,
                                        })
                                        if rear_distance < float(nav_cfg.get("reverse_prefer_min_distance_cm", 2.0)) + back_step * 2:
                                            self.debug.event(
                                                "reverse_short_target_selected",
                                                target_local_forward_cm=round(float(forward), 2),
                                                target_local_lateral_cm=round(float(lateral), 2),
                                                action_cycles=selected_cycles,
                                                next_distance_cm=round(float(selected_next_dist), 2),
                                            )
        self.debug.event(
            "reverse_preference_evaluated",
            navigation_mode="normal",
            selected_action="reverse" if any(item["kind"] == "reverse" for item in options) else None,
            target_local_forward_cm=round(float(forward), 2),
            target_local_lateral_cm=round(float(lateral), 2),
            target_rear_angle_error_deg=round(float(rear_angle_error), 2),
            reverse_preferred=any(item["kind"] == "reverse" for item in options),
            reverse_rejected_reason=reverse_rejected_reason or None,
            movement_corridor_clear=any(item["kind"] == "reverse" for item in options),
        )

        min_forward = float(nav_cfg.get("translation_min_forward_cm", 6.0))
        if forward >= min_forward:
            requested = min(float(forward), current_dist)
            planned = self.planned_forward_step_cm(requested)
            if self.forward_clear_for_distance(
                pose,
                planned,
                exact_goal_xy=waypoint if allow_goal_high_cost else None,
            ):
                travel = min(float(forward), planned)
                next_xy = self.translated_pose_xy(pose, forward_cm=travel)
                progress = current_dist - distance_xy(next_xy, waypoint)
                if progress >= min_progress:
                    options.append(
                        {
                            "kind": "forward",
                            "distance_cm": requested,
                            "planned_cm": planned,
                            "progress_cm": progress,
                            "forward_cm": forward,
                            "lateral_cm": lateral,
                        }
                    )

        min_lateral = float(nav_cfg.get("strafe_min_lateral_cm", 7.0))
        max_yaw = float(nav_cfg.get("strafe_prefer_max_yaw_deg", 115.0))
        max_backward = float(nav_cfg.get("translation_max_backward_cm", 8.0))
        desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
        diff_yaw = angle_diff_deg(desired_yaw, pose.yaw_deg)
        if forward >= -max_backward and abs(lateral) >= min_lateral and abs(diff_yaw) <= max_yaw:
            planned = self.planned_lateral_step_cm(lateral)
            lateral_target = self.translated_pose_xy(pose, lateral_cm=planned)
            lateral_reaches_goal = (
                allow_goal_high_cost
                and distance_xy(lateral_target, waypoint)
                <= float(self.config["navigation"].get("target_arrival_radius_cm", 4.0))
            )
            if abs(planned) > 0.0 and self.path_segments_clear(
                [pose.xy(), lateral_target],
                allow_goal_high_cost=lateral_reaches_goal,
            ):
                next_xy = self.translated_pose_xy(pose, lateral_cm=planned)
                progress = current_dist - distance_xy(next_xy, waypoint)
                if progress >= min_progress:
                    options.append(
                        {
                            "kind": "strafe",
                            "distance_cm": lateral,
                            "planned_cm": planned,
                            "progress_cm": progress,
                            "forward_cm": forward,
                            "lateral_cm": lateral,
                        }
                    )

        if not options:
            return None
        reverse_option = next((item for item in options if item["kind"] == "reverse"), None)
        forward_option = next((item for item in options if item["kind"] == "forward"), None)
        selected = reverse_option or forward_option or max(options, key=lambda item: item["progress_cm"])
        corridor = selected.get("corridor_metrics") or self.movement_corridor_metrics(
            pose.xy(),
            self.translated_pose_xy(
                pose,
                forward_cm=float(selected["planned_cm"]) if selected["kind"] != "strafe" else 0.0,
                lateral_cm=float(selected["planned_cm"]) if selected["kind"] == "strafe" else 0.0,
            ),
        )
        self.debug.event(
            "translation_preferred",
            navigation_mode="body_translation",
            action=selected["kind"],
            selected_action=selected["kind"],
            target_local_forward_cm=round(float(forward), 2),
            target_local_lateral_cm=round(float(lateral), 2),
            target_rear_angle_error_deg=round(float(rear_angle_error), 2),
            reverse_preferred=selected["kind"] == "reverse",
            reverse_rejected_reason=reverse_rejected_reason or None,
            translation_candidate_cost=round(float(selected["planned_cm"]), 2),
            turn_candidate_cost=float(nav_cfg.get("action_planner_turn_fixed_cost_cm", 20.0)),
            turn_penalty=float(nav_cfg.get("action_planner_turn_fixed_cost_cm", 20.0)),
            path_length_cm=round(abs(float(selected["planned_cm"])), 2),
            path_obstacle_cost=round(float(corridor.get("path_obstacle_cost", 0.0)), 2),
            minimum_wall_clearance_cm=round(
                float(corridor.get("minimum_wall_clearance_cm", 0.0)), 2
            ),
            wall_clearance_penalty=round(
                max(
                    0.0,
                    float(nav_cfg.get("normal_wall_clearance_target_cm", 25.0))
                    - float(corridor.get("minimum_wall_clearance_cm", 0.0)),
                )
                * float(nav_cfg.get("normal_wall_clearance_penalty_scale", 4.0)),
                2,
            ),
            target_direct_cost_exemption=False,
            movement_corridor_clear=bool(corridor.get("clear", False)),
        )
        return selected

    def execute_translation_action(
        self,
        action: dict,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        goal_dist_cm: float,
        context: dict,
    ) -> str:
        pose_before_action = self.copy_pose(pose)
        detail = dict(context)
        detail.update(
            {
                "action": action["kind"],
                "selected_action": action["kind"],
                "progress_cm": round(float(action.get("progress_cm", 0.0)), 1),
                "planned_cm": round(float(action.get("planned_cm", 0.0)), 1),
                "forward_component_cm": round(float(action.get("forward_cm", 0.0)), 1),
                "lateral_component_cm": round(float(action.get("lateral_cm", 0.0)), 1),
                "target_local_forward_cm": round(float(action.get("forward_cm", 0.0)), 1),
                "target_local_lateral_cm": round(float(action.get("lateral_cm", 0.0)), 1),
                "waypoint": (round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
            }
        )
        self.debug.event("translation_step", **detail)
        if action["kind"] == "reverse":
            self.forward_map_block_count = 0
            key = "back_fast"
            step_cm = abs(float(self.config["motion"]["actions"][key].get("forward_cm", -2.5)))
            requested = self.motion.reverse_cycles_for_distance(float(action["distance_cm"]))
            near_wall = self.near_wall_now(pose)
            cycles, batch_reason = self.select_adaptive_action_batch(
                "reverse",
                requested,
                step_cm,
                abs(float(action["distance_cm"])),
                goal_dist_cm,
                near_wall=near_wall,
            )
            travel = -cycles * step_cm
            end_xy = self.translated_pose_xy(pose, forward_cm=travel)
            corridor = self.movement_corridor_metrics(pose.xy(), end_xy)
            if not corridor["clear"]:
                self.debug.event(
                    "reverse_rejected",
                    navigation_mode="normal",
                    selected_action=None,
                    reverse_preferred=False,
                    reverse_rejected_reason="rear_corridor_blocked_before_execute",
                    movement_corridor_clear=False,
                    target_local_forward_cm=round(float(action.get("forward_cm", 0.0)), 2),
                    target_local_lateral_cm=round(float(action.get("lateral_cm", 0.0)), 2),
                )
                self.localize_scan()
                return "recovered"
            self.debug.event(
                "action_batch_started",
                action=key,
                selected_action="reverse",
                requested_action_cycles=requested,
                selected_action_cycles=cycles,
                adaptive_batch_reason=batch_reason,
                movement_corridor_clear=True,
                minimum_wall_clearance_cm=round(
                    float(corridor["minimum_wall_clearance_cm"]), 2
                ),
                **context
            )
            result = self.motion.run(key, times_override=cycles)
            self.debug.event(
                "action_batch_completed",
                action=key,
                selected_action="reverse",
                actual_action_cycles=getattr(
                    result, "executed_times", result.times if result.ok else 0
                ),
                ok=result.ok,
                **context
            )
            if not result.ok:
                self.last_navigation_failure_reason = "hardware_failure"
                return "failed"
            self.clear_turn_progress_watchdog("successful_reverse")
            self.post_action_relocalize("translation_reverse", pose_before_action, result, waypoint)
            return "moved"

        if action["kind"] == "strafe":
            self.forward_map_block_count = 0
            key = "strafe_left_fast" if float(action["distance_cm"]) > 0.0 else "strafe_right_fast"
            step_cm = abs(float(self.config["motion"]["actions"][key].get("lateral_cm", 4.0)))
            requested = self.motion.lateral_cycles_for_distance(float(action["distance_cm"]))
            near_wall = self.near_wall_now(pose)
            cycles, batch_reason = self.select_adaptive_action_batch(
                "strafe", requested, step_cm, abs(float(action["distance_cm"])), goal_dist_cm,
                near_wall=near_wall,
            )
            travel = math.copysign(cycles * step_cm, float(action["distance_cm"]))
            end_xy = self.translated_pose_xy(pose, lateral_cm=travel)
            corridor = self.movement_corridor_metrics(pose.xy(), end_xy)
            if not corridor["clear"]:
                self.debug.event(
                    "translation_corridor_blocked",
                    selected_action="strafe",
                    movement_corridor_clear=False,
                    **context
                )
                self.localize_scan()
                return "recovered"
            detail.update(requested_action_cycles=requested, adaptive_batch_reason=batch_reason)
            self.debug.event("action_batch_started", action=key, requested_action_cycles=requested, selected_action_cycles=cycles, **context)
            result = self.motion.run(key, times_override=cycles)
            self.debug.event("action_batch_completed", action=key, actual_action_cycles=getattr(result, "executed_times", result.times if result.ok else 0), ok=result.ok, **context)
            if not result.ok:
                self.last_navigation_failure_reason = "hardware_failure"
                return "failed"
            self.clear_turn_progress_watchdog("successful_translation")
            self.post_action_relocalize("translation_strafe", pose_before_action, result, waypoint)
            return "moved"

        if self.front_obstacle_visible():
            self.debug.event("front_obstacle_recover", **context)
            reason = str(context.get("reason", "front_obstacle_visible"))
            self.recover_toward_field_center(reason + ":front_obstacle_visible", backoff=True)
            return "recovered"

        forward_dist = min(float(action["distance_cm"]), goal_dist_cm)
        planned_forward_cm = self.planned_forward_step_cm(forward_dist)
        map_check_min = float(self.config["navigation"].get("forward_map_check_min_cm", 16.0))
        if planned_forward_cm >= map_check_min and not self.forward_clear_for_distance(pose, planned_forward_cm):
            self.forward_map_block_count += 1
            block_detail = dict(context)
            block_detail.update(
                {
                    "requested_forward_cm": round(forward_dist, 1),
                    "checked_forward_cm": round(planned_forward_cm, 1),
                    "count": self.forward_map_block_count,
                    "recover_limit": int(self.config["navigation"].get("forward_map_block_recover_limit", 2)),
                    "outward_facing": self.is_facing_outside(pose),
                    "exit_dist_cm": round(self.distance_to_field_exit_ahead(pose), 1),
                }
            )
            self.debug.event("forward_blocked_by_map", **block_detail)
            if self.forward_map_block_count < int(self.config["navigation"].get("forward_map_block_recover_limit", 2)):
                self.localize_scan()
            else:
                self.forward_map_block_count = 0
                reason = str(context.get("reason", "forward_map_blocked"))
                self.recover_toward_field_center(reason + ":forward_map_blocked", backoff=True)
            return "recovered"

        self.forward_map_block_count = 0
        pose_before_forward = RobotPose(
            pose.x_cm,
            pose.y_cm,
            pose.yaw_deg,
            confidence=pose.confidence,
            source=pose.source,
            last_update_s=pose.last_update_s,
        )
        visual_before = None
        if self.visual_progress_check_enabled():
            visual_before = self.capture_visual_progress_frame()
        step_cm = abs(float(self.config["motion"]["actions"]["forward_fast"].get("forward_cm", 3.5)))
        requested = self.motion.forward_cycles_for_distance(forward_dist)
        near_wall = self.near_wall_now(pose)
        cycles, batch_reason = self.select_adaptive_action_batch(
            "forward", requested, step_cm, forward_dist, goal_dist_cm,
            near_wall=near_wall,
        )
        self.debug.event("action_batch_started", action="forward_fast", requested_action_cycles=requested, selected_action_cycles=cycles, adaptive_batch_reason=batch_reason, **context)
        result = self.motion.run("forward_fast", times_override=cycles)
        self.debug.event("action_batch_completed", action="forward_fast", actual_action_cycles=getattr(result, "executed_times", result.times if result.ok else 0), ok=result.ok, **context)
        if not result.ok:
            self.last_navigation_failure_reason = "hardware_failure"
            return "failed"
        self.clear_turn_progress_watchdog("successful_translation")
        self.set_pending_forward_progress(pose_before_forward, abs(float(result.model_forward_cm)))
        self.evaluate_visual_forward_progress(visual_before, abs(float(result.model_forward_cm)))
        if self.collision_recovery_pending:
            reason = str(context.get("reason", "visual_forward_no_progress"))
            self.recover_toward_field_center(reason + ":visual_forward_no_progress", backoff=True)
            return "recovered"
        self.post_action_relocalize("translation_forward", pose_before_forward, result, waypoint)
        return "moved"

    def clear_navigation_noop(self) -> None:
        self.navigation_noop_count = 0

    @staticmethod
    def copy_pose(pose: RobotPose) -> RobotPose:
        return RobotPose(
            pose.x_cm,
            pose.y_cm,
            pose.yaw_deg,
            confidence=pose.confidence,
            source=pose.source,
            last_update_s=pose.last_update_s,
        )

    def clear_turn_progress_watchdog(self, reason: str = "") -> None:
        previous = self.turn_no_progress_count
        self.turn_no_progress_count = 0
        self.turn_progress_failure_start_diff = None
        if previous:
            self.debug.event("turn_progress_restored", reason=reason, previous_count=previous)

    def monitor_turn_result(
        self,
        before_pose: RobotPose,
        target_yaw: float,
        action_result,
        reason: str,
    ) -> bool:
        """Scan after a turn and stop navigation after repeated visual failure."""
        if not bool(getattr(action_result, "ok", True)):
            self.last_navigation_failure_reason = "hardware_failure"
            self.debug.event(
                "action_batch_completed",
                action=getattr(action_result, "key", "turn"),
                actual_action_cycles=getattr(action_result, "executed_times", 0) or 0,
                ok=False,
                error=getattr(action_result, "error", "turn_failed"),
            )
            return False
        outcome = self.scan_after_turn(
            reason,
            action_result.key,
            action_result,
            before_pose=before_pose,
            target_yaw=target_yaw,
        )
        self.debug.event(
            "post_action_replan",
            action=action_result.key,
            post_action_relocalized=bool(outcome.get("accepted")),
            post_action_replanned=False,
            replan_requested=True,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
        )
        self.pending_post_action_replan = True
        failed = bool(outcome.get("turn_no_progress") or outcome.get("direction_conflict"))
        if not failed:
            improvement = outcome.get("target_improvement_deg")
            if outcome.get("accepted") and (improvement is None or float(improvement) >= 2.0):
                self.clear_turn_progress_watchdog("turn_progress")
            return True

        self.turn_no_progress_count += 1
        if self.turn_no_progress_count == 1:
            self.turn_progress_failure_start_diff = outcome.get("diff_before")
        if self.turn_no_progress_count < 2:
            return True

        baseline_diff = self.turn_progress_failure_start_diff
        self.debug.event(
            "turn_progress_relocalize",
            reason=reason,
            count=self.turn_no_progress_count,
            target_yaw=round(float(target_yaw), 2),
        )
        localized = self.localize_scan(reset_turn_watchdog=False)
        relocalized_pose = self.state.pose if localized else None
        relocalized_diff = None
        improved = False
        if relocalized_pose is not None and baseline_diff is not None:
            relocalized_diff = angle_diff_deg(target_yaw, relocalized_pose.yaw_deg)
            improved = abs(float(baseline_diff)) - abs(relocalized_diff) >= 2.0
        if improved:
            self.clear_turn_progress_watchdog("forced_relocalize_improved")
            return True

        detail = {
            "before_yaw": round(float(outcome.get("before_yaw", before_pose.yaw_deg)), 3),
            "after_yaw": None if outcome.get("after_yaw") is None else round(float(outcome["after_yaw"]), 3),
            "expected_delta": round(float(outcome.get("expected_delta", action_result.model_yaw_deg)), 3),
            "actual_delta": None if outcome.get("actual_delta") is None else round(float(outcome["actual_delta"]), 3),
            "target_yaw": round(float(target_yaw), 3),
            "diff_before": None if baseline_diff is None else round(float(baseline_diff), 3),
            "diff_after": None if relocalized_diff is None else round(float(relocalized_diff), 3),
            "count": self.turn_no_progress_count,
            "localized": bool(localized),
        }
        self.debug.event("turn_progress_failed", **detail)
        print("[turn_progress_failed] {}".format(detail))
        self.turn_navigation_abort = True
        self.last_navigation_failure_reason = "RECOVERY_NO_PROGRESS"
        return False

    def handle_navigation_noop(self, reason: str, waypoint: Tuple[float, float], diff: float, extra: Optional[dict] = None) -> None:
        self.navigation_noop_count += 1
        detail = dict(extra or {})
        detail.update(
            {
                "reason": reason,
                "stop_reason": "within_turn_tolerance_no_translation",
                "count": self.navigation_noop_count,
                "diff_yaw": round(float(diff), 1),
                "waypoint": (round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
            }
        )
        self.debug.event("turn_last_resort_noop", **detail)
        if self.navigation_noop_count >= 2:
            self.debug.event("navigation_noop_recover", **detail)
            self.clear_navigation_noop()
            self.localize_scan()

    def should_strafe_toward(self, pose: RobotPose, waypoint: Tuple[float, float], diff_yaw: float) -> Tuple[bool, float]:
        forward, lateral = self.local_vector_to(pose, waypoint)
        min_lateral = float(self.config["navigation"].get("strafe_min_lateral_cm", 7.0))
        max_yaw = float(self.config["navigation"].get("strafe_prefer_max_yaw_deg", 70.0))
        if forward < -4.0:
            return False, lateral
        if abs(lateral) < min_lateral:
            return False, lateral
        if abs(diff_yaw) > max_yaw:
            return False, lateral
        if not self.strafe_target_clear(pose, lateral):
            return False, lateral
        return True, lateral

    def field_center_xy(self) -> Tuple[float, float]:
        return self.map.width_cm / 2.0, self.map.height_cm / 2.0

    def distance_to_field_exit_ahead(self, pose: RobotPose) -> float:
        x = float(pose.x_cm)
        y = float(pose.y_cm)
        dx = math.cos(math.radians(pose.yaw_deg))
        dy = math.sin(math.radians(pose.yaw_deg))
        candidates = []
        if abs(dx) > 1e-6:
            candidates.append((0.0 - x) / dx)
            candidates.append((self.map.width_cm - x) / dx)
        if abs(dy) > 1e-6:
            candidates.append((0.0 - y) / dy)
            candidates.append((self.map.height_cm - y) / dy)
        forward_hits = [item for item in candidates if item > 0.0]
        if not forward_hits:
            return float("inf")
        return min(forward_hits)

    def is_facing_outside(self, pose: Optional[RobotPose] = None) -> bool:
        pose = self.state.pose if pose is None else pose
        if pose is None:
            return False
        exit_dist = self.distance_to_field_exit_ahead(pose)
        threshold = float(self.config["navigation"].get("outward_exit_distance_cm", 62.0))
        return exit_dist <= threshold

    def distance_to_nearest_boundary(self, pose: RobotPose) -> float:
        return min(
            float(pose.x_cm),
            float(pose.y_cm),
            self.map.width_cm - float(pose.x_cm),
            self.map.height_cm - float(pose.y_cm),
        )

    def is_near_boundary(self, pose: Optional[RobotPose] = None) -> bool:
        pose = self.state.pose if pose is None else pose
        if pose is None:
            return False
        margin = float(self.config["navigation"].get("boundary_trapped_margin_cm", 45.0))
        return self.distance_to_nearest_boundary(pose) <= margin

    def is_boundary_trapped(self, pose: Optional[RobotPose] = None, reason: str = "") -> bool:
        pose = self.state.pose if pose is None else pose
        if pose is None:
            return False
        if not self.is_near_boundary(pose):
            return False
        if self.is_facing_outside(pose):
            return True
        limit = int(self.config["navigation"].get("no_tag_recovery_failures", 2))
        if self.consecutive_no_tag_scans >= limit or self.consecutive_localize_failures >= limit:
            return True
        return any(key in reason for key in ("near_wall", "map_blocked", "no_tag", "front_obstacle"))

    def yaw_toward_field_center(self, pose: RobotPose) -> float:
        cx, cy = self.field_center_xy()
        return math.degrees(math.atan2(cy - pose.y_cm, cx - pose.x_cm))

    def distance_to_field_exit_for_yaw(self, pose: RobotPose, yaw_deg: float) -> float:
        probe = RobotPose(
            pose.x_cm,
            pose.y_cm,
            yaw_deg,
            confidence=pose.confidence,
            source=pose.source,
            last_update_s=pose.last_update_s,
        )
        return self.distance_to_field_exit_ahead(probe)

    def choose_boundary_safe_turn_key(self, pose: RobotPose, target_yaw: float) -> str:
        actions = self.config["motion"]["actions"]
        left_step = abs(float(actions.get("turn_left_large", actions["turn_left_fast"]).get("yaw_deg", 45.0)))
        right_step = abs(float(actions.get("turn_right_large", actions["turn_right_fast"]).get("yaw_deg", -45.0)))
        candidates = [
            ("turn_left_large" if "turn_left_large" in actions else "turn_left_fast", pose.yaw_deg + left_step),
            ("turn_right_large" if "turn_right_large" in actions else "turn_right_fast", pose.yaw_deg - right_step),
        ]
        best_key = candidates[0][0]
        best_score = -float("inf")
        for key, yaw in candidates:
            yaw = normalize_angle_deg(yaw)
            target_error = abs(angle_diff_deg(target_yaw, yaw))
            exit_dist = self.distance_to_field_exit_for_yaw(pose, yaw)
            score = -target_error + 0.35 * min(exit_dist, 120.0)
            if score > best_score:
                best_score = score
                best_key = key
        return best_key

    def turn_toward_yaw_for_recovery(self, target_yaw: float) -> bool:
        attempts = max(1, int(self.config["navigation"].get("collision_recovery_turn_attempts", 3)))
        tolerance = float(self.config["navigation"].get("turn_tolerance_deg", 20.0))
        for _ in range(attempts):
            pose = self.state.pose
            if pose is None:
                result = self.motion.run("turn_left_large")
                self.scan_after_turn("recovery_pose_missing_turn", "turn_left_large", result)
                continue
            diff = angle_diff_deg(target_yaw, pose.yaw_deg)
            if abs(diff) <= tolerance:
                return True
            before_pose = self.copy_pose(pose)
            if bool(self.config["navigation"].get("boundary_safe_turn_enabled", True)) and self.is_near_boundary(pose):
                key = self.choose_boundary_safe_turn_key(pose, target_yaw)
                self.debug.event(
                    "boundary_safe_turn",
                    key=key,
                    current_yaw=round(pose.yaw_deg, 1),
                    target_yaw=round(float(target_yaw), 1),
                    exit_dist_cm=round(self.distance_to_field_exit_ahead(pose), 1),
                )
                result = self.motion.run(key)
                if not self.monitor_turn_result(before_pose, target_yaw, result, "boundary_safe_recovery_turn"):
                    return False
            else:
                result = self.motion.turn_toward(diff)
                if result is not None:
                    if not self.monitor_turn_result(before_pose, target_yaw, result, "recovery_turn_toward"):
                        return False
        return not self.turn_navigation_abort

    def turn_toward_yaw_boundary_aware(self, target_yaw: float) -> bool:
        pose = self.state.pose
        if pose is None:
            result = self.motion.run("turn_left_large")
            self.scan_after_turn("pose_missing_turn", "turn_left_large", result)
            return not self.turn_navigation_abort
        diff = angle_diff_deg(target_yaw, pose.yaw_deg)
        if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
            return True
        if bool(self.config["navigation"].get("boundary_safe_turn_enabled", True)) and self.is_near_boundary(pose):
            return self.turn_toward_yaw_for_recovery(target_yaw)
        before_pose = self.copy_pose(pose)
        result = self.motion.turn_toward(diff)
        if result is not None:
            return self.monitor_turn_result(before_pose, target_yaw, result, "turn_toward")
        return True

    def indoor_recovery_candidates(self, pose: RobotPose):
        max_dist = float(self.config["navigation"].get("boundary_recovery_max_target_distance_cm", 170.0))
        center_xy = self.field_center_xy()
        pull = float(self.config["navigation"].get("boundary_recovery_center_pull_cm", 65.0))
        dx = center_xy[0] - pose.x_cm
        dy = center_xy[1] - pose.y_cm
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            inward_xy = (
                pose.x_cm + min(pull, dist) * dx / dist,
                pose.y_cm + min(pull, dist) * dy / dist,
            )
            inward_xy = self.map.nearest_free_xy(inward_xy)
            path = self.map.plan(pose.xy(), inward_xy)
            if path:
                path_dist = sum(distance_xy(a, b) for a, b in zip(path, path[1:])) if len(path) > 1 else distance_xy(pose.xy(), inward_xy)
                yield {
                    "kind": "inward",
                    "xy": inward_xy,
                    "path": path,
                    "score": path_dist,
                    "distance_cm": path_dist,
                }
        for screen in self.map.screens.values():
            target_xy = screen.target_xy
            dx = target_xy[0] - pose.x_cm
            dy = target_xy[1] - pose.y_cm
            dist = math.hypot(dx, dy)
            if dist > max_dist:
                continue
            path = self.plan_navigation_path(pose, target_xy)
            if not path:
                continue
            path_dist = sum(distance_xy(a, b) for a, b in zip(path, path[1:])) if len(path) > 1 else distance_xy(pose.xy(), target_xy)
            score = path_dist + 0.2 * abs(angle_diff_deg(screen.interaction_yaw_deg, pose.yaw_deg))
            yield {
                "kind": "screen_viewpoint",
                "screen_id": int(screen.screen_id),
                "xy": target_xy,
                "path": path,
                "score": score,
                "distance_cm": path_dist,
            }

    def choose_boundary_recovery_target(self, pose: RobotPose):
        candidates = list(self.indoor_recovery_candidates(pose))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["score"])
        return candidates[0]

    def blind_navigate_to_xy(self, target_xy: Tuple[float, float], reason: str) -> bool:
        max_steps = int(self.config["navigation"].get("boundary_recovery_max_steps", 18))
        radius = float(self.config["navigation"].get("boundary_recovery_goal_radius_cm", 20.0))
        self.debug.event("boundary_blind_nav_start", reason=reason, target_xy=target_xy, max_steps=max_steps)
        for step in range(max_steps):
            pose = self.state.pose
            if pose is None:
                return False
            dist = distance_xy(pose.xy(), target_xy)
            if dist <= radius:
                self.debug.event("boundary_blind_nav_arrived", step=step, distance_cm=round(dist, 1))
                return True
            path = self.plan_navigation_path(pose, target_xy)
            waypoint = self.select_navigation_waypoint(pose, path, target_xy)
            desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
            diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
            self.debug.event(
                "boundary_blind_nav_step",
                step=step + 1,
                distance_cm=round(dist, 1),
                waypoint=waypoint,
                desired_yaw=round(desired_yaw, 1),
                diff_yaw=round(diff, 1),
            )
            if abs(diff) > float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
                self.turn_toward_yaw_for_recovery(desired_yaw)
                continue
            forward_cm = min(dist, distance_xy(pose.xy(), waypoint))
            planned_forward_cm = self.planned_forward_step_cm(forward_cm)
            if not self.forward_clear_for_distance(pose, planned_forward_cm):
                self.debug.event("boundary_blind_nav_forward_blocked", checked_forward_cm=round(planned_forward_cm, 1))
                return False
            self.motion.move_forward(forward_cm)
            self.publish_state(path=path)
        self.debug.event("boundary_blind_nav_failed", target_xy=target_xy, max_steps=max_steps)
        return False

    def recover_via_indoor_waypoint(self, reason: str) -> bool:
        if not bool(self.config["navigation"].get("boundary_recovery_enabled", True)):
            return False
        pose = self.state.pose
        if pose is None or not self.is_boundary_trapped(pose, reason):
            return False
        target = self.choose_boundary_recovery_target(pose)
        if target is None:
            self.debug.event("boundary_recovery_no_indoor_waypoint", reason=reason, pose=pose.as_dict())
            return False
        self.last_recovery["boundary_target"] = {
            "kind": target.get("kind"),
            "screen_id": target.get("screen_id"),
            "xy": list(target["xy"]),
            "distance_cm": round(float(target["distance_cm"]), 1),
            "score": round(float(target["score"]), 1),
        }
        self.debug.event("boundary_recovery_target_selected", **self.last_recovery["boundary_target"])
        ok = self.blind_navigate_to_xy(tuple(target["xy"]), reason=reason)
        if not ok:
            return False
        return self.localize_scan()

    def wall_clearance_cm(self, pose: RobotPose, yaw_deg: Optional[float] = None) -> float:
        """Measure free map distance along a body-relative ray."""
        if yaw_deg is None and hasattr(self.map, "robot_clearance_cm"):
            return float(self.map.robot_clearance_cm(pose.xy()))
        yaw = float(pose.yaw_deg if yaw_deg is None else yaw_deg)
        nav = self.config["navigation"]
        max_distance = max(
            40.0,
            float(nav.get("safe_wall_distance_cm", 17.0))
            + float(nav.get("near_wall_backoff_step_cm", 5.0))
            + 10.0,
        )
        rad = math.radians(yaw)
        distance = 0.0
        while distance < max_distance:
            distance += 1.0
            point = (
                pose.x_cm + distance * math.cos(rad),
                pose.y_cm + distance * math.sin(rad),
            )
            if not self.map.is_free_xy(point):
                return max(0.0, distance - 1.0)
        return max_distance

    def recovery_translation_clear(self, pose: RobotPose, forward_cm: float = 0.0, lateral_cm: float = 0.0) -> bool:
        target = self.translated_pose_xy(pose, forward_cm=forward_cm, lateral_cm=lateral_cm)
        return self.movement_corridor_clear(pose.xy(), target)

    def near_wall_now(self, pose: RobotPose) -> bool:
        if hasattr(self.map, "robot_clearance_cm"):
            return self.map.robot_clearance_cm(pose.xy()) < float(
                self.config["navigation"]["safe_wall_distance_cm"]
            )
        return self.map.is_dangerously_close_to_wall(
            pose.xy(),
            pose.yaw_deg,
            float(self.config["navigation"]["safe_wall_distance_cm"]),
        )

    def recovery_action_cycles(self, key: str, requested: float, model_field: str) -> int:
        configured = abs(float(self.config["motion"]["actions"][key].get(model_field, requested)))
        return max(1, int(math.ceil(abs(float(requested)) / max(0.1, configured))))

    def execute_near_wall_recovery_action(
        self,
        key: str,
        phase: str,
        attempt: int,
        times: int = 1,
    ) -> NearWallRecoveryResult:
        """Execute one recovery action and verify it using a fresh localization."""
        before = self.state.pose
        if before is None:
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        total_limit = max(1, int(self.config["navigation"].get("near_wall_recovery_max_total_actions", 12)))
        if getattr(self, "near_wall_recovery_actions", 0) >= total_limit:
            self.debug.event(
                "near_wall_recovery_relocalize",
                reason="recovery_action_budget_exhausted",
                near_wall_recovery_actions=self.near_wall_recovery_actions,
                current_target_screen_id=getattr(self, "current_target_screen_id", None),
                target_preserved=getattr(self, "current_target_screen_id", None) is not None,
            )
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        before = self.copy_pose(before)
        before_clearance = self.wall_clearance_cm(before)
        self.debug.event(
            "near_wall_recovery_before",
            phase=phase,
            attempt=attempt,
            action=key,
            pose=before.as_dict(),
            yaw=before.yaw_deg,
            wall_distance_cm=round(before_clearance, 2),
        )
        self.debug.event(
            "near_wall_recovery_action",
            phase=phase,
            attempt=attempt,
            action=key,
            times=times,
        )
        max_cycles = max(1, int(self.config["navigation"].get("recovery_max_action_cycles", 1)))
        times = min(max_cycles, max(1, int(times)))
        result = self.motion.run(key, times_override=times)
        self.near_wall_recovery_actions = getattr(self, "near_wall_recovery_actions", 0) + int(
            getattr(result, "executed_times", result.times if result.ok else 0) or 0
        )
        if not result.ok:
            self.last_navigation_failure_reason = "hardware_failure"
            return NearWallRecoveryResult.HARDWARE_FAILURE
        self.hardware.center_head()
        localized = bool(self.localize_scan())
        after = self.state.pose
        after_clearance = None if after is None else self.wall_clearance_cm(after)
        position_delta = 0.0
        yaw_delta = 0.0
        clearance_delta = 0.0
        if localized and after is not None:
            position_delta = distance_xy(before.xy(), after.xy())
            yaw_delta = angle_diff_deg(after.yaw_deg, before.yaw_deg)
            clearance_delta = float(after_clearance) - before_clearance
        no_progress = (
            not localized
            or after is None
            or (
                position_delta < 1.0
                and abs(yaw_delta) < 1.0
                and abs(clearance_delta) < 1.0
            )
        )
        if no_progress:
            self.near_wall_recovery_no_progress_count = getattr(
                self, "near_wall_recovery_no_progress_count", 0
            ) + 1
        else:
            self.near_wall_recovery_no_progress_count = 0
        self.debug.event(
            "near_wall_recovery_after",
            phase=phase,
            attempt=attempt,
            action=key,
            localized=localized,
            pose=None if after is None else after.as_dict(),
            yaw=None if after is None else after.yaw_deg,
            wall_distance_cm=None if after_clearance is None else round(after_clearance, 2),
            position_delta_cm=round(position_delta, 2),
            yaw_delta_deg=round(yaw_delta, 2),
            wall_distance_delta_cm=round(clearance_delta, 2),
            no_progress=no_progress,
            no_progress_count=self.near_wall_recovery_no_progress_count,
            requested_action_cycles=times,
            actual_action_cycles=getattr(result, "executed_times", result.times if result.ok else 0),
            wall_clearance_before=round(before_clearance, 2),
            wall_clearance_after=None if after_clearance is None else round(after_clearance, 2),
        )
        threshold = max(
            1,
            int(self.config["navigation"].get("near_wall_recovery_no_progress_threshold", 2)),
        )
        if self.near_wall_recovery_no_progress_count >= threshold:
            self.last_navigation_failure_reason = "near_wall_recovery_exhausted"
            self.debug.event(
                "near_wall_recovery_no_progress",
                error="RECOVERY_NO_PROGRESS",
                phase=phase,
                attempt=attempt,
                action=key,
                count=self.near_wall_recovery_no_progress_count,
                before_pose=before.as_dict(),
                after_pose=None if after is None else after.as_dict(),
                before_wall_distance_cm=round(before_clearance, 2),
                after_wall_distance_cm=None if after_clearance is None else round(after_clearance, 2),
            )
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED if not localized else NearWallRecoveryResult.STILL_NEAR_WALL
        if localized and after is not None and not self.near_wall_now(after):
            self.near_wall_recovery_no_progress_count = 0
            self.clear_turn_progress_watchdog("near_wall_recovery_success")
            self.debug.event(
                "near_wall_recovery_success",
                phase=phase,
                attempt=attempt,
                action=key,
                pose=after.as_dict(),
                wall_distance_cm=round(float(after_clearance), 2),
            )
            return NearWallRecoveryResult.RECOVERED
        if not localized or after is None:
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        improvement = float(self.config["navigation"].get("near_wall_min_clearance_improvement_cm", 1.5))
        if clearance_delta >= improvement:
            self.debug.event(
                "near_wall_clearance_improved",
                phase=phase,
                improvement_cm=round(clearance_delta, 2),
                current_target_screen_id=getattr(self, "current_target_screen_id", None),
            )
            return NearWallRecoveryResult.RETRY_WITH_NEW_POSE
        return NearWallRecoveryResult.STILL_NEAR_WALL

    def choose_near_wall_lateral_direction(self, pose: RobotPose, step_cm: float, excluded=None) -> Optional[float]:
        excluded = set(excluded or [])
        map_obj = getattr(self, "map", None)
        candidates = []
        for direction, yaw in ((1.0, pose.yaw_deg + 90.0), (-1.0, pose.yaw_deg - 90.0)):
            if direction in excluded:
                continue
            distance = direction * abs(float(step_cm))
            if self.recovery_translation_clear(pose, lateral_cm=distance):
                xy = self.translated_pose_xy(pose, lateral_cm=distance)
                candidate_pose = RobotPose(xy[0], xy[1], pose.yaw_deg, pose.confidence, pose.source, pose.last_update_s)
                front_clearance = self.wall_clearance_cm(candidate_pose)
                boundary_clearance = min(xy[0], map_obj.width_cm - xy[0], xy[1], map_obj.height_cm - xy[1]) if map_obj is not None and hasattr(map_obj, "width_cm") else 0.0
                target = getattr(self, "current_target_screen_id", None)
                target_screen = map_obj.screens.get(target) if target is not None and map_obj is not None and hasattr(map_obj, "screens") else None
                away = 0.0
                if target_screen is not None:
                    target_xy = target_screen.task_target_xy or target_screen.target_xy
                    away = max(0.0, distance_xy(xy, target_xy) - distance_xy(pose.xy(), target_xy))
                candidates.append((front_clearance, boundary_clearance, -away, direction))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    def register_near_wall_recovery_stall(self, reason: str, action: str = "none") -> bool:
        """Count a recovery episode where no safe action could make progress."""
        self.near_wall_recovery_no_progress_count = getattr(
            self, "near_wall_recovery_no_progress_count", 0
        ) + 1
        threshold = max(1, int(
            self.config["navigation"].get("near_wall_recovery_no_progress_threshold", 2)
        ))
        self.debug.event(
            "near_wall_recovery_stall",
            reason=reason,
            action=action,
            count=self.near_wall_recovery_no_progress_count,
            threshold=threshold,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
        )
        if self.near_wall_recovery_no_progress_count >= threshold:
            self.last_navigation_failure_reason = "near_wall_recovery_exhausted"
            self.debug.event(
                "near_wall_recovery_aborted",
                reason=reason,
                action=action,
                count=self.near_wall_recovery_no_progress_count,
            )
            return True
        return False

    def register_navigation_stall(
        self,
        pose: RobotPose,
        target_xy: Tuple[float, float],
        action: str,
        failure_reason: str,
    ) -> bool:
        signature = (
            getattr(self, "current_target_screen_id", None),
            round(float(target_xy[0]), 1),
            round(float(target_xy[1]), 1),
            round(float(pose.x_cm), 1),
            round(float(pose.y_cm), 1),
            round(float(pose.yaw_deg), 1),
            action,
            failure_reason,
        )
        if signature == getattr(self, "navigation_stall_signature", None):
            self.navigation_stall_count = getattr(self, "navigation_stall_count", 0) + 1
        else:
            self.navigation_stall_signature = signature
            self.navigation_stall_count = 1
        if self.navigation_stall_count < 2:
            return False
        self.last_navigation_failure_reason = "local_planner_stalled"
        self.debug.event(
            "navigation_stall_aborted",
            action=action,
            failure_reason=failure_reason,
            count=self.navigation_stall_count,
            pose=pose.as_dict(),
            target_xy=target_xy,
        )
        return True

    def recover_from_near_wall(self, reason: str) -> NearWallRecoveryResult:
        """Recover in a fixed backoff -> lateral -> small-turn order."""
        pose = self.state.pose
        if pose is None:
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        nav = self.config["navigation"]
        self.recovery_count += 1
        actions_before = int(getattr(self, "near_wall_recovery_actions", 0))
        map_obj = getattr(self, "map", None)
        target_screen = map_obj.screens.get(getattr(self, "current_target_screen_id", None)) if map_obj is not None and hasattr(map_obj, "screens") else None
        if target_screen is not None:
            self.preserve_current_target(target_screen, "near_wall_recovery_started")
        self.debug.event(
            "near_wall_recovery_started",
            reason=reason,
            pose=pose.as_dict(),
            yaw=pose.yaw_deg,
            wall_distance_cm=round(self.wall_clearance_cm(pose), 2),
            recovery_count=self.recovery_count,
            near_wall_source="non_target_wall",
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
            target_preserved=target_screen is not None,
        )

        back_step = abs(float(nav.get("near_wall_backoff_step_cm", 5.0)))
        for attempt in range(1, max(0, int(nav.get("near_wall_backoff_max_attempts", 2))) + 1):
            pose = self.state.pose
            if pose is None or not self.recovery_translation_clear(pose, forward_cm=-back_step):
                self.debug.event(
                    "near_wall_recovery_action",
                    phase="backoff",
                    attempt=attempt,
                    action="back_fast",
                    executed=False,
                    reason="rear_path_unsafe",
                )
                break
            self.debug.event("near_wall_backoff_attempt", attempt=attempt, step_cm=back_step)
            outcome = self.execute_near_wall_recovery_action("back_fast", "backoff", attempt, 1)
            if outcome == NearWallRecoveryResult.RECOVERED:
                return outcome
            if outcome in (NearWallRecoveryResult.LOCALIZATION_REQUIRED, NearWallRecoveryResult.HARDWARE_FAILURE):
                return outcome
            if outcome == NearWallRecoveryResult.STILL_NEAR_WALL:
                break

        lateral_step = abs(float(nav.get("near_wall_lateral_step_cm", 4.0)))
        tried_directions = set()
        for attempt in range(1, max(0, int(nav.get("near_wall_lateral_max_attempts", 2))) + 1):
            pose = self.state.pose
            if pose is None:
                break
            direction = self.choose_near_wall_lateral_direction(pose, lateral_step, tried_directions)
            if direction is None:
                self.debug.event(
                    "near_wall_recovery_action",
                    phase="lateral",
                    attempt=attempt,
                    action=None,
                    executed=False,
                    reason="no_safe_lateral_path",
                )
                break
            tried_directions.add(direction)
            key = "strafe_left_fast" if direction > 0.0 else "strafe_right_fast"
            self.debug.event("near_wall_lateral_attempt", attempt=attempt, action=key, direction=direction)
            outcome = self.execute_near_wall_recovery_action(key, "lateral", attempt, 1)
            if outcome == NearWallRecoveryResult.RECOVERED:
                return outcome
            if outcome in (NearWallRecoveryResult.LOCALIZATION_REQUIRED, NearWallRecoveryResult.HARDWARE_FAILURE):
                return outcome

        pose = self.state.pose
        if pose is not None:
            left_clearance = self.wall_clearance_cm(pose, pose.yaw_deg + 90.0)
            right_clearance = self.wall_clearance_cm(pose, pose.yaw_deg - 90.0)
            key = "turn_left_fast" if left_clearance >= right_clearance else "turn_right_fast"
            turn_step = abs(float(nav.get("near_wall_turn_step_deg", 7.5)))
            turn_safe = map_obj is None or not hasattr(map_obj, "rotation_sweep_clear") or map_obj.rotation_sweep_clear(
                pose.xy(),
                float(nav.get("turn_sweep_radius_cm", 10.0)),
                float(nav.get("normal_navigation_max_cost", 55.0)),
            )
            if not turn_safe:
                self.debug.event(
                    "near_wall_recovery_action",
                    phase="small_turn_last_resort",
                    action=key,
                    executed=False,
                    reason="rotation_sweep_blocked",
                )
                outcome = NearWallRecoveryResult.STILL_NEAR_WALL
            else:
                outcome = self.execute_near_wall_recovery_action(key, "small_turn_last_resort", 1, 1)
            if outcome in (NearWallRecoveryResult.RECOVERED, NearWallRecoveryResult.HARDWARE_FAILURE, NearWallRecoveryResult.LOCALIZATION_REQUIRED):
                return outcome

        self.debug.event(
            "near_wall_recovery_no_progress",
            error="near_wall_recovery_exhausted",
            reason=reason,
            count=self.near_wall_recovery_no_progress_count,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
            near_wall_recovery_result=NearWallRecoveryResult.STILL_NEAR_WALL.value,
            target_preserved=target_screen is not None,
        )
        if int(getattr(self, "near_wall_recovery_actions", 0)) == actions_before:
            self.register_near_wall_recovery_stall(reason, "all_recovery_actions_rejected")
        if self.near_wall_recovery_no_progress_count >= max(
            1, int(nav.get("near_wall_recovery_no_progress_threshold", 2))
        ):
            self.last_navigation_failure_reason = "near_wall_recovery_exhausted"
            self.debug.event(
                "near_wall_recovery_aborted",
                reason=reason,
                count=self.near_wall_recovery_no_progress_count,
            )
        self.debug.event(
            "near_wall_recovery_continue_same_target",
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
            target_preserved=target_screen is not None,
        )
        return NearWallRecoveryResult.STILL_NEAR_WALL

    def recover_toward_field_center(self, reason: str, backoff: bool = False) -> bool:
        if not bool(self.config["navigation"].get("collision_recovery_enabled", True)):
            return False
        pose = self.state.pose
        self.pending_progress_check = None
        self.collision_recovery_pending = False
        self.visual_no_progress_count = 0
        self.recovery_count += 1
        outward = False if pose is None else self.is_facing_outside(pose)
        max_attempts = max(1, int(self.config["navigation"].get("collision_recovery_localize_attempts", 3)))
        back_steps = max(1, int(self.config["navigation"].get("collision_recovery_back_steps", 1)))
        self.last_recovery = {
            "t": round(now_s(), 3),
            "reason": reason,
            "strategy": "repeat_backoff_then_localize",
            "backoff": bool(backoff),
            "back_steps": back_steps,
            "max_attempts": max_attempts,
            "outward_facing": bool(outward),
            "target_yaw": None,
            "pose": None if pose is None else pose.as_dict(),
            "count": self.recovery_count,
        }
        self.debug.event("recovery_start", **self.last_recovery)
        self.hardware.center_head()
        localized = False
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            if self.time_left_s() <= 0:
                break
            attempts_used = attempt
            do_backoff = bool(backoff) or attempt > 1
            if do_backoff:
                self.motion.run("back_fast", times_override=back_steps)
                self.hardware.center_head()
            localized = self.localize_scan(reason="no_tag_recovery", allow_pan_search=True)
            self.debug.event(
                "recovery_backoff_localize_attempt",
                reason=reason,
                attempt=attempt,
                max_attempts=max_attempts,
                backoff=do_backoff,
                back_steps=back_steps if do_backoff else 0,
                localized=localized,
                pose=None if self.state.pose is None else self.state.pose.as_dict(),
            )
            if localized:
                break
        self.debug.event(
            "recovery_done",
            reason=reason,
            localized=localized,
            attempts=attempts_used,
            via="repeat_backoff_then_localize",
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
        )
        return localized

    def no_tag_recovery_needed(self) -> bool:
        if not bool(self.config["navigation"].get("no_tag_recovery_enabled", True)):
            return False
        limit = int(self.config["navigation"].get("no_tag_recovery_failures", 2))
        if self.consecutive_no_tag_scans < limit:
            return False
        cooldown = float(self.config["navigation"].get("no_tag_recovery_cooldown_s", 4.0))
        if now_s() - self.last_no_tag_recovery_s < cooldown:
            return False
        pose = self.state.pose
        return pose is None or self.is_facing_outside(pose) or self.is_boundary_trapped(pose, "no_tag")

    def recover_from_no_tag_if_needed(self, reason: str) -> bool:
        if not self.no_tag_recovery_needed():
            return False
        self.last_no_tag_recovery_s = now_s()
        pose = self.state.pose
        backoff = True
        self.debug.event(
            "no_tag_recovery_triggered",
            reason=reason,
            no_tag_scans=self.consecutive_no_tag_scans,
            seconds_since_tag=round(now_s() - self.last_any_tag_seen_s, 2),
            outward_facing=False if pose is None else self.is_facing_outside(pose),
        )
        return self.recover_toward_field_center("no_tag:" + reason, backoff=backoff)

    def forward_clear_for_distance(
        self,
        pose: RobotPose,
        distance_cm: float,
        exact_goal_xy: Optional[Tuple[float, float]] = None,
    ) -> bool:
        if exact_goal_xy is not None:
            return self.movement_corridor_clear(
                pose.xy(),
                exact_goal_xy,
                allow_goal_high_cost=True,
            )
        margin = float(self.config["navigation"].get("forward_clearance_margin_cm", 10.0))
        travel = max(0.0, float(distance_cm) + margin)
        yaw = math.radians(pose.yaw_deg)
        target_xy = (
            pose.x_cm + travel * math.cos(yaw),
            pose.y_cm + travel * math.sin(yaw),
        )
        if not (0.0 <= target_xy[0] <= self.map.width_cm and 0.0 <= target_xy[1] <= self.map.height_cm):
            return False
        return self.movement_corridor_clear(pose.xy(), target_xy)

    def planned_forward_step_cm(self, distance_cm: float) -> float:
        cycles = self.motion.forward_cycles_for_distance(distance_cm)
        step = abs(float(self.config["motion"]["actions"]["forward_fast"].get("forward_cm", 3.5)))
        return cycles * step

    def set_pending_forward_progress(self, pose: RobotPose, expected_cm: float) -> None:
        if not bool(self.config["navigation"].get("progress_check_enabled", True)):
            return
        min_expected = float(self.config["navigation"].get("progress_check_expected_cm", 18.0))
        if expected_cm < min_expected:
            return
        self.pending_progress_check = {
            "start_xy": pose.xy(),
            "expected_cm": float(expected_cm),
            "t": now_s(),
        }

    def evaluate_pending_progress(self, pose: RobotPose) -> None:
        check = self.pending_progress_check
        if not check:
            return
        self.pending_progress_check = None
        actual = distance_xy(tuple(check["start_xy"]), pose.xy())
        expected = float(check["expected_cm"])
        min_actual = float(self.config["navigation"].get("progress_check_min_actual_cm", 6.0))
        if actual < min_actual:
            self.no_progress_count += 1
            self.debug.event(
                "forward_no_progress",
                expected_cm=round(expected, 1),
                actual_cm=round(actual, 1),
                count=self.no_progress_count,
            )
            if self.no_progress_count >= int(self.config["navigation"].get("progress_check_fail_limit", 2)):
                self.collision_recovery_pending = True
        else:
            if self.no_progress_count:
                self.debug.event("forward_progress_restored", actual_cm=round(actual, 1))
            self.no_progress_count = 0

    def visual_progress_check_enabled(self, expected_cm: float = 0.0) -> bool:
        nav = self.config["navigation"]
        if self.args.dry_run or not bool(nav.get("visual_progress_check_enabled", True)):
            return False
        return True

    def capture_visual_progress_frame(self):
        if self.args.dry_run:
            return None
        if not self.boundary_safe_pan_angles([100], reason="visual_progress_check", emit_event=False):
            self.debug.event("visual_progress_check_skipped_boundary_outward")
            return None
        self.hardware.set_head_pan_angle(100)
        return self.camera.capture_settled(discard_frames=1)

    def visual_progress_metrics(self, before, after) -> Optional[Dict[str, float]]:
        if before is None or after is None:
            return None
        import cv2
        import numpy as np

        width = int(self.config["navigation"].get("visual_progress_resize_width", 160))

        def prepare(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if width > 0 and w > width:
                scale = width / float(w)
                gray = cv2.resize(gray, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            return cv2.GaussianBlur(gray, (5, 5), 0)

        g0 = prepare(before)
        g1 = prepare(after)
        if g0.shape != g1.shape:
            g1 = cv2.resize(g1, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_AREA)
        mean_absdiff = float(np.mean(cv2.absdiff(g0, g1)))
        pts = cv2.goodFeaturesToTrack(g0, maxCorners=120, qualityLevel=0.01, minDistance=7)
        tracked = 0
        median_shift = 0.0
        if pts is not None and len(pts) > 0:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
            if nxt is not None and status is not None:
                mask = status.reshape(-1) == 1
                if np.any(mask):
                    shifts = np.linalg.norm(nxt.reshape(-1, 2)[mask] - pts.reshape(-1, 2)[mask], axis=1)
                    if shifts.size:
                        tracked = int(shifts.size)
                        median_shift = float(np.median(shifts))
        return {
            "mean_absdiff": mean_absdiff,
            "tracked_features": float(tracked),
            "median_feature_shift_px": median_shift,
        }

    def evaluate_visual_forward_progress(self, before_frame, expected_cm: float = 0.0) -> bool:
        if before_frame is None or not self.visual_progress_check_enabled():
            return False
        after_frame = self.capture_visual_progress_frame()
        metrics = self.visual_progress_metrics(before_frame, after_frame)
        if metrics is None:
            self.debug.event("visual_progress_check_inconclusive", expected_cm=round(float(expected_cm), 1), reason="missing_frame")
            return False
        nav = self.config["navigation"]
        min_absdiff = float(nav.get("visual_progress_min_absdiff", 4.5))
        min_shift = float(nav.get("visual_progress_min_feature_shift_px", 1.8))
        min_features = int(nav.get("visual_progress_min_tracked_features", 12))
        enough_features = int(metrics["tracked_features"]) >= min_features
        low_diff = float(metrics["mean_absdiff"]) < min_absdiff
        low_shift = (not enough_features) or float(metrics["median_feature_shift_px"]) < min_shift
        if not enough_features and not low_diff:
            self.debug.event(
                "visual_progress_check_inconclusive",
                expected_cm=round(float(expected_cm), 1),
                mean_absdiff=round(float(metrics["mean_absdiff"]), 3),
                tracked_features=int(metrics["tracked_features"]),
                reason="few_features",
            )
            return False
        if low_diff and low_shift:
            self.visual_no_progress_count += 1
            reason = "low_diff_low_shift" if enough_features else "low_diff_few_features"
            self.debug.event(
                "visual_forward_no_progress",
                reason=reason,
                expected_cm=round(float(expected_cm), 1),
                count=self.visual_no_progress_count,
                fail_limit=int(nav.get("visual_progress_fail_limit", 2)),
                mean_absdiff=round(float(metrics["mean_absdiff"]), 3),
                tracked_features=int(metrics["tracked_features"]),
                median_feature_shift_px=round(float(metrics["median_feature_shift_px"]), 3),
            )
            if self.visual_no_progress_count >= int(nav.get("visual_progress_fail_limit", 2)):
                self.collision_recovery_pending = True
            return True
        if self.visual_no_progress_count:
            self.debug.event(
                "visual_forward_progress_restored",
                mean_absdiff=round(float(metrics["mean_absdiff"]), 3),
                tracked_features=int(metrics["tracked_features"]),
                median_feature_shift_px=round(float(metrics["median_feature_shift_px"]), 3),
            )
        self.visual_no_progress_count = 0
        return False

    def navigate_to_xy(
        self,
        target_xy: Tuple[float, float],
        reason: str = "navigate_xy",
        arrival_radius_cm: Optional[float] = None,
        max_steps: Optional[int] = None,
        target_yaw_deg: Optional[float] = None,
        target_yaw_tolerance_deg: Optional[float] = None,
        allow_goal_high_cost: bool = False,
        target_screen: Optional[Screen] = None,
    ) -> bool:
        self.turn_navigation_abort = False
        self.last_navigation_failure_reason = ""
        self.near_wall_recovery_no_progress_count = 0
        self.near_wall_recovery_actions = 0
        self.navigation_stall_signature = None
        self.navigation_stall_count = 0
        self.clear_turn_progress_watchdog("navigate_xy_start")
        target_xy = (float(target_xy[0]), float(target_xy[1]))
        if allow_goal_high_cost and (
            not self.map.in_bounds_xy(target_xy) or not self.map.is_free_xy(target_xy)
        ):
            self.last_navigation_failure_reason = "exact_target_not_physically_free"
            self.debug.event(
                "navigate_xy_failed",
                reason=reason,
                failure_reason=self.last_navigation_failure_reason,
                target_xy=target_xy,
            )
            return False
        if not allow_goal_high_cost and not self.map.is_free_xy(target_xy):
            adjusted = self.map.nearest_free_xy(target_xy)
            self.debug.event("navigate_xy_target_adjusted", reason=reason, target_xy=target_xy, adjusted_xy=adjusted)
            target_xy = adjusted
        radius = float(arrival_radius_cm if arrival_radius_cm is not None else self.config["navigation"]["arrival_radius_cm"])
        max_steps = int(max_steps if max_steps is not None else self.config["navigation"]["max_steps_per_target"])
        for step in range(max_steps):
            if self.time_left_s() <= 0:
                self.debug.event("navigate_xy_stop", reason=reason, stop_reason="time_limit", target_xy=target_xy, step=step)
                return False
            if self.state.pose is None:
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed(reason + ":pose_missing")
                if self.state.pose is None:
                    self.last_navigation_failure_reason = "localization_required"
                    continue
            pose = self.state.pose
            if pose is None:
                continue
            localization_stop_threshold = max(
                2, int(self.config["navigation"].get("no_tag_recovery_failures", 2))
            )
            if int(getattr(self, "consecutive_localize_failures", 0)) >= localization_stop_threshold:
                self.debug.event(
                    "post_action_relocalize",
                    reason="consecutive_localization_failures",
                    post_action_relocalized=False,
                    normal_navigation_paused=True,
                    current_target_screen_id=getattr(self, "current_target_screen_id", None),
                    target_preserved=getattr(self, "current_target_screen_id", None) is not None,
                )
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed(reason + ":consecutive_localization_failures")
                continue
            dist = distance_xy(pose.xy(), target_xy)
            if dist <= radius:
                # Check facing direction if target_yaw_deg is specified
                if target_yaw_deg is not None:
                    yaw_diff = abs(angle_diff_deg(float(target_yaw_deg), pose.yaw_deg))
                    arrival_yaw_tolerance = float(
                        target_yaw_tolerance_deg
                        if target_yaw_tolerance_deg is not None
                        else self.config["navigation"].get("arrival_yaw_tolerance_deg", 30.0)
                    )
                    if yaw_diff > arrival_yaw_tolerance:
                        self.debug.event(
                            "navigate_xy_wrong_yaw",
                            reason=reason,
                            target_xy=target_xy,
                            distance_cm=round(dist, 1),
                            desired_yaw=round(float(target_yaw_deg), 1),
                            current_yaw=round(pose.yaw_deg, 1),
                            yaw_diff=round(yaw_diff, 1),
                            tolerance=arrival_yaw_tolerance,
                        )
                        general_tolerance = float(
                            self.config["navigation"].get("turn_tolerance_deg", 20.0)
                        )
                        if yaw_diff <= general_tolerance:
                            key = "turn_left_micro" if angle_diff_deg(float(target_yaw_deg), pose.yaw_deg) > 0.0 else "turn_right_micro"
                            before_pose = self.copy_pose(pose)
                            result = self.motion.run(key, times_override=1)
                            if not self.monitor_turn_result(
                                before_pose,
                                float(target_yaw_deg),
                                result,
                                "target_arrival_yaw",
                            ):
                                return False
                        elif not self.turn_toward_yaw_boundary_aware(float(target_yaw_deg)):
                            return False
                        continue
                self.clear_navigation_noop()
                self.debug.event("navigate_xy_arrived", reason=reason, target_xy=target_xy, distance_cm=round(dist, 1), step=step)
                return True
            direct_path = self.target_direct_approach_path(pose, target_screen, target_xy)
            direct_mode = bool(direct_path)
            self.debug.event(
                "navigation_mode",
                navigation_mode="target_direct_approach" if direct_mode else "normal",
                screen_id=None if target_screen is None else target_screen.screen_id,
                distance_cm=round(dist, 2),
                direct_target_corridor_clear=direct_mode,
                final_target_distance_cm=float(self.config["interaction"]["target_distance_cm"]),
                turn_penalty={
                    "per_deg": self.config["navigation"].get("action_planner_turn_cost_cm_per_deg"),
                    "fixed": self.config["navigation"].get("action_planner_turn_fixed_cost_cm"),
                },
            )
            if self.collision_recovery_pending:
                if direct_mode:
                    self.collision_recovery_pending = False
                    self.debug.event("target_direct_recovery_suppressed", reason="collision_recovery_pending")
                else:
                    self.recover_toward_field_center(reason + ":forward_no_progress", backoff=True)
                    continue
            if (
                not direct_mode
                and self.near_wall_now(pose)
            ):
                recovery_result = self.recover_from_near_wall(reason + ":near_wall_pre_forward")
                if recovery_result is False:  # Compatibility with injected legacy test doubles.
                    recovery_result = NearWallRecoveryResult.STILL_NEAR_WALL
                if recovery_result == NearWallRecoveryResult.HARDWARE_FAILURE:
                    self.debug.event(
                        "navigate_xy_failed",
                        reason=reason,
                        failure_reason=self.last_navigation_failure_reason,
                        target_xy=target_xy,
                        step=step,
                    )
                    return False
                if recovery_result == NearWallRecoveryResult.LOCALIZATION_REQUIRED:
                    self.debug.event(
                        "near_wall_recovery_relocalize",
                        current_target_screen_id=getattr(self, "current_target_screen_id", None),
                        target_preserved=getattr(self, "current_target_screen_id", None) is not None,
                    )
                    self.localize_scan()
                if (
                    recovery_result == NearWallRecoveryResult.STILL_NEAR_WALL
                    and self.last_navigation_failure_reason == "near_wall_recovery_exhausted"
                ):
                    return False
                continue
            if direct_mode:
                if self.mission_state != MissionState.TARGET_DIRECT_APPROACH:
                    self.set_mission_state(MissionState.TARGET_DIRECT_APPROACH)
                path = direct_path
            else:
                path = self.plan_navigation_path(
                    pose,
                    target_xy,
                    allow_goal_high_cost=allow_goal_high_cost,
                    target_screen=target_screen,
                )
            if not path:
                self.last_navigation_failure_reason = "no_safe_path_to_exact_target"
                self.local_replan_failures = getattr(self, "local_replan_failures", 0) + 1
                self.debug.event(
                    "post_action_replan",
                    reason=reason,
                    failure_reason=self.last_navigation_failure_reason,
                    target_xy=target_xy,
                    step=step,
                    post_action_replanned=False,
                    local_replan_failures=self.local_replan_failures,
                    current_target_screen_id=getattr(self, "current_target_screen_id", None),
                    target_preserved=getattr(self, "current_target_screen_id", None) is not None,
                )
                localized = self.localize_scan()
                if localized and self.state.pose is not None and self.near_wall_now(self.state.pose):
                    recovery_result = self.recover_from_near_wall(reason + ":local_replan_failed")
                    if recovery_result == NearWallRecoveryResult.HARDWARE_FAILURE:
                        return False
                continue
            if bool(getattr(self, "pending_post_action_replan", False)):
                self.debug.event(
                    "post_action_replan",
                    target_xy=target_xy,
                    pose=None if self.state.pose is None else self.state.pose.as_dict(),
                    navigation_mode="target_direct_approach" if direct_mode else "normal",
                    post_action_replanned=True,
                    current_target_screen_id=getattr(self, "current_target_screen_id", None),
                    target_preserved=getattr(self, "current_target_screen_id", None) is not None,
                )
                self.pending_post_action_replan = False
            waypoint = (
                path[1]
                if direct_mode and len(path) >= 2
                else self.select_navigation_waypoint(
                    pose,
                    path,
                    path[-1]
                    if target_screen is not None and distance_xy(path[-1], target_xy) > 1.0
                    else target_xy,
                    allow_goal_high_cost=(
                        allow_goal_high_cost and distance_xy(path[-1], target_xy) <= 1.0
                    ),
                )
            )
            desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
            diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
            self.debug.render_map(self.map, pose=pose, path=path)
            self.publish_state(path=path)
            waypoint_is_exact_goal = allow_goal_high_cost and distance_xy(waypoint, target_xy) <= 0.1
            if direct_mode and target_screen is not None:
                direct_action = self.choose_target_direct_action(pose, waypoint, target_screen)
                if direct_action is not None:
                    if not self.execute_target_direct_action(direct_action, target_screen, target_xy):
                        if not self.last_navigation_failure_reason:
                            self.last_navigation_failure_reason = "target_direct_action_failed"
                        return False
                    self.clear_navigation_noop()
                    continue
            action = self.choose_translation_action(
                pose,
                waypoint,
                allow_goal_high_cost=waypoint_is_exact_goal,
            )
            if action is not None:
                status = self.execute_translation_action(
                    action,
                    pose,
                    waypoint,
                    dist,
                    {"reason": reason, "diff_yaw": round(diff, 1)},
                )
                if status == "recovered":
                    self.clear_navigation_noop()
                    continue
                if status == "failed":
                    return False
                self.clear_navigation_noop()
            else:
                self.forward_map_block_count = 0
                rotation_clear = not hasattr(self.map, "rotation_sweep_clear") or self.map.rotation_sweep_clear(
                    pose.xy(),
                    float(self.config["navigation"].get("turn_sweep_radius_cm", 10.0)),
                    float(self.config["navigation"].get("normal_navigation_max_cost", 55.0)),
                )
                self.debug.event(
                    "turn_last_resort",
                    reason=reason,
                    desired_yaw=round(desired_yaw, 1),
                    diff_yaw=round(diff, 1),
                    waypoint=(round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
                    movement_corridor_clear=rotation_clear,
                )
                if not rotation_clear:
                    self.debug.event(
                        "turn_rejected",
                        reason="rotation_sweep_blocked",
                        navigation_mode="normal",
                        selected_action=None,
                        movement_corridor_clear=False,
                    )
                    if self.register_navigation_stall(
                        pose, target_xy, "turn_last_resort", "rotation_sweep_blocked"
                    ):
                        return False
                    recovery_result = self.recover_from_near_wall(reason + ":rotation_sweep_blocked")
                    if self.last_navigation_failure_reason == "near_wall_recovery_exhausted":
                        return False
                    continue
                if not self.turn_toward_yaw_boundary_aware(desired_yaw):
                    return False
                if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
                    self.handle_navigation_noop(
                        reason=reason,
                        waypoint=waypoint,
                        diff=diff,
                    )
                else:
                    self.clear_navigation_noop()
            if not direct_mode and self.turn_no_progress_count == 0 and self.state.needs_relocalize():
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed(reason + ":scheduled_relocalize")
                if self.collision_recovery_pending:
                    self.recover_toward_field_center(reason + ":forward_no_progress_after_localize", backoff=True)
        self.last_navigation_failure_reason = "navigation_step_limit"
        self.debug.event("navigate_xy_failed", reason=reason, failure_reason=self.last_navigation_failure_reason, target_xy=target_xy, max_steps=max_steps)
        return False

    def navigate_to_screen(self, screen: Screen) -> bool:
        return self.navigate_directly_to_target(screen)

    def update_dynamic_obstacles(self, tags, pan: float = 100.0) -> bool:
        """Update dynamic obstacles on the map from already-detected tags.

        Clears previous dynamic obstacles first, then adds new ones for any
        robot tags (id >= tag_min_id). Returns True if any obstacle was added.
        """
        self.map.clear_dynamic_obstacles()
        if not bool(self.config.get("obstacle", {}).get("enabled", True)):
            return False
        if self.state.pose is None:
            return False
        pose = self.state.pose
        min_id = int(self.config["obstacle"].get("tag_min_id", 81))
        obstacle_size = float(self.config["obstacle"].get("dynamic_size_cm", 20.0))
        found = False
        for tag in tags:
            if int(tag.tag_id) < min_id:
                continue
            world_xy = self.localizer.estimate_tag_world_xy(tag, pose, head_pan_angle=pan)
            if world_xy is None:
                continue
            if not self.map.in_bounds_xy(world_xy):
                continue
            self.map.add_dynamic_obstacle(world_xy, size_cm=obstacle_size)
            self.debug.event(
                "dynamic_obstacle_added",
                tag_id=int(tag.tag_id),
                world_xy=(round(world_xy[0], 1), round(world_xy[1], 1)),
                size_cm=obstacle_size,
            )
            found = True
        return found

    def front_obstacle_visible(self) -> bool:
        if self.args.dry_run or not bool(self.config.get("obstacle", {}).get("enabled", True)):
            return False
        if self.state.pose is None:
            return False
        if not self.boundary_safe_pan_angles([100], reason="front_obstacle_visible"):
            self.debug.event("front_obstacle_check_skipped_boundary_outward")
            return False
        frame, tags = self.capture_with_tags(100)
        if frame is None:
            return False
        # Update dynamic obstacles on map from all visible robot tags
        self.update_dynamic_obstacles(tags, pan=100.0)
        # Check if any robot tag is in the central band and close enough
        h, w = frame.shape[:2]
        band = float(self.config["obstacle"].get("center_band_ratio", 0.55))
        x0 = w * (0.5 - band / 2.0)
        x1 = w * (0.5 + band / 2.0)
        min_area = float(self.config["obstacle"].get("min_area_px", 350.0))
        min_id = int(self.config["obstacle"].get("tag_min_id", 81))
        for tag in tags:
            if int(tag.tag_id) < min_id:
                continue
            area = self.localizer.tag_area(tag)
            if area >= min_area and x0 <= float(tag.center[0]) <= x1:
                self.debug.event("front_obstacle_seen", tag_id=int(tag.tag_id), area=round(area, 1))
                return True
        return False

    def publish_state(self, target_screen: Optional[Screen] = None, path=None):
        target_screen = target_screen or self.map.screens.get(self.current_target_screen_id)
        target_distance = None
        if target_screen is not None and self.state.pose is not None:
            target_distance = round(
                distance_xy(
                    self.state.pose.xy(),
                    target_screen.task_target_xy or target_screen.interaction_xy,
                ),
                2,
            )
        state_now = now_s()
        recent_bound_state = {}
        for screen_id, observation in getattr(
            self, "recent_bound_flower_observations", {}
        ).items():
            item = observation.as_dict()
            item["age_s"] = round(max(0.0, state_now - observation.captured_s), 3)
            recent_bound_state[str(screen_id)] = item
        data = {
            "mode": self.args.mode,
            "target_flower": self.target_flower,
            "time_left_s": round(self.time_left_s(), 1),
            "completed_count": self.map.completed_count(),
            "processed_count": self.map.processed_count(),
            "remaining_target_ids": [screen.screen_id for screen in self.map.unfinished_screens()],
            "mission_state": self.mission_state.value,
            "current_target_tag_id": self.current_target_screen_id,
            "current_target_screen_id": self.current_target_screen_id,
            "current_target_distance_cm": target_distance,
            "arrived_at_target": self.arrived_at_target,
            "classifier_allowed": self.classifier_allowed,
            "visual_authorization": None if self.visual_authorization is None else self.visual_authorization.as_dict(),
            "target_visual_confirmation": None if self.target_visual_confirmation is None else self.target_visual_confirmation.as_dict(),
            "target_distance_cm": float(self.config["interaction"]["target_distance_cm"]),
            "target_final_forward_cm": float(self.config["interaction"]["target_final_forward_cm"]),
            "target_confirmation_retry_count": getattr(self, "target_confirmation_retry_count", 0),
            "target_confirmation_max_retries": int(self.config["interaction"].get("target_confirmation_max_retries", 3)),
            "target_confirmation_recovery_cycle": getattr(self, "target_confirmation_recovery_cycle", 0),
            "target_confirmation_diagnostics": getattr(self, "last_target_confirmation_diagnostics", {}),
            "final_forward_executed": self.final_forward_executed,
            "transit_bindings": self.transit_bindings,
            "recent_bound_flower_observations": recent_bound_state,
            "robot": self.state.as_dict(),
            "target_screen": None if target_screen is None else target_screen.as_dict(),
            "last_vote_summary": self.last_vote_summary,
            "latest_interaction": self.latest_interaction_result,
            "recent_interactions": self.recent_interaction_results,
            "interaction": {
                "phase": self.interaction_phase,
                "ready": bool(self.last_interaction_check and self.last_interaction_check.get("ready")),
                "left_hand_lifted": self.left_hand_lifted,
                "last_check": self.last_interaction_check,
            },
            "last_target_plan": self.last_target_plan,
            "localization_health": {
                "seconds_since_any_tag": round(now_s() - self.last_any_tag_seen_s, 2),
                "seconds_since_localize": None if self.last_localize_success_s <= 0 else round(now_s() - self.last_localize_success_s, 2),
                "consecutive_no_tag_scans": self.consecutive_no_tag_scans,
                "consecutive_localize_failures": self.consecutive_localize_failures,
                "localization_failures": self.localization_failures,
                "last_localization_tag_count": self.last_localization_tag_count,
                "last_localization_quality": self.last_localization_quality,
                "last_localization_pose_conflict": self.last_localization_pose_conflict,
                "actions_since_localize": self.state.actions_since_localize,
                "motion_uncertainty": round(self.state.motion_uncertainty, 3),
            },
            "recovery": {
                "count": self.recovery_count,
                "last": self.last_recovery,
                "no_progress_count": self.no_progress_count,
                "visual_no_progress_count": self.visual_no_progress_count,
                "collision_recovery_pending": self.collision_recovery_pending,
                "facing_outside": False if self.state.pose is None else self.is_facing_outside(self.state.pose),
                "field_exit_ahead_cm": None if self.state.pose is None else round(self.distance_to_field_exit_ahead(self.state.pose), 1),
                "forward_map_block_count": self.forward_map_block_count,
                "local_replan_failures": self.local_replan_failures,
                "near_wall_recovery_actions": self.near_wall_recovery_actions,
                "near_wall_recovery_no_progress": self.near_wall_recovery_no_progress_count,
                "fatal_target_failures": self.fatal_target_failures,
            },
            "interaction_log_path": self.interaction_audit_path,
            "screens": {sid: screen.as_dict() for sid, screen in sorted(self.map.screens.items())},
        }
        self.debug.state(data)
        self.debug.render_map(self.map, pose=self.state.pose, path=path, target_screen=target_screen)

    def close(self):
        try:
            self.hardware.close()
        finally:
            if self.interaction_audit_file is not None:
                self.interaction_audit_file.close()
            self.camera.release()
            self.debug.close()
