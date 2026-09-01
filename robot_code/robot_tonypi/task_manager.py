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
    TargetGoal,
    TargetTagConfirmation,
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
        self.classifier = ClassifierClient(
            args.classifier_url,
            dry_run=args.dry_run,
            mode=getattr(args, "classifier_mode", "direct"),
            student_id=getattr(args, "classifier_student_id", None),
            password=getattr(args, "classifier_password", None),
        )
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
        self.current_target_goal: Optional[TargetGoal] = None
        self.target_generation_counter = 0
        self.active_recovery_waypoint = None
        self.arrived_at_target = False
        self.classifier_allowed = False
        self.target_visual_confirmation: Optional[TargetVisualConfirmation] = None
        self.target_tag_confirmation: Optional[TargetTagConfirmation] = None
        self.visual_authorization: Optional[VisualAuthorization] = None
        self.final_forward_executed = False
        self.nfc_interaction_status = {}
        self.nfc_interaction_stopped_for_mission_timeout = False
        self.nfc_interaction_gave_up = False
        self.nfc_gave_up_screen_ids = set()
        self.post_interaction_retreat_pending = False
        self.post_interaction_retreat_completed = False
        self.post_interaction_retreat_blocked = False
        self.post_interaction_screen_id = None
        self.target_confirmation_retry_count = 0
        self.target_confirmation_recovery_cycle = 0
        self.last_target_confirmation_diagnostics = {}
        self.last_target_confirmation_failure_kind = ""
        self.classifier_available = True
        self.last_classifier_error = ""
        self.last_classifier_error_kind = ""
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
        self.no_tag_recovery_active = False
        self.no_tag_recovery_exhausted = False
        self.localization_recovery_exhausted = False
        self.last_localization_attempt_result = "never_attempted"
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
        self.near_wall_recovery_rejection_count = 0
        self.near_wall_recovery_actions = 0
        self.navigation_stall_signature = None
        self.navigation_stall_count = 0
        self.decision_stall_signature = None
        self.decision_stall_count = 0
        self.local_replan_failures = 0
        self.plan_failure_signature = None
        self.identical_plan_failure_count = 0
        self.last_recovered_deterministic_failure_key = None
        self.active_navigation_plan = None
        self.last_navigation_mode = ""
        self.last_motion_action = ""
        self.localization_failures = 0
        self.fatal_target_failures = 0
        self.temporarily_failed_targets: Dict[int, dict] = {}
        self.target_failure_counts: Dict[int, int] = {}
        self.global_recovery_cycles = 0
        self.mission_completion_announced = False
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

    def resolve_target_goal(self, screen_id: int, generation_id: Optional[int] = None) -> TargetGoal:
        """Resolve every consumer's target pose from the same canonical Screen."""
        screen_id = int(screen_id)
        screen = self.map.screens.get(screen_id)
        if screen is None:
            raise ValueError("unknown target screen {}".format(screen_id))
        worker_id = int(screen.worker_id or screen.screen_id)
        if worker_id != screen_id:
            raise ValueError("screen/tag/worker identity mismatch for {}".format(screen_id))
        goal_xy = screen.task_target_xy or screen.target_xy
        desired_yaw = (
            screen.task_target_yaw_deg
            if screen.task_target_yaw_deg is not None
            else screen.interaction_yaw_deg
        )
        return TargetGoal(
            screen_id=screen_id,
            tag_id=screen_id,
            anchor_xy=(float(screen.center_xy[0]), float(screen.center_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            desired_yaw_deg=float(desired_yaw),
            source="map.screens.cardinal_task_target",
            generation_id=int(
                self.target_generation_counter if generation_id is None else generation_id
            ),
        )

    def target_goal_from_screen(self, screen: Screen, generation_id: int) -> TargetGoal:
        """Compatibility resolver for isolated tests; production uses map.screens."""
        goal_xy = screen.task_target_xy or screen.target_xy
        yaw = screen.task_target_yaw_deg if screen.task_target_yaw_deg is not None else screen.interaction_yaw_deg
        return TargetGoal(
            screen_id=int(screen.screen_id),
            tag_id=int(screen.screen_id),
            anchor_xy=(float(screen.center_xy[0]), float(screen.center_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            desired_yaw_deg=float(yaw),
            source="screen.cardinal_task_target",
            generation_id=int(generation_id),
        )

    def lock_target_goal(self, screen: Screen) -> TargetGoal:
        """Atomically lock identity and pose; only a real target change advances generation."""
        existing = getattr(self, "current_target_goal", None)
        if existing is None or int(existing.screen_id) != int(screen.screen_id):
            self.target_generation_counter = int(getattr(self, "target_generation_counter", 0)) + 1
        if hasattr(self, "map") and int(screen.screen_id) in self.map.screens:
            goal = self.resolve_target_goal(screen.screen_id, self.target_generation_counter)
        else:
            goal = self.target_goal_from_screen(screen, self.target_generation_counter)
        self.current_target_goal = goal
        self.current_target_screen_id = goal.screen_id
        if hasattr(self, "debug"):
            self.debug.event("target_goal_resolved", **goal.as_dict())
        return goal

    def validate_target_goal(
        self,
        goal: Optional[TargetGoal] = None,
        *,
        requested_xy: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """Reject stale ID/coordinate pairings before motion and before ARRIVED."""
        goal = goal or getattr(self, "current_target_goal", None)
        if goal is None:
            return False
        try:
            canonical = (
                self.resolve_target_goal(goal.screen_id, goal.generation_id)
                if hasattr(self, "map")
                else goal
            )
        except (KeyError, ValueError) as exc:
            if hasattr(self, "debug"):
                self.debug.event("target_pose_mismatch", reason=str(exc))
            return False
        compared_xy = goal.goal_xy if requested_xy is None else requested_xy
        difference = distance_xy(tuple(compared_xy), canonical.goal_xy)
        tolerance = float(self.config["navigation"].get("target_goal_consistency_tolerance_cm", 0.5))
        valid = bool(
            int(self.current_target_screen_id or -1) == goal.screen_id
            and int(goal.tag_id) == goal.screen_id
            and int(goal.generation_id) == int(getattr(self, "target_generation_counter", -1))
            and distance_xy(goal.anchor_xy, canonical.anchor_xy) <= tolerance
            and distance_xy(goal.goal_xy, canonical.goal_xy) <= tolerance
            and difference <= tolerance
            and abs(angle_diff_deg(goal.desired_yaw_deg, canonical.desired_yaw_deg)) <= 0.1
        )
        event = "target_goal_validated" if valid else "target_pose_mismatch"
        if hasattr(self, "debug"):
            self.debug.event(
                event,
                screen_id=goal.screen_id,
                tag_id=goal.tag_id,
                stored_goal_xy=goal.goal_xy,
                resolved_goal_xy=canonical.goal_xy,
                anchor_xy=canonical.anchor_xy,
                difference_cm=round(difference, 3),
                source=goal.source,
                generation_id=goal.generation_id,
            )
        return valid

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

    def finish_mission_on_timeout(self) -> bool:
        """The global deadline is the only automatic terminal mission event."""
        self.set_mission_state(MissionState.MISSION_TIMEOUT)
        self.debug.event(
            "mission_timeout",
            successful=self.map.processed_count(),
            changed=self.map.completed_count(),
            temporary_failed=sorted(getattr(self, "temporarily_failed_targets", {})),
            recovery_cycles=int(getattr(self, "global_recovery_cycles", 0)),
        )
        self.hardware.stop()
        self.publish_state()
        return True

    def mission_retry_pause(self, config_key: str = "recovery_retry_interval_s") -> None:
        delay = max(0.0, float(self.config["mission"].get(config_key, 0.5)))
        time.sleep(min(delay, max(0.0, self.time_left_s())))

    def run_mission(self) -> bool:
        self.set_mission_state(MissionState.LOCALIZE)
        while self.state.pose is None and self.time_left_s() > 0:
            self.debug.event("relocalization_started", reason="initial_localize")
            if self.initial_localize():
                self.debug.event("relocalization_success", reason="initial_localize")
                break
            self.debug.event(
                "relocalization_failed",
                reason="initial_localize_failed",
                remaining_time=round(self.time_left_s(), 1),
                mission_continues=True,
            )
            self.mission_retry_pause()
        if self.time_left_s() <= 0:
            return self.finish_mission_on_timeout()
        loops = 0
        loop_watchdog = max(1, int(self.config["mission"].get("max_main_loops", 10000)))
        while self.time_left_s() > 0:
            loops += 1
            if loops > loop_watchdog:
                self.debug.event(
                    "global_recovery_started",
                    reason="main_loop_watchdog",
                    loops=loops,
                    remaining_time=round(self.time_left_s(), 1),
                )
                self.perform_global_recovery("main_loop_watchdog")
                loops = 0
            if getattr(self, "mission_completion_announced", False):
                if self.mission_state != MissionState.MISSION_COMPLETE:
                    self.set_mission_state(MissionState.MISSION_COMPLETE)
                self.mission_retry_pause("completed_idle_interval_s")
                continue
            # Never hand a close post-interaction pose to the normal planner.
            # After the physical retreat succeeds, localization retries do not
            # repeat that retreat.
            if getattr(self, "post_interaction_retreat_pending", False):
                pending_id = self.post_interaction_screen_id
                pending_target = self.map.screens.get(pending_id)
                if pending_target is None:
                    self.post_interaction_retreat_blocked = True
                    self.set_mission_state(MissionState.MISSION_BLOCKED)
                    self.debug.event(
                        "interaction_retreat_failed",
                        screen_id=pending_id,
                        reason="pending_screen_missing",
                    )
                elif self.complete_post_interaction_retreat(pending_target):
                    self.set_mission_state(MissionState.MARK_TARGET_COMPLETE)
                    self.publish_state(pending_target)
                    self.clear_current_target_context()
                    continue
                self.publish_state(pending_target)
                time.sleep(max(0.0, float(
                    self.config["interaction"].get(
                        "post_interaction_retry_interval_s", 1.0
                    )
                )))
                continue
            if self.target_reached():
                if not getattr(self, "mission_completion_announced", False):
                    self.mission_completion_announced = True
                    self.set_mission_state(MissionState.MISSION_COMPLETE)
                    self.debug.event("mission_success", completed=self.map.completed_count())
                self.mission_retry_pause("completed_idle_interval_s")
                continue
            self.set_mission_state(MissionState.SELECT_NEAREST_TARGET)
            target = self.choose_nearest_screen()
            if target is None:
                self.finish_mission_without_available_targets()
                self.mission_retry_pause()
                continue
            new_target = self.current_target_screen_id != target.screen_id
            target_goal = self.lock_target_goal(target)
            self.arrived_at_target = False
            self.classifier_allowed = False
            if new_target:
                self.target_tag_confirmation = None
                self.target_visual_confirmation = None
                self.visual_authorization = None
                self.final_forward_executed = False
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
                target_goal=target_goal.as_dict(),
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
                self.register_temporary_target_failure(target, failure_reason)
                self.fatal_target_failures += 1
                self.clear_current_target_context()
                continue
            self.arrived_at_target = True
            self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
            if not self.confirm_target_with_visibility_recovery(target):
                if self.mission_state in (
                    MissionState.TARGET_CLASSIFICATION_WAIT,
                    MissionState.TARGET_CLASSIFICATION_DEGRADED,
                    MissionState.MISSION_BLOCKED,
                ):
                    self.publish_state(target)
                    time.sleep(max(0.0, float(
                        self.config["interaction"].get("classifier_retry_interval_s", 1.0)
                    )))
                    continue
                continue
            classified = bool(target.last_classification and self.visual_authorization is not None)
            if target.status == ScreenStatus.ALREADY_TARGET:
                classified = True
            elif classified and not self.args.skip_change and not self.execute_final_forward(target):
                self.register_target_failure(target, "target_final_forward_failed")
                self.set_mission_state(MissionState.MARK_TARGET_COMPLETE)
                self.publish_state(target)
                self.current_target_screen_id = None
                self.current_target_goal = None
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
                if (
                    not changed
                    and getattr(
                        self,
                        "nfc_interaction_stopped_for_mission_timeout",
                        False,
                    )
                ):
                    return self.finish_mission_on_timeout()
                if (
                    not changed
                    and getattr(self, "nfc_interaction_gave_up", False)
                ):
                    self.debug.event(
                        "target_nfc_gave_up",
                        screen_id=target.screen_id,
                        attempts=2,
                        mission_continues=True,
                    )
                elif not changed and target.status == ScreenStatus.NEEDS_CHANGE:
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
            if getattr(self, "post_interaction_retreat_pending", False):
                if not self.complete_post_interaction_retreat(target):
                    self.publish_state(target)
                    time.sleep(max(0.0, float(
                        self.config["interaction"].get(
                            "post_interaction_retry_interval_s", 1.0
                        )
                    )))
                    continue
            self.set_mission_state(MissionState.MARK_TARGET_COMPLETE)
            self.publish_state(target)
            self.clear_current_target_context()
        return self.finish_mission_on_timeout()

    def clear_current_target_context(self) -> None:
        self.current_target_screen_id = None
        self.current_target_goal = None
        self.arrived_at_target = False
        self.classifier_allowed = False
        self.target_visual_confirmation = None
        self.visual_authorization = None
        self.final_forward_executed = False

    def mark_target_terminal_failed(self, target: Screen, reason: str) -> None:
        """Compatibility entry point: runtime failures are never permanent."""
        self.register_temporary_target_failure(target, reason)

    def register_target_failure(self, target: Screen, reason: str, relocalize: bool = False) -> bool:
        """Record a bounded local retry; True means rotate the target temporarily."""
        target.attempts += 1
        target.notes.append(reason)
        max_attempts = max(1, int(self.config["mission"].get("max_target_attempts", 2)))
        if target.attempts >= max_attempts:
            self.register_temporary_target_failure(
                target,
                reason,
                increment_attempt=False,
            )
            return True
        self.debug.event(
            "target_retry",
            screen_id=target.screen_id,
            reason=reason,
            attempts=target.attempts,
            max_attempts=max_attempts,
        )
        if relocalize:
            self.debug.event("relocalization_started", reason=reason, target_id=target.screen_id)
            localized = bool(self.localize_scan())
            self.debug.event(
                "relocalization_success" if localized else "relocalization_failed",
                reason=reason,
                target_id=target.screen_id,
            )
        return False

    def is_retryable_target_failure(self, reason: str) -> bool:
        """Navigation/localization/runtime failures never permanently blacklist a target."""
        return str(reason) not in ("target_completed", "already_target")

    def register_temporary_target_failure(
        self,
        target: Screen,
        reason: str,
        *,
        increment_attempt: bool = True,
    ) -> None:
        """Temporarily rotate an unfinished target without blacklisting it."""
        target_id = int(target.screen_id)
        if not hasattr(self, "target_failure_counts"):
            self.target_failure_counts = {}
        if not hasattr(self, "temporarily_failed_targets"):
            self.temporarily_failed_targets = {}
        if increment_attempt:
            target.attempts += 1
            target.notes.append(reason)
        self.target_failure_counts[target_id] = int(
            getattr(self, "target_failure_counts", {}).get(target_id, 0)
        ) + 1
        self.temporarily_failed_targets[target_id] = {
            "reason": str(reason),
            "failed_s": now_s(),
            "count": self.target_failure_counts[target_id],
        }
        # Undo legacy permanent failure state while preserving successful work.
        if target.status == ScreenStatus.FAILED:
            target.status = ScreenStatus.UNKNOWN
        self.debug.event(
            "target_navigation_failed",
            screen_id=target_id,
            reason=reason,
            attempts=target.attempts,
            temporary=True,
            mission_failed=False,
        )
        self.debug.event(
            "target_temporarily_failed",
            target_id=target_id,
            pose=(
                None
                if getattr(getattr(self, "state", None), "pose", None) is None
                else self.state.pose.as_dict()
            ),
            reason=reason,
            failure_count=self.target_failure_counts[target_id],
            temporary_failed=sorted(self.temporarily_failed_targets),
            remaining_time=round(self.time_left_s(), 1) if hasattr(self, "args") else None,
        )

    def release_temporary_target_failures(self, reason: str) -> List[int]:
        released = sorted(getattr(self, "temporarily_failed_targets", {}))
        for target_id in released:
            target = self.map.screens.get(target_id)
            if target is not None:
                if target.status == ScreenStatus.FAILED:
                    target.status = ScreenStatus.UNKNOWN
                if not target.done():
                    target.attempts = 0
        self.temporarily_failed_targets.clear()
        if released:
            self.debug.event(
                "target_retry_released",
                target_ids=released,
                reason=reason,
                remaining_time=round(self.time_left_s(), 1),
            )
        return released

    def preserve_current_target(self, target: Screen, reason: str) -> None:
        goal = getattr(self, "current_target_goal", None)
        if goal is None or int(goal.screen_id) != int(target.screen_id):
            goal = self.lock_target_goal(target)
        self.current_target_screen_id = int(goal.screen_id)
        self.debug.event(
            "current_target_preserved",
            current_target_screen_id=target.screen_id,
            current_target_xy=target.task_target_xy or target.target_xy,
            current_target_yaw=target.task_target_yaw_deg,
            target_goal=goal.as_dict(),
            target_preserved=True,
            reason=reason,
            target_attempts=target.attempts,
        )

    def set_mission_state(self, state: MissionState) -> None:
        self.mission_state = state
        self.debug.event("mission_state", state=state.value)

    def finish_mission_without_available_targets(self) -> MissionState:
        """Recover/release temporary failures; never terminate the mission."""
        unfinished = [item for item in self.map.screens.values() if not item.done()]
        if not unfinished:
            self.set_mission_state(MissionState.MISSION_COMPLETE)
            if not getattr(self, "mission_completion_announced", False):
                self.mission_completion_announced = True
                self.debug.event(
                    "mission_complete",
                    processed=self.map.processed_count(),
                    changed=self.map.completed_count(),
                    waiting_for_timeout=True,
                )
            return self.mission_state
        temporary = sorted(getattr(self, "temporarily_failed_targets", {}))
        legacy_failed = [
            item.screen_id for item in unfinished
            if item.status == ScreenStatus.FAILED
        ]
        self.debug.event(
            "all_targets_temporarily_failed",
            target_ids=sorted(item.screen_id for item in unfinished),
            temporary_failed=temporary,
            legacy_failed=legacy_failed,
            remaining_time=round(self.time_left_s(), 1),
        )
        self.perform_global_recovery("all_targets_temporarily_failed")
        self.release_temporary_target_failures("global_recovery_complete")
        self.set_mission_state(MissionState.SELECT_NEAREST_TARGET)
        return self.mission_state

    def perform_global_recovery(self, reason: str) -> bool:
        """Run one bounded recovery cycle; callers keep the mission alive."""
        self.global_recovery_cycles = int(getattr(self, "global_recovery_cycles", 0)) + 1
        self.set_mission_state(MissionState.NAVIGATION_RECOVERY)
        self.debug.event(
            "global_recovery_started",
            reason=reason,
            cycle=self.global_recovery_cycles,
            target_id=getattr(self, "current_target_screen_id", None),
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
            temporary_failed=sorted(getattr(self, "temporarily_failed_targets", {})),
            remaining_time=round(self.time_left_s(), 1),
        )
        self.debug.event("relocalization_started", reason="global_recovery:" + reason)
        localized = bool(self.localize_scan(
            reason="global_recovery:" + reason,
            allow_pan_search=True,
            allow_failure_escalation=False,
        ))
        self.debug.event(
            "relocalization_success" if localized else "relocalization_failed",
            reason="global_recovery:" + reason,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
        )
        recovered = localized
        pose = self.state.pose
        if self.time_left_s() > 0 and pose is not None:
            if self.near_wall_now(pose):
                result = self.recover_from_near_wall("global_recovery:" + reason)
                recovered = result in (
                    NearWallRecoveryResult.RECOVERED,
                    NearWallRecoveryResult.RETRY_WITH_NEW_POSE,
                )
            else:
                recovered = bool(self.recover_via_indoor_waypoint(
                    "global_recovery:" + reason
                )) or recovered
        elif self.time_left_s() > 0:
            actions = self.config["localization"].get("startup_search_actions", [])
            recovered = bool(self.run_localization_search_sequence(
                reason="global_recovery_pose_missing:" + reason,
                max_search_actions=len(actions),
                runtime_safety=False,
                no_tag_recovery=True,
            ))
        self.debug.event(
            "global_recovery_finished",
            reason=reason,
            cycle=self.global_recovery_cycles,
            success=bool(recovered),
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
            mission_continues=self.time_left_s() > 0,
            remaining_time=round(self.time_left_s(), 1),
        )
        return bool(recovered)

    def run_harvest_mode(self) -> bool:
        """Navigate directly to one nearest configured target and classify it."""
        self.set_mission_state(MissionState.SELECT_NEAREST_TARGET)
        target = self.choose_nearest_screen()
        if target is None:
            return True
        self.lock_target_goal(target)
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
        while not self.confirm_target_with_visibility_recovery(target):
            if self.mission_state not in (
                MissionState.TARGET_CLASSIFICATION_WAIT,
                MissionState.TARGET_CLASSIFICATION_DEGRADED,
                MissionState.MISSION_BLOCKED,
            ) or self.time_left_s() <= 0:
                return False
            self.publish_state(target)
            time.sleep(max(0.0, float(
                self.config["interaction"].get("classifier_retry_interval_s", 1.0)
            )))
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
        return self.run_localization_search_sequence(
            reason="initial_localize",
            max_search_actions=attempts,
            runtime_safety=False,
            no_tag_recovery=False,
        )

    def accept_visual_localization(self, pose: RobotPose, reason: str) -> None:
        """Install an accepted visual pose and consume accumulated dead reckoning."""
        actions_before = int(getattr(self.state, "actions_since_localize", 0))
        uncertainty_before = float(getattr(self.state, "motion_uncertainty", 0.0))
        self.state.set_pose(pose)
        self.last_localize_success_s = now_s()
        self.no_tag_recovery_exhausted = False
        self.localization_recovery_exhausted = False
        self.last_localization_attempt_result = "accepted_visual_pose"
        self.debug.event(
            "localization_state_reset",
            reason="accepted_visual_pose",
            localization_reason=reason,
            actions_before_reset=actions_before,
            uncertainty_before_reset=round(uncertainty_before, 3),
        )

    def execute_localization_search_action(
        self,
        requested_action: str,
        *,
        runtime_safety: bool,
    ) -> dict:
        """Execute one configured search action through the normal motion layer."""
        action = str(requested_action)
        pose = self.state.pose
        safety_result = "startup_pose_unknown"
        if runtime_safety and pose is None:
            return {
                "executed": False,
                "requested_action": requested_action,
                "action": None,
                "safety_result": "pose_missing_no_safe_action",
                "ok": False,
            }
        if runtime_safety and pose is not None:
            nav = self.config["navigation"]
            actions = self.config["motion"]["actions"]
            if action.startswith("turn_"):
                rotation_clear = not hasattr(self.map, "rotation_sweep_clear") or self.map.rotation_sweep_clear(
                    pose.xy(),
                    float(nav.get("turn_sweep_radius_cm", 10.0)),
                    float(nav.get("normal_navigation_max_cost", 55.0)),
                )
                if rotation_clear:
                    safety_result = "rotation_sweep_clear"
                    if self.is_near_boundary(pose):
                        target_yaw = self.yaw_toward_field_center(pose)
                        action = self.choose_boundary_safe_turn_key(pose, target_yaw)
                        safety_result = "boundary_safe_turn_selected"
                else:
                    fallback = "back_fast"
                    model = actions.get(fallback, {})
                    travel = float(model.get("forward_cm", -2.5)) * max(
                        1, int(model.get("times", 1))
                    )
                    yaw = math.radians(pose.yaw_deg)
                    fallback_xy = (
                        pose.x_cm + travel * math.cos(yaw),
                        pose.y_cm + travel * math.sin(yaw),
                    )
                    if self.escape_corridor_metrics(pose.xy(), fallback_xy).get("clear"):
                        action = fallback
                        safety_result = "rotation_blocked_safe_back_substitute"
                    else:
                        return {
                            "executed": False,
                            "requested_action": requested_action,
                            "action": None,
                            "safety_result": "rotation_and_back_substitute_blocked",
                            "ok": False,
                        }
            elif action == "back_fast":
                model = actions.get(action, {})
                travel = float(model.get("forward_cm", -2.5)) * max(
                    1, int(model.get("times", 1))
                )
                yaw = math.radians(pose.yaw_deg)
                end_xy = (
                    pose.x_cm + travel * math.cos(yaw),
                    pose.y_cm + travel * math.sin(yaw),
                )
                if self.escape_corridor_metrics(pose.xy(), end_xy).get("clear"):
                    safety_result = "rear_escape_corridor_clear"
                else:
                    fallback = self.choose_boundary_safe_turn_key(
                        pose, self.yaw_toward_field_center(pose)
                    )
                    rotation_clear = not hasattr(self.map, "rotation_sweep_clear") or self.map.rotation_sweep_clear(
                        pose.xy(),
                        float(nav.get("turn_sweep_radius_cm", 10.0)),
                        float(nav.get("normal_navigation_max_cost", 55.0)),
                    )
                    if not rotation_clear:
                        return {
                            "executed": False,
                            "requested_action": requested_action,
                            "action": None,
                            "safety_result": "rear_and_rotation_substitute_blocked",
                            "ok": False,
                        }
                    action = fallback
                    safety_result = "rear_blocked_safe_turn_substitute"
            else:
                safety_result = "configured_non_translation_action"
        result = self.motion.run(action)
        return {
            "executed": bool(getattr(result, "ok", False)),
            "requested_action": requested_action,
            "action": action,
            "safety_result": safety_result,
            "ok": bool(getattr(result, "ok", False)),
            "result": result,
        }

    def run_localization_search_sequence(
        self,
        reason: str,
        *,
        max_search_actions: int,
        runtime_safety: bool,
        no_tag_recovery: bool,
    ) -> bool:
        """Shared full-pan plus configured-action strategy for startup and recovery."""
        pans = list(self.config["localization"].get("scan_pan_angles", []))
        actions = list(self.config["localization"].get("startup_search_actions", []))
        target_id = getattr(self, "current_target_screen_id", None)
        target_goal = getattr(self, "current_target_goal", None)
        target_generation = None if target_goal is None else target_goal.generation_id

        def full_pan(stage: str) -> bool:
            if no_tag_recovery:
                self.debug.event(
                    "localization_full_pan_start",
                    reason=reason,
                    stage=stage,
                    pan_angles=pans,
                    failure_kind=str(getattr(
                        self, "last_localization_attempt_result", "unknown"
                    )),
                )
                self.debug.event(
                    "no_tag_full_pan_start",
                    reason=reason,
                    stage=stage,
                    pan_angles=pans,
                )
            success = bool(self.localize_scan(
                reason=reason + ":" + stage,
                allow_pan_search=True,
                pan_angles=pans,
                allow_failure_escalation=False,
            ))
            if no_tag_recovery:
                self.debug.event(
                    "localization_full_pan_result",
                    reason=reason,
                    stage=stage,
                    success=success,
                    localization_attempt_result=str(getattr(
                        self, "last_localization_attempt_result", "unknown"
                    )),
                    accepted_visual_pose=success,
                )
                self.debug.event(
                    "no_tag_full_pan_result",
                    reason=reason,
                    stage=stage,
                    success=success,
                    saw_any_tag=(
                        self.last_localization_attempt_result
                        not in ("no_tag", "capture_failed")
                    ),
                    accepted_visual_pose=success,
                )
            return success

        if full_pan("initial_full_pan"):
            return True
        if not actions:
            return False
        for index in range(max(0, int(max_search_actions))):
            if self.time_left_s() <= 0:
                break
            if not no_tag_recovery:
                self.debug.event(
                    "initial_localize_attempt",
                    attempt=index + 1,
                    max_attempts=max_search_actions,
                )
            requested = actions[index % len(actions)]
            action_detail = self.execute_localization_search_action(
                requested,
                runtime_safety=runtime_safety,
            )
            if no_tag_recovery:
                self.debug.event(
                    "localization_recovery_action",
                    index=index,
                    requested_action=requested,
                    action=action_detail.get("action"),
                    executed=action_detail.get("executed", False),
                    safety_result=action_detail.get("safety_result"),
                )
                self.debug.event(
                    "no_tag_recovery_action",
                    index=index,
                    requested_action=requested,
                    action=action_detail.get("action"),
                    executed=action_detail.get("executed", False),
                    safety_result=action_detail.get("safety_result"),
                    target_preserved=(
                        getattr(self, "current_target_screen_id", None) == target_id
                        and (
                            getattr(self, "current_target_goal", None) is target_goal
                            or (
                                target_goal is not None
                                and getattr(self.current_target_goal, "generation_id", None)
                                == target_generation
                            )
                        )
                    ),
                )
            if not action_detail.get("executed", False):
                continue
            success = full_pan("after_action_{}".format(index))
            if no_tag_recovery:
                self.debug.event(
                    "localization_recovery_attempt_result",
                    index=index,
                    action=action_detail.get("action"),
                    success=success,
                    localization_attempt_result=str(getattr(
                        self, "last_localization_attempt_result", "unknown"
                    )),
                )
                self.debug.event(
                    "no_tag_recovery_localize_result",
                    index=index,
                    action=action_detail.get("action"),
                    success=success,
                    accepted_visual_pose=(
                        self.last_localization_attempt_result == "accepted_visual_pose"
                    ),
                    pose_confidence=(
                        None if self.state.pose is None
                        else self.state.pose.confidence.value
                    ),
                )
            if success:
                return True
        return False

    def assess_visual_localization(self, pose: RobotPose, tags, prior_pose: Optional[RobotPose]) -> dict:
        """Describe Tag quality and disagreement with the currently trusted pose."""
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
        if prior_pose is not None:
            position_conflict_cm = distance_xy(prior_pose.xy(), pose.xy())
            yaw_conflict_deg = abs(angle_diff_deg(pose.yaw_deg, prior_pose.yaw_deg))
            conflict = (
                position_conflict_cm > float(self.config["navigation"].get("localization_pose_conflict_distance_cm", 15.0))
                or yaw_conflict_deg > float(self.config["navigation"].get("localization_pose_conflict_yaw_deg", 25.0))
            )
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

    def capture_visual_pose_once(self, pan: float, reason: str) -> dict:
        """Capture and estimate one visual pose without moving or mutating RobotState."""
        frame, tags = self.capture_with_tags(pan)
        if frame is None:
            return {
                "pose": None,
                "frame": None,
                "tags": [],
                "annotated": None,
                "result": "capture_failed",
            }
        pose, annotated = self.localizer.estimate_from_frame(
            frame,
            tags,
            head_pan_angle=pan,
            annotate=True,
        )
        return {
            "pose": pose,
            "frame": frame,
            "tags": tags,
            "annotated": annotated,
            "result": (
                "pose_available"
                if pose is not None
                else ("pose_unavailable_with_tags" if tags else "no_tag")
            ),
            "reason": reason,
        }

    def evaluate_and_accept_visual_pose(
        self,
        pose: RobotPose,
        tags,
        pan: float,
        reason: str,
        prior_pose: Optional[RobotPose],
    ) -> dict:
        """Atomically accept a normal pose or confirm a suspect visual jump."""
        detail = self.assess_visual_localization(pose, tags, prior_pose)
        enabled = bool(self.config["navigation"].get(
            "localization_suspect_confirmation_enabled", True
        ))
        if not detail.get("visual_odometry_conflict") or not enabled:
            self.accept_visual_localization(pose, reason)
            return {
                "accepted": True,
                "pose": pose,
                "tags": tags,
                "frame": None,
                "annotated": None,
                "localization_detail": detail,
                "decision": "accepted_normal_visual_pose",
            }

        suspect_pose = pose
        self.debug.event(
            "visual_dead_reckoning_conflict_observed",
            reason=reason,
            position_conflict_cm=detail["visual_odometry_position_delta_cm"],
            yaw_conflict_deg=detail["visual_odometry_yaw_delta_deg"],
            dead_reckoning_pose=(
                None if prior_pose is None else prior_pose.as_dict()
            ),
            visual_pose=suspect_pose.as_dict(),
        )
        self.debug.event(
            "visual_pose_suspect",
            reason=reason,
            prior_pose=None if prior_pose is None else prior_pose.as_dict(),
            suspect_visual_pose=suspect_pose.as_dict(),
            position_conflict_cm=detail["visual_odometry_position_delta_cm"],
            yaw_conflict_deg=detail["visual_odometry_yaw_delta_deg"],
            localization_quality=detail["localization_quality"],
            localization_tag_count=detail["localization_tag_count"],
            localization_best_tag_area_px=detail[
                "localization_best_tag_area_px"
            ],
            pan=float(pan),
        )
        attempts = max(1, int(self.config["navigation"].get(
            "localization_suspect_confirmation_attempts", 1
        )))
        max_position_delta = float(self.config["navigation"].get(
            "localization_suspect_confirmation_distance_cm", 10.0
        ))
        max_yaw_delta = float(self.config["navigation"].get(
            "localization_suspect_confirmation_yaw_deg", 15.0
        ))
        self.debug.event(
            "visual_pose_confirmation_started",
            reason=reason,
            pan=float(pan),
            suspect_pose=suspect_pose.as_dict(),
            attempts=attempts,
        )

        last_confirmation = None
        last_decision = "confirmation_pose_unavailable"
        for attempt in range(1, attempts + 1):
            confirmation = self.capture_visual_pose_once(
                pan, reason + ":suspect_confirmation"
            )
            last_confirmation = confirmation
            confirmation_pose = confirmation.get("pose")
            confirmation_position_delta = None
            confirmation_yaw_delta = None
            accepted_pose = None
            confirmation_detail = None
            if confirmation_pose is None:
                decision = "confirmation_pose_unavailable"
            else:
                confirmation_position_delta = distance_xy(
                    suspect_pose.xy(), confirmation_pose.xy()
                )
                confirmation_yaw_delta = abs(angle_diff_deg(
                    confirmation_pose.yaw_deg, suspect_pose.yaw_deg
                ))
                if (
                    confirmation_position_delta <= max_position_delta
                    and confirmation_yaw_delta <= max_yaw_delta
                ):
                    confirmation_detail = self.assess_visual_localization(
                        confirmation_pose, confirmation.get("tags", []), prior_pose
                    )
                    decision = "confirmed_visual_jump"
                    accepted_pose = confirmation_pose
                else:
                    confirmation_detail = self.assess_visual_localization(
                        confirmation_pose, confirmation.get("tags", []), prior_pose
                    )
                    if not confirmation_detail.get("visual_odometry_conflict"):
                        decision = "confirmation_pose_matches_prior"
                        accepted_pose = confirmation_pose
                    else:
                        decision = "rejected_inconsistent_visual_pose"
            last_decision = decision
            self.debug.event(
                "visual_pose_confirmation_result",
                reason=reason,
                attempt=attempt,
                success=accepted_pose is not None,
                suspect_pose=suspect_pose.as_dict(),
                confirmation_pose=(
                    None if confirmation_pose is None else confirmation_pose.as_dict()
                ),
                confirmation_position_delta_cm=(
                    None
                    if confirmation_position_delta is None
                    else round(float(confirmation_position_delta), 2)
                ),
                confirmation_yaw_delta_deg=(
                    None
                    if confirmation_yaw_delta is None
                    else round(float(confirmation_yaw_delta), 2)
                ),
                decision=decision,
                pose_installed=accepted_pose is not None,
            )
            if accepted_pose is not None:
                self.accept_visual_localization(accepted_pose, reason)
                if decision == "confirmed_visual_jump":
                    self.debug.event(
                        "visual_pose_jump_confirmed",
                        reason=reason,
                        suspect_pose=suspect_pose.as_dict(),
                        confirmed_pose=accepted_pose.as_dict(),
                        pose_installed=True,
                    )
                else:
                    self.debug.event(
                        "visual_pose_jump_rejected",
                        reason=reason,
                        decision=decision,
                        suspect_pose=suspect_pose.as_dict(),
                        confirmation_pose=accepted_pose.as_dict(),
                        suspect_pose_installed=False,
                        confirmation_pose_installed=True,
                    )
                return {
                    "accepted": True,
                    "pose": accepted_pose,
                    "tags": confirmation.get("tags", []),
                    "frame": confirmation.get("frame"),
                    "annotated": confirmation.get("annotated"),
                    "localization_detail": confirmation_detail,
                    "decision": decision,
                }

        self.debug.event(
            "visual_pose_jump_rejected",
            reason=reason,
            decision=last_decision,
            suspect_pose=suspect_pose.as_dict(),
            confirmation_pose=(
                None
                if last_confirmation is None or last_confirmation.get("pose") is None
                else last_confirmation["pose"].as_dict()
            ),
            suspect_pose_installed=False,
            confirmation_pose_installed=False,
            prior_pose_retained=True,
        )
        return {
            "accepted": False,
            "pose": None,
            "tags": tags,
            "frame": None,
            "annotated": None,
            "localization_detail": detail,
            "decision": last_decision,
        }

    def emit_localization_diagnostics(
        self,
        tags,
        pan: float,
        reason: str,
        accepted: bool,
        additional_rejection: Optional[dict] = None,
    ) -> None:
        """Publish the Localizer's per-Tag reasons without changing pose math."""
        diagnostics = dict(getattr(
            self.localizer, "last_estimation_diagnostics", {}
        ) or {})
        rejected = list(diagnostics.get("rejected_tags", []))
        if additional_rejection is not None:
            rejected.append(dict(additional_rejection))
        if not diagnostics and tags and not accepted:
            for tag in tags:
                try:
                    area = round(float(self.localizer.tag_area(tag)), 1)
                except Exception:
                    area = None
                rejected.append({
                    "tag_id": int(tag.tag_id),
                    "tag_area_px": area,
                    "tag_center_px": [
                        round(float(tag.center[0]), 1),
                        round(float(tag.center[1]), 1),
                    ] if getattr(tag, "center", None) is not None else None,
                    "stage": "pose_estimation",
                    "reason": "pose_not_produced",
                })
        for item in rejected:
            self.debug.event(
                "localization_tag_rejected",
                pan=float(pan),
                localization_reason=reason,
                **item
            )
        if tags and not accepted:
            self.debug.event(
                "localization_frame_failed",
                pan=float(pan),
                localization_reason=reason,
                detected_tag_ids=diagnostics.get(
                    "detected_tag_ids", [int(tag.tag_id) for tag in tags]
                ),
                candidate_localization_tag_ids=diagnostics.get(
                    "candidate_localization_tag_ids", []
                ),
                rejected_tags=rejected,
                result=(
                    "pose_rejected_by_localizer"
                    if additional_rejection is not None
                    else "pose_unavailable_with_tags"
                ),
            )

    def record_localization_failure(
        self,
        attempt_result: str,
        *,
        saw_any_tag: bool,
        reason: str,
    ) -> None:
        """Count every attempt that did not install an accepted visual pose."""
        self.consecutive_localize_failures = int(getattr(
            self, "consecutive_localize_failures", 0
        )) + 1
        self.localization_failures = int(getattr(
            self, "localization_failures", 0
        )) + 1
        if saw_any_tag:
            self.consecutive_no_tag_scans = 0
        else:
            self.consecutive_no_tag_scans = int(getattr(
                self, "consecutive_no_tag_scans", 0
            )) + 1
        self.last_localization_attempt_result = str(attempt_result)
        self.debug.event(
            "localize_failed",
            reason=reason,
            saw_any_tag=bool(saw_any_tag),
            no_tag_scans=self.consecutive_no_tag_scans,
            failures=self.consecutive_localize_failures,
            localization_attempt_result=str(attempt_result),
            actions_since_localize=int(getattr(
                self.state, "actions_since_localize", 0
            )),
            motion_uncertainty=round(float(getattr(
                self.state, "motion_uncertainty", 0.0
            )), 3),
            last_successful_localization_s=float(getattr(
                self, "last_localize_success_s", 0.0
            )),
        )

    def localize_scan(
        self,
        reset_turn_watchdog: bool = True,
        *,
        reason: str = "routine",
        allow_pan_search: bool = False,
        pan_angles: Optional[List[float]] = None,
        allow_failure_escalation: bool = True,
        required_target_screen_id: Optional[int] = None,
    ) -> bool:
        """Localize normally, or keep scanning until one required target is bound."""
        saw_any_tag = False
        captured_frame = False
        accepted_any_pose = False
        rejected_suspect_pose = False
        last_scan_pan = None
        required_target_id = (
            None
            if required_target_screen_id is None
            else int(required_target_screen_id)
        )
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        requested_pans = list(
            pan_angles
            if pan_angles is not None
            else self.config["localization"].get("scan_pan_angles", [center])
        )
        scan_pans = (
            self.unique_pan_angles(requested_pans)
            if required_target_id is not None
            else self.boundary_safe_pan_angles(requested_pans, reason=reason)
        )
        self.debug.event(
            "localize_scan_started",
            reason=reason,
            pan_angles=scan_pans,
        )
        try:
            for pan in scan_pans:
                last_scan_pan = pan
                frame, tags = self.capture_with_tags(pan)
                if frame is None:
                    continue
                captured_frame = True
                if tags:
                    saw_any_tag = True
                    self.update_dynamic_obstacles(tags, pan=pan)
                pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=pan, annotate=True)
                if pose is not None:
                    prior_pose = None if self.state.pose is None else self.copy_pose(self.state.pose)
                    acceptance = self.evaluate_and_accept_visual_pose(
                        pose, tags, pan, reason, prior_pose
                    )
                    if not acceptance["accepted"]:
                        rejected_suspect_pose = True
                        self.emit_localization_diagnostics(
                            tags,
                            pan,
                            reason,
                            False,
                            additional_rejection={
                                "stage": "temporal_consistency",
                                "reason": acceptance["decision"],
                            },
                        )
                        annotated = self.observe_transit_bindings(
                            frame, tags, annotated, pan, reason
                        )
                        self.debug.save_image(
                            "latest_annotated.jpg", annotated, force=True
                        )
                        if required_target_id is not None:
                            detected_ids = [int(tag.tag_id) for tag in tags]
                            bound_ids = set(getattr(
                                self, "last_transit_binding_screen_ids", set()
                            ))
                            target_bound = required_target_id in bound_ids
                            self.debug.event(
                                "nfc_retry_localization_only",
                                target_screen_id=required_target_id,
                                localization_tag_ids=detected_ids,
                                localization_success=False,
                                target_seen=required_target_id in detected_ids,
                                target_bound=target_bound,
                                target_reacquired=target_bound,
                                pan=float(pan),
                            )
                            if target_bound:
                                self.debug.event(
                                    "nfc_retry_target_reacquired",
                                    target_screen_id=required_target_id,
                                    pan=float(pan),
                                    localization_tag_ids=detected_ids,
                                    binding_screen_ids=sorted(bound_ids),
                                )
                                self.publish_state()
                                return True
                        continue
                    accepted_any_pose = True
                    pose = acceptance["pose"]
                    localization_detail = acceptance["localization_detail"]
                    if acceptance.get("frame") is None:
                        accepted_frame = frame
                        accepted_tags = tags
                        accepted_annotated = annotated
                    else:
                        accepted_frame = acceptance["frame"]
                        accepted_tags = acceptance.get("tags", [])
                        accepted_annotated = acceptance.get("annotated")
                        if accepted_tags:
                            saw_any_tag = True
                            self.update_dynamic_obstacles(accepted_tags, pan=pan)
                    self.emit_localization_diagnostics(
                        accepted_tags, pan, reason, True
                    )
                    annotated = self.observe_transit_bindings(
                        accepted_frame,
                        accepted_tags,
                        accepted_annotated,
                        pan,
                        reason,
                    )
                    if reset_turn_watchdog:
                        self.clear_turn_progress_watchdog("normal_relocalize")
                    self.consecutive_localize_failures = 0
                    self.consecutive_no_tag_scans = 0
                    self.evaluate_pending_progress(pose)
                    self.debug.event("pose_update", **pose.as_dict(), head_pan_angle=pan, **localization_detail)
                    self.debug.save_image("latest_annotated.jpg", annotated, force=True)
                    if required_target_id is not None:
                        detected_ids = [int(tag.tag_id) for tag in accepted_tags]
                        bound_ids = set(getattr(
                            self, "last_transit_binding_screen_ids", set()
                        ))
                        target_bound = required_target_id in bound_ids
                        self.debug.event(
                            "nfc_retry_localization_only",
                            target_screen_id=required_target_id,
                            localization_tag_ids=detected_ids,
                            localization_success=True,
                            target_seen=required_target_id in detected_ids,
                            target_bound=target_bound,
                            target_reacquired=target_bound,
                            pan=float(pan),
                        )
                        if not target_bound:
                            continue
                        self.debug.event(
                            "nfc_retry_target_reacquired",
                            target_screen_id=required_target_id,
                            pan=float(pan),
                            localization_tag_ids=detected_ids,
                            binding_screen_ids=sorted(bound_ids),
                        )
                    self.debug.event(
                        "pan_search_stopped_on_success",
                        reason=reason,
                        successful_pan=float(pan),
                        visited_through=float(pan),
                        stop_condition=(
                            "required_target_reacquired"
                            if required_target_id is not None
                            else "accepted_visual_pose"
                        ),
                    )
                    self.publish_state()
                    return True
                # Even when this frame cannot produce a new pose, the old
                # dead-reckoning pose may still safely support visual evidence.
                self.emit_localization_diagnostics(
                    tags, pan, reason, False
                )
                annotated = self.observe_transit_bindings(frame, tags, annotated, pan, reason)
                self.debug.save_image("latest_annotated.jpg", annotated, force=True)
                if required_target_id is not None:
                    detected_ids = [int(tag.tag_id) for tag in tags]
                    bound_ids = set(getattr(
                        self, "last_transit_binding_screen_ids", set()
                    ))
                    target_bound = required_target_id in bound_ids
                    self.debug.event(
                        "nfc_retry_localization_only",
                        target_screen_id=required_target_id,
                        localization_tag_ids=detected_ids,
                        localization_success=False,
                        target_seen=required_target_id in detected_ids,
                        target_bound=target_bound,
                        target_reacquired=target_bound,
                        pan=float(pan),
                    )
                    if target_bound:
                        self.debug.event(
                            "nfc_retry_target_reacquired",
                            target_screen_id=required_target_id,
                            pan=float(pan),
                            localization_tag_ids=detected_ids,
                            binding_screen_ids=sorted(bound_ids),
                        )
                        self.debug.event(
                            "pan_search_stopped_on_success",
                            reason=reason,
                            successful_pan=float(pan),
                            visited_through=float(pan),
                            stop_condition="required_target_reacquired",
                        )
                        self.publish_state()
                        return True
            if required_target_id is not None and accepted_any_pose:
                self.debug.event(
                    "nfc_retry_target_not_seen",
                    target_screen_id=required_target_id,
                    pan_angles=scan_pans,
                    localization_success=True,
                    target_reacquired=False,
                )
                return False
            if rejected_suspect_pose:
                attempt_result = "suspect_visual_pose_rejected"
            elif saw_any_tag:
                attempt_result = "pose_unavailable_with_tags"
            elif captured_frame:
                attempt_result = "no_tag"
            else:
                attempt_result = "capture_failed"
            self.record_localization_failure(
                attempt_result,
                saw_any_tag=saw_any_tag,
                reason=reason,
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
        if (
            pose is None
            or not hasattr(self, "map")
            or not bool(self.config["navigation"].get("boundary_safe_turn_enabled", True))
        ):
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
        self.last_transit_binding_screen_ids = set()
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
        self.last_transit_binding_screen_ids = {int(screen_id) for screen_id in seen}
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
            if not bool(getattr(self, "classifier_available", True)):
                self.last_target_confirmation_failure_kind = "classifier_unavailable"
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
            self.classifier_available = False
            self.last_classifier_error = str(exc)
            self.last_classifier_error_kind = "service_unavailable"
            self.last_target_confirmation_failure_kind = "classifier_unavailable"
            self.debug.event(
                "target_classifier_unavailable",
                screen_id=screen_id,
                tag_id=tag_id,
                retryable=True,
                error=str(exc),
            )
            self.debug.event(
                "bound_flower_observation_failed",
                screen_id=screen_id,
                tag_id=tag_id,
                reason=reason,
                error=str(exc),
            )
            return None
        confidence = float(result.confidence) if result.ok else 0.0
        if not result.ok:
            self.last_classifier_error = result.error
            self.last_classifier_error_kind = getattr(result, "error_kind", "") or "classification_failed"
            self.classifier_available = self.last_classifier_error_kind != "service_unavailable"
            self.last_target_confirmation_failure_kind = (
                "classifier_unavailable"
                if not self.classifier_available or bool(getattr(result, "retryable", False))
                else "classifier_result_invalid"
            )
            event = (
                "target_classifier_unavailable"
                if self.last_target_confirmation_failure_kind == "classifier_unavailable"
                else "target_classifier_result_invalid"
            )
            self.debug.event(
                event,
                screen_id=screen_id,
                tag_id=tag_id,
                error=result.error,
                error_kind=self.last_classifier_error_kind,
                retryable=bool(getattr(result, "retryable", False)),
                target_tag_confirmed=getattr(self, "target_tag_confirmation", None) is not None,
            )
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
        was_unavailable = not bool(getattr(self, "classifier_available", True))
        self.classifier_available = True
        self.last_classifier_error = ""
        self.last_classifier_error_kind = ""
        if was_unavailable:
            self.debug.event("target_classifier_recovered", screen_id=screen_id, tag_id=tag_id)
        self.debug.event(
            "target_classifier_result",
            screen_id=screen_id,
            tag_id=tag_id,
            flower=result.flower_api,
            confidence=round(confidence, 4),
        )
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
        """Confirm the locked Tag live; the caller recenters after consuming its frame."""
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        pans = self.unique_pan_angles([center, 130.0, 70.0])
        self._last_target_live_frame = None
        self._last_target_live_tags = []
        self._last_target_live_pan = center
        for pan in pans:
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
                self.target_tag_confirmation = TargetTagConfirmation(
                    screen_id=screen.screen_id,
                    tag_id=screen.screen_id,
                    captured_s=seen_s,
                    pan=float(pan),
                )
                self.set_mission_state(MissionState.TARGET_TAG_CONFIRMED)
                self.debug.event(
                    "target_tag_live_confirmed",
                    screen_id=screen.screen_id,
                    tag_id=screen.screen_id,
                    pan=float(pan),
                    current_tag_seen_s=seen_s,
                    frame_retained=True,
                )
                self.debug.event("target_tag_confirmed", **self.target_tag_confirmation.as_dict())
                return True
        self.target_tag_confirmation = None
        self.debug.event(
            "target_tag_live_missing",
            screen_id=screen.screen_id,
            pan_angles=pans,
        )
        return False

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
                retry_pan = float(getattr(
                    self,
                    "_last_target_live_pan",
                    self.config["camera"].get("head_center_angle", 100.0),
                ))
                self.debug.event(
                    "target_classifier_retry",
                    screen_id=screen.screen_id,
                    attempt=attempt + 1,
                    pan=retry_pan,
                )
                frame, tags = self.capture_with_tags(retry_pan)
                frames.append((frame, tags, retry_pan))
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
                        if getattr(self, "last_target_confirmation_failure_kind", "") == "classifier_unavailable":
                            break
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
        self.last_target_confirmation_failure_kind = ""
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
                self.last_target_confirmation_failure_kind = "target_tag_missing"
                self.center_head_after_scan(
                    "confirm_target_tag_now_missing", getattr(self, "_last_target_live_pan", None)
                )
                return False
            seen_s = float(getattr(self, "_last_target_tag_seen_s", now_s()))
            try:
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
                if observation is None:
                    if not self.last_target_confirmation_failure_kind:
                        self.last_target_confirmation_failure_kind = "target_screen_binding_missing"
                    return False
                if not self.adopt_cached_target_observation(
                    screen,
                    observation,
                    current_tag_seen_s=seen_s,
                    source=source,
                ):
                    self.last_target_confirmation_failure_kind = "target_evidence_mismatch"
                    return False
                self.target_confirmation_retry_count = 0
            finally:
                self.center_head_after_scan(
                    "confirm_target_observation_consumed",
                    getattr(self, "_last_target_live_pan", None),
                )
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
            recovered = self.recover_via_indoor_waypoint(
                "target_visibility_recovery:no_localization"
            )
            self.debug.event(
                "target_visibility_recovery_action",
                screen_id=screen.screen_id,
                action="interior_recovery_waypoint",
                ok=bool(recovered),
            )
            return bool(recovered)
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
        failure_kind = getattr(self, "last_target_confirmation_failure_kind", "")
        if failure_kind in ("classifier_unavailable", "classifier_result_invalid"):
            state = (
                MissionState.TARGET_CLASSIFICATION_WAIT
                if failure_kind == "classifier_unavailable"
                else MissionState.TARGET_CLASSIFICATION_DEGRADED
            )
            self.set_mission_state(state)
            self.preserve_current_target(screen, failure_kind)
            self.debug.event(
                "target_classifier_unavailable" if failure_kind == "classifier_unavailable" else "target_classifier_degraded",
                screen_id=screen.screen_id,
                tag_id=screen.screen_id,
                target_preserved=True,
                target_tag_confirmed=getattr(self, "target_tag_confirmation", None) is not None,
                error=getattr(self, "last_classifier_error", ""),
                error_kind=getattr(self, "last_classifier_error_kind", ""),
                mission_failed=False,
            )
            return False
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
            failure_kind = getattr(self, "last_target_confirmation_failure_kind", "")
            if failure_kind in ("classifier_unavailable", "classifier_result_invalid"):
                self.set_mission_state(
                    MissionState.TARGET_CLASSIFICATION_WAIT
                    if failure_kind == "classifier_unavailable"
                    else MissionState.TARGET_CLASSIFICATION_DEGRADED
                )
                self.preserve_current_target(screen, failure_kind)
                return False
        screen.attempts += 1
        self.last_navigation_failure_reason = "target_screen_confirmation_unresolved"
        self.classifier_allowed = False
        self.target_visual_confirmation = None
        self.visual_authorization = None
        self.preserve_current_target(screen, self.last_navigation_failure_reason)
        self.set_mission_state(MissionState.MISSION_BLOCKED)
        self.debug.event(
            "target_screen_confirmation_unresolved",
            screen_id=screen.screen_id,
            target_confirmation_retry_count=self.target_confirmation_retry_count,
            target_confirmation_max_retries=int(self.config["interaction"].get("target_confirmation_max_retries", 3)),
            target_confirmation_recovery_cycle=self.target_confirmation_recovery_cycle,
            final_forward_executed=self.final_forward_executed,
            target_preserved=True,
            mission_failed=False,
        )
        self.publish_state(screen)
        return False

    def classify_after_final_forward(self, screen: Screen, *, allow_without_forward: bool = False) -> int:
        """Capture once and classify after the configured final forward motion."""
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
            self.record_localization_failure(
                "capture_failed",
                saw_any_tag=False,
                reason="scan_after_turn:" + reason,
            )
            self.debug.event("scan_after_turn_failed", reason=reason, action_key=action_key, error="capture_failed")
            return outcome
        pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=center, annotate=True)
        localized = pose is not None
        outcome["localized"] = localized
        if pose is not None:
            prior_pose = (
                None if self.state.pose is None else self.copy_pose(self.state.pose)
            )
            acceptance = self.evaluate_and_accept_visual_pose(
                pose,
                tags,
                center,
                "scan_after_turn:" + reason,
                prior_pose,
            )
            if acceptance["accepted"]:
                pose = acceptance["pose"]
                localization_detail = acceptance["localization_detail"]
                if watchdog_scan:
                    progress = evaluate_turn_progress(
                        before_pose,
                        pose,
                        float(action_result.model_yaw_deg),
                        target_yaw,
                    )
                    outcome.update(progress)
                    if progress["suspect_stale_pose"]:
                        self.debug.event(
                            "suspect_stale_pose_after_turn",
                            reason=reason,
                            action_key=action_key,
                            **progress
                        )
                    if progress["direction_conflict"]:
                        self.debug.event(
                            "turn_direction_conflict",
                            reason=reason,
                            action_key=action_key,
                            **progress
                        )
                    if progress["turn_no_progress"]:
                        self.debug.event(
                            "turn_no_progress",
                            reason=reason,
                            action_key=action_key,
                            **progress
                        )
                    if progress["reject_visual_pose"]:
                        self.debug.event(
                            "scan_after_turn_pose_conflict_observed",
                            reason=reason,
                            action_key=action_key,
                            dead_reckoning_pose=(
                                None
                                if prior_pose is None
                                else prior_pose.as_dict()
                            ),
                            visual_pose=pose.as_dict(),
                            **progress
                        )
                if acceptance.get("frame") is not None:
                    frame = acceptance["frame"]
                    tags = acceptance.get("tags", [])
                    annotated = acceptance.get("annotated")
                self.emit_localization_diagnostics(
                    tags, center, "scan_after_turn:" + reason, True
                )
                outcome["accepted"] = True
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
            else:
                self.emit_localization_diagnostics(
                    tags,
                    center,
                    "scan_after_turn:" + reason,
                    False,
                    additional_rejection={
                        "stage": "temporal_consistency",
                        "reason": acceptance["decision"],
                    },
                )
                self.record_localization_failure(
                    "suspect_visual_pose_rejected",
                    saw_any_tag=bool(tags),
                    reason="scan_after_turn:" + reason,
                )
        else:
            attempt_result = (
                "pose_unavailable_with_tags" if tags else "no_tag"
            )
            self.emit_localization_diagnostics(
                tags, center, "scan_after_turn:" + reason, False
            )
            self.record_localization_failure(
                attempt_result,
                saw_any_tag=bool(tags),
                reason="scan_after_turn:" + reason,
            )
        annotated = self.observe_transit_bindings(frame, tags, annotated, center, "scan_after_turn:" + reason)
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        self.debug.event(
            "scan_after_turn_done",
            reason=reason,
            action_key=action_key,
            localized=localized,
            accepted_visual_pose=bool(outcome.get("accepted")),
            localization_attempt_result=str(getattr(
                self, "last_localization_attempt_result", "unknown"
            )),
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
        goal = getattr(self, "current_target_goal", None)
        if goal is None or int(goal.screen_id) != int(screen.screen_id):
            goal = self.lock_target_goal(screen)
        if not self.validate_target_goal(goal):
            self.last_navigation_failure_reason = "target_pose_mismatch"
            return False
        return self.navigate_to_xy(
            goal.goal_xy,
            reason="task_target",
            arrival_radius_cm=float(self.config["navigation"]["target_arrival_radius_cm"]),
            max_steps=int(self.config["navigation"]["max_steps_per_target"]),
            target_yaw_deg=goal.desired_yaw_deg,
            target_yaw_tolerance_deg=float(
                self.config["navigation"]["target_arrival_yaw_tolerance_deg"]
            ),
            allow_goal_high_cost=True,
            target_screen=screen,
            target_goal=goal,
            bypass_action_safety=True,
        )

    def execute_final_forward(self, screen: Screen) -> bool:
        """Execute the configured final forward action exactly once before classification."""
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
        self.final_forward_executed = bool(result.ok)
        if result.ok:
            self.post_interaction_retreat_pending = True
            self.post_interaction_retreat_completed = False
            self.post_interaction_retreat_blocked = False
            self.post_interaction_screen_id = int(screen.screen_id)
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

    def complete_post_interaction_retreat(
        self,
        screen: Screen,
        *,
        reason: str = "post_interaction",
    ) -> bool:
        """Reverse out of the close pose exactly once, then relocalize."""
        if not getattr(self, "post_interaction_retreat_pending", False):
            return True
        if getattr(self, "post_interaction_retreat_blocked", False):
            self.set_mission_state(MissionState.MISSION_BLOCKED)
            return False

        interaction = self.config["interaction"]
        requested_cm = float(interaction.get("post_interaction_retreat_cm", 10.0))
        action_key = str(interaction.get("post_interaction_retreat_action", "back_fast"))
        if not getattr(self, "post_interaction_retreat_completed", False):
            spec = self.config["motion"]["actions"].get(action_key)
            modeled_cycle_cm = 0.0 if spec is None else abs(float(spec.get("forward_cm", 0.0)))
            if requested_cm <= 0.0 or modeled_cycle_cm <= 0.0:
                self.post_interaction_retreat_blocked = True
                self.set_mission_state(MissionState.MISSION_BLOCKED)
                self.debug.event(
                    "interaction_retreat_failed",
                    screen_id=screen.screen_id,
                    requested_cm=requested_cm,
                    action=action_key,
                    reason="invalid_retreat_configuration",
                )
                return False
            cycles = max(1, int(math.ceil(requested_cm / modeled_cycle_cm)))
            self.set_mission_state(MissionState.POST_INTERACTION_RETREAT)
            self.debug.event(
                "interaction_retreat_started",
                screen_id=screen.screen_id,
                retreat_cm=requested_cm,
                action=action_key,
                modeled_cycle_cm=modeled_cycle_cm,
                action_cycles=cycles,
                reason=reason,
            )
            stand_result = self.motion.run("stand", times_override=1)
            if not stand_result.ok:
                self.post_interaction_retreat_blocked = True
                self.set_mission_state(MissionState.MISSION_BLOCKED)
                self.debug.event(
                    "interaction_retreat_failed",
                    screen_id=screen.screen_id,
                    requested_cm=requested_cm,
                    action="stand",
                    reason=stand_result.error or "stand_failed",
                )
                return False
            result = self.motion.run(action_key, times_override=cycles)
            actual_cycles_value = getattr(result, "executed_times", None)
            if actual_cycles_value is None:
                actual_cycles_value = result.times if result.ok else 0
            actual_cycles = int(actual_cycles_value)
            actual_cm = abs(float(result.model_forward_cm)) if result.ok else 0.0
            self.debug.event(
                "interaction_retreat_completed" if result.ok else "interaction_retreat_failed",
                screen_id=screen.screen_id,
                requested_cm=requested_cm,
                actual_action_cycles=actual_cycles,
                actual_modeled_cm=round(actual_cm, 3),
                action=action_key,
                ok=bool(result.ok),
                error=result.error,
            )
            if not result.ok:
                # The blocking ActionGroup API cannot report a partial prefix;
                # do not blindly issue another retreat after an exception.
                self.post_interaction_retreat_blocked = True
                self.set_mission_state(MissionState.MISSION_BLOCKED)
                return False
            self.post_interaction_retreat_completed = True
            self.final_forward_executed = False

        self.set_mission_state(MissionState.POST_INTERACTION_RELOCALIZE)
        localized = self.localize_scan(
            reason="{}_retreat".format(reason),
            allow_pan_search=True,
            allow_failure_escalation=False,
        )
        self.debug.event(
            "post_interaction_relocalize",
            screen_id=screen.screen_id,
            success=bool(localized),
            center_first=True,
            pan_search_on_center_failure=True,
            reason=reason,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
        )
        if not localized:
            self.set_mission_state(MissionState.MISSION_BLOCKED)
            return False
        self.post_interaction_retreat_pending = False
        self.post_interaction_retreat_completed = False
        self.post_interaction_retreat_blocked = False
        self.post_interaction_screen_id = None
        return True

    def update_nfc_interaction_status(
        self,
        screen: Screen,
        attempt: int,
        state: str,
        *,
        started_s: Optional[float] = None,
        last_failure_reason: str = "",
        seq=None,
    ) -> None:
        elapsed = 0.0 if started_s is None else max(0.0, time.monotonic() - started_s)
        self.nfc_interaction_status = {
            "screen_id": int(screen.screen_id),
            "attempt": int(attempt),
            "state": str(state),
            "seq": seq,
            "elapsed_s": round(elapsed, 3),
            "last_failure_reason": str(last_failure_reason or ""),
        }

    def nfc_change_is_terminal(
        self,
        screen: Screen,
        *,
        attempt: int,
        source: str,
        seq=None,
    ) -> bool:
        """Make CHANGED an unconditional exit from the current NFC flow."""
        if screen.status != ScreenStatus.CHANGED:
            return False
        self.update_nfc_interaction_status(
            screen,
            attempt,
            "SUCCESS",
            last_failure_reason="",
            seq=seq,
        )
        self.debug.event(
            "nfc_change_terminal_success",
            screen_id=screen.screen_id,
            attempt=int(attempt),
            seq=seq,
            source=source,
            next_action="retreat_then_mark_target_complete",
            retry_allowed=False,
        )
        return True

    def restore_nfc_physical_contact(
        self,
        screen: Screen,
        *,
        completed_attempt: int,
        seq,
        failure_reason: str,
    ) -> str:
        """Retreat, reclassify this target, then conditionally reapproach."""
        retry_attempt = int(completed_attempt) + 1
        retry_visual_after_s = now_s()
        self.debug.event(
            "nfc_interaction_retry_started",
            screen_id=screen.screen_id,
            attempt=retry_attempt,
            previous_attempt=completed_attempt,
            seq=seq,
            elapsed_s=0.0,
            reason=failure_reason,
            target_preserved=True,
        )
        if self.nfc_change_is_terminal(
            screen,
            attempt=completed_attempt,
            source="status_before_retry_recovery",
            seq=seq,
        ):
            return "already_changed"
        max_cycles = max(1, int(self.config["interaction"].get(
            "nfc_retry_target_reacquire_max_cycles", 3
        )))
        self.debug.event(
            "nfc_retry_target_reacquire_started",
            screen_id=screen.screen_id,
            target_screen_id=screen.screen_id,
            attempt=retry_attempt,
            max_cycles=max_cycles,
            seq=seq,
        )
        self.update_nfc_interaction_status(
            screen,
            retry_attempt,
            "RETRY_RETREAT",
            last_failure_reason=failure_reason,
            seq=seq,
        )
        self.debug.event(
            "nfc_retry_retreat",
            screen_id=screen.screen_id,
            attempt=retry_attempt,
            seq=seq,
            elapsed_s=0.0,
            reason=failure_reason,
            retreat_cm=float(
                self.config["interaction"].get(
                    "post_interaction_retreat_cm", 10.0
                )
            ),
        )
        retreated = self.complete_post_interaction_retreat(
            screen,
            reason="nfc_retry",
        )
        self.debug.event(
            "nfc_retry_relocalize",
            screen_id=screen.screen_id,
            attempt=retry_attempt,
            seq=seq,
            elapsed_s=0.0,
            localization_success=bool(retreated),
            target_reacquired=False,
            reason=("accepted_visual_pose" if retreated else "retreat_localization_failed"),
            target_preserved=True,
        )

        def latest_fresh_target_observation():
            observation = getattr(
                self, "recent_bound_flower_observations", {}
            ).get(int(screen.screen_id))
            if not bool(
                observation is not None
                and int(observation.screen_id) == int(screen.screen_id)
                and int(observation.tag_id) == int(screen.screen_id)
                and bool(observation.binding_ok)
                and float(observation.captured_s) >= float(retry_visual_after_s)
                and float(observation.confidence) >= float(
                    self.config["vision"].get("min_confidence", 0.2)
                )
            ):
                return None
            return observation

        for reacquire_cycle in range(1, max_cycles + 1):
            if self.time_left_s() <= 0.0:
                return "mission_timeout"
            observation = latest_fresh_target_observation()
            target_reacquired = observation is not None
            if not target_reacquired:
                self.update_nfc_interaction_status(
                    screen,
                    retry_attempt,
                    "RETRY_VISUAL_CHECK",
                    last_failure_reason="current_target_not_reacquired",
                    seq=seq,
                )
                target_reacquired = bool(self.localize_scan(
                    reason="nfc_retry_visual_check",
                    allow_pan_search=True,
                    allow_failure_escalation=False,
                    required_target_screen_id=screen.screen_id,
                ))
                observation = latest_fresh_target_observation()
                self.debug.event(
                    "nfc_retry_relocalize",
                    screen_id=screen.screen_id,
                    attempt=retry_attempt,
                    cycle=reacquire_cycle,
                    max_cycles=max_cycles,
                    seq=seq,
                    elapsed_s=0.0,
                    localization_success=(
                        getattr(self, "last_localization_attempt_result", "")
                        == "accepted_visual_pose"
                    ),
                    target_reacquired=bool(target_reacquired),
                    reason=(
                        "current_target_reacquired"
                        if target_reacquired
                        else "current_target_not_seen"
                    ),
                    target_preserved=True,
                )

            valid_observation = observation is not None
            if not valid_observation:
                self.update_nfc_interaction_status(
                    screen,
                    retry_attempt,
                    "RETRY_VISUAL_CHECK",
                    last_failure_reason="fresh_target_classification_missing",
                    seq=seq,
                )
                self.debug.event(
                    "nfc_retry_visual_check",
                    screen_id=screen.screen_id,
                    tag_id=screen.screen_id,
                    attempt=retry_attempt,
                    flower=None,
                    target_flower=self.target_flower,
                    cycle=reacquire_cycle,
                    max_cycles=max_cycles,
                    target_reacquired=bool(target_reacquired),
                    decision="current_target_not_reacquired",
                )
                self.debug.event(
                    "nfc_retry_target_not_seen",
                    target_screen_id=screen.screen_id,
                    attempt=retry_attempt,
                    cycle=reacquire_cycle,
                    max_cycles=max_cycles,
                    target_reacquired=False,
                )
                if reacquire_cycle < max_cycles:
                    self.mission_retry_pause("recovery_retry_interval_s")
                continue

            adopted = self.adopt_cached_target_observation(
                screen,
                observation,
                current_tag_seen_s=float(observation.captured_s),
                source="nfc_retry_relocalization",
            )
            if not adopted:
                self.debug.event(
                    "nfc_retry_visual_check",
                    screen_id=screen.screen_id,
                    tag_id=observation.tag_id,
                    attempt=retry_attempt,
                    flower=observation.flower,
                    target_flower=self.target_flower,
                    decision="target_binding_rejected",
                )
                self.mission_retry_pause("recovery_retry_interval_s")
                continue

            already_changed = observation.flower == self.target_flower
            self.debug.event(
                "nfc_retry_visual_check",
                screen_id=screen.screen_id,
                tag_id=observation.tag_id,
                attempt=retry_attempt,
                flower=observation.flower,
                target_flower=self.target_flower,
                confidence=round(float(observation.confidence), 4),
                captured_s=float(observation.captured_s),
                decision=(
                    "already_changed_skip_retry"
                    if already_changed
                    else "still_needs_change_reapproach"
                ),
            )
            if already_changed:
                screen.status = ScreenStatus.CHANGED
                screen.notes.append("nfc_change_verified_visually_after_timeout")
                self.update_nfc_interaction_status(
                    screen,
                    completed_attempt,
                    "SUCCESS",
                    last_failure_reason="ack_missing_change_verified_visually",
                    seq=seq,
                )
                self.nfc_change_is_terminal(
                    screen,
                    attempt=completed_attempt,
                    source="fpga_confirmed_after_nfc_failure",
                    seq=seq,
                )
                return "already_changed"

            if self.nfc_change_is_terminal(
                screen,
                attempt=completed_attempt,
                source="status_before_retry_reapproach",
                seq=seq,
            ):
                return "already_changed"

            self.update_nfc_interaction_status(
                screen,
                retry_attempt,
                "RETRY_REAPPROACH",
                last_failure_reason=failure_reason,
                seq=seq,
            )
            self.debug.event(
                "nfc_retry_reapproach",
                screen_id=screen.screen_id,
                attempt=retry_attempt,
                seq=seq,
                elapsed_s=0.0,
                reason="repeat_existing_final_forward",
                forward_cm=float(
                    self.config["interaction"].get(
                        "target_final_forward_cm", 17.0
                    )
                ),
            )
            if self.recalibrate_target_for_nfc_retry(screen, retry_attempt):
                return "reapproached"
            self.debug.event(
                "nfc_interaction_invalid_response",
                screen_id=screen.screen_id,
                attempt=retry_attempt,
                seq=seq,
                elapsed_s=0.0,
                reason="nfc_retry_reapproach_failed",
            )
            self.mission_retry_pause("recovery_retry_interval_s")
        self.update_nfc_interaction_status(
            screen,
            retry_attempt,
            "TARGET_REACQUIRE_FAILED",
            last_failure_reason="current_target_reacquire_exhausted",
            seq=seq,
        )
        self.debug.event(
            "nfc_retry_target_reacquire_exhausted",
            screen_id=screen.screen_id,
            target_screen_id=screen.screen_id,
            attempt=retry_attempt,
            cycles=max_cycles,
            seq=seq,
            target_preserved=True,
            next_action="give_up_current_interaction",
        )
        return "target_reacquire_failed"

    def recalibrate_target_for_nfc_retry(
        self,
        screen: Screen,
        retry_attempt: int,
    ) -> bool:
        """Rebuild the configured target pose and identity lock before NFC attempt 2."""
        self.debug.event(
            "nfc_retry_target_recalibration_started",
            screen_id=screen.screen_id,
            tag_id=screen.screen_id,
            attempt=int(retry_attempt),
            task_target_xy=screen.task_target_xy or screen.target_xy,
            desired_yaw=screen.task_target_yaw_deg,
        )
        goal = self.lock_target_goal(screen)
        self.arrived_at_target = False
        self.set_mission_state(MissionState.NAVIGATE_TO_TARGET)
        if not self.navigate_to_screen(screen):
            self.debug.event(
                "nfc_retry_target_recalibration_failed",
                screen_id=screen.screen_id,
                tag_id=screen.screen_id,
                attempt=int(retry_attempt),
                stage="navigate_to_task_target",
                target_goal=goal.as_dict(),
                reason=self.last_navigation_failure_reason or "navigation_failed",
            )
            return False
        self.arrived_at_target = True
        self.set_mission_state(MissionState.ARRIVED_AT_TARGET)
        if not self.confirm_target_tag_now(screen):
            self.debug.event(
                "nfc_retry_target_recalibration_failed",
                screen_id=screen.screen_id,
                tag_id=screen.screen_id,
                attempt=int(retry_attempt),
                stage="confirm_current_target_tag",
                target_goal=goal.as_dict(),
                reason="target_tag_not_confirmed",
            )
            return False
        if not self.execute_final_forward(screen):
            self.debug.event(
                "nfc_retry_target_recalibration_failed",
                screen_id=screen.screen_id,
                tag_id=screen.screen_id,
                attempt=int(retry_attempt),
                stage="final_forward",
                target_goal=goal.as_dict(),
                reason="final_forward_failed",
            )
            return False
        self.debug.event(
            "nfc_retry_target_recalibration_completed",
            screen_id=screen.screen_id,
            tag_id=screen.screen_id,
            attempt=int(retry_attempt),
            target_goal=goal.as_dict(),
            tag_confirmed=True,
            final_forward_executed=True,
        )
        return True

    def give_up_nfc_change(self, screen: Screen, attempts: int, reason: str) -> None:
        """Retire one Screen after the bounded two physical NFC attempts."""
        if not hasattr(self, "nfc_gave_up_screen_ids"):
            self.nfc_gave_up_screen_ids = set()
        self.nfc_gave_up_screen_ids.add(int(screen.screen_id))
        self.nfc_interaction_gave_up = True
        screen.status = ScreenStatus.FAILED
        screen.attempts = max(int(screen.attempts), int(attempts))
        screen.notes.append("nfc_change_gave_up:{}".format(reason))
        self.update_nfc_interaction_status(
            screen,
            attempts,
            "GAVE_UP",
            last_failure_reason=reason,
        )
        self.debug.event(
            "nfc_change_give_up",
            screen_id=screen.screen_id,
            attempts=int(attempts),
            reason=reason,
            mission_continues=True,
        )

    def process_screen_interaction(self, screen: Screen) -> bool:
        if self.nfc_change_is_terminal(
            screen,
            attempt=max(1, int(screen.attempts)),
            source="status_before_interaction",
        ):
            return True
        worker_id = self.worker_id_for_screen(screen)
        if not screen.last_classification or screen.last_classification == self.target_flower:
            self.debug.event("interaction_skipped", screen_id=screen.screen_id, reason="flower_not_changeable")
            return False
        self.nfc_interaction_stopped_for_mission_timeout = False
        self.nfc_interaction_gave_up = False
        attempt = 1
        max_physical_attempts = 2
        while attempt <= max_physical_attempts and self.time_left_s() > 0.0:
            from_flower = screen.last_classification
            authorization_check = self.visual_authorization_check(
                screen,
                expected_from_flower=from_flower,
            )
            self.last_interaction_check = authorization_check.as_dict()
            if not authorization_check.ready:
                self.debug.event(
                    "interaction_safety_gate_blocked",
                    screen_id=screen.screen_id,
                    stage="visual_authorization",
                    attempt=attempt,
                    check=authorization_check.as_dict(),
                )
                return False
            pose_snapshot = None if self.state.pose is None else self.state.pose.as_dict()
            screen.status = ScreenStatus.INTERACTING
            self.set_mission_state(MissionState.EXECUTE_CHANGE)
            attempt_started = time.monotonic()
            attempt_timeout = min(
                float(self.config["interaction"].get(
                    "flower_change_attempt_timeout_s", 15.0
                )),
                max(0.1, self.time_left_s()),
            )
            self.update_nfc_interaction_status(
                screen,
                attempt,
                "WAITING_NFC",
                started_s=attempt_started,
            )
            result = self.interaction.change_flower(
                screen_id=screen.screen_id,
                worker_id=worker_id,
                from_flower=from_flower,
                to_flower=self.target_flower,
                safety_gate=lambda: self.visual_authorization_check(
                    screen,
                    expected_from_flower=from_flower,
                ),
                attempt=attempt,
                attempt_timeout_s=attempt_timeout,
            )
            response = dict(result.response or {})
            seq = response.get("seq")
            if seq is None:
                seq = (response.get("request") or {}).get("nonce")
            elapsed = max(0.0, time.monotonic() - attempt_started)
            record = {
                "t": round(time.time(), 3),
                "screen_id": screen.screen_id,
                "worker_id": worker_id,
                "attempt": attempt,
                "seq": seq,
                "elapsed_s": round(elapsed, 3),
                "attempt_timeout_s": attempt_timeout,
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
                "post_interaction_retreat": {
                    "pending": bool(getattr(self, "post_interaction_retreat_pending", False)),
                    "physical_retreat_completed": bool(getattr(
                        self, "post_interaction_retreat_completed", False
                    )),
                    "blocked": bool(getattr(self, "post_interaction_retreat_blocked", False)),
                    "screen_id": getattr(self, "post_interaction_screen_id", None),
                    "requested_cm": float(
                        self.config["interaction"].get("post_interaction_retreat_cm", 10.0)
                    ),
                },
                "interaction_check": authorization_check.as_dict(),
            }
            self.write_interaction_audit(record)
            self.latest_interaction_result = record
            self.recent_interaction_results.append(record)
            self.recent_interaction_results = self.recent_interaction_results[-5:]
            if result.success:
                changed = apply_worker_change_result(screen, result)
                self.update_nfc_interaction_status(
                    screen,
                    attempt,
                    "SUCCESS",
                    started_s=attempt_started,
                    seq=seq,
                )
                self.debug.event("interaction_changed", **record)
                self.nfc_change_is_terminal(
                    screen,
                    attempt=attempt,
                    source="nfc_ack_success",
                    seq=seq,
                )
                return bool(changed)

            failure_reason = result.error or "nfc_invalid_response"
            self.update_nfc_interaction_status(
                screen,
                attempt,
                "RETRY_RETREAT",
                started_s=attempt_started,
                last_failure_reason=failure_reason,
                seq=seq,
            )
            self.debug.event(
                "interaction_not_changed",
                retrying_same_target=True,
                **record
            )
            if self.time_left_s() <= 0.0:
                break
            if attempt >= max_physical_attempts:
                self.give_up_nfc_change(
                    screen,
                    attempts=attempt,
                    reason=failure_reason,
                )
                return False
            retry_outcome = self.restore_nfc_physical_contact(
                screen,
                completed_attempt=attempt,
                seq=seq,
                failure_reason=failure_reason,
            )
            if retry_outcome == "already_changed" or self.nfc_change_is_terminal(
                screen,
                attempt=attempt,
                source="status_after_retry_recovery",
                seq=seq,
            ):
                return True
            if retry_outcome == "target_reacquire_failed":
                self.give_up_nfc_change(
                    screen,
                    attempts=attempt,
                    reason="current_target_reacquire_exhausted",
                )
                return False
            if retry_outcome != "reapproached":
                break
            if self.nfc_change_is_terminal(
                screen,
                attempt=attempt,
                source="status_before_attempt_2",
                seq=seq,
            ):
                return True
            attempt += 1

        screen.status = ScreenStatus.NEEDS_CHANGE
        self.nfc_interaction_stopped_for_mission_timeout = True
        self.update_nfc_interaction_status(
            screen,
            attempt,
            "MISSION_TIMEOUT",
            last_failure_reason="global_mission_timeout",
        )
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

    def normal_navigation_clearance_ok(self, clearance_cm: float) -> bool:
        minimum = float(
            self.config["navigation"].get("normal_navigation_min_clearance_cm", 25.0)
        )
        return float(clearance_cm) >= minimum

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
        minimum_required = float(nav.get("normal_navigation_min_clearance_cm", 25.0))
        wall_scale = float(nav.get("normal_wall_clearance_penalty_scale", 4.0))
        headings = []
        rejection_reasons = []
        for index, (start, end) in enumerate(zip(path, path[1:]), start=1):
            is_goal = allow_goal_high_cost and index == len(path) - 1
            metrics = self.movement_corridor_metrics(start, end, allow_goal_high_cost=is_goal)
            segment_length = float(metrics["path_length_cm"])
            length += segment_length
            obstacle_integral += float(metrics["path_obstacle_cost"]) * segment_length / 10.0
            clearance = float(metrics["minimum_wall_clearance_cm"])
            minimum_clearance = min(minimum_clearance, clearance)
            wall_penalty += max(0.0, wall_target - clearance) * wall_scale * segment_length / 10.0
            clearance_ok = is_goal or self.normal_navigation_clearance_ok(clearance)
            if not bool(metrics["clear"]):
                if bool(metrics.get("physical_collision", False)):
                    rejection_reasons.append("corridor_physical_collision")
                elif bool(metrics.get("soft_cost_rejected", False)):
                    rejection_reasons.append("corridor_soft_cost_rejected")
                else:
                    rejection_reasons.append("corridor_blocked")
            if not clearance_ok:
                rejection_reasons.append("corridor_clearance_below_minimum")
            clear = clear and bool(metrics["clear"]) and clearance_ok
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
            "minimum_required_clearance_cm": minimum_required,
            "clearance_traversable": minimum_clearance >= minimum_required,
            "wall_clearance_penalty": wall_penalty,
            "turn_cost": turn_cost,
            "action_switch_penalty": switch_penalty,
            "reachability_rejection_reason": (
                None if clear else "+".join(dict.fromkeys(rejection_reasons))
            ),
        }

    def path_segments_clear(
        self,
        points: List[Tuple[float, float]],
        allow_goal_high_cost: bool = False,
        minimum_clearance_cm: Optional[float] = None,
    ) -> bool:
        if len(points) < 2:
            return False
        required_clearance = float(
            self.config["navigation"].get("normal_navigation_min_clearance_cm", 25.0)
            if minimum_clearance_cm is None else minimum_clearance_cm
        )
        for index, pt in enumerate(points[1:], start=1):
            is_goal = allow_goal_high_cost and index == len(points) - 1
            if is_goal:
                if not self.map.is_free_xy(pt):
                    return False
            elif required_clearance > 0.0:
                detail = self.navigation_point_diagnostics(pt)
                if (
                    not detail["footprint_free"]
                    or detail["footprint_max_cost"] >= float(
                        self.config["navigation"].get("normal_navigation_max_cost", 55.0)
                    )
                    or float(detail["clearance_cm"]) < required_clearance
                ):
                    return False
            elif not self.map.is_traversable_xy(
                pt,
                max_cost=float(
                    self.config["navigation"].get("normal_navigation_max_cost", 55.0)
                ),
            ):
                return False
        for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
            is_goal = allow_goal_high_cost and index == len(points) - 1
            metrics = self.movement_corridor_metrics(
                start, end, allow_goal_high_cost=is_goal
            )
            if not metrics.get("clear"):
                return False
            if not is_goal and float(
                metrics.get("minimum_wall_clearance_cm", 0.0)
            ) < required_clearance:
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
            return self.map.target_direct_corridor_metrics(
                start,
                end,
                screen.screen_id,
                half_width,
                max_cost,
                # Final target motion intentionally has narrower clearance than
                # normal navigation.  Physical obstacles and unrelated soft
                # inflation are still enforced by the checks above.
                minimum_non_target_clearance_cm=0.0,
            )["clear"]

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

    def target_owned_approach_metrics(
        self,
        pose: RobotPose,
        screen: Screen,
        goal_xy: Tuple[float, float],
    ) -> dict:
        """Score an approach corridor while exempting only its target inflation."""
        nav = self.config["navigation"]
        metrics = self.map.target_direct_corridor_metrics(
            pose.xy(),
            goal_xy,
            screen.screen_id,
            float(nav.get("target_direct_corridor_half_width_cm", 6.0)),
            float(nav.get("target_direct_non_target_max_cost", 60.0)),
            # These candidates are the hand-off into the bounded final target
            # phase, so do not re-apply normal-navigation clearance to the
            # target-side corridor.  Unrelated inflation and all physical
            # obstacles remain hard constraints.
            minimum_non_target_clearance_cm=0.0,
        )
        reason = None
        if not metrics.get("clear"):
            if metrics.get("physical_collision"):
                reason = "target_approach_physical_collision"
            elif metrics.get("soft_cost_rejected"):
                reason = "target_approach_unrelated_soft_cost_rejected"
            elif metrics.get("clearance_rejected"):
                reason = "target_approach_unrelated_clearance_rejected"
            else:
                reason = "target_approach_corridor_blocked"
        length = float(metrics.get("path_length_cm", distance_xy(pose.xy(), goal_xy)))
        obstacle_component = (
            float(metrics.get("path_obstacle_cost", 0.0))
            * float(nav.get("normal_path_obstacle_cost_scale", 0.35))
        )
        desired_heading = math.degrees(math.atan2(
            float(goal_xy[1]) - pose.y_cm,
            float(goal_xy[0]) - pose.x_cm,
        ))
        heading_delta = abs(angle_diff_deg(desired_heading, pose.yaw_deg))
        turn_cost = 0.0
        if heading_delta > float(nav.get("turn_tolerance_deg", 20.0)):
            turn_cost = float(nav.get("action_planner_turn_fixed_cost_cm", 20.0))
            turn_cost += heading_delta * float(
                nav.get("action_planner_turn_cost_cm_per_deg", 0.8)
            )
        metrics.update({
            "total_cost": length + obstacle_component + turn_cost,
            "turn_cost": turn_cost,
            "reachability_rejection_reason": reason,
            "target_obstacle_soft_cost_exempted": True,
        })
        return metrics

    def navigation_point_diagnostics(self, xy: Tuple[float, float]) -> dict:
        """Describe physical occupancy and inflated footprint cost at one world point."""
        nav = self.config["navigation"]
        in_bounds = bool(self.map.in_bounds_xy(xy))
        grid = self.map.grid_pos(xy)
        free = bool(in_bounds and self.map.is_free_xy(xy))
        max_cost = float(nav.get("normal_navigation_max_cost", 55.0))
        half_width = float(nav.get("translation_corridor_half_width_cm", 8.0))
        offsets = [(0.0, 0.0)]
        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            offsets.append((half_width * math.cos(radians), half_width * math.sin(radians)))
        footprint_free = True
        footprint_max_cost = 0.0
        for dx, dy in offsets:
            sample = (float(xy[0]) + dx, float(xy[1]) + dy)
            if not self.map.is_free_xy(sample):
                footprint_free = False
                footprint_max_cost = float("inf")
                break
            node = self.map.grid_pos(sample)
            footprint_max_cost = max(
                footprint_max_cost, float(self.map.cost[node[0], node[1]])
            )
        clearance_cm = round(float(self.map.robot_clearance_cm(xy)), 3)
        clearance_traversable = self.normal_navigation_clearance_ok(clearance_cm)
        if not free or not footprint_free:
            occupancy_class = "HARD_BLOCKED"
        elif footprint_max_cost >= max_cost or not clearance_traversable:
            occupancy_class = "SOFT_HIGH_COST"
        else:
            occupancy_class = "SAFE"
        return {
            "xy": (float(xy[0]), float(xy[1])),
            "grid": grid,
            "in_bounds": in_bounds,
            "blocked": not free,
            "traversable": bool(free and float(self.map.cost[grid]) < max_cost),
            "cost": None if not in_bounds else round(float(self.map.cost[grid]), 3),
            "clearance_cm": clearance_cm,
            "minimum_clearance_cm": float(
                nav.get("normal_navigation_min_clearance_cm", 25.0)
            ),
            "clearance_traversable": clearance_traversable,
            "occupancy_class": occupancy_class,
            "footprint_free": footprint_free,
            "footprint_max_cost": footprint_max_cost,
            "footprint_traversable": (
                footprint_free
                and footprint_max_cost < max_cost
                and clearance_traversable
            ),
            "free_neighbor_count": sum(
                1 for node in self.map._neighbors(grid, include_diagonal=True)
                if self.map.is_free_grid(node)
            ),
        }

    def escape_corridor_metrics(self, start_xy, end_xy) -> dict:
        """Allow only a physically clear, non-worsening move out of soft inflation."""
        nav = self.config["navigation"]
        physical = self.map.translation_corridor_metrics(
            start_xy,
            end_xy,
            float(nav.get("translation_corridor_half_width_cm", 8.0)),
            float("inf"),
        )
        start = self.navigation_point_diagnostics(start_xy)
        end = self.navigation_point_diagnostics(end_xy)
        improvement = float(start["footprint_max_cost"]) - float(end["footprint_max_cost"])
        clearance_improvement = float(end["clearance_cm"]) - float(start["clearance_cm"])
        minimum = float(nav.get("planner_start_escape_min_cost_improvement", 2.0))
        physical["clear"] = bool(
            physical.get("clear")
            and end["footprint_free"]
            and (
                improvement >= minimum
                or clearance_improvement >= 1.0
                or end["footprint_traversable"]
            )
            and float(end["footprint_max_cost"]) <= float(start["footprint_max_cost"])
        )
        physical.update({
            "start_footprint_max_cost": start["footprint_max_cost"],
            "end_footprint_max_cost": end["footprint_max_cost"],
            "cost_improvement": improvement,
            "clearance_improvement_cm": clearance_improvement,
        })
        return physical

    def safe_start_projection(self, pose: RobotPose) -> Optional[Tuple[float, float]]:
        start = self.navigation_point_diagnostics(pose.xy())
        if start["footprint_traversable"]:
            return None
        maximum = float(self.config["navigation"].get("planner_start_projection_max_cm", 22.0))
        radius = int(math.ceil(maximum / self.map.res))
        start_grid = self.map.grid_pos(pose.xy())
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                node = (start_grid[0] + dx, start_grid[1] + dy)
                if not (0 <= node[0] < self.map.rows and 0 <= node[1] < self.map.cols):
                    continue
                xy = self.map.xy_from_grid(node)
                travel = distance_xy(pose.xy(), xy)
                if travel < 1.0 or travel > maximum:
                    continue
                detail = self.navigation_point_diagnostics(xy)
                if not detail["footprint_traversable"]:
                    continue
                escape = self.escape_corridor_metrics(pose.xy(), xy)
                if not escape.get("clear"):
                    continue
                score = travel - 0.15 * max(0.0, escape["cost_improvement"])
                candidates.append((score, travel, xy, detail, escape))
        if not candidates:
            return None
        _, travel, xy, detail, escape = min(candidates, key=lambda item: item[:2])
        self.debug.event(
            "start_grid_projected",
            raw_start_xy=pose.xy(),
            raw_start_grid=start["grid"],
            raw_start_footprint_max_cost=start["footprint_max_cost"],
            projected_start_xy=xy,
            projected_start_grid=detail["grid"],
            projected_footprint_max_cost=detail["footprint_max_cost"],
            distance_cm=round(travel, 3),
            clearance_cm=detail["clearance_cm"],
            cost_improvement=round(float(escape["cost_improvement"]), 3),
        )
        return xy

    def reachable_navigation_goal_candidates(
        self,
        pose: RobotPose,
        target_screen: Optional[Screen],
        final_goal_xy: Tuple[float, float],
        allow_goal_high_cost: bool,
    ) -> List[dict]:
        candidates = []
        direct_limit = float(self.config["navigation"].get("target_direct_approach_distance_cm", 40.0))
        goal_distance = distance_xy(pose.xy(), final_goal_xy)
        if target_screen is not None and goal_distance > 1.0:
            radius = min(max(0.0, direct_limit - 2.0), goal_distance)
            scale = radius / max(1e-6, goal_distance)
            line_stage = (
                float(final_goal_xy[0]) + (float(pose.x_cm) - float(final_goal_xy[0])) * scale,
                float(final_goal_xy[1]) + (float(pose.y_cm) - float(final_goal_xy[1])) * scale,
            )
            if distance_xy(pose.xy(), line_stage) >= 3.0:
                candidates.append({"goal_type": "staging", "xy": line_stage, "source": "line_to_target"})
        if target_screen is not None and target_screen.face_center_xy is not None:
            nav = self.config["navigation"]
            base = float(self.config["interaction"]["target_distance_cm"])
            maximum = max(base, float(nav.get("reachable_approach_max_standoff_cm", 40.0)))
            step = max(2.0, float(nav.get("reachable_approach_step_cm", 5.0)))
            lateral_step = max(0.0, float(nav.get("reachable_approach_lateral_step_cm", 6.0)))
            configured_lateral = float(self.config["interaction"].get("target_lateral_offset_cm", -1.0))
            lateral_values = [configured_lateral]
            if lateral_step > 0.0:
                lateral_values.extend((configured_lateral - lateral_step, configured_lateral + lateral_step))
            standoff = base + step
            while standoff <= maximum + 1e-6:
                for lateral in lateral_values:
                    xy = (
                        float(target_screen.face_center_xy[0]) + target_screen.normal_xy[0] * standoff + target_screen.screen_left_tangent_xy[0] * lateral,
                        float(target_screen.face_center_xy[1]) + target_screen.normal_xy[1] * standoff + target_screen.screen_left_tangent_xy[1] * lateral,
                    )
                    candidates.append({
                        "goal_type": "approach",
                        "xy": xy,
                        "source": "face_standoff",
                        "standoff_cm": standoff,
                        "lateral_offset_cm": lateral,
                    })
                standoff += step
        candidates.append({
            "goal_type": "exact",
            "xy": (float(final_goal_xy[0]), float(final_goal_xy[1])),
            "source": "canonical_target_goal",
            "allow_goal_high_cost": bool(allow_goal_high_cost),
        })
        unique = []
        seen = set()
        for item in candidates:
            key = (round(item["xy"][0], 2), round(item["xy"][1], 2), item["goal_type"])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def emit_navigation_plan_diagnostic(
        self,
        pose: RobotPose,
        target_screen: Optional[Screen],
        final_goal_xy: Tuple[float, float],
        selected_goal_type: str,
        selected_goal_xy: Optional[Tuple[float, float]],
        path_found: bool,
    ) -> None:
        """Emit one self-contained robot/anchor/approach/staging grid diagnosis."""
        start = self.navigation_point_diagnostics(pose.xy())
        anchor_xy = None if target_screen is None else target_screen.center_xy
        anchor = None if anchor_xy is None else self.navigation_point_diagnostics(anchor_xy)
        approach = self.navigation_point_diagnostics(final_goal_xy)
        selected = (
            None if selected_goal_xy is None
            else self.navigation_point_diagnostics(selected_goal_xy)
        )
        self.debug.event(
            "navigation_plan_diagnostic",
            screen_id=None if target_screen is None else int(target_screen.screen_id),
            robot_xy=pose.xy(),
            robot_yaw=pose.yaw_deg,
            start_grid=start["grid"],
            start_blocked=start["blocked"],
            start_footprint_blocked=not start["footprint_free"],
            start_footprint_max_cost=start["footprint_max_cost"],
            target_anchor_xy=anchor_xy,
            anchor_grid=None if anchor is None else anchor["grid"],
            anchor_blocked=None if anchor is None else anchor["blocked"],
            resolved_approach_xy=final_goal_xy,
            approach_grid=approach["grid"],
            approach_blocked=approach["blocked"],
            approach_footprint_max_cost=approach["footprint_max_cost"],
            staging_xy=(
                selected_goal_xy
                if selected_goal_type in ("start_projection", "staging", "approach")
                else None
            ),
            staging_grid=(
                None
                if selected is None
                or selected_goal_type not in ("start_projection", "staging", "approach")
                else selected["grid"]
            ),
            staging_blocked=(
                None
                if selected is None
                or selected_goal_type not in ("start_projection", "staging", "approach")
                else selected["blocked"]
            ),
            selected_goal_type=selected_goal_type,
            selected_goal_xy=selected_goal_xy,
            selected_goal_grid=None if selected is None else selected["grid"],
            selected_goal_blocked=None if selected is None else selected["blocked"],
            path_found=bool(path_found),
        )

    def plan_navigation_path(
        self,
        pose: RobotPose,
        goal_xy: Tuple[float, float],
        allow_goal_high_cost: bool = False,
        target_screen: Optional[Screen] = None,
    ) -> List[Tuple[float, float]]:
        direct = self.target_direct_approach_path(pose, target_screen, goal_xy)
        screen_id = None if target_screen is None else int(target_screen.screen_id)
        target_goal = getattr(self, "current_target_goal", None)
        anchor_xy = target_screen.center_xy if target_screen is not None else None
        if direct:
            self.active_navigation_plan = {
                "goal_type": "exact",
                "goal_xy": list(goal_xy),
                "final_target_xy": list(goal_xy),
                "staging_xy": None,
                "direct_corridor_clear": True,
            }
            self.debug.event(
                "target_direct_approach",
                navigation_mode="target_direct_approach",
                current_target_screen_id=screen_id,
                target_direct_corridor_clear=True,
                target_direct_cost_exemption=True,
                target_xy=goal_xy,
            )
            self.emit_navigation_plan_diagnostic(
                pose, target_screen, goal_xy, "exact", goal_xy, True
            )
            return direct

        projected = self.safe_start_projection(pose)
        if projected is not None:
            self.active_navigation_plan = {
                "goal_type": "start_projection",
                "goal_xy": list(projected),
                "final_target_xy": list(goal_xy),
                "staging_xy": list(projected),
                "direct_corridor_clear": False,
            }
            self.debug.event(
                "path_plan_requested",
                screen_id=screen_id,
                robot_xy=pose.xy(),
                robot_grid=self.map.grid_pos(pose.xy()),
                target_anchor_xy=anchor_xy,
                target_goal_xy=goal_xy,
                target_grid=self.map.grid_pos(goal_xy),
                goal_type="start_projection",
                plan_goal_xy=projected,
                staging_xy=projected,
                staging_grid=self.map.grid_pos(projected),
                direct_corridor_clear=False,
            )
            self.debug.event(
                "path_plan_success",
                screen_id=screen_id,
                goal_type="start_projection",
                goal_xy=projected,
                path_length_cm=round(distance_xy(pose.xy(), projected), 3),
                path_nodes=2,
            )
            self.emit_navigation_plan_diagnostic(
                pose, target_screen, goal_xy, "start_projection", projected, True
            )
            return [pose.xy(), projected]

        viable = []
        action_fallback_candidates = []
        candidate_evaluations = []
        minimum_clearance = float(
            self.config["navigation"].get("normal_navigation_min_clearance_cm", 25.0)
        )
        for candidate in self.reachable_navigation_goal_candidates(
            pose, target_screen, goal_xy, allow_goal_high_cost
        ):
            plan_goal = candidate["xy"]
            detail = self.navigation_point_diagnostics(plan_goal)
            candidate_allow_high = bool(candidate.get("allow_goal_high_cost", False))
            valid = bool(
                detail["in_bounds"]
                and not detail["blocked"]
                and (candidate_allow_high or detail["footprint_traversable"])
                and (candidate_allow_high or detail["clearance_cm"] >= minimum_clearance)
            )
            path = []
            metrics = {}
            path_source = "astar"
            astar = {}
            raw_path = []
            rejection_reason = None
            if not detail["in_bounds"]:
                rejection_reason = "goal_out_of_bounds"
            elif detail["blocked"] or not detail["footprint_free"]:
                rejection_reason = "goal_physical_footprint_collision"
            elif not detail["footprint_traversable"] and not candidate_allow_high:
                rejection_reason = (
                    "goal_soft_cost_rejected"
                    if detail["footprint_max_cost"] >= float(
                        self.config["navigation"].get(
                            "normal_navigation_max_cost", 55.0
                        )
                    )
                    else "goal_clearance_below_minimum"
                )
            if valid:
                raw_path = self.map.plan(
                    pose.xy(), plan_goal, allow_goal_high_cost=candidate_allow_high
                )
                astar = dict(getattr(self.map, "last_astar_metrics", {}))
                if raw_path:
                    path = self.compact_path_points([pose.xy()] + list(raw_path))
                    metrics = self.normal_path_metrics(
                        pose, path, allow_goal_high_cost=candidate_allow_high
                    )
                    if not metrics.get("clear"):
                        rejection_reason = metrics.get(
                            "reachability_rejection_reason"
                        ) or "astar_path_post_validation_failed"
                        path = []
                else:
                    rejection_reason = "astar_{}".format(
                        astar.get("reason", "no_path")
                    )

            # Approach poses are generated from the locked target face.  They
            # may ignore only that target building's soft inflation; physical
            # occupancy, field boundaries, unrelated buildings, and dynamic
            # obstacles remain hard constraints.  Exact goals still enter via
            # the existing bounded target_direct_approach path.
            target_approach_eligible = bool(
                target_screen is not None
                and candidate["goal_type"] == "approach"
                and self.current_target_screen_id == int(target_screen.screen_id)
                and detail["in_bounds"]
                and not detail["blocked"]
            )
            if not path and target_approach_eligible:
                target_metrics = self.target_owned_approach_metrics(
                    pose, target_screen, plan_goal
                )
                if target_metrics.get("clear"):
                    path = [pose.xy(), plan_goal]
                    metrics = target_metrics
                    path_source = "target_owned_approach"
                    rejection_reason = None
                else:
                    rejection_reason = target_metrics.get(
                        "reachability_rejection_reason"
                    ) or rejection_reason

            cost = float(metrics.get("total_cost", float("inf")))
            if path:
                cost += 0.25 * distance_xy(plan_goal, goal_xy)
                viable.append((
                    cost, candidate, path, metrics, detail, astar, path_source
                ))
            evaluation = {
                "candidate": candidate,
                "detail": detail,
                "astar": astar,
                "path": path,
                "metrics": metrics,
                "path_source": path_source if path else None,
                "cost": cost,
                "rejection_reason": rejection_reason,
                "astar_path_found": bool(raw_path),
            }
            candidate_evaluations.append(evaluation)
            if valid and not path:
                action_fallback_candidates.append(evaluation)
        # Prefer the established A* approach/staging ordering.  Only when all
        # center-cell A* routes fail footprint clearance do we invoke the
        # action-space planner as a second way to reach those same candidates.
        if (
            not viable
            and action_fallback_candidates
            and bool(self.config["navigation"].get("action_planner_enabled", True))
        ):
            for evaluation in action_fallback_candidates:
                candidate = evaluation["candidate"]
                detail = evaluation["detail"]
                astar = evaluation["astar"]
                plan_goal = candidate["xy"]
                candidate_allow_high = bool(candidate.get("allow_goal_high_cost", False))
                alternate = self.map.plan_action_path(
                    pose,
                    plan_goal,
                    self.config["navigation"],
                    self.config["motion"],
                    allow_goal_high_cost=candidate_allow_high,
                )
                if not alternate:
                    evaluation["rejection_reason"] = "+".join(filter(None, (
                        evaluation.get("rejection_reason"),
                        "action_planner_no_path",
                    )))
                    continue
                metrics = self.normal_path_metrics(
                    pose,
                    alternate,
                    allow_goal_high_cost=candidate_allow_high,
                )
                if not metrics.get("clear"):
                    evaluation["rejection_reason"] = "+".join(filter(None, (
                        evaluation.get("rejection_reason"),
                        metrics.get("reachability_rejection_reason")
                        or "action_planner_path_post_validation_failed",
                    )))
                    continue
                planner_metrics = getattr(self.map, "last_action_plan_metrics", {})
                if planner_metrics.get("total_cost") is not None:
                    metrics["total_cost"] = float(planner_metrics["total_cost"])
                cost = float(metrics.get("total_cost", float("inf")))
                cost += 0.25 * distance_xy(plan_goal, goal_xy)
                viable.append((
                    cost,
                    candidate,
                    alternate,
                    metrics,
                    detail,
                    astar,
                    "action_planner",
                ))
                evaluation.update({
                    "path": alternate,
                    "metrics": metrics,
                    "path_source": "action_planner",
                    "cost": cost,
                    "rejection_reason": None,
                })
        for evaluation in candidate_evaluations:
            candidate = evaluation["candidate"]
            detail = evaluation["detail"]
            astar = evaluation["astar"]
            path = evaluation["path"]
            cost = evaluation["cost"]
            self.debug.event(
                "staging_candidate_generated",
                screen_id=screen_id,
                goal_type=candidate["goal_type"],
                candidate_xy=candidate["xy"],
                candidate_grid=detail["grid"],
                blocked=detail["blocked"],
                occupancy_class=detail["occupancy_class"],
                footprint_traversable=detail["footprint_traversable"],
                footprint_free=detail["footprint_free"],
                footprint_max_cost=detail["footprint_max_cost"],
                clearance_cm=detail["clearance_cm"],
                reachable=bool(path),
                reachability_rejection_reason=evaluation["rejection_reason"],
                selected_path_source=evaluation["path_source"],
                cost=None if not path else round(float(cost), 3),
                astar_path_found=evaluation["astar_path_found"],
                astar_reason=astar.get("reason"),
                astar_expanded_nodes=astar.get("expanded_nodes"),
            )
        if not viable:
            candidate_rejections = [
                {
                    "goal_type": item["candidate"]["goal_type"],
                    "candidate_xy": item["candidate"]["xy"],
                    "reason": item["rejection_reason"],
                    "astar_reason": item["astar"].get("reason"),
                    "astar_path_found": item["astar_path_found"],
                }
                for item in candidate_evaluations
            ]
            self.active_navigation_plan = {
                "goal_type": "none",
                "goal_xy": None,
                "final_target_xy": list(goal_xy),
                "staging_xy": None,
                "direct_corridor_clear": False,
                "candidate_rejections": candidate_rejections,
            }
            self.emit_navigation_plan_diagnostic(
                pose, target_screen, goal_xy, "none", None, False
            )
            return []

        (
            _, selected, astar_path, astar_metrics, selected_detail,
            astar_debug, initial_path_source,
        ) = min(
            viable, key=lambda item: item[0]
        )
        normal_goal = selected["xy"]
        normal_allow_goal_high_cost = bool(selected.get("allow_goal_high_cost", False))
        goal_type = selected["goal_type"]
        self.active_navigation_plan = {
            "goal_type": goal_type,
            "goal_xy": list(normal_goal),
            "final_target_xy": list(goal_xy),
            "staging_xy": list(normal_goal) if goal_type in ("staging", "approach") else None,
            "direct_corridor_clear": False,
            "source": selected.get("source"),
            "path_source": initial_path_source,
        }
        self.debug.event(
            "normal_navigation_staging_target",
            navigation_mode="normal",
            target_xy=goal_xy,
            staging_xy=normal_goal,
            staging_goal_type=goal_type,
            target_direct_cost_exemption=False,
        )
        self.debug.event(
            "path_plan_requested",
            screen_id=screen_id,
            robot_xy=pose.xy(),
            robot_grid=self.map.grid_pos(pose.xy()),
            target_anchor_xy=anchor_xy,
            target_goal_xy=goal_xy,
            target_grid=self.map.grid_pos(goal_xy),
            goal_type=goal_type,
            plan_goal_xy=normal_goal,
            staging_xy=normal_goal if goal_type in ("staging", "approach") else None,
            staging_grid=self.map.grid_pos(normal_goal),
            direct_corridor_clear=False,
            target_generation=None if target_goal is None else target_goal.generation_id,
        )

        candidates = [(initial_path_source, astar_path, astar_metrics)]
        for translation_path in self.body_translation_candidate_paths(
            pose, normal_goal, allow_goal_high_cost=normal_allow_goal_high_cost
        ):
            metrics = self.normal_path_metrics(
                pose, translation_path,
                allow_goal_high_cost=normal_allow_goal_high_cost,
                translation_only=True,
            )
            if metrics.get("clear"):
                candidates.append(("body_translation", translation_path, metrics))
        if bool(self.config["navigation"].get("action_planner_enabled", True)):
            action_path = self.map.plan_action_path(
                pose, normal_goal, self.config["navigation"], self.config["motion"],
                allow_goal_high_cost=normal_allow_goal_high_cost,
            )
            if action_path:
                metrics = self.normal_path_metrics(
                    pose, action_path, allow_goal_high_cost=normal_allow_goal_high_cost
                )
                if metrics.get("clear"):
                    planner_metrics = getattr(self.map, "last_action_plan_metrics", {})
                    if planner_metrics.get("total_cost") is not None:
                        metrics["total_cost"] = float(planner_metrics["total_cost"])
                        metrics["turn_cost"] = float(planner_metrics.get("turn_cost", 0.0))
                        metrics["selected_actions"] = list(planner_metrics.get("selected_actions", []))
                    candidates.append(("action_planner", action_path, metrics))
        priority = {"body_translation": 0, "action_planner": 1, "astar": 2}
        selected_name, selected_path, selected_metrics = min(
            candidates,
            key=lambda item: (float(item[2].get("total_cost", float("inf"))), priority.get(item[0], 9)),
        )
        self.debug.event(
            "staging_path_selected" if goal_type in ("staging", "approach") else "navigation_path_selected",
            screen_id=screen_id,
            goal_type=goal_type,
            staging_xy=normal_goal if goal_type in ("staging", "approach") else None,
            target_xy=goal_xy,
            selected_path_type=selected_name,
            path_length_cm=round(float(selected_metrics.get("path_length_cm", 0.0)), 2),
            path_nodes=len(selected_path),
            total_cost=round(float(selected_metrics.get("total_cost", 0.0)), 2),
        )
        self.debug.event(
            "path_plan_success",
            screen_id=screen_id,
            goal_type=goal_type,
            goal_xy=normal_goal,
            path_length_cm=round(float(selected_metrics.get("path_length_cm", 0.0)), 2),
            path_nodes=len(selected_path),
            astar_expanded_nodes=astar_debug.get("expanded_nodes"),
        )
        self.emit_navigation_plan_diagnostic(
            pose, target_screen, goal_xy, goal_type, normal_goal, True
        )
        return selected_path

    def choose_target_direct_action(
        self,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        screen: Screen,
        bypass_action_safety: bool = False,
        final_goal_distance_cm: Optional[float] = None,
    ) -> Optional[dict]:
        """Choose forward first, then lateral, with a shortened final step."""
        nav = self.config["navigation"]
        forward, lateral = self.local_vector_to(pose, waypoint)
        minimum = float(nav.get("target_direct_min_component_cm", 2.0))
        radius = float(nav.get("target_arrival_radius_cm", 4.0))
        half_width = float(nav.get("target_direct_corridor_half_width_cm", 6.0))
        max_cost = float(nav.get("target_direct_non_target_max_cost", 60.0))

        def corridor_for(forward_cm=0.0, lateral_cm=0.0):
            if bypass_action_safety:
                return True
            end = self.translated_pose_xy(pose, forward_cm=forward_cm, lateral_cm=lateral_cm)
            return self.map.target_direct_corridor_clear(
                pose.xy(), end, screen.screen_id, half_width, max_cost
            )

        if forward < 0.0:
            reverse_max_goal_distance_cm = float(
                nav.get("reverse_prefer_max_goal_distance_cm", 10.0)
            )
            reverse_allowed_by_goal_distance = (
                final_goal_distance_cm is None
                or float(final_goal_distance_cm) <= reverse_max_goal_distance_cm
            )
            rear_angle_error = math.degrees(
                math.atan2(abs(float(lateral)), max(1e-6, -float(forward)))
            )
            rear_tolerance = float(nav.get("reverse_prefer_rear_angle_tolerance_deg", 30.0))
            max_lateral = float(nav.get("reverse_prefer_max_lateral_cm", 8.0))
            step = abs(float(self.config["motion"]["actions"]["back_fast"].get("forward_cm", -2.5)))
            next_xy = self.translated_pose_xy(pose, forward_cm=-step)
            next_distance = distance_xy(next_xy, waypoint)
            current_distance = distance_xy(pose.xy(), waypoint)
            reverse_rejected_reason = None
            if not reverse_allowed_by_goal_distance:
                reverse_rejected_reason = "goal_too_far_for_reverse"
            elif rear_angle_error > rear_tolerance:
                reverse_rejected_reason = "rear_angle_exceeds_tolerance"
            elif abs(lateral) > max_lateral:
                reverse_rejected_reason = "lateral_error_too_large"
            elif next_distance >= current_distance:
                reverse_rejected_reason = "would_not_reduce_goal_distance"
            elif not corridor_for(forward_cm=-step):
                reverse_rejected_reason = "rear_corridor_blocked"
            self.debug.event(
                "reverse_preference_evaluated",
                navigation_mode="target_direct_approach",
                selected_action="reverse" if reverse_rejected_reason is None else None,
                target_local_forward_cm=round(float(forward), 2),
                target_local_lateral_cm=round(float(lateral), 2),
                target_rear_angle_error_deg=round(float(rear_angle_error), 2),
                final_goal_distance_cm=(
                    None
                    if final_goal_distance_cm is None
                    else round(float(final_goal_distance_cm), 2)
                ),
                reverse_max_goal_distance_cm=round(reverse_max_goal_distance_cm, 2),
                reverse_allowed_by_goal_distance=reverse_allowed_by_goal_distance,
                reverse_preferred=reverse_rejected_reason is None,
                reverse_rejected_reason=reverse_rejected_reason,
            )
            if reverse_rejected_reason is None:
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
            self.post_action_relocalize(
                "target_direct_approach",
                pose_before,
                result,
                target_xy,
                navigation_mode="target_direct_approach",
            )
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

    def navigation_relocalization_mode(
        self,
        navigation_mode: Optional[str] = None,
        *,
        recovery: bool = False,
    ) -> str:
        """Normalize navigation state into one configured localization phase."""
        value = str(navigation_mode or "").strip().lower()
        if recovery or value in ("recovery", "navigation_recovery"):
            return "recovery"
        if value in ("target_direct", "target_direct_approach", "final_approach"):
            return "target_direct"
        plan = getattr(self, "active_navigation_plan", None) or {}
        if value == "staging" or plan.get("goal_type") in (
            "start_projection", "staging", "approach"
        ):
            return "staging"
        return "normal"

    def relocalization_action_budget(self, phase: str, confidence: Confidence) -> int:
        nav = self.config["navigation"]
        suffix = str(getattr(confidence, "value", confidence)).lower()
        fallback = int(nav.get(
            "relocalize_after_actions_{}".format(suffix),
            nav.get("relocalize_after_actions", 1),
        ))
        return max(1, int(nav.get(
            "relocalize_action_budget_{}_{}".format(phase, suffix), fallback
        )))

    def visual_pose_age_s(self) -> float:
        pose = self.state.pose
        if pose is None:
            return float("inf")
        last_success = float(getattr(self, "last_localize_success_s", 0.0))
        if last_success <= 0.0 and pose.source not in ("DEAD_RECKONING", "UNKNOWN"):
            last_success = float(pose.last_update_s)
        return float("inf") if last_success <= 0.0 else max(0.0, now_s() - last_success)

    def visual_pose_is_fresh(self, max_age_s: float) -> bool:
        pose = self.state.pose
        return bool(
            pose is not None
            and pose.confidence != Confidence.LOW
            and pose.source not in ("DEAD_RECKONING", "UNKNOWN")
            and int(getattr(self.state, "actions_since_localize", 0)) == 0
            and self.visual_pose_age_s() <= float(max_age_s)
        )

    def adaptive_relocalization_decision(
        self,
        navigation_mode: Optional[str] = None,
        *,
        last_action: str = "",
        action_result=None,
        obstacle_tight: bool = False,
        recovery: bool = False,
        force_reason: str = "",
        emit: bool = True,
    ) -> dict:
        """Decide whether dead reckoning remains safe for the current phase."""
        nav = self.config["navigation"]
        phase = self.navigation_relocalization_mode(navigation_mode, recovery=recovery)
        pose = self.state.pose
        confidence = Confidence.LOW if pose is None else pose.confidence
        actions = int(getattr(self.state, "actions_since_localize", 0))
        uncertainty = float(getattr(self.state, "motion_uncertainty", 0.0))
        action_budget = self.relocalization_action_budget(phase, confidence)
        uncertainty_limit = float(nav.get(
            "relocalize_uncertainty_limit_{}".format(phase),
            nav.get("relocalize_uncertainty_threshold", 7.0),
        ))
        action_key = str(last_action or getattr(self, "last_motion_action", "") or "")
        yaw_per_cycle = 0.0
        if action_result is not None:
            actual = max(1, int(
                getattr(action_result, "executed_times", 0)
                or getattr(action_result, "times", 1)
                or 1
            ))
            yaw_per_cycle = abs(float(getattr(action_result, "model_yaw_deg", 0.0))) / actual
        large_turn = bool(
            "large" in action_key.lower()
            or yaw_per_cycle >= float(nav.get("large_turn_threshold_deg", 35.0))
        )

        decision = "continue_dead_reckoning"
        reason = "within_action_and_uncertainty_budget"
        if force_reason:
            decision, reason = "relocalize_now", str(force_reason)
        elif pose is None:
            decision, reason = "relocalize_now", "pose_missing"
        elif confidence == Confidence.LOW:
            decision, reason = "relocalize_now", "pose_confidence_low"
        elif large_turn and actions > 0:
            decision, reason = "relocalize_now", "large_turn"
        elif obstacle_tight:
            decision, reason = "relocalize_now", "obstacle_tight_navigation"
        elif uncertainty >= uncertainty_limit:
            decision, reason = "relocalize_now", "motion_uncertainty_limit"
        elif actions >= action_budget:
            decision, reason = "relocalize_now", "action_budget_exhausted"
        elif not bool(nav.get("adaptive_relocalization_enabled", True)) and actions > 0:
            decision, reason = "relocalize_now", "adaptive_policy_disabled"

        detail = {
            "actions_since_localize": actions,
            "actions_since_last_successful_localization": actions,
            "motion_uncertainty": round(uncertainty, 3),
            "last_successful_localization_s": float(getattr(
                self, "last_localize_success_s", 0.0
            )),
            "localization_attempt_result": str(getattr(
                self, "last_localization_attempt_result", "unknown"
            )),
            "pose_confidence": None if pose is None else pose.confidence.value,
            "effective_pose_confidence": confidence.value,
            "navigation_mode": phase,
            "last_action": action_key or None,
            "large_turn_relocalization_pending": bool(large_turn and actions > 0),
            "action_budget": action_budget,
            "uncertainty_limit": uncertainty_limit,
            "decision": decision,
            "reason": reason,
        }
        if emit:
            self.debug.event("relocalization_decision", **detail)
        return detail

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
        phase = self.navigation_relocalization_mode(
            navigation_mode,
            recovery=recovery or near_wall,
        )
        action_budget = self.relocalization_action_budget(phase, confidence)
        remaining_actions = max(
            1,
            action_budget - int(getattr(self.state, "actions_since_localize", 0)),
        )
        cap = min(cap, remaining_actions)
        reasons.append("{}_budget".format(phase))
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
            uncertainty_limit = float(nav.get(
                "relocalize_uncertainty_limit_{}".format(phase),
                nav.get("relocalize_uncertainty_threshold", 6.0),
            ))
            budget = max(0.0, uncertainty_limit - float(getattr(self.state, "motion_uncertainty", 0.0)))
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

    def post_action_relocalize(
        self,
        reason: str,
        pose_before: RobotPose,
        result,
        target_xy,
        *,
        navigation_mode: Optional[str] = None,
        obstacle_tight: bool = False,
        force_reason: str = "",
    ) -> bool:
        """Adaptively localize after motion while always requesting a local replan."""
        actual = getattr(result, "executed_times", None)
        if actual is None:
            actual = int(getattr(result, "times", 0)) if bool(getattr(result, "ok", False)) else 0
        self.last_motion_action = str(getattr(result, "key", "") or reason)
        relocalization = self.adaptive_relocalization_decision(
            navigation_mode,
            last_action=self.last_motion_action,
            action_result=result,
            obstacle_tight=obstacle_tight,
            force_reason=force_reason,
        )
        should_localize = relocalization["decision"] == "relocalize_now"
        localized = False
        if should_localize:
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
            relocalization_skipped=not should_localize,
            relocalization_decision=relocalization["decision"],
            relocalization_reason=relocalization["reason"],
        )
        self.debug.event(
            "post_action_replan",
            target_xy=target_xy,
            post_action_replanned=False,
            replan_requested=True,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
        )
        self.pending_post_action_replan = True
        return bool(localized or not should_localize)

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
        temporary = set(getattr(self, "temporarily_failed_targets", {}))
        nfc_gave_up = set(getattr(self, "nfc_gave_up_screen_ids", set()))
        locked = self.map.screens.get(getattr(self, "current_target_screen_id", None))
        if (
            locked is not None
            and not locked.terminal()
            and int(locked.screen_id) not in temporary
            and int(locked.screen_id) not in nfc_gave_up
        ):
            self.last_target_plan = {
                "selection_rule": "preserve_locked_target",
                "screen_id": locked.screen_id,
                "task_target_xy": list(locked.task_target_xy or locked.interaction_xy),
            }
            return locked
        ranked = sorted(
            (
                screen
                for screen in self.map.screens.values()
                if (
                    not screen.done()
                    and int(screen.screen_id) not in temporary
                    and int(screen.screen_id) not in nfc_gave_up
                )
            ),
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
        bypass_action_safety: bool = False,
        final_goal_distance_cm: Optional[float] = None,
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
        escape_mode = bool(
            getattr(self, "active_navigation_plan", None)
            and self.active_navigation_plan.get("goal_type") == "start_projection"
        )

        def corridor_metrics_to(next_xy):
            if bypass_action_safety:
                return {
                    "clear": True,
                    "path_obstacle_cost": 0.0,
                    "minimum_wall_clearance_cm": 9999.0,
                    "safety_bypassed": True,
                }
            if escape_mode:
                return self.escape_corridor_metrics(pose.xy(), next_xy)
            return self.movement_corridor_metrics(pose.xy(), next_xy)

        reverse_rejected_reason = "disabled"
        reverse_max_goal_distance_cm = float(
            nav_cfg.get("reverse_prefer_max_goal_distance_cm", 10.0)
        )
        reverse_allowed_by_goal_distance = (
            final_goal_distance_cm is None
            or escape_mode
            or float(final_goal_distance_cm) <= reverse_max_goal_distance_cm
        )
        rear_angle_error = math.degrees(
            math.atan2(abs(float(lateral)), max(1e-6, -float(forward)))
        ) if forward < 0.0 else 180.0
        if bool(nav_cfg.get("reverse_prefer_enabled", True)) and "back_fast" in self.config["motion"]["actions"]:
            reverse_rejected_reason = (
                "target_not_behind"
                if reverse_allowed_by_goal_distance
                else "goal_too_far_for_reverse"
            )
            rear_distance = -float(forward)
            if reverse_allowed_by_goal_distance and forward < 0.0:
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
                                metrics = corridor_metrics_to(next_xy)
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
            final_goal_distance_cm=(
                None
                if final_goal_distance_cm is None
                else round(float(final_goal_distance_cm), 2)
            ),
            reverse_max_goal_distance_cm=round(reverse_max_goal_distance_cm, 2),
            reverse_goal_distance_limit_enforced=(
                final_goal_distance_cm is not None and not escape_mode
            ),
            reverse_allowed_by_goal_distance=reverse_allowed_by_goal_distance,
            reverse_preferred=any(item["kind"] == "reverse" for item in options),
            reverse_rejected_reason=reverse_rejected_reason or None,
            movement_corridor_clear=any(item["kind"] == "reverse" for item in options),
        )

        min_forward = float(nav_cfg.get("translation_min_forward_cm", 6.0))
        if forward >= min_forward:
            requested = min(float(forward), current_dist)
            planned = self.planned_forward_step_cm(requested)
            planned_xy = self.translated_pose_xy(pose, forward_cm=planned)
            forward_metrics = corridor_metrics_to(planned_xy)
            forward_clear = bypass_action_safety or (
                bool(forward_metrics.get("clear")) if escape_mode else self.forward_clear_for_distance(
                    pose, planned, exact_goal_xy=waypoint if allow_goal_high_cost else None
                )
            )
            if forward_clear:
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
                            "corridor_metrics": forward_metrics,
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
            lateral_metrics = corridor_metrics_to(lateral_target)
            lateral_clear = bypass_action_safety or (
                bool(lateral_metrics.get("clear")) if escape_mode else self.path_segments_clear(
                    [pose.xy(), lateral_target], allow_goal_high_cost=lateral_reaches_goal
                )
            )
            if abs(planned) > 0.0 and lateral_clear:
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
                            "corridor_metrics": lateral_metrics,
                        }
                    )

        if not options:
            return None
        reverse_option = next((item for item in options if item["kind"] == "reverse"), None)
        forward_option = next((item for item in options if item["kind"] == "forward"), None)
        selected = reverse_option or forward_option or max(options, key=lambda item: item["progress_cm"])
        corridor = selected.get("corridor_metrics") or corridor_metrics_to(
            self.translated_pose_xy(
                pose,
                forward_cm=float(selected["planned_cm"]) if selected["kind"] != "strafe" else 0.0,
                lateral_cm=float(selected["planned_cm"]) if selected["kind"] == "strafe" else 0.0,
            )
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
            start_escape_mode=escape_mode,
        )
        return selected

    def execute_translation_action(
        self,
        action: dict,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        goal_dist_cm: float,
        context: dict,
        bypass_action_safety: bool = False,
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
            near_wall = False if bypass_action_safety else self.near_wall_now(pose)
            cycles, batch_reason = self.select_adaptive_action_batch(
                "reverse",
                requested,
                step_cm,
                abs(float(action["distance_cm"])),
                goal_dist_cm,
                near_wall=near_wall,
            )
            # The planner has already collision-checked the selected reverse
            # prefix and attached the authoritative corridor metrics. In
            # particular, a start-projection escape is deliberately evaluated
            # with escape_corridor_metrics(), whose soft-inflation policy
            # permits a physically clear, non-worsening escape. Rechecking it
            # here with the normal-navigation policy could veto forever the
            # exact action the planner had just accepted.
            corridor = action.get("corridor_metrics") or {}
            self.debug.event(
                "action_batch_started",
                action=key,
                selected_action="reverse",
                requested_action_cycles=requested,
                selected_action_cycles=cycles,
                adaptive_batch_reason=batch_reason,
                movement_corridor_clear=bool(corridor.get("clear", True)),
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
            self.clear_decision_stall()
            self.clear_turn_progress_watchdog("successful_reverse")
            tight_limit = float(self.config["navigation"].get(
                "relocalize_obstacle_tight_clearance_cm", 20.0
            ))
            self.post_action_relocalize(
                "translation_reverse",
                pose_before_action,
                result,
                waypoint,
                navigation_mode=self.navigation_relocalization_mode(),
                obstacle_tight=bool(
                    near_wall
                    or float(corridor.get("minimum_wall_clearance_cm", float("inf")))
                    <= tight_limit
                ),
            )
            return "moved"

        if action["kind"] == "strafe":
            self.forward_map_block_count = 0
            key = "strafe_left_fast" if float(action["distance_cm"]) > 0.0 else "strafe_right_fast"
            step_cm = abs(float(self.config["motion"]["actions"][key].get("lateral_cm", 4.0)))
            requested = self.motion.lateral_cycles_for_distance(float(action["distance_cm"]))
            near_wall = False if bypass_action_safety else self.near_wall_now(pose)
            cycles, batch_reason = self.select_adaptive_action_batch(
                "strafe", requested, step_cm, abs(float(action["distance_cm"])), goal_dist_cm,
                near_wall=near_wall,
            )
            travel = math.copysign(cycles * step_cm, float(action["distance_cm"]))
            end_xy = self.translated_pose_xy(pose, lateral_cm=travel)
            corridor = action.get("corridor_metrics") or (
                {
                    "clear": True,
                    "path_obstacle_cost": 0.0,
                    "minimum_wall_clearance_cm": 9999.0,
                    "safety_bypassed": True,
                }
                if bypass_action_safety
                else self.movement_corridor_metrics(pose.xy(), end_xy)
            )
            if not bypass_action_safety and not corridor["clear"]:
                self.debug.event(
                    "translation_corridor_blocked",
                    selected_action="strafe",
                    movement_corridor_clear=False,
                    **context
                )
                if self.register_decision_stall(
                    pose,
                    waypoint,
                    "strafe",
                    "translation_corridor_blocked_before_execute",
                ):
                    self.recover_from_near_wall(
                        str(context.get("reason", "translation"))
                        + ":decision_stall"
                    )
                else:
                    self.localize_scan()
                return "recovered"
            detail.update(requested_action_cycles=requested, adaptive_batch_reason=batch_reason)
            self.debug.event("action_batch_started", action=key, requested_action_cycles=requested, selected_action_cycles=cycles, **context)
            result = self.motion.run(key, times_override=cycles)
            self.debug.event("action_batch_completed", action=key, actual_action_cycles=getattr(result, "executed_times", result.times if result.ok else 0), ok=result.ok, **context)
            if not result.ok:
                self.last_navigation_failure_reason = "hardware_failure"
                return "failed"
            self.clear_decision_stall()
            self.clear_turn_progress_watchdog("successful_translation")
            tight_limit = float(self.config["navigation"].get(
                "relocalize_obstacle_tight_clearance_cm", 20.0
            ))
            self.post_action_relocalize(
                "translation_strafe",
                pose_before_action,
                result,
                waypoint,
                navigation_mode=self.navigation_relocalization_mode(),
                obstacle_tight=bool(
                    near_wall
                    or float(corridor.get("minimum_wall_clearance_cm", float("inf")))
                    <= tight_limit
                ),
            )
            return "moved"

        if not bypass_action_safety and self.front_obstacle_visible():
            self.register_decision_stall(
                pose,
                waypoint,
                "forward",
                "front_obstacle_visible_before_execute",
            )
            self.debug.event("front_obstacle_recover", **context)
            reason = str(context.get("reason", "front_obstacle_visible"))
            self.recover_toward_field_center(reason + ":front_obstacle_visible", backoff=True)
            return "recovered"

        forward_dist = min(float(action["distance_cm"]), goal_dist_cm)
        planned_forward_cm = self.planned_forward_step_cm(forward_dist)
        map_check_min = float(self.config["navigation"].get("forward_map_check_min_cm", 16.0))
        if (
            not bypass_action_safety
            and planned_forward_cm >= map_check_min
            and not self.forward_clear_for_distance(pose, planned_forward_cm)
        ):
            self.forward_map_block_count += 1
            self.register_decision_stall(
                pose,
                waypoint,
                "forward",
                "forward_map_blocked_before_execute",
            )
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
        near_wall = False if bypass_action_safety else self.near_wall_now(pose)
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
        self.clear_decision_stall()
        self.clear_turn_progress_watchdog("successful_translation")
        self.set_pending_forward_progress(pose_before_forward, abs(float(result.model_forward_cm)))
        self.evaluate_visual_forward_progress(visual_before, abs(float(result.model_forward_cm)))
        if self.collision_recovery_pending and not bypass_action_safety:
            reason = str(context.get("reason", "visual_forward_no_progress"))
            self.recover_toward_field_center(reason + ":visual_forward_no_progress", backoff=True)
            return "recovered"
        if bypass_action_safety:
            self.collision_recovery_pending = False
        action_corridor = action.get("corridor_metrics") or (
            {
                "clear": True,
                "path_obstacle_cost": 0.0,
                "minimum_wall_clearance_cm": 9999.0,
                "safety_bypassed": True,
            }
            if bypass_action_safety
            else self.movement_corridor_metrics(
                pose_before_forward.xy(),
                self.state.pose.xy() if self.state.pose is not None else waypoint,
            )
        )
        tight_limit = float(self.config["navigation"].get(
            "relocalize_obstacle_tight_clearance_cm", 20.0
        ))
        self.post_action_relocalize(
            "translation_forward",
            pose_before_forward,
            result,
            waypoint,
            navigation_mode=self.navigation_relocalization_mode(),
            obstacle_tight=bool(
                near_wall
                or float(action_corridor.get("minimum_wall_clearance_cm", float("inf")))
                <= tight_limit
            ),
        )
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
        self.last_motion_action = str(getattr(action_result, "key", "turn"))
        if hasattr(self, "config") and hasattr(self.state, "actions_since_localize"):
            self.adaptive_relocalization_decision(
                self.navigation_relocalization_mode(),
                last_action=self.last_motion_action,
                action_result=action_result,
                force_reason=(
                    "large_turn"
                    if "large" in self.last_motion_action.lower()
                    else "turn_progress_validation"
                ),
            )
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
        """Yield short, obstacle-safe points in the field's inward-shrunk region."""
        nav = self.config["navigation"]
        margin = float(nav.get("interior_recovery_margin_cm", 55.0))
        clearance_required = float(nav.get("interior_recovery_min_clearance_cm", 20.0))
        max_dist = float(nav.get("interior_recovery_max_distance_cm", 85.0))
        sample_step = max(5.0, float(nav.get("interior_recovery_sample_step_cm", 10.0)))
        center_xy = self.field_center_xy()
        xmin, xmax = margin, self.map.width_cm - margin
        ymin, ymax = margin, self.map.height_cm - margin
        if xmin > xmax or ymin > ymax:
            return
        preferred = (
            min(max(pose.x_cm, xmin), xmax),
            min(max(pose.y_cm, ymin), ymax),
        )
        candidates = {preferred, center_xy}
        rings = int(math.ceil(max_dist / sample_step))
        for ix in range(-rings, rings + 1):
            for iy in range(-rings, rings + 1):
                xy = (preferred[0] + ix * sample_step, preferred[1] + iy * sample_step)
                if xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax:
                    candidates.add(xy)
        for xy in candidates:
            travel = distance_xy(pose.xy(), xy)
            if travel < 1.0 or travel > max_dist:
                continue
            if not self.map.is_free_xy(xy):
                continue
            clearance = float(self.map.robot_clearance_cm(xy))
            if clearance < clearance_required:
                continue
            forward, lateral = self.local_vector_to(pose, xy)
            lateral_mid = self.translated_pose_xy(pose, lateral_cm=lateral)
            forward_mid = self.translated_pose_xy(pose, forward_cm=forward)
            # Recovery is specifically allowed to escape a low-clearance start;
            # its destination still has the dedicated recovery clearance gate.
            lateral_first = self.path_segments_clear(
                [pose.xy(), lateral_mid, xy], minimum_clearance_cm=0.0
            )
            forward_first = self.path_segments_clear(
                [pose.xy(), forward_mid, xy], minimum_clearance_cm=0.0
            )
            if not lateral_first and not forward_first:
                continue
            order = "lateral_then_longitudinal" if lateral_first else "longitudinal_then_lateral"
            center_distance = distance_xy(xy, center_xy)
            score = travel + 0.04 * center_distance - 0.08 * min(clearance, 80.0)
            yield {
                "kind": "interior_safe",
                "xy": (float(xy[0]), float(xy[1])),
                "path": [pose.xy(), lateral_mid, xy] if lateral_first else [pose.xy(), forward_mid, xy],
                "score": score,
                "distance_cm": travel,
                "clearance_cm": clearance,
                "boundary_margin_cm": margin,
                "recovery_yaw": float(pose.yaw_deg),
                "component_order": order,
            }

    def choose_boundary_recovery_target(self, pose: RobotPose):
        candidates = list(self.indoor_recovery_candidates(pose))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["score"])
        return candidates[0]

    def blind_navigate_to_xy(
        self,
        target_xy: Tuple[float, float],
        reason: str,
        component_order: str = "lateral_then_longitudinal",
    ) -> bool:
        """Move to recovery target without changing yaw, using body-axis translations."""
        max_steps = int(self.config["navigation"].get("boundary_recovery_max_steps", 18))
        radius = float(self.config["navigation"].get("interior_recovery_arrival_radius_cm", 6.0))
        recovery_yaw = None if self.state.pose is None else float(self.state.pose.yaw_deg)
        self.debug.event("boundary_blind_nav_start", reason=reason, target_xy=target_xy, max_steps=max_steps)
        for step in range(max_steps):
            pose = self.state.pose
            if pose is None:
                return False
            dist = distance_xy(pose.xy(), target_xy)
            if dist <= radius:
                self.debug.event("boundary_blind_nav_arrived", step=step, distance_cm=round(dist, 1))
                return True
            forward, lateral = self.local_vector_to(pose, target_xy)
            component_tolerance = radius / math.sqrt(2.0)
            yaw_error = 0.0 if recovery_yaw is None else angle_diff_deg(recovery_yaw, pose.yaw_deg)
            self.debug.event(
                "boundary_blind_nav_step",
                step=step + 1,
                distance_cm=round(dist, 1),
                target_xy=target_xy,
                recovery_yaw=recovery_yaw,
                current_yaw=pose.yaw_deg,
                yaw_error=round(yaw_error, 1),
                forward_component_cm=round(forward, 1),
                lateral_component_cm=round(lateral, 1),
            )
            action = None
            lateral_first = component_order != "longitudinal_then_lateral"
            use_lateral = abs(lateral) > component_tolerance and (
                lateral_first or abs(forward) <= component_tolerance
            )
            use_longitudinal = abs(forward) > component_tolerance
            if use_lateral:
                action = "strafe_left_fast" if lateral > 0.0 else "strafe_right_fast"
                requested = float(self.config["motion"]["actions"][action].get("lateral_cm", 0.0))
                safe = self.recovery_translation_clear(pose, lateral_cm=requested)
            elif use_longitudinal:
                action = "forward_fast" if forward > 0.0 else "back_fast"
                requested = float(self.config["motion"]["actions"][action].get("forward_cm", 0.0))
                safe = self.recovery_translation_clear(pose, forward_cm=requested)
            else:
                return True
            if not safe:
                self.debug.event("recovery_action", action=action, executed=False, reason="corridor_blocked")
                return False
            result = self.motion.run(action, times_override=1)
            self.debug.event(
                "recovery_action",
                action=action,
                executed=bool(result.ok),
                recovery_yaw=recovery_yaw,
                turn_used=False,
                error=result.error,
            )
            if not result.ok:
                return False
            self.last_motion_action = action
            recovery_relocalization = self.adaptive_relocalization_decision(
                "recovery",
                last_action=action,
                action_result=result,
                recovery=True,
            )
            if recovery_relocalization["decision"] == "relocalize_now":
                self.hardware.center_head()
                if not self.localize_scan(
                    reason="interior_recovery_adaptive",
                    allow_failure_escalation=False,
                ):
                    return False
            self.publish_state(path=[pose.xy(), target_xy])
        self.debug.event("boundary_blind_nav_failed", target_xy=target_xy, max_steps=max_steps)
        return False

    def recover_via_indoor_waypoint(self, reason: str) -> bool:
        if not bool(self.config["navigation"].get("boundary_recovery_enabled", True)):
            return False
        pose = self.state.pose
        if pose is None:
            return False
        original_goal = getattr(self, "current_target_goal", None)
        if not isinstance(getattr(self, "last_recovery", None), dict):
            self.last_recovery = {}
        self.last_recovery.update({
            "t": round(now_s(), 3),
            "reason": reason,
            "strategy": "interior_waypoint_preserve_yaw",
            "original_target": None if original_goal is None else original_goal.as_dict(),
        })
        target = self.choose_boundary_recovery_target(pose)
        if target is None:
            self.debug.event("boundary_recovery_no_indoor_waypoint", reason=reason, pose=pose.as_dict())
            return False
        self.last_recovery["boundary_target"] = {
            "kind": target.get("kind"),
            "xy": list(target["xy"]),
            "distance_cm": round(float(target["distance_cm"]), 1),
            "score": round(float(target["score"]), 1),
            "clearance_cm": round(float(target["clearance_cm"]), 1),
            "boundary_margin_cm": float(target["boundary_margin_cm"]),
            "recovery_yaw": float(target["recovery_yaw"]),
            "component_order": target["component_order"],
        }
        self.active_recovery_waypoint = dict(self.last_recovery["boundary_target"])
        self.debug.event(
            "recovery_waypoint_selected",
            original_target_screen_id=None if original_goal is None else original_goal.screen_id,
            original_target_generation=None if original_goal is None else original_goal.generation_id,
            current_pose=pose.as_dict(),
            recovery_xy=target["xy"],
            recovery_yaw=target["recovery_yaw"],
            boundary_margin=target["boundary_margin_cm"],
            reason=reason,
            component_order=target["component_order"],
        )
        ok = self.blind_navigate_to_xy(
            tuple(target["xy"]),
            reason=reason,
            component_order=target["component_order"],
        )
        if not ok:
            return False
        localized = self.localize_scan(reason="interior_recovery", allow_pan_search=True)
        if localized:
            self.debug.event("recovery_localization_success", recovery_xy=target["xy"])
            self.active_recovery_waypoint = None
            if original_goal is not None:
                self.current_target_goal = original_goal
                self.current_target_screen_id = original_goal.screen_id
                self.debug.event("resume_original_target", **original_goal.as_dict())
        return localized

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
        executed_cycles = int(
            getattr(result, "executed_times", result.times if result.ok else 0) or 0
        )
        self.near_wall_recovery_actions = getattr(
            self, "near_wall_recovery_actions", 0
        ) + executed_cycles
        if not result.ok:
            self.last_navigation_failure_reason = "hardware_failure"
            self.debug.event(
                "recovery_action_rejected",
                phase=phase,
                action=key,
                reason="hardware_action_failed",
                executed=False,
            )
            return NearWallRecoveryResult.HARDWARE_FAILURE
        self.debug.event(
            "recovery_action_executed",
            phase=phase,
            action=key,
            executed=True,
            actual_action_cycles=executed_cycles,
            target_id=getattr(self, "current_target_screen_id", None),
        )
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
        # A failed observation is not proof that a physical action made no
        # progress. Only a successfully re-localized, unchanged pose counts.
        verified_no_progress = bool(
            localized
            and after is not None
            and (
                position_delta < 1.0
                and abs(yaw_delta) < 1.0
                and abs(clearance_delta) < 1.0
            )
        )
        if verified_no_progress:
            self.near_wall_recovery_no_progress_count = getattr(
                self, "near_wall_recovery_no_progress_count", 0
            ) + 1
        elif localized and after is not None:
            self.near_wall_recovery_no_progress_count = 0
            self.near_wall_recovery_rejection_count = 0
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
            no_progress=verified_no_progress,
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
        if verified_no_progress and self.near_wall_recovery_no_progress_count >= threshold:
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
        """Track planner vetoes separately from verified physical no-progress."""
        self.near_wall_recovery_rejection_count = int(getattr(
            self, "near_wall_recovery_rejection_count", 0
        )) + 1
        threshold = max(1, int(
            self.config["navigation"].get("near_wall_recovery_rejection_threshold", 2)
        ))
        self.debug.event(
            "near_wall_recovery_stall",
            reason=reason,
            action=action,
            count=self.near_wall_recovery_rejection_count,
            threshold=threshold,
            physical_no_progress_count=self.near_wall_recovery_no_progress_count,
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
        )
        if self.near_wall_recovery_rejection_count >= threshold:
            self.debug.event(
                "recovery_decision",
                reason=reason,
                action=action,
                decision="forced_escape",
                rejected_count=self.near_wall_recovery_rejection_count,
                physical_no_progress_count=self.near_wall_recovery_no_progress_count,
            )
            return True
        return False

    def forced_escape_translation_candidate(
        self,
        pose: RobotPose,
        *,
        name: str,
        action: str,
        forward_cm: float = 0.0,
        lateral_cm: float = 0.0,
    ) -> dict:
        """Evaluate a bounded escape while tolerating only pre-existing overlap."""
        result = {
            "name": name,
            "action": action,
            "endpoint": None,
            "cost": None,
            "clearance": 0.0,
            "valid": False,
            "reason": "map_metrics_unavailable",
            "new_hard_collision": False,
            "allow_initial_overlap": True,
            "require_clearance_improvement": True,
        }
        map_obj = getattr(self, "map", None)
        required = ("is_free_xy", "in_bounds_xy", "grid_pos", "cost")
        if map_obj is None or any(not hasattr(map_obj, item) for item in required):
            return result
        endpoint = self.translated_pose_xy(
            pose,
            forward_cm=float(forward_cm),
            lateral_cm=float(lateral_cm),
        )
        result["endpoint"] = (round(endpoint[0], 3), round(endpoint[1], 3))
        if not map_obj.in_bounds_xy(endpoint):
            result["reason"] = "endpoint_out_of_bounds"
            return result

        nav = self.config["navigation"]
        half_width = max(
            0.0,
            float(nav.get("translation_corridor_half_width_cm", 8.0)),
        )
        sample_step = max(0.5, min(1.0, float(getattr(map_obj, "res", 1.0))))
        dx = float(endpoint[0]) - float(pose.x_cm)
        dy = float(endpoint[1]) - float(pose.y_cm)
        length = math.hypot(dx, dy)
        if length < 0.1:
            result["reason"] = "zero_modeled_motion"
            return result
        tangent = (dx / length, dy / length)
        normal = (-tangent[1], tangent[0])
        offsets = [0.0]
        offset_steps = max(1, int(math.ceil(half_width / sample_step)))
        for index in range(1, offset_steps + 1):
            offset = min(half_width, index * sample_step)
            offsets.extend((-offset, offset))

        def footprint_blocked_cells(center_xy):
            blocked = set()
            for offset in offsets:
                sample = (
                    float(center_xy[0]) + normal[0] * offset,
                    float(center_xy[1]) + normal[1] * offset,
                )
                if not map_obj.in_bounds_xy(sample) or not map_obj.is_free_xy(sample):
                    blocked.add(map_obj.grid_pos(sample))
            return blocked

        initial_overlap = footprint_blocked_cells(pose.xy())
        endpoint_overlap = footprint_blocked_cells(endpoint)
        new_collision_cells = set()
        longitudinal_steps = max(1, int(math.ceil(length / sample_step)))
        for index in range(1, longitudinal_steps + 1):
            along = min(length, index * sample_step)
            center = (
                float(pose.x_cm) + tangent[0] * along,
                float(pose.y_cm) + tangent[1] * along,
            )
            for offset in offsets:
                sample = (
                    center[0] + normal[0] * offset,
                    center[1] + normal[1] * offset,
                )
                if not map_obj.in_bounds_xy(sample):
                    new_collision_cells.add(("out_of_bounds",))
                    continue
                if not map_obj.is_free_xy(sample):
                    node = map_obj.grid_pos(sample)
                    if node not in initial_overlap:
                        new_collision_cells.add(node)

        current = self.navigation_point_diagnostics(pose.xy())
        candidate = self.navigation_point_diagnostics(endpoint)
        current_cost = float(current["cost"] if current["cost"] is not None else float("inf"))
        endpoint_cost = float(candidate["cost"] if candidate["cost"] is not None else float("inf"))
        current_clearance = float(current["clearance_cm"])
        endpoint_clearance = float(candidate["clearance_cm"])
        cost_improvement = current_cost - endpoint_cost
        clearance_improvement = endpoint_clearance - current_clearance
        overlap_improvement = len(initial_overlap) - len(endpoint_overlap)
        minimum_cost = float(nav.get("planner_start_escape_min_cost_improvement", 2.0))
        minimum_clearance = float(nav.get("near_wall_min_clearance_improvement_cm", 1.5))
        endpoint_center_free = bool(map_obj.is_free_xy(endpoint))
        overlap_not_worse = (
            not endpoint_overlap
            if not initial_overlap
            else len(endpoint_overlap) < len(initial_overlap)
        )
        safety_improved = bool(
            cost_improvement >= minimum_cost
            or clearance_improvement >= minimum_clearance
            or overlap_improvement > 0
            or (not bool(current["free_neighbor_count"]) and endpoint_center_free)
            or (not bool(map_obj.is_free_xy(pose.xy())) and endpoint_center_free)
        )
        valid = bool(
            endpoint_center_free
            and not new_collision_cells
            and overlap_not_worse
            and safety_improved
        )
        reason = "safer_endpoint" if valid else (
            "endpoint_hard_occupied" if not endpoint_center_free else
            "new_hard_collision" if new_collision_cells else
            "footprint_overlap_not_improved" if not overlap_not_worse else
            "insufficient_safety_improvement"
        )
        priority_bonus = 2.0 if name in ("left", "right") else (1.0 if name == "backward" else 0.5)
        score = (
            25.0 * float(overlap_improvement)
            + 10.0 * float(clearance_improvement)
            + max(-100.0, min(100.0, float(cost_improvement)))
            + priority_bonus
        )
        result.update({
            "cost": round(endpoint_cost, 3),
            "clearance": round(endpoint_clearance, 3),
            "valid": valid,
            "reason": reason,
            "new_hard_collision": bool(new_collision_cells),
            "current_cost": round(current_cost, 3),
            "current_clearance": round(current_clearance, 3),
            "cost_improvement": round(cost_improvement, 3),
            "clearance_improvement": round(clearance_improvement, 3),
            "initial_overlap_cells": len(initial_overlap),
            "endpoint_overlap_cells": len(endpoint_overlap),
            "overlap_improvement": overlap_improvement,
            "score": round(score, 3),
        })
        return result

    def forced_escape_translation_candidates(self, pose: RobotPose) -> List[dict]:
        """Evaluate both lateral directions before longitudinal escape moves."""
        actions = self.config["motion"]["actions"]
        return [
            self.forced_escape_translation_candidate(
                pose,
                name="left",
                action="strafe_left_fast",
                lateral_cm=float(actions["strafe_left_fast"].get("lateral_cm", 0.0)),
            ),
            self.forced_escape_translation_candidate(
                pose,
                name="right",
                action="strafe_right_fast",
                lateral_cm=float(actions["strafe_right_fast"].get("lateral_cm", 0.0)),
            ),
            self.forced_escape_translation_candidate(
                pose,
                name="backward",
                action="back_fast",
                forward_cm=float(actions["back_fast"].get("forward_cm", 0.0)),
            ),
            self.forced_escape_translation_candidate(
                pose,
                name="forward",
                action="forward_micro",
                forward_cm=float(actions["forward_micro"].get("forward_cm", 0.0)),
            ),
        ]

    def execute_bounded_escape(self, reason: str) -> NearWallRecoveryResult:
        """Use tiny configured turns when conservative normal recovery vetoes all moves."""
        nav = self.config["navigation"]
        if not bool(nav.get("forced_escape_enabled", True)):
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        target_id = getattr(self, "current_target_screen_id", None)
        self.debug.event(
            "forced_escape_started",
            reason=reason,
            target_id=target_id,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
            rejected_count=int(getattr(self, "near_wall_recovery_rejection_count", 0)),
        )
        self.debug.event("relocalization_started", reason="forced_escape_head_scan")
        localized = bool(self.localize_scan(
            reason="forced_escape_head_scan",
            allow_pan_search=True,
            allow_failure_escalation=False,
        ))
        self.debug.event(
            "relocalization_success" if localized else "relocalization_failed",
            reason="forced_escape_head_scan",
            target_id=target_id,
        )
        pose = self.state.pose
        if localized and pose is not None and not self.near_wall_now(pose):
            self.near_wall_recovery_rejection_count = 0
            self.debug.event(
                "forced_escape_finished",
                success=True,
                reason="head_scan_pose_cleared_near_wall",
                target_id=target_id,
            )
            return NearWallRecoveryResult.RECOVERED
        if pose is None:
            self.debug.event(
                "forced_escape_finished",
                success=False,
                reason="pose_unavailable_after_full_pan",
                target_id=target_id,
            )
            return NearWallRecoveryResult.LOCALIZATION_REQUIRED
        current_detail = (
            self.navigation_point_diagnostics(pose.xy())
            if hasattr(self.map, "grid_pos") and hasattr(self.map, "cost")
            else {}
        )
        candidates = self.forced_escape_translation_candidates(pose)
        valid_candidates = [item for item in candidates if item.get("valid")]
        selected = max(valid_candidates, key=lambda item: item["score"]) if valid_candidates else None
        evaluation = {item["name"]: item for item in candidates}
        self.debug.event(
            "forced_escape_candidate_evaluation",
            current_pose=pose.as_dict(),
            current_cost=current_detail.get("cost"),
            current_clearance=current_detail.get("clearance_cm"),
            current_center_free=(
                None if not hasattr(self.map, "is_free_xy")
                else bool(self.map.is_free_xy(pose.xy()))
            ),
            left=evaluation.get("left"),
            right=evaluation.get("right"),
            forward=evaluation.get("forward"),
            backward=evaluation.get("backward"),
            selected_action=None if selected is None else selected["action"],
            reason=(
                "best_safety_improvement"
                if selected is not None
                else "no_safe_translation_candidate"
            ),
        )
        if selected is not None:
            self.debug.event(
                "forced_escape_action_selected",
                action=selected["action"],
                candidate=selected["name"],
                endpoint=selected["endpoint"],
                cost=selected["cost"],
                clearance=selected["clearance"],
                reason=selected["reason"],
            )
            outcome = self.execute_near_wall_recovery_action(
                selected["action"], "forced_escape", 1, 1
            )
            self.debug.event(
                "forced_escape_action_executed",
                action=selected["action"],
                candidate=selected["name"],
                executed=outcome != NearWallRecoveryResult.HARDWARE_FAILURE,
                outcome=outcome.value,
            )
            if outcome in (
                NearWallRecoveryResult.RECOVERED,
                NearWallRecoveryResult.RETRY_WITH_NEW_POSE,
                NearWallRecoveryResult.LOCALIZATION_REQUIRED,
                NearWallRecoveryResult.HARDWARE_FAILURE,
            ):
                if outcome in (
                    NearWallRecoveryResult.RECOVERED,
                    NearWallRecoveryResult.RETRY_WITH_NEW_POSE,
                ):
                    self.near_wall_recovery_rejection_count = 0
                self.debug.event(
                    "forced_escape_finished",
                    success=outcome in (
                        NearWallRecoveryResult.RECOVERED,
                        NearWallRecoveryResult.RETRY_WITH_NEW_POSE,
                    ),
                    reason="translation_executed_replan",
                    action=selected["action"],
                    target_id=target_id,
                )
                return outcome
            self.near_wall_recovery_rejection_count = 0
            self.debug.event(
                "forced_escape_finished",
                success=True,
                reason="bounded_translation_changed_pose_replan",
                action=selected["action"],
                target_id=target_id,
            )
            return NearWallRecoveryResult.RETRY_WITH_NEW_POSE

        left_clearance = self.wall_clearance_cm(pose, pose.yaw_deg + 90.0)
        right_clearance = self.wall_clearance_cm(pose, pose.yaw_deg - 90.0)
        actions = (
            ["turn_left_fast", "turn_right_fast"]
            if left_clearance >= right_clearance
            else ["turn_right_fast", "turn_left_fast"]
        )
        max_actions = 1
        any_executed = False
        for index, action in enumerate(actions[:max_actions], 1):
            self.debug.event(
                "forced_escape_action",
                index=index,
                action=action,
                target_id=target_id,
                override="soft_rotation_sweep_only",
                actual_action_cycles=1,
            )
            outcome = self.execute_near_wall_recovery_action(
                action, "forced_escape", index, 1
            )
            any_executed = outcome != NearWallRecoveryResult.HARDWARE_FAILURE
            if outcome == NearWallRecoveryResult.HARDWARE_FAILURE:
                self.debug.event(
                    "forced_escape_finished",
                    success=False,
                    reason="hardware_failure",
                    target_id=target_id,
                )
                return outcome
            if outcome == NearWallRecoveryResult.RECOVERED:
                self.near_wall_recovery_rejection_count = 0
                self.debug.event(
                    "forced_escape_finished",
                    success=True,
                    reason="escaped_near_wall",
                    target_id=target_id,
                )
                return outcome
            if outcome == NearWallRecoveryResult.RETRY_WITH_NEW_POSE:
                self.near_wall_recovery_rejection_count = 0
                self.debug.event(
                    "forced_escape_finished",
                    success=True,
                    reason="clearance_improved_replan",
                    target_id=target_id,
                )
                return outcome
            if outcome == NearWallRecoveryResult.LOCALIZATION_REQUIRED:
                self.debug.event("relocalization_started", reason="forced_escape_after_action")
                relocalized = bool(self.localize_scan(
                    reason="forced_escape_after_action",
                    allow_pan_search=True,
                    allow_failure_escalation=False,
                ))
                self.debug.event(
                    "relocalization_success" if relocalized else "relocalization_failed",
                    reason="forced_escape_after_action",
                    target_id=target_id,
                )
                if not relocalized:
                    continue
            elif outcome == NearWallRecoveryResult.STILL_NEAR_WALL:
                # The action was physically verified and changed the pose.
                # Replan from that new orientation instead of undoing it with
                # the opposite turn in the same recovery episode.
                self.near_wall_recovery_rejection_count = 0
                self.debug.event(
                    "forced_escape_finished",
                    success=True,
                    reason="bounded_turn_changed_pose_replan",
                    target_id=target_id,
                )
                return NearWallRecoveryResult.RETRY_WITH_NEW_POSE
            if self.state.pose is not None and not self.near_wall_now(self.state.pose):
                self.near_wall_recovery_rejection_count = 0
                self.debug.event(
                    "forced_escape_finished",
                    success=True,
                    reason="fresh_pose_cleared_near_wall",
                    target_id=target_id,
                )
                return NearWallRecoveryResult.RECOVERED
        self.debug.event(
            "forced_escape_finished",
            success=False,
            reason="bounded_actions_exhausted",
            action_executed=any_executed,
            target_id=target_id,
        )
        return (
            NearWallRecoveryResult.STILL_NEAR_WALL
            if any_executed and self.state.pose is not None
            else NearWallRecoveryResult.LOCALIZATION_REQUIRED
        )

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

    def register_decision_stall(
        self,
        pose: RobotPose,
        target_xy: Tuple[float, float],
        selected_action: str,
        failure_reason: str,
    ) -> bool:
        """Escalate a repeated executor veto with an unchanged planner input."""
        signature = (
            getattr(self, "current_target_screen_id", None),
            round(float(target_xy[0]), 1),
            round(float(target_xy[1]), 1),
            round(float(pose.x_cm), 1),
            round(float(pose.y_cm), 1),
            round(float(pose.yaw_deg), 1),
            selected_action,
            failure_reason,
        )
        if signature == getattr(self, "decision_stall_signature", None):
            self.decision_stall_count = int(getattr(
                self, "decision_stall_count", 0
            )) + 1
        else:
            self.decision_stall_signature = signature
            self.decision_stall_count = 1
        stalled = self.decision_stall_count >= 2
        if stalled:
            self.debug.event(
                "decision_stall_detected",
                selected_action=selected_action,
                executed=False,
                failure_reason=failure_reason,
                count=self.decision_stall_count,
                pose=pose.as_dict(),
                target_xy=target_xy,
                current_target_screen_id=getattr(
                    self, "current_target_screen_id", None
                ),
                next_strategy="near_wall_recovery",
            )
        return stalled

    def clear_decision_stall(self) -> None:
        """A physical action invalidates any prior no-motion decision stall."""
        self.decision_stall_signature = None
        self.decision_stall_count = 0

    def map_planning_signature(self):
        dynamic = []
        for item in getattr(self.map, "dynamic_obstacles", []):
            dynamic.append(tuple(
                round(float(item.get(key, 0.0)), 1)
                for key in ("x_min", "x_max", "y_min", "y_max")
            ))
        return tuple(sorted(dynamic))

    def register_plan_failure(
        self,
        pose: RobotPose,
        final_goal_xy: Tuple[float, float],
        reason: str,
    ) -> Tuple[int, tuple]:
        plan = getattr(self, "active_navigation_plan", None) or {}
        plan_goal = plan.get("goal_xy") or final_goal_xy
        staging = plan.get("staging_xy")
        target_goal = getattr(self, "current_target_goal", None)
        signature = (
            getattr(self, "current_target_screen_id", None),
            None if target_goal is None else target_goal.generation_id,
            self.map.grid_pos(pose.xy()),
            self.map.grid_pos(plan_goal),
            None if staging is None else self.map.grid_pos(staging),
            reason,
            self.map_planning_signature(),
        )
        if signature == getattr(self, "plan_failure_signature", None):
            self.identical_plan_failure_count = int(
                getattr(self, "identical_plan_failure_count", 0)
            ) + 1
        else:
            self.plan_failure_signature = signature
            self.identical_plan_failure_count = 1
        threshold = max(1, int(
            self.config["navigation"].get("identical_local_replan_failure_threshold", 3)
        ))
        self.local_replan_failures = min(self.identical_plan_failure_count, threshold)
        self.debug.event(
            "repeated_plan_failure_detected",
            count=self.identical_plan_failure_count,
            threshold=threshold,
            screen_id=getattr(self, "current_target_screen_id", None),
            target_generation=None if target_goal is None else target_goal.generation_id,
            start_xy=pose.xy(),
            start_grid=self.map.grid_pos(pose.xy()),
            goal_xy=plan_goal,
            goal_grid=self.map.grid_pos(plan_goal),
            staging_xy=staging,
            map_signature=self.map_planning_signature(),
            failure_reason=reason,
            failure_signature=repr(signature),
        )
        return self.identical_plan_failure_count, signature

    def deterministic_plan_failure_key(
        self,
        final_goal_xy: Tuple[float, float],
        reason: str,
    ) -> tuple:
        """Identify failures that recovery motion cannot change semantically."""
        target_goal = getattr(self, "current_target_goal", None)
        return (
            getattr(self, "current_target_screen_id", None),
            None if target_goal is None else target_goal.generation_id,
            self.map.grid_pos(final_goal_xy),
            str(reason),
            self.map_planning_signature(),
        )

    def clear_plan_failure_watchdog(self, reason: str) -> None:
        if getattr(self, "identical_plan_failure_count", 0):
            self.debug.event(
                "plan_failure_watchdog_reset",
                reason=reason,
                previous_count=self.identical_plan_failure_count,
            )
        self.plan_failure_signature = None
        self.identical_plan_failure_count = 0
        self.local_replan_failures = 0

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
                self.debug.event(
                    "recovery_action_rejected",
                    phase="backoff",
                    action="back_fast",
                    executed=False,
                    reason="rear_path_unsafe",
                    target_id=getattr(self, "current_target_screen_id", None),
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
                self.debug.event(
                    "recovery_action_rejected",
                    phase="lateral",
                    action=None,
                    executed=False,
                    reason="no_safe_lateral_path",
                    target_id=getattr(self, "current_target_screen_id", None),
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
                self.debug.event(
                    "recovery_action_rejected",
                    phase="small_turn_last_resort",
                    action=key,
                    executed=False,
                    reason="rotation_sweep_blocked",
                    target_id=getattr(self, "current_target_screen_id", None),
                )
                outcome = NearWallRecoveryResult.STILL_NEAR_WALL
            else:
                outcome = self.execute_near_wall_recovery_action(key, "small_turn_last_resort", 1, 1)
            if outcome in (NearWallRecoveryResult.RECOVERED, NearWallRecoveryResult.HARDWARE_FAILURE, NearWallRecoveryResult.LOCALIZATION_REQUIRED):
                return outcome

        if int(getattr(self, "near_wall_recovery_actions", 0)) == actions_before:
            self.register_near_wall_recovery_stall(reason, "all_recovery_actions_rejected")
            forced = self.execute_bounded_escape(reason + ":all_actions_vetoed")
            if forced != NearWallRecoveryResult.STILL_NEAR_WALL:
                return forced
            # A bounded physical action occurred; subsequent handling is based
            # on its fresh localization, not on the preceding planner vetoes.
            if int(getattr(self, "near_wall_recovery_actions", 0)) > actions_before:
                return forced
        physical_stalled = self.near_wall_recovery_no_progress_count >= max(
            1, int(nav.get("near_wall_recovery_no_progress_threshold", 2))
        )
        if physical_stalled:
            self.debug.event(
                "near_wall_recovery_no_progress",
                error="near_wall_recovery_exhausted",
                reason=reason,
                count=self.near_wall_recovery_no_progress_count,
                rejected_count=int(getattr(self, "near_wall_recovery_rejection_count", 0)),
                pose=None if self.state.pose is None else self.state.pose.as_dict(),
                near_wall_recovery_result=NearWallRecoveryResult.STILL_NEAR_WALL.value,
                target_preserved=target_screen is not None,
            )
            self.last_navigation_failure_reason = "near_wall_recovery_exhausted"
            self.debug.event(
                "near_wall_recovery_aborted",
                reason=reason,
                count=self.near_wall_recovery_no_progress_count,
            )
        else:
            self.debug.event(
                "recovery_decision",
                reason=reason,
                decision="continue_with_fresh_pose",
                physical_no_progress_count=self.near_wall_recovery_no_progress_count,
                rejected_count=int(getattr(self, "near_wall_recovery_rejection_count", 0)),
            )
        self.debug.event(
            "near_wall_recovery_continue_same_target",
            current_target_screen_id=getattr(self, "current_target_screen_id", None),
            target_preserved=target_screen is not None,
        )
        return NearWallRecoveryResult.STILL_NEAR_WALL

    def recover_toward_field_center(self, reason: str, backoff: bool = False) -> bool:
        if (
            not str(reason).startswith("no_tag:")
            and not bool(self.config["navigation"].get("collision_recovery_enabled", True))
        ):
            return False
        pose = self.state.pose
        self.pending_progress_check = None
        self.collision_recovery_pending = False
        self.visual_no_progress_count = 0
        self.recovery_count = int(getattr(self, "recovery_count", 0)) + 1
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
            localized = self.localize_scan(reason="recovery:" + reason)
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

    def localization_failure_recovery_needed(self) -> bool:
        """Compatibility alias: only genuine consecutive no-Tag scans recover."""
        return self.no_tag_recovery_needed()

    def no_tag_recovery_needed(self) -> bool:
        nav = self.config["navigation"]
        if not bool(nav.get("no_tag_recovery_enabled", True)):
            return False
        if bool(getattr(self, "no_tag_recovery_active", False)):
            return False
        limit = int(nav.get("no_tag_recovery_failures", 2))
        if int(getattr(self, "consecutive_no_tag_scans", 0)) < limit:
            return False
        cooldown = float(nav.get("no_tag_recovery_cooldown_s", 4.0))
        if now_s() - float(getattr(self, "last_no_tag_recovery_s", 0.0)) < cooldown:
            return False
        pose = self.state.pose
        boundary_trapped = bool(
            pose is not None
            and hasattr(self, "map")
            and self.is_boundary_trapped(pose, "no_tag")
        )
        return (
            pose is None
            or self.is_facing_outside(pose)
            or boundary_trapped
        )

    def recover_from_localization_failure_if_needed(self, reason: str) -> bool:
        return self.recover_from_no_tag_if_needed(reason)

    def recover_from_no_tag_if_needed(self, reason: str) -> bool:
        if not self.no_tag_recovery_needed():
            return False
        self.last_no_tag_recovery_s = now_s()
        self.no_tag_recovery_active = True
        self.no_tag_recovery_exhausted = False
        self.localization_recovery_exhausted = False
        pose = self.state.pose
        self.debug.event(
            "no_tag_recovery_triggered",
            reason=reason,
            no_tag_scans=self.consecutive_no_tag_scans,
            outward_facing=False if pose is None else self.is_facing_outside(pose),
        )
        try:
            return self.recover_toward_field_center(
                "no_tag:" + reason,
                backoff=True,
            )
        finally:
            self.no_tag_recovery_active = False

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
        target_goal: Optional[TargetGoal] = None,
        bypass_action_safety: bool = False,
    ) -> bool:
        self.turn_navigation_abort = False
        self.last_navigation_failure_reason = ""
        self.near_wall_recovery_no_progress_count = 0
        self.near_wall_recovery_rejection_count = 0
        self.near_wall_recovery_actions = 0
        self.navigation_stall_signature = None
        self.navigation_stall_count = 0
        episode = (
            None if target_goal is None else target_goal.screen_id,
            None if target_goal is None else target_goal.generation_id,
            round(float(target_xy[0]), 1),
            round(float(target_xy[1]), 1),
        )
        if episode != getattr(self, "navigation_plan_episode", None):
            self.navigation_plan_episode = episode
            self.clear_plan_failure_watchdog("new_navigation_target")
            self.last_recovered_deterministic_failure_key = None
        self.clear_turn_progress_watchdog("navigate_xy_start")
        target_xy = (float(target_xy[0]), float(target_xy[1]))
        if target_goal is not None and not self.validate_target_goal(
            target_goal, requested_xy=target_xy
        ):
            self.last_navigation_failure_reason = "target_pose_mismatch"
            return False
        if not bypass_action_safety and allow_goal_high_cost and (
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
                    return False
            pose = self.state.pose
            if pose is None:
                continue
            if target_goal is not None and not self.validate_target_goal(
                target_goal, requested_xy=target_xy
            ):
                self.last_navigation_failure_reason = "target_pose_mismatch"
                return False
            pre_action_relocalization = self.adaptive_relocalization_decision(
                self.navigation_relocalization_mode(),
                last_action=getattr(self, "last_motion_action", ""),
                emit=False,
            )
            if pre_action_relocalization["decision"] == "relocalize_now":
                self.adaptive_relocalization_decision(
                    self.navigation_relocalization_mode(),
                    last_action=getattr(self, "last_motion_action", ""),
                )
                localized = self.localize_scan(
                    reason="adaptive_navigation_budget",
                    allow_failure_escalation=False,
                )
                if not localized:
                    self.recover_from_no_tag_if_needed(reason + ":adaptive_relocalization")
                else:
                    continue
            dist = distance_xy(pose.xy(), target_xy)
            if dist <= radius:
                arrival_max_age = float(self.config["navigation"].get(
                    "arrival_visual_pose_max_age_s", 3.0
                ))
                if not self.visual_pose_is_fresh(arrival_max_age):
                    self.adaptive_relocalization_decision(
                        "target_direct_approach",
                        last_action=getattr(self, "last_motion_action", ""),
                        force_reason="before_arrived_at_target",
                    )
                    if not self.localize_scan(
                        reason="before_arrived_at_target",
                        allow_failure_escalation=False,
                    ):
                        self.last_navigation_failure_reason = "arrival_visual_localization_required"
                    continue
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
                        elif bypass_action_safety:
                            before_pose = self.copy_pose(pose)
                            target_diff = angle_diff_deg(float(target_yaw_deg), pose.yaw_deg)
                            result = self.motion.turn_toward(target_diff)
                            if result is not None and not self.monitor_turn_result(
                                before_pose,
                                float(target_yaw_deg),
                                result,
                                "task_target_arrival_yaw",
                            ):
                                return False
                        elif not self.turn_toward_yaw_boundary_aware(float(target_yaw_deg)):
                            return False
                        continue
                self.clear_navigation_noop()
                arrival = {
                    "reason": reason,
                    "screen_id": None if target_goal is None else target_goal.screen_id,
                    "tag_id": None if target_goal is None else target_goal.tag_id,
                    "robot_xy": pose.xy(),
                    "goal_xy": target_xy,
                    "anchor_xy": None if target_goal is None else target_goal.anchor_xy,
                    "distance_to_goal": round(dist, 3),
                    "distance_to_anchor": None if target_goal is None else round(distance_xy(pose.xy(), target_goal.anchor_xy), 3),
                    "desired_yaw": target_yaw_deg,
                    "current_yaw": pose.yaw_deg,
                    "generation_id": None if target_goal is None else target_goal.generation_id,
                    "step": step,
                }
                self.debug.event("navigate_xy_arrival_check", **arrival)
                self.debug.event("navigate_xy_arrived", target_xy=target_xy, distance_cm=round(dist, 1), **arrival)
                return True
            direct_path = self.target_direct_approach_path(pose, target_screen, target_xy)
            bypass_planned_path = None
            if bypass_action_safety and not direct_path:
                # Preserve the old planner whenever it can provide a route.
                # For a task target only, a safety-only planning rejection is
                # not allowed to leave the robot stationary: fall back to the
                # same direct target vector used by target-direct approach.
                bypass_planned_path = self.plan_navigation_path(
                    pose,
                    target_xy,
                    allow_goal_high_cost=allow_goal_high_cost,
                    target_screen=target_screen,
                )
                if not bypass_planned_path:
                    direct_path = [pose.xy(), target_xy]
                    self.debug.event(
                        "task_target_safety_bypass",
                        reason="no_safe_path",
                        target_xy=target_xy,
                    )
            direct_mode = bool(direct_path)
            current_navigation_mode = (
                "target_direct"
                if direct_mode
                else self.navigation_relocalization_mode()
            )
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
            if (
                direct_mode
                and getattr(self, "last_navigation_mode", "") != "target_direct"
            ):
                direct_max_age = float(self.config["navigation"].get(
                    "target_direct_entry_visual_pose_max_age_s", 4.0
                ))
                if not self.visual_pose_is_fresh(direct_max_age):
                    self.adaptive_relocalization_decision(
                        "target_direct_approach",
                        last_action=getattr(self, "last_motion_action", ""),
                        force_reason="before_target_final_approach",
                    )
                    if not self.localize_scan(
                        reason="before_target_final_approach",
                        allow_failure_escalation=False,
                    ):
                        self.last_navigation_failure_reason = "target_direct_visual_localization_required"
                    continue
            self.last_navigation_mode = current_navigation_mode
            if self.collision_recovery_pending and bypass_action_safety:
                self.collision_recovery_pending = False
                self.debug.event(
                    "task_target_safety_bypass",
                    reason="collision_recovery_pending",
                    target_xy=target_xy,
                )
            elif self.collision_recovery_pending:
                if direct_mode:
                    self.collision_recovery_pending = False
                    self.debug.event("target_direct_recovery_suppressed", reason="collision_recovery_pending")
                else:
                    self.recover_toward_field_center(reason + ":forward_no_progress", backoff=True)
                    continue
            if (
                not bypass_action_safety
                and not direct_mode
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
                path = bypass_planned_path or self.plan_navigation_path(
                    pose,
                    target_xy,
                    allow_goal_high_cost=allow_goal_high_cost,
                    target_screen=target_screen,
                )
            if not path:
                self.last_navigation_failure_reason = "no_reachable_approach_or_staging"
                count, signature = self.register_plan_failure(
                    pose, target_xy, self.last_navigation_failure_reason
                )
                plan = getattr(self, "active_navigation_plan", None) or {}
                plan_goal = tuple(plan.get("goal_xy") or target_xy)
                staging_xy = plan.get("staging_xy")
                start_detail = self.navigation_point_diagnostics(pose.xy())
                goal_detail = self.navigation_point_diagnostics(plan_goal)
                staging_detail = None if staging_xy is None else self.navigation_point_diagnostics(tuple(staging_xy))
                astar = dict(getattr(self.map, "last_astar_metrics", {}))
                anchor_xy = None if target_screen is None else target_screen.center_xy
                self.debug.event(
                    "path_plan_failed",
                    screen_id=None if target_screen is None else target_screen.screen_id,
                    start_xy=pose.xy(),
                    start_grid=start_detail["grid"],
                    start_blocked=start_detail["blocked"],
                    start_footprint_max_cost=start_detail["footprint_max_cost"],
                    target_anchor_xy=anchor_xy,
                    goal_xy=plan_goal,
                    goal_grid=goal_detail["grid"],
                    goal_type=plan.get("goal_type", "none"),
                    goal_in_bounds=goal_detail["in_bounds"],
                    goal_blocked=goal_detail["blocked"],
                    goal_clearance_cm=goal_detail["clearance_cm"],
                    staging_xy=staging_xy,
                    staging_grid=None if staging_detail is None else staging_detail["grid"],
                    staging_blocked=None if staging_detail is None else staging_detail["blocked"],
                    free_neighbor_count=start_detail["free_neighbor_count"],
                    astar_expanded_nodes=astar.get("expanded_nodes"),
                    astar_reason=astar.get("reason"),
                    candidate_rejections=plan.get("candidate_rejections", []),
                    local_replan_failures=self.local_replan_failures,
                    failure_signature=repr(signature),
                )
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
                threshold = max(1, int(
                    self.config["navigation"].get("identical_local_replan_failure_threshold", 3)
                ))
                if count >= threshold:
                    deterministic_key = self.deterministic_plan_failure_key(
                        target_xy, self.last_navigation_failure_reason
                    )
                    self.set_mission_state(MissionState.NAVIGATION_RECOVERY)
                    if deterministic_key == getattr(
                        self, "last_recovered_deterministic_failure_key", None
                    ):
                        self.last_navigation_failure_reason = "navigation_blocked"
                        self.set_mission_state(MissionState.NAVIGATION_BLOCKED)
                        self.debug.event(
                            "deterministic_recovery_repeat_blocked",
                            screen_id=getattr(self, "current_target_screen_id", None),
                            target_generation=(
                                None if target_goal is None
                                else target_goal.generation_id
                            ),
                            target_xy=target_xy,
                            map_signature=self.map_planning_signature(),
                            failure_reason="no_reachable_approach_or_staging",
                            recovery_repeated=False,
                            target_preserved=True,
                        )
                        return False
                    self.debug.event(
                        "local_replan_escalated",
                        failure_count=count,
                        reason=self.last_navigation_failure_reason,
                        next_strategy="interior_recovery_waypoint",
                        screen_id=getattr(self, "current_target_screen_id", None),
                    )
                    recovered = self.recover_via_indoor_waypoint(
                        reason + ":repeated_plan_failure"
                    )
                    if recovered:
                        self.last_recovered_deterministic_failure_key = deterministic_key
                        self.clear_plan_failure_watchdog("interior_recovery_success")
                        self.debug.event(
                            "interior_recovery_waypoint_selected",
                            screen_id=getattr(self, "current_target_screen_id", None),
                            target_preserved=True,
                            recovery=getattr(self, "last_recovery", {}),
                        )
                        continue
                    self.last_navigation_failure_reason = "navigation_blocked"
                    self.set_mission_state(MissionState.NAVIGATION_BLOCKED)
                    self.debug.event(
                        "local_replan_escalated",
                        failure_count=count,
                        reason="all_safe_planning_fallbacks_exhausted",
                        next_strategy="navigation_blocked",
                        screen_id=getattr(self, "current_target_screen_id", None),
                        target_preserved=True,
                    )
                    return False
                localized = self.localize_scan()
                if localized and self.state.pose is not None and self.near_wall_now(self.state.pose):
                    recovery_result = self.recover_from_near_wall(reason + ":local_replan_failed")
                    if recovery_result == NearWallRecoveryResult.HARDWARE_FAILURE:
                        return False
                continue
            self.clear_plan_failure_watchdog("path_plan_success")
            self.last_recovered_deterministic_failure_key = None
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
            self.debug.render_map(
                self.map,
                pose=pose,
                path=path,
                target_screen=target_screen,
                target_goal=getattr(self, "current_target_goal", None),
                recovery_waypoint=getattr(self, "active_recovery_waypoint", None),
                navigation_plan=getattr(self, "active_navigation_plan", None),
            )
            self.publish_state(path=path)
            waypoint_is_exact_goal = allow_goal_high_cost and distance_xy(waypoint, target_xy) <= 0.1
            if direct_mode and target_screen is not None:
                direct_action = self.choose_target_direct_action(
                    pose,
                    waypoint,
                    target_screen,
                    bypass_action_safety=bypass_action_safety,
                    final_goal_distance_cm=dist,
                )
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
                bypass_action_safety=bypass_action_safety,
                final_goal_distance_cm=dist,
            )
            if action is not None:
                status = self.execute_translation_action(
                    action,
                    pose,
                    waypoint,
                    dist,
                    {"reason": reason, "diff_yaw": round(diff, 1)},
                    bypass_action_safety=bypass_action_safety,
                )
                if status == "recovered":
                    self.clear_navigation_noop()
                    continue
                if status == "failed":
                    return False
                self.clear_navigation_noop()
            else:
                self.forward_map_block_count = 0
                rotation_clear = bypass_action_safety or not hasattr(self.map, "rotation_sweep_clear") or self.map.rotation_sweep_clear(
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
                if bypass_action_safety:
                    before_pose = self.copy_pose(pose)
                    result = self.motion.turn_toward(diff)
                    if result is not None and not self.monitor_turn_result(
                        before_pose, desired_yaw, result, "task_target_turn"
                    ):
                        return False
                elif not self.turn_toward_yaw_boundary_aware(desired_yaw):
                    return False
                if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
                    self.handle_navigation_noop(
                        reason=reason,
                        waypoint=waypoint,
                        diff=diff,
                    )
                else:
                    self.clear_navigation_noop()
            scheduled_relocalization = self.adaptive_relocalization_decision(
                self.navigation_relocalization_mode(),
                last_action=getattr(self, "last_motion_action", ""),
                emit=False,
            )
            if (
                not direct_mode
                and self.turn_no_progress_count == 0
                and scheduled_relocalization["decision"] == "relocalize_now"
            ):
                self.adaptive_relocalization_decision(
                    self.navigation_relocalization_mode(),
                    last_action=getattr(self, "last_motion_action", ""),
                )
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
        target_goal = getattr(self, "current_target_goal", None)
        if target_goal is not None:
            target_screen = self.map.screens.get(target_goal.screen_id)
        else:
            target_screen = target_screen or self.map.screens.get(self.current_target_screen_id)
        target_distance = None
        if target_screen is not None and self.state.pose is not None:
            target_distance = round(
                distance_xy(
                    self.state.pose.xy(),
                    target_goal.goal_xy if target_goal is not None else (
                        target_screen.task_target_xy or target_screen.interaction_xy
                    ),
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
            "temporarily_failed_targets": {
                str(target_id): dict(detail)
                for target_id, detail in sorted(
                    getattr(self, "temporarily_failed_targets", {}).items()
                )
            },
            "target_failure_counts": dict(getattr(self, "target_failure_counts", {})),
            "mission_state": self.mission_state.value,
            "current_target_tag_id": self.current_target_screen_id,
            "current_target_screen_id": self.current_target_screen_id,
            "current_target_goal": None if target_goal is None else target_goal.as_dict(),
            "current_navigation_plan": getattr(self, "active_navigation_plan", None),
            "current_target_distance_cm": target_distance,
            "arrived_at_target": self.arrived_at_target,
            "classifier_allowed": self.classifier_allowed,
            "visual_authorization": None if self.visual_authorization is None else self.visual_authorization.as_dict(),
            "target_visual_confirmation": None if self.target_visual_confirmation is None else self.target_visual_confirmation.as_dict(),
            "target_tag_confirmation": None if getattr(self, "target_tag_confirmation", None) is None else self.target_tag_confirmation.as_dict(),
            "classifier_health": {
                "available": bool(getattr(self, "classifier_available", True)),
                "last_error": getattr(self, "last_classifier_error", ""),
                "last_error_kind": getattr(self, "last_classifier_error_kind", ""),
            },
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
                "nfc": dict(getattr(self, "nfc_interaction_status", {})),
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
                "last_localization_attempt_result": self.last_localization_attempt_result,
                "localization_recovery_active": self.no_tag_recovery_active,
                "localization_recovery_exhausted": self.localization_recovery_exhausted,
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
                "identical_plan_failure_count": getattr(self, "identical_plan_failure_count", 0),
                "plan_failure_signature": getattr(self, "plan_failure_signature", None),
                "near_wall_recovery_actions": self.near_wall_recovery_actions,
                "near_wall_recovery_no_progress": self.near_wall_recovery_no_progress_count,
                "near_wall_recovery_rejections": int(getattr(
                    self, "near_wall_recovery_rejection_count", 0
                )),
                "fatal_target_failures": self.fatal_target_failures,
                "global_recovery_cycles": int(getattr(self, "global_recovery_cycles", 0)),
                "active_waypoint": getattr(self, "active_recovery_waypoint", None),
            },
            "interaction_log_path": self.interaction_audit_path,
            "screens": {sid: screen.as_dict() for sid, screen in sorted(self.map.screens.items())},
        }
        self.debug.state(data)
        self.debug.render_map(
            self.map,
            pose=self.state.pose,
            path=path,
            target_screen=target_screen,
            target_goal=getattr(self, "current_target_goal", None),
            recovery_waypoint=getattr(self, "active_recovery_waypoint", None),
            navigation_plan=getattr(self, "active_navigation_plan", None),
        )

    def close(self):
        try:
            self.hardware.close()
        finally:
            if self.interaction_audit_file is not None:
                self.interaction_audit_file.close()
            self.camera.release()
            self.debug.close()
