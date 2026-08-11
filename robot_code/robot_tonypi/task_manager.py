#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main competition decision tree."""

import collections
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .classifier import ClassifierClient
from .debug import DebugReporter
from .hardware import RealtimeCamera, TonyPiHardware
from .interaction_client import RobotInteractionClient
from .interaction_logic import apply_worker_change_result, evaluate_interaction_pose, store_flower_observation
from .localizer import AprilTagDetector, Localizer
from .map_model import MapModel, load_tag_positions
from .models import Confidence, InteractionPoseCheck, RobotPose, Screen, ScreenStatus
from .motion import MotionController, RobotState
from .utils import angle_diff_deg, distance_xy, ensure_dir, normalize_angle_deg, now_s
from .vision import ScreenDetector


class TaskManager:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.start_time = time.monotonic()
        self.target_flower = args.target_flower
        self.tag_poses = load_tag_positions(args.load_pos)
        self.map = MapModel(self.tag_poses, config)
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
        self.last_opportunistic_s = 0.0
        self.last_opportunistic_xy = None
        self.last_full_scan_s = 0.0
        self.last_scan_after_turn_s = 0.0
        self.active_passby_stop = None
        self.last_route_scan_stops = []
        self.last_passby_plan = {}
        self.passby_scan_records = {}
        self.last_any_tag_seen_s = now_s()
        self.last_localize_success_s = 0.0
        self.consecutive_no_tag_scans = 0
        self.consecutive_localize_failures = 0
        self.last_no_tag_recovery_s = 0.0
        self.recovery_count = 0
        self.last_recovery = {}
        self.last_initial_discovery_scan = {}
        self.pending_progress_check = None
        self.no_progress_count = 0
        self.visual_no_progress_count = 0
        self.collision_recovery_pending = False
        self.forward_map_block_count = 0
        self.navigation_noop_count = 0
        self.interaction_audit_file = None
        self.interaction_audit_path = ""
        self.open_interaction_audit_log()
        if args.start_x is not None and args.start_y is not None and args.start_yaw is not None:
            self.state.set_manual_pose(args.start_x, args.start_y, args.start_yaw, source="START_ARG")

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
                self.harvest_visible()
                return True
            return self.run_mission()
        finally:
            self.close()

    def run_mission(self) -> bool:
        if self.state.pose is None:
            if not self.initial_localize():
                self.debug.event("mission_abort", reason="initial_localize_failed")
                return False
        self.run_initial_discovery_scan()
        loops = 0
        while loops < int(self.config["mission"]["max_main_loops"]):
            loops += 1
            if self.time_left_s() <= 0:
                self.debug.event("mission_stop", reason="time_limit")
                break
            observed = self.harvest_opportunistic(reason="loop_start")
            if observed == 0 and self.should_periodic_full_scan():
                observed = self.harvest_visible(reason="periodic_full_scan")
                self.last_full_scan_s = now_s()
            if self.target_reached():
                self.debug.event("mission_success", completed=self.map.completed_count())
                break
            target = self.choose_next_screen()
            if target is None:
                self.debug.event("mission_stop", reason="no_unfinished_screen")
                break
            target_mode = "interaction" if target.needs_interaction() else "observation"
            target_xy = target.interaction_staging_xy if target.needs_interaction() else target.observation_xy
            self.debug.event("target_selected", screen_id=target.screen_id, target_mode=target_mode, target_xy=target_xy, plan=self.last_target_plan)
            if target.needs_interaction():
                self.process_screen_interaction(target)
                self.publish_state(target)
                continue
            ok = self.navigate_to_screen(target)
            if not ok:
                target.attempts += 1
                target.notes.append("navigation_failed")
                target.status = ScreenStatus.FAILED
                continue
            self.harvest_visible(prefer_screen_id=target.screen_id, reason="arrived_target")
            if target.status == ScreenStatus.UNKNOWN and self.body_reaim_to_screen(target, reason="arrived_target_no_detection"):
                self.harvest_visible(pan_angles=[100], frames=1, reason="arrived_target_body_reaim")
            if target.needs_interaction():
                self.process_screen_interaction(target)
            elif target.status == ScreenStatus.UNKNOWN:
                target.notes.append("arrived_without_stable_decision")
                target.status = ScreenStatus.FAILED
                self.debug.event(
                    "target_not_completed_after_arrival",
                    screen_id=target.screen_id,
                    attempts=target.attempts,
                    status=target.status.value,
                )
            self.publish_state(target)
        return self.map.completed_count() > 0

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

    def run_initial_discovery_scan(self) -> int:
        """One optional body/head scan after initial localization.

        This is independent of the removed four-waypoint scripted route and may
        only update visual observations; it never performs a flower interaction.
        """
        if not bool(self.config["mission"].get("initial_discovery_scan_enabled", True)):
            return 0
        if self.time_left_s() <= 0:
            return 0
        mission_cfg = self.config["mission"]
        pan_angles = self.unique_pan_angles(
            list(mission_cfg.get("initial_discovery_scan_pan_angles", [130, 100, 70]))
        )
        frames = max(1, int(mission_cfg.get("initial_discovery_scan_frames", 1)))
        left_cycles = max(1, int(mission_cfg.get("initial_discovery_scan_left_turn_cycles", 4)))
        right_cycles = max(1, int(mission_cfg.get("initial_discovery_scan_right_turn_cycles", 5)))
        self.last_initial_discovery_scan = {
            "started_s": round(now_s(), 3),
            "pan_angles": pan_angles,
            "frames": frames,
            "left_turn_cycles": left_cycles,
            "right_turn_cycles": right_cycles,
            "relocalize_after": bool(mission_cfg.get("initial_discovery_scan_relocalize_after", True)),
            "relocalized": None,
            "observed": 0,
        }
        self.debug.event(
            "initial_discovery_scan_start",
            pan_angles=pan_angles,
            frames=frames,
            left_turn_cycles=left_cycles,
            right_turn_cycles=right_cycles,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
        )
        total_observed = 0
        self.hardware.center_head()
        self.debug.event("initial_discovery_scan_turn", direction="left", key="turn_left_fast", cycles=left_cycles)
        result = self.motion.run("turn_left_fast", times_override=left_cycles)
        self.scan_after_turn("initial_discovery_scan_left_turn", "turn_left_fast", result)
        total_observed += int(self.harvest_visible(pan_angles=pan_angles, frames=frames, reason="initial_discovery_scan_left"))
        if self.time_left_s() > 0:
            self.debug.event("initial_discovery_scan_turn", direction="right", key="turn_right_fast", cycles=right_cycles)
            result = self.motion.run("turn_right_fast", times_override=right_cycles)
            self.scan_after_turn("initial_discovery_scan_right_turn", "turn_right_fast", result)
            total_observed += int(self.harvest_visible(pan_angles=pan_angles, frames=frames, reason="initial_discovery_scan_right"))
        if self.time_left_s() > 0 and bool(mission_cfg.get("initial_discovery_scan_relocalize_after", True)):
            self.debug.event("initial_discovery_scan_relocalize_start")
            relocalized = self.localize_scan()
            self.last_initial_discovery_scan["relocalized"] = bool(relocalized)
            self.debug.event(
                "initial_discovery_scan_relocalize_done",
                ok=bool(relocalized),
                pose=None if self.state.pose is None else self.state.pose.as_dict(),
            )
        self.last_initial_discovery_scan["observed"] = total_observed
        self.last_initial_discovery_scan["done_s"] = round(now_s(), 3)
        self.debug.event(
            "initial_discovery_scan_done",
            observed=total_observed,
            pose=None if self.state.pose is None else self.state.pose.as_dict(),
        )
        self.publish_state()
        return total_observed

    def initial_localize(self) -> bool:
        if self.args.dry_run and self.state.pose is None:
            self.state.set_manual_pose(150.0, 150.0, 0.0, source="DRY_RUN_DEFAULT")
            return True
        attempts = int(self.config["localization"]["startup_attempts"])
        search_actions = list(self.config["localization"]["startup_search_actions"])
        for attempt in range(1, attempts + 1):
            self.debug.event("initial_localize_attempt", attempt=attempt, max_attempts=attempts)
            if self.localize_scan():
                return True
            action = search_actions[(attempt - 1) % len(search_actions)]
            result = self.motion.run(action)
            if str(action).startswith("turn_"):
                self.scan_after_turn("initial_localize_search", str(action), result)
                if self.state.pose is not None:
                    return True
        return False

    def localize_scan(self) -> bool:
        saw_any_tag = False
        last_scan_pan = None
        pan_angles = self.boundary_safe_pan_angles(
            list(self.config["localization"]["scan_pan_angles"]),
            reason="localize_scan",
        )
        if not pan_angles:
            self.debug.event("localize_skipped_boundary_outward", pose=None if self.state.pose is None else self.state.pose.as_dict())
        try:
            for pan in pan_angles:
                last_scan_pan = pan
                frame, tags = self.capture_with_tags(pan)
                if frame is None:
                    continue
                if tags:
                    saw_any_tag = True
                    self.update_dynamic_obstacles(tags, pan=pan)
                pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=pan, annotate=True)
                if pose is not None:
                    self.state.set_pose(pose)
                    self.last_localize_success_s = now_s()
                    self.consecutive_localize_failures = 0
                    self.consecutive_no_tag_scans = 0
                    self.evaluate_pending_progress(pose)
                    self.debug.event("pose_update", **pose.as_dict(), head_pan_angle=pan)
                    annotated = self.harvest_during_localize(frame, tags, annotated, pan)
                    self.debug.save_image("latest_annotated.jpg", annotated, force=True)
                    self.publish_state()
                    return True
                self.debug.save_image("latest_annotated.jpg", annotated, force=True)
            self.consecutive_localize_failures += 1
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
            self.center_head_after_scan("localize_scan", last_scan_pan)

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

    def should_periodic_full_scan(self) -> bool:
        period = float(self.config["vision"].get("periodic_full_scan_s", 0.0))
        if period <= 0.0:
            return False
        return now_s() - self.last_full_scan_s >= period

    def should_recheck_completed_screens(self, reason: str) -> bool:
        vision_cfg = self.config.get("vision", {})
        if not bool(vision_cfg.get("recheck_completed_screens_enabled", True)):
            return False
        prefixes = vision_cfg.get(
            "recheck_completed_screen_reasons",
            ["route_passby", "navigate_step", "loop_start", "periodic_full_scan"],
        )
        return any(str(reason).startswith(str(prefix)) for prefix in prefixes)

    def harvest_opportunistic(self, reason: str = "opportunistic") -> int:
        if self.args.dry_run or not bool(self.config["vision"].get("opportunistic_enabled", True)):
            return 0
        pose = self.state.pose
        if pose is None:
            return 0
        pan_angles = list(self.config["vision"].get("opportunistic_pan_angles", [100]))
        include_done = self.should_recheck_completed_screens(reason)
        predicted = self.predict_visible_screens(pose, pan_angles=pan_angles, include_done=include_done)
        if not predicted:
            return 0
        t = now_s()
        if t - self.last_opportunistic_s < float(self.config["vision"].get("opportunistic_min_interval_s", 2.0)):
            return 0
        if self.last_opportunistic_xy is not None:
            moved = distance_xy(self.last_opportunistic_xy, pose.xy())
            if moved < float(self.config["vision"].get("opportunistic_min_move_cm", 12.0)):
                return 0
        max_pans = max(1, int(self.config["vision"].get("opportunistic_max_pans", 1)))
        best_pans = self.unique_pan_angles([item[2] for item in predicted[:max_pans]])
        if not best_pans:
            best_pans = self.unique_pan_angles(pan_angles)
        self.debug.event(
            "opportunistic_harvest",
            reason=reason,
            include_done=include_done,
            predicted=[{"screen_id": sid, "score": round(score, 3), "pan": pan} for sid, score, pan in predicted[:6]],
            pan_angles=best_pans,
        )
        self.last_opportunistic_s = t
        self.last_opportunistic_xy = pose.xy()
        return self.harvest_visible(pan_angles=best_pans, frames=1, reason=reason)

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

    def observe_frame(
        self,
        frame,
        tags,
        annotated,
        pan: float,
        include_completed_screens: bool = False,
        vote_summary: Optional[dict] = None,
    ) -> Tuple[object, int]:
        """Detect and classify screens, updating observations only.

        Returns (annotated_frame, observed_count). This function is deliberately
        incapable of lifting the hand or sending a Worker request.
        """
        min_conf = float(self.config["vision"]["min_confidence"])
        candidates = self.screen_detector.detect(frame, tags, self.state.pose)
        annotated = self.screen_detector.annotate(annotated, candidates, tags)
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        observed = 0
        for cand in candidates:
            screen = self.map.screens.get(cand.screen_id)
            if screen is None or (screen.done() and not include_completed_screens):
                continue
            # Skip screens already being processed in this scan
            if screen.done() and not include_completed_screens:
                continue
            result = self.classifier.classify_crop(cand.crop_28x28)
            suffix = "imm_{}".format(pan)
            self.debug.save_crop(cand.screen_id, cand.crop_28x28, suffix)
            if vote_summary is not None:
                vote_entry = self._vote_entry(vote_summary, cand.screen_id)
                vote_entry["observations"].append(
                    {
                        "pan": pan,
                        "ok": result.ok,
                        "flower": result.flower_api if result.ok else None,
                        "flower_cn": result.flower_cn if result.ok else None,
                        "confidence": round(float(result.confidence), 4) if result.ok else 0.0,
                        "error": result.error if not result.ok else "",
                    }
                )
            if not result.ok:
                self.debug.event("classification_failed", screen_id=cand.screen_id, error=result.error)
                continue
            self.debug.event(
                "classified",
                screen_id=cand.screen_id,
                flower=result.flower_api,
                flower_cn=result.flower_cn,
                confidence=result.confidence,
            )
            if float(result.confidence) < min_conf:
                self.debug.event(
                    "classification_low_confidence",
                    screen_id=cand.screen_id,
                    flower=result.flower_api,
                    flower_cn=result.flower_cn,
                    confidence=result.confidence,
                    threshold=min_conf,
                )
                continue
            entry = self._vote_entry(vote_summary, cand.screen_id) if vote_summary else None
            if entry is not None:
                entry["best"] = {
                    "flower": result.flower_api,
                    "count": 1,
                    "avg_confidence": round(float(result.confidence), 4),
                }
            self.record_flower_observation(screen, result.flower_api, float(result.confidence), entry)
            observed += 1
        return annotated, observed

    def collect_harvest_votes_from_frame(
        self,
        frame,
        tags,
        annotated,
        votes,
        confs,
        vote_summary,
        round_idx: int,
        pan: float,
        min_conf: float,
        include_completed_screens: bool = False,
    ):
        candidates = self.screen_detector.detect(frame, tags, self.state.pose)
        annotated = self.screen_detector.annotate(annotated, candidates, tags)
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        for cand in candidates:
            screen = self.map.screens.get(cand.screen_id)
            if screen is None or (screen.done() and not include_completed_screens):
                continue
            result = self.classifier.classify_crop(cand.crop_28x28)
            suffix = "{}_{}".format(round_idx, pan)
            self.debug.save_crop(cand.screen_id, cand.crop_28x28, suffix)
            vote_entry = self._vote_entry(vote_summary, cand.screen_id)
            if not result.ok:
                vote_entry["observations"].append(
                    {
                        "round": round_idx,
                        "pan": pan,
                        "ok": False,
                        "error": result.error,
                    }
                )
                self.debug.event("classification_failed", screen_id=cand.screen_id, error=result.error)
                continue
            flower_vote = vote_entry["votes"].setdefault(
                result.flower_api,
                {"count": 0, "confidences": []},
            )
            observation = {
                "round": round_idx,
                "pan": pan,
                "ok": True,
                "flower": result.flower_api,
                "flower_cn": result.flower_cn,
                "confidence": round(float(result.confidence), 4),
            }
            if float(result.confidence) < min_conf:
                observation["accepted_vote"] = False
                observation["reject_reason"] = "low_confidence"
                vote_entry["observations"].append(observation)
                self.debug.event(
                    "classification_low_confidence",
                    screen_id=cand.screen_id,
                    flower=result.flower_api,
                    flower_cn=result.flower_cn,
                    confidence=result.confidence,
                    threshold=min_conf,
                )
                continue
            votes[cand.screen_id][result.flower_api] += 1
            confs[cand.screen_id][result.flower_api].append(result.confidence)
            flower_vote["count"] += 1
            flower_vote["confidences"].append(round(float(result.confidence), 4))
            observation["accepted_vote"] = True
            vote_entry["observations"].append(observation)
            self.debug.event(
                "classified",
                screen_id=cand.screen_id,
                flower=result.flower_api,
                flower_cn=result.flower_cn,
                confidence=result.confidence,
            )
        return annotated

    def finalize_harvest_votes(self, votes, confs, vote_summary, min_conf: float, publish: bool = True) -> int:
        observed = 0
        min_votes = int(self.config["vision"]["min_votes"])
        for screen_id, counter in votes.items():
            flower = max(
                counter,
                key=lambda name: (
                    int(counter[name]),
                    sum(confs[screen_id][name]) / max(1, len(confs[screen_id][name])),
                ),
            )
            count = counter[flower]
            avg_conf = sum(confs[screen_id][flower]) / max(1, len(confs[screen_id][flower]))
            screen = self.map.screens[screen_id]
            vote_entry = self._vote_entry(vote_summary, screen_id)
            vote_entry["best"] = {
                "flower": flower,
                "count": int(count),
                "avg_confidence": round(float(avg_conf), 4),
            }
            screen.last_seen_s = now_s()
            screen.last_classification = flower
            screen.last_confidence = avg_conf
            if count < min_votes or avg_conf < min_conf:
                if screen.status not in (ScreenStatus.NEEDS_CHANGE, ScreenStatus.CHANGED, ScreenStatus.ALREADY_TARGET):
                    screen.status = ScreenStatus.FAILED
                screen.notes.append("unstable_vote:{}x conf={:.2f}".format(count, avg_conf))
                vote_entry["decision"] = "unstable"
                continue
            self.record_flower_observation(screen, flower, avg_conf, vote_entry)
            observed += 1
        for entry in vote_summary["screens"].values():
            if entry["best"] is None and entry["observations"]:
                entry["decision"] = "no_stable_success"
            for item in entry["votes"].values():
                values = item.get("confidences", [])
                item["avg_confidence"] = round(sum(values) / max(1, len(values)), 4) if values else 0.0
        self.last_vote_summary = vote_summary
        if publish:
            self.publish_state()
        return observed

    def harvest_during_localize(self, frame, tags, annotated, pan: float):
        if self.args.dry_run or not bool(self.config["vision"].get("harvest_during_localize_enabled", True)):
            return annotated
        include_done = bool(self.config["vision"].get("harvest_during_localize_recheck_completed", True))
        vote_summary = {
            "started_s": round(now_s(), 3),
            "reason": "localize_scan",
            "prefer_screen_id": None,
            "pan_angles": [pan],
            "min_confidence": float(self.config["vision"]["min_confidence"]),
            "source": "localize_scan",
            "include_completed_screens": include_done,
            "screens": {},
        }
        try:
            annotated, observed = self.observe_frame(
                frame,
                tags,
                annotated,
                pan=pan,
                include_completed_screens=include_done,
                vote_summary=vote_summary,
            )
            self.debug.event(
                "localize_harvest_done",
                pan=pan,
                observed=observed,
                screens=len(vote_summary["screens"]),
            )
        except Exception as exc:
            self.debug.event("localize_harvest_failed", pan=pan, error=str(exc))
        return annotated

    def scan_after_turn(self, reason: str, action_key: str = "", action_result=None) -> int:
        if self.args.dry_run or not bool(self.config["vision"].get("scan_after_turn_enabled", True)):
            return 0
        if self.time_left_s() <= 0:
            return 0
        t = now_s()
        min_interval = float(self.config["vision"].get("scan_after_turn_min_interval_s", 1.0))
        if t - self.last_scan_after_turn_s < min_interval:
            return 0
        self.last_scan_after_turn_s = t
        center = float(self.config["camera"].get("head_center_angle", 100.0))
        frame, tags = self.capture_with_tags(center)
        if frame is None:
            self.debug.event("scan_after_turn_failed", reason=reason, action_key=action_key, error="capture_failed")
            return 0
        pose, annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=center, annotate=True)
        localized = pose is not None
        if pose is not None:
            self.state.set_pose(pose)
            self.last_localize_success_s = now_s()
            self.consecutive_localize_failures = 0
            self.consecutive_no_tag_scans = 0
            self.evaluate_pending_progress(pose)
            self.debug.event("pose_update", **pose.as_dict(), head_pan_angle=center, reason="scan_after_turn")
        include_done = self.should_recheck_completed_screens("scan_after_turn")
        vote_summary = {
            "started_s": round(now_s(), 3),
            "reason": "scan_after_turn",
            "turn_reason": reason,
            "action_key": action_key,
            "action_times": None if action_result is None else int(getattr(action_result, "times", 0)),
            "prefer_screen_id": None,
            "pan_angles": [center],
            "min_confidence": float(self.config["vision"]["min_confidence"]),
            "include_completed_screens": include_done,
            "screens": {},
        }
        annotated, observed = self.observe_frame(
            frame,
            tags,
            annotated,
            pan=center,
            include_completed_screens=include_done,
            vote_summary=vote_summary,
        )
        self.debug.save_image("latest_annotated.jpg", annotated, force=True)
        self.debug.event(
            "scan_after_turn_done",
            reason=reason,
            action_key=action_key,
            localized=localized,
            tag_count=len(tags),
            observed=observed,
            screens=len(vote_summary["screens"]),
            include_done=include_done,
        )
        self.publish_state()
        return int(observed)

    def harvest_visible(
        self,
        prefer_screen_id: Optional[int] = None,
        pan_angles: Optional[List[float]] = None,
        frames: Optional[int] = None,
        reason: str = "harvest_visible",
    ) -> int:
        if self.args.dry_run:
            return 0
        last_scan_pan = None
        if pan_angles is None:
            pan_angles = list(self.config["vision"]["harvest_pan_angles"])
        else:
            pan_angles = list(pan_angles)
        include_done = self.should_recheck_completed_screens(reason)
        if prefer_screen_id is not None:
            pan_angles = self.pan_angles_for_screen(prefer_screen_id, fallback=pan_angles)
        pan_angles = self.boundary_safe_pan_angles(pan_angles, reason=reason)
        if not pan_angles:
            self.debug.event(
                "harvest_skipped_boundary_outward",
                reason=reason,
                prefer_screen_id=prefer_screen_id,
                pose=None if self.state.pose is None else self.state.pose.as_dict(),
            )
            return 0
        vote_summary = {
            "started_s": round(now_s(), 3),
            "reason": reason,
            "prefer_screen_id": prefer_screen_id,
            "pan_angles": pan_angles,
            "min_confidence": float(self.config["vision"]["min_confidence"]),
            "include_completed_screens": include_done,
            "screens": {},
        }

        total_observed = 0
        for pan in pan_angles:
            last_scan_pan = pan
            frame, tags = self.capture_with_tags(pan)
            if frame is None:
                continue
            pose, loc_annotated = self.localizer.estimate_from_frame(frame, tags, head_pan_angle=pan, annotate=True)
            if pose is not None:
                self.state.set_pose(pose)
                self.debug.event("pose_update", **pose.as_dict(), head_pan_angle=pan)
            annotated, observed = self.observe_frame(
                frame,
                tags,
                loc_annotated,
                pan=pan,
                include_completed_screens=include_done,
                vote_summary=vote_summary,
            )
            total_observed += observed

        self.last_vote_summary = vote_summary
        self.center_head_after_scan("harvest_visible:{}".format(reason), last_scan_pan)
        self.publish_state()
        return total_observed

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

    def worker_id_for_screen(self, screen: Screen) -> Optional[int]:
        """Resolve the explicit screen-to-Worker mapping; never assume identity."""
        return screen.worker_id

    def interaction_pose_check(
        self,
        screen: Screen,
        pose: Optional[RobotPose] = None,
        expected_from_flower: Optional[str] = None,
    ) -> InteractionPoseCheck:
        pose = self.state.pose if pose is None else pose
        return evaluate_interaction_pose(
            screen,
            pose,
            self.target_flower,
            self.config["interaction"],
            self.worker_id_for_screen(screen),
            expected_from_flower=expected_from_flower,
        )

    def align_for_screen_interaction(self, screen: Screen) -> bool:
        cfg = self.config["interaction"]
        max_steps = max(1, int(cfg.get("interaction_max_alignment_steps", 20)))
        for step in range(max_steps):
            if self.time_left_s() <= 0:
                return False
            should_localize = bool(cfg.get("interaction_relocalize_each_step", True)) or step == 0
            if should_localize and not self.args.dry_run:
                self.localize_scan()
            check = self.interaction_pose_check(screen)
            self.last_interaction_check = check.as_dict()
            self.debug.event(
                "interaction_alignment_check",
                screen_id=screen.screen_id,
                worker_id=self.worker_id_for_screen(screen),
                step=step,
                target_xy=screen.interaction_xy,
                target_yaw_deg=screen.interaction_yaw_deg,
                **check.as_dict(),
            )
            self.publish_state(screen)
            if check.ready:
                return True
            if not check.pose_valid:
                if self.args.dry_run:
                    return False
                continue
            if check.yaw_error_deg is not None and abs(check.yaw_error_deg) > float(cfg["interaction_yaw_tolerance_deg"]):
                key = "turn_left_micro" if check.yaw_error_deg > 0.0 else "turn_right_micro"
                self.motion.run(key, times_override=1)
                continue
            if check.lateral_error_cm is not None and abs(check.lateral_error_cm) > float(cfg["interaction_lateral_tolerance_cm"]):
                self.motion.move_lateral(-check.lateral_error_cm)
                continue
            if check.distance_error_cm is not None and abs(check.distance_error_cm) > float(cfg["interaction_distance_tolerance_cm"]):
                if check.distance_error_cm > 0.0:
                    self.motion.run("forward_micro", times_override=1)
                else:
                    self.motion.run("back_fast", times_override=1)
                continue
            return False
        self.debug.event("interaction_alignment_failed", screen_id=screen.screen_id, max_steps=max_steps)
        return False

    def navigate_to_interaction_pose(self, screen: Screen) -> bool:
        cfg = self.config["interaction"]
        return self.navigate_to_xy(
            screen.interaction_staging_xy,
            reason="interaction_staging",
            arrival_radius_cm=float(cfg.get("interaction_staging_arrival_radius_cm", 8.0)),
            max_steps=int(self.config["navigation"]["max_steps_per_target"]),
            target_yaw_deg=screen.interaction_yaw_deg,
        )

    def process_screen_interaction(self, screen: Screen) -> bool:
        worker_id = self.worker_id_for_screen(screen)
        if not screen.last_classification or screen.last_classification == self.target_flower:
            self.debug.event("interaction_skipped", screen_id=screen.screen_id, reason="flower_not_changeable")
            return False
        if worker_id is None:
            screen.notes.append("worker_mapping_missing")
            self.debug.event("interaction_skipped", screen_id=screen.screen_id, reason="worker_mapping_missing")
            return False
        if not self.navigate_to_interaction_pose(screen):
            screen.notes.append("interaction_staging_navigation_failed")
            return False
        if not self.align_for_screen_interaction(screen):
            screen.notes.append("interaction_alignment_failed")
            return False
        if bool(self.config["interaction"].get("interaction_relocalize_before_action", True)) and not self.args.dry_run:
            self.localize_scan()
        # Navigation/relocalization may have produced a newer visual result.
        # Bind the transaction to that final flower and require it to remain
        # unchanged through the lower-level pre-send safety gate.
        from_flower = screen.last_classification
        if not from_flower or from_flower == self.target_flower:
            self.debug.event("interaction_skipped", screen_id=screen.screen_id, reason="flower_changed_before_action")
            return False
        final_check = self.interaction_pose_check(screen, expected_from_flower=from_flower)
        self.last_interaction_check = final_check.as_dict()
        if not final_check.ready:
            self.debug.event("interaction_safety_gate_blocked", screen_id=screen.screen_id, stage="task_manager_final", check=final_check.as_dict())
            return False

        pose_snapshot = None if self.state.pose is None else self.state.pose.as_dict()
        screen.status = ScreenStatus.INTERACTING
        result = self.interaction.change_flower(
            screen_id=screen.screen_id,
            worker_id=worker_id,
            from_flower=from_flower,
            to_flower=self.target_flower,
            safety_gate=lambda: self.interaction_pose_check(screen, expected_from_flower=from_flower),
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
            "interaction_check": final_check.as_dict(),
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

    def path_segments_clear(self, points: List[Tuple[float, float]]) -> bool:
        if len(points) < 2:
            return False
        for pt in points[1:]:
            if not self.map.is_traversable_xy(pt):
                return False
        for start, end in zip(points, points[1:]):
            if not self.map.line_clear(start, end):
                return False
        return True

    def body_translation_candidate_paths(self, pose: RobotPose, goal_xy: Tuple[float, float]) -> List[List[Tuple[float, float]]]:
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
        return [path for path in candidates if self.path_segments_clear(path)]

    def plan_navigation_path(self, pose: RobotPose, goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        if bool(self.config["navigation"].get("action_planner_enabled", True)):
            action_path = self.map.plan_action_path(
                pose,
                goal_xy,
                self.config["navigation"],
                self.config["motion"],
            )
            if action_path:
                return action_path
            return self.map.plan(pose.xy(), goal_xy)

        astar_path = self.map.plan(pose.xy(), goal_xy)
        candidates = self.body_translation_candidate_paths(pose, goal_xy)
        if not candidates:
            return astar_path
        fallback_len = self.path_length_cm(astar_path, fallback_start=pose.xy(), fallback_goal=goal_xy)
        max_ratio = float(self.config["navigation"].get("translation_path_max_detour_ratio", 1.45))
        best = min(candidates, key=lambda item: self.path_length_cm(item, fallback_start=pose.xy(), fallback_goal=goal_xy))
        best_len = self.path_length_cm(best, fallback_start=pose.xy(), fallback_goal=goal_xy)
        if not astar_path or best_len <= fallback_len * max_ratio:
            return best
        return astar_path

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
            if not self.map.line_clear(pose.xy(), point):
                continue
            last_clear = point
            if self.waypoint_has_navigation_action(pose, point):
                last_actionable = point
                if dist >= lookahead:
                    far_actionable = point
            if dist >= max_lookahead:
                break
        if self.map.line_clear(pose.xy(), target_xy):
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

    def predict_route_screen_ids(
        self,
        pose: RobotPose,
        path: List[Tuple[float, float]],
        exclude_id: Optional[int] = None,
    ) -> Set[int]:
        if not path:
            return set()
        points = [pose.xy()] + list(path)
        found: Set[int] = set()
        sample_step = float(self.config["navigation"].get("route_sample_step_cm", 24.0))
        pans = list(self.config["vision"].get("opportunistic_pan_angles", [100]))
        for idx in range(1, len(points)):
            start = points[idx - 1]
            end = points[idx]
            seg_len = distance_xy(start, end)
            if seg_len < 1e-6:
                continue
            yaw = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
            samples = max(1, int(seg_len / max(5.0, sample_step)))
            for sample_idx in range(samples + 1):
                t = sample_idx / float(samples)
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                sample_pose = RobotPose(x, y, yaw, confidence=pose.confidence, source="ROUTE_PREDICT")
                for sid, _, _ in self.predict_visible_screens(sample_pose, pan_angles=pans):
                    if exclude_id is not None and sid == int(exclude_id):
                        continue
                    found.add(int(sid))
        return found

    def passby_recently_scanned(self, screen_id: int, xy: Tuple[float, float]) -> bool:
        record = self.passby_scan_records.get(int(screen_id))
        if not record:
            return False
        cooldown = float(self.config["navigation"].get("route_passby_cooldown_s", 22.0))
        repeat_dist = float(self.config["navigation"].get("route_passby_repeat_distance_cm", 45.0))
        if now_s() - float(record.get("t", 0.0)) < cooldown:
            return True
        last_xy = tuple(record.get("xy", xy))
        return distance_xy(last_xy, xy) < repeat_dist

    def mark_passby_scan_attempt(self, stop) -> None:
        pose = self.state.pose
        xy = pose.xy() if pose is not None else tuple(stop.get("xy", (0.0, 0.0)))
        t = now_s()
        for screen_id in stop.get("screen_ids", []):
            self.passby_scan_records[int(screen_id)] = {"t": t, "xy": xy}

    def describe_passby_side(self, rel_angle: float) -> str:
        if rel_angle > 22.0:
            return "left"
        if rel_angle < -22.0:
            return "right"
        return "front"

    def merge_passby_stops(self, raw_stops: List[dict]) -> List[dict]:
        if not raw_stops:
            return []
        merge_dist = float(self.config["navigation"].get("route_passby_merge_distance_cm", 28.0))
        max_pans = max(1, int(self.config["navigation"].get("route_passby_max_pans", 2)))
        merged = []
        for stop in sorted(raw_stops, key=lambda item: item["ahead_cm"]):
            if merged and abs(stop["ahead_cm"] - merged[-1]["ahead_cm"]) <= merge_dist:
                current = merged[-1]
                current["score"] = max(current["score"], stop["score"])
                current["screen_ids"] = sorted(set(current["screen_ids"]) | set(stop["screen_ids"]))
                current["sides"] = sorted(set(current["sides"]) | set(stop["sides"]))
                current["pan_angles"] = self.unique_pan_angles(current["pan_angles"] + stop["pan_angles"])[:max_pans]
                current["predicted"] = (current.get("predicted", []) + stop.get("predicted", []))[:10]
                continue
            stop = dict(stop)
            stop["pan_angles"] = self.unique_pan_angles(stop["pan_angles"])[:max_pans]
            merged.append(stop)
        return merged

    def find_passby_stops_for_segment(
        self,
        pose: RobotPose,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        target_screen_id: Optional[int] = None,
    ) -> List[dict]:
        if not bool(self.config["navigation"].get("route_passby_scan_enabled", True)):
            return []
        seg_len = distance_xy(start_xy, end_xy)
        if seg_len < 1e-6:
            return []
        min_ahead = float(self.config["navigation"].get("route_passby_min_ahead_cm", 50.0))
        max_ahead = min(seg_len, float(self.config["navigation"].get("route_passby_max_ahead_cm", 95.0)))
        if max_ahead < min_ahead:
            return []
        sample_step = max(6.0, float(self.config["navigation"].get("route_sample_step_cm", 24.0)))
        min_score = float(self.config["navigation"].get("route_passby_min_score", 0.22))
        pans = list(self.config["navigation"].get("route_passby_pan_angles", [130, 70, 100]))
        include_done = bool(self.config["navigation"].get("route_passby_include_completed_screens", True))
        yaw = math.degrees(math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0]))
        samples = max(1, int(math.ceil((max_ahead - min_ahead) / sample_step)))
        raw_stops = []
        for sample_idx in range(samples + 1):
            ahead = min_ahead + (max_ahead - min_ahead) * sample_idx / float(samples)
            t = ahead / seg_len
            x = start_xy[0] + (end_xy[0] - start_xy[0]) * t
            y = start_xy[1] + (end_xy[1] - start_xy[1]) * t
            sample_pose = RobotPose(x, y, yaw, confidence=pose.confidence, source="ROUTE_PASSBY")
            visible = self.predict_visible_screens(sample_pose, pan_angles=pans, include_done=include_done)
            predicted = []
            screen_ids = []
            pan_angles = []
            sides = []
            best_score = 0.0
            for sid, score, pan in visible:
                if score < min_score:
                    continue
                screen = self.map.screens.get(int(sid))
                if screen is None or (screen.done() and not include_done):
                    continue
                if self.passby_recently_scanned(int(sid), (x, y)):
                    continue
                dx = screen.center_xy[0] - x
                dy = screen.center_xy[1] - y
                rel = angle_diff_deg(math.degrees(math.atan2(dy, dx)), yaw)
                side = self.describe_passby_side(rel)
                screen_ids.append(int(sid))
                pan_angles.append(float(pan))
                sides.append(side)
                best_score = max(best_score, float(score))
                predicted.append(
                    {
                        "screen_id": int(sid),
                        "score": round(float(score), 3),
                        "pan": round(float(pan), 1),
                        "side": side,
                        "rel_angle": round(float(rel), 1),
                    }
                )
            if screen_ids:
                if bool(self.config["navigation"].get("route_passby_require_side", True)) and not any(side in ("left", "right") for side in sides):
                    continue
                raw_stops.append(
                    {
                        "xy": (round(float(x), 2), round(float(y), 2)),
                        "ahead_cm": round(float(ahead), 1),
                        "yaw_deg": round(float(yaw), 1),
                        "score": round(float(best_score), 3),
                        "screen_ids": sorted(set(screen_ids)),
                        "pan_angles": self.unique_pan_angles(pan_angles),
                        "sides": sorted(set(sides)),
                        "target_screen_id": None if target_screen_id is None else int(target_screen_id),
                        "predicted": predicted[:10],
                    }
                )
        return self.merge_passby_stops(raw_stops)

    def find_passby_stops_for_path(
        self,
        pose: RobotPose,
        path: List[Tuple[float, float]],
        target_screen_id: Optional[int] = None,
    ) -> List[dict]:
        if not bool(self.config["navigation"].get("route_passby_scan_enabled", True)):
            return []
        points = [pose.xy()] + [tuple(item) for item in path]
        segments = []
        total_len = 0.0
        for idx in range(1, len(points)):
            start = points[idx - 1]
            end = points[idx]
            seg_len = distance_xy(start, end)
            if seg_len < 1e-6:
                continue
            segments.append((total_len, total_len + seg_len, start, end, seg_len))
            total_len += seg_len
        if not segments:
            return []
        min_ahead = float(self.config["navigation"].get("route_passby_min_ahead_cm", 50.0))
        max_ahead = min(total_len, float(self.config["navigation"].get("route_passby_max_ahead_cm", 95.0)))
        if max_ahead < min_ahead:
            return []
        sample_step = max(6.0, float(self.config["navigation"].get("route_sample_step_cm", 24.0)))
        min_score = float(self.config["navigation"].get("route_passby_min_score", 0.22))
        pans = list(self.config["navigation"].get("route_passby_pan_angles", [130, 70, 100]))
        include_done = bool(self.config["navigation"].get("route_passby_include_completed_screens", True))
        samples = max(1, int(math.ceil((max_ahead - min_ahead) / sample_step)))
        raw_stops = []
        for sample_idx in range(samples + 1):
            ahead = min_ahead + (max_ahead - min_ahead) * sample_idx / float(samples)
            seg = segments[-1]
            for candidate in segments:
                if ahead <= candidate[1] + 1e-6:
                    seg = candidate
                    break
            seg_start, _, start_xy, end_xy, seg_len = seg
            t = max(0.0, min(1.0, (ahead - seg_start) / seg_len))
            x = start_xy[0] + (end_xy[0] - start_xy[0]) * t
            y = start_xy[1] + (end_xy[1] - start_xy[1]) * t
            yaw = math.degrees(math.atan2(end_xy[1] - start_xy[1], end_xy[0] - start_xy[0]))
            sample_pose = RobotPose(x, y, yaw, confidence=pose.confidence, source="ROUTE_PASSBY")
            visible = self.predict_visible_screens(sample_pose, pan_angles=pans, include_done=include_done)
            predicted = []
            screen_ids = []
            pan_angles = []
            sides = []
            best_score = 0.0
            for sid, score, pan in visible:
                if score < min_score:
                    continue
                screen = self.map.screens.get(int(sid))
                if screen is None or (screen.done() and not include_done):
                    continue
                if self.passby_recently_scanned(int(sid), (x, y)):
                    continue
                dx = screen.center_xy[0] - x
                dy = screen.center_xy[1] - y
                rel = angle_diff_deg(math.degrees(math.atan2(dy, dx)), yaw)
                side = self.describe_passby_side(rel)
                screen_ids.append(int(sid))
                pan_angles.append(float(pan))
                sides.append(side)
                best_score = max(best_score, float(score))
                predicted.append(
                    {
                        "screen_id": int(sid),
                        "score": round(float(score), 3),
                        "pan": round(float(pan), 1),
                        "side": side,
                        "rel_angle": round(float(rel), 1),
                    }
                )
            if screen_ids:
                if bool(self.config["navigation"].get("route_passby_require_side", True)) and not any(side in ("left", "right") for side in sides):
                    continue
                raw_stops.append(
                    {
                        "xy": (round(float(x), 2), round(float(y), 2)),
                        "ahead_cm": round(float(ahead), 1),
                        "yaw_deg": round(float(yaw), 1),
                        "score": round(float(best_score), 3),
                        "screen_ids": sorted(set(screen_ids)),
                        "pan_angles": self.unique_pan_angles(pan_angles),
                        "sides": sorted(set(sides)),
                        "target_screen_id": None if target_screen_id is None else int(target_screen_id),
                        "predicted": predicted[:10],
                    }
                )
        return self.merge_passby_stops(raw_stops)

    def current_passby_stop_valid(self, pose: RobotPose, target_screen_id: int) -> bool:
        stop = self.active_passby_stop
        if not stop:
            return False
        if int(stop.get("target_screen_id", target_screen_id)) != int(target_screen_id):
            return False
        stop_xy = tuple(stop["xy"])
        desired = math.degrees(math.atan2(stop_xy[1] - pose.y_cm, stop_xy[0] - pose.x_cm))
        if abs(angle_diff_deg(desired, pose.yaw_deg)) > 115.0 and distance_xy(pose.xy(), stop_xy) > 20.0:
            return False
        return True

    def plan_next_passby_stop(self, pose: RobotPose, path: List[Tuple[float, float]], target_screen_id: int):
        if not path:
            self.last_route_scan_stops = []
            return None
        stops = self.find_passby_stops_for_path(pose, path, target_screen_id=target_screen_id)
        self.last_route_scan_stops = stops[:5]
        if stops:
            self.debug.event(
                "route_passby_candidates",
                target_screen_id=target_screen_id,
                stops=stops[:5],
            )
            return stops[0]
        return None

    def execute_passby_scan(self, stop) -> int:
        pan_angles = self.unique_pan_angles(list(stop.get("pan_angles", [])))
        if not pan_angles:
            pan_angles = self.unique_pan_angles(list(self.config["navigation"].get("route_passby_pan_angles", [130, 70, 100])))
        self.debug.event(
            "route_passby_scan_start",
            screen_ids=stop.get("screen_ids", []),
            sides=stop.get("sides", []),
            pan_angles=pan_angles,
            xy=stop.get("xy"),
            predicted=stop.get("predicted", []),
        )
        self.mark_passby_scan_attempt(stop)
        observed = self.harvest_visible(pan_angles=pan_angles, frames=1, reason="route_passby")
        self.debug.event(
            "route_passby_scan_done",
            observed=observed,
            screen_ids=stop.get("screen_ids", []),
        )
        return observed

    def choose_next_screen(self) -> Optional[Screen]:
        if self.state.pose is None:
            return None
        pose = self.state.pose
        best = None
        best_score = float("inf")
        best_plan = {}
        for screen in self.map.unfinished_screens():
            goal = screen.interaction_staging_xy if screen.needs_interaction() else screen.observation_xy
            path = self.plan_navigation_path(pose, goal)
            travel = self.path_length_cm(path, fallback_start=pose.xy(), fallback_goal=goal)
            route_screens = self.predict_route_screen_ids(pose, path, exclude_id=screen.screen_id)
            desired_yaw = math.degrees(math.atan2(goal[1] - pose.y_cm, goal[0] - pose.x_cm))
            turn = abs(angle_diff_deg(desired_yaw, pose.yaw_deg))
            route_bonus = float(self.config["navigation"].get("route_screen_bonus_cm", 55.0)) * len(route_screens)
            turn_cost = float(self.config["navigation"].get("turn_cost_cm_per_deg", 0.65))
            score = travel + turn_cost * turn - route_bonus
            if score < best_score:
                best = screen
                best_score = score
                best_plan = {
                    "score": round(float(score), 2),
                    "travel_cm": round(float(travel), 1),
                    "turn_deg": round(float(turn), 1),
                    "turn_cost": round(float(turn_cost * turn), 1),
                    "route_bonus": round(float(route_bonus), 1),
                    "route_screen_ids": sorted(route_screens),
                    "path_points": len(path),
                    "target_mode": "interaction" if screen.needs_interaction() else "observation",
                    "target_xy": [round(float(goal[0]), 2), round(float(goal[1]), 2)],
                }
        self.last_target_plan = best_plan
        return best

    def body_reaim_to_screen(self, screen: Screen, reason: str = "body_reaim") -> bool:
        if not bool(self.config["navigation"].get("target_body_reaim_enabled", True)):
            return False
        pose = self.state.pose
        if pose is None:
            return False
        desired_yaw = math.degrees(math.atan2(screen.center_xy[1] - pose.y_cm, screen.center_xy[0] - pose.x_cm))
        diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
        min_deg = float(self.config["navigation"].get("target_body_reaim_min_deg", 12.0))
        if abs(diff) < min_deg:
            self.debug.event(
                "target_body_reaim_skipped",
                reason=reason,
                screen_id=screen.screen_id,
                diff_yaw=round(diff, 1),
            )
            return False
        self.debug.event(
            "target_body_reaim",
            reason=reason,
            screen_id=screen.screen_id,
            desired_yaw=round(desired_yaw, 1),
            current_yaw=round(pose.yaw_deg, 1),
            diff_yaw=round(diff, 1),
        )
        self.hardware.center_head()
        self.turn_toward_yaw_boundary_aware(desired_yaw)
        return True

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

    def choose_translation_action(self, pose: RobotPose, waypoint: Tuple[float, float]) -> Optional[dict]:
        nav_cfg = self.config["navigation"]
        if not bool(nav_cfg.get("translation_prefer_enabled", True)):
            return None
        forward, lateral = self.local_vector_to(pose, waypoint)
        current_dist = distance_xy(pose.xy(), waypoint)
        if current_dist < 1.0:
            return None
        min_progress = float(nav_cfg.get("translation_min_progress_cm", 2.0))
        options = []

        min_forward = float(nav_cfg.get("translation_min_forward_cm", 6.0))
        if forward >= min_forward:
            requested = min(float(forward), current_dist)
            planned = self.planned_forward_step_cm(requested)
            map_check_min = float(nav_cfg.get("forward_map_check_min_cm", 16.0))
            if planned < map_check_min or self.forward_clear_for_distance(pose, planned):
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
            if abs(planned) > 0.0 and self.path_segments_clear([pose.xy(), lateral_target]):
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
        return max(options, key=lambda item: (item["progress_cm"], item["kind"] == "forward"))

    def execute_translation_action(
        self,
        action: dict,
        pose: RobotPose,
        waypoint: Tuple[float, float],
        goal_dist_cm: float,
        context: dict,
    ) -> str:
        detail = dict(context)
        detail.update(
            {
                "action": action["kind"],
                "progress_cm": round(float(action.get("progress_cm", 0.0)), 1),
                "planned_cm": round(float(action.get("planned_cm", 0.0)), 1),
                "forward_component_cm": round(float(action.get("forward_cm", 0.0)), 1),
                "lateral_component_cm": round(float(action.get("lateral_cm", 0.0)), 1),
                "waypoint": (round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
            }
        )
        self.debug.event("translation_step", **detail)
        if action["kind"] == "strafe":
            self.forward_map_block_count = 0
            self.motion.move_lateral(float(action["distance_cm"]))
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
            self.active_passby_stop = None
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
        result = self.motion.move_forward(forward_dist)
        self.set_pending_forward_progress(pose_before_forward, abs(float(result.model_forward_cm)))
        if self.evaluate_visual_forward_progress(visual_before, abs(float(result.model_forward_cm))):
            self.active_passby_stop = None
        if self.collision_recovery_pending:
            reason = str(context.get("reason", "visual_forward_no_progress"))
            self.recover_toward_field_center(reason + ":visual_forward_no_progress", backoff=True)
            return "recovered"
        return "moved"

    def clear_navigation_noop(self) -> None:
        self.navigation_noop_count = 0

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

    def turn_toward_yaw_for_recovery(self, target_yaw: float) -> None:
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
                break
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
                self.scan_after_turn("boundary_safe_recovery_turn", key, result)
            else:
                result = self.motion.turn_toward(diff)
                if result is not None:
                    self.scan_after_turn("recovery_turn_toward", result.key, result)

    def turn_toward_yaw_boundary_aware(self, target_yaw: float) -> None:
        pose = self.state.pose
        if pose is None:
            result = self.motion.run("turn_left_large")
            self.scan_after_turn("pose_missing_turn", "turn_left_large", result)
            return
        diff = angle_diff_deg(target_yaw, pose.yaw_deg)
        if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
            return
        if bool(self.config["navigation"].get("boundary_safe_turn_enabled", True)) and self.is_near_boundary(pose):
            self.turn_toward_yaw_for_recovery(target_yaw)
            return
        result = self.motion.turn_toward(diff)
        if result is not None:
            self.scan_after_turn("turn_toward", result.key, result)

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
            target_xy = screen.observation_xy
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

    def recover_toward_field_center(self, reason: str, backoff: bool = False) -> bool:
        if not bool(self.config["navigation"].get("collision_recovery_enabled", True)):
            return False
        pose = self.state.pose
        self.active_passby_stop = None
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
            localized = self.localize_scan()
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

    def forward_clear_for_distance(self, pose: RobotPose, distance_cm: float) -> bool:
        margin = float(self.config["navigation"].get("forward_clearance_margin_cm", 10.0))
        travel = max(0.0, float(distance_cm) + margin)
        yaw = math.radians(pose.yaw_deg)
        target_xy = (
            pose.x_cm + travel * math.cos(yaw),
            pose.y_cm + travel * math.sin(yaw),
        )
        if not (0.0 <= target_xy[0] <= self.map.width_cm and 0.0 <= target_xy[1] <= self.map.height_cm):
            return False
        if not bool(self.config["navigation"].get("forward_map_obstacle_check_enabled", False)):
            return True
        return self.map.is_free_xy(target_xy) and self.map.line_clear(pose.xy(), target_xy)

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
    ) -> bool:
        target_xy = (float(target_xy[0]), float(target_xy[1]))
        if not self.map.is_free_xy(target_xy):
            adjusted = self.map.nearest_free_xy(target_xy)
            self.debug.event("navigate_xy_target_adjusted", reason=reason, target_xy=target_xy, adjusted_xy=adjusted)
            target_xy = adjusted
        radius = float(arrival_radius_cm if arrival_radius_cm is not None else self.config["navigation"]["arrival_radius_cm"])
        max_steps = int(max_steps if max_steps is not None else self.config["navigation"]["max_steps_per_target"])
        self.active_passby_stop = None
        for step in range(max_steps):
            if self.time_left_s() <= 0:
                self.debug.event("navigate_xy_stop", reason=reason, stop_reason="time_limit", target_xy=target_xy, step=step)
                return False
            if self.state.pose is None:
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed(reason + ":pose_missing")
                if self.state.pose is None:
                    return False
            pose = self.state.pose
            self.harvest_opportunistic(reason=reason + "_step")
            pose = self.state.pose
            if pose is None:
                continue
            if self.collision_recovery_pending:
                self.recover_toward_field_center(reason + ":forward_no_progress", backoff=True)
                continue
            dist = distance_xy(pose.xy(), target_xy)
            if dist <= radius:
                # Check facing direction if target_yaw_deg is specified
                if target_yaw_deg is not None:
                    yaw_diff = abs(angle_diff_deg(float(target_yaw_deg), pose.yaw_deg))
                    arrival_yaw_tolerance = float(self.config["navigation"].get("arrival_yaw_tolerance_deg", 30.0))
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
                        self.turn_toward_yaw_boundary_aware(float(target_yaw_deg))
                        continue
                self.clear_navigation_noop()
                self.debug.event("navigate_xy_arrived", reason=reason, target_xy=target_xy, distance_cm=round(dist, 1), step=step)
                return True
            if self.map.is_dangerously_close_to_wall(pose.xy(), pose.yaw_deg, float(self.config["navigation"]["safe_wall_distance_cm"])):
                self.debug.event(
                    "near_wall_recover",
                    reason=reason,
                    outward_facing=self.is_facing_outside(pose),
                    exit_dist_cm=round(self.distance_to_field_exit_ahead(pose), 1),
                )
                self.recover_toward_field_center(reason + ":near_wall_pre_forward", backoff=True)
                continue
            path = self.plan_navigation_path(pose, target_xy)
            waypoint = self.select_navigation_waypoint(pose, path, target_xy)
            desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
            diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
            self.debug.render_map(self.map, pose=pose, path=path)
            self.publish_state(path=path)
            action = self.choose_translation_action(pose, waypoint)
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
                self.clear_navigation_noop()
            else:
                self.forward_map_block_count = 0
                self.debug.event(
                    "turn_last_resort",
                    reason=reason,
                    desired_yaw=round(desired_yaw, 1),
                    diff_yaw=round(diff, 1),
                    waypoint=(round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
                )
                self.turn_toward_yaw_boundary_aware(desired_yaw)
                if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
                    self.handle_navigation_noop(
                        reason=reason,
                        waypoint=waypoint,
                        diff=diff,
                    )
                else:
                    self.clear_navigation_noop()
            if self.state.needs_relocalize():
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed(reason + ":scheduled_relocalize")
                if self.collision_recovery_pending:
                    self.recover_toward_field_center(reason + ":forward_no_progress_after_localize", backoff=True)
        self.debug.event("navigate_xy_failed", reason=reason, target_xy=target_xy, max_steps=max_steps)
        return False

    def navigate_to_screen(self, screen: Screen) -> bool:
        max_steps = int(self.config["navigation"]["max_steps_per_target"])
        self.active_passby_stop = None
        self.last_route_scan_stops = []
        for step in range(max_steps):
            if self.state.pose is None:
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed("pose_missing")
                if self.state.pose is None:
                    return False
            pose = self.state.pose
            self.harvest_opportunistic(reason="navigate_step")
            pose = self.state.pose
            if pose is None:
                continue
            if self.collision_recovery_pending:
                self.recover_toward_field_center("forward_no_progress", backoff=True)
                continue
            if self.active_passby_stop is not None:
                if not self.current_passby_stop_valid(pose, screen.screen_id):
                    self.debug.event("route_passby_stop_cleared", reason="invalid_or_behind", stop=self.active_passby_stop)
                    self.active_passby_stop = None
                else:
                    stop_dist = distance_xy(pose.xy(), tuple(self.active_passby_stop["xy"]))
                    if stop_dist <= float(self.config["navigation"].get("route_passby_arrival_radius_cm", 12.0)):
                        self.execute_passby_scan(self.active_passby_stop)
                        self.active_passby_stop = None
                        continue
            goal = screen.observation_xy
            dist = distance_xy(pose.xy(), goal)
            if dist <= float(self.config["navigation"]["arrival_radius_cm"]):
                # Check facing direction: robot must be roughly facing the screen
                desired_yaw = screen.interaction_yaw_deg
                yaw_diff = abs(angle_diff_deg(desired_yaw, pose.yaw_deg))
                arrival_yaw_tolerance = float(self.config["navigation"].get("arrival_yaw_tolerance_deg", 30.0))
                if yaw_diff > arrival_yaw_tolerance:
                    self.debug.event(
                        "arrived_position_wrong_yaw",
                        screen_id=screen.screen_id,
                        distance_cm=round(dist, 1),
                        desired_yaw=round(desired_yaw, 1),
                        current_yaw=round(pose.yaw_deg, 1),
                        yaw_diff=round(yaw_diff, 1),
                        tolerance=arrival_yaw_tolerance,
                    )
                    self.turn_toward_yaw_boundary_aware(desired_yaw)
                    continue
                self.clear_navigation_noop()
                self.debug.event("arrived_viewpoint", screen_id=screen.screen_id, distance_cm=round(dist, 1), yaw_diff=round(yaw_diff, 1))
                return True
            if self.map.is_dangerously_close_to_wall(pose.xy(), pose.yaw_deg, float(self.config["navigation"]["safe_wall_distance_cm"])):
                self.debug.event(
                    "near_wall_recover",
                    outward_facing=self.is_facing_outside(pose),
                    exit_dist_cm=round(self.distance_to_field_exit_ahead(pose), 1),
                )
                self.recover_toward_field_center("near_wall_pre_forward", backoff=True)
                continue
            path = self.plan_navigation_path(pose, goal)
            waypoint = self.select_navigation_waypoint(pose, path, goal)
            if self.active_passby_stop is None:
                self.active_passby_stop = self.plan_next_passby_stop(pose, path, screen.screen_id)
                if self.active_passby_stop is not None:
                    self.last_passby_plan = dict(self.active_passby_stop)
                    self.debug.event(
                        "route_passby_stop_selected",
                        target_screen_id=screen.screen_id,
                        stop=self.active_passby_stop,
                    )
            if self.active_passby_stop is not None:
                waypoint = tuple(self.active_passby_stop["xy"])
            desired_yaw = math.degrees(math.atan2(waypoint[1] - pose.y_cm, waypoint[0] - pose.x_cm))
            diff = angle_diff_deg(desired_yaw, pose.yaw_deg)
            self.debug.render_map(self.map, pose=pose, path=path, target_screen=screen, scan_stops=self.last_route_scan_stops)
            self.publish_state(screen, path)
            action = self.choose_translation_action(pose, waypoint)
            if action is not None:
                status = self.execute_translation_action(
                    action,
                    pose,
                    waypoint,
                    dist,
                    {"screen_id": screen.screen_id, "diff_yaw": round(diff, 1)},
                )
                if status == "recovered":
                    self.clear_navigation_noop()
                    continue
                self.clear_navigation_noop()
            else:
                self.forward_map_block_count = 0
                self.debug.event(
                    "turn_last_resort",
                    screen_id=screen.screen_id,
                    desired_yaw=round(desired_yaw, 1),
                    diff_yaw=round(diff, 1),
                    waypoint=(round(float(waypoint[0]), 1), round(float(waypoint[1]), 1)),
                )
                self.turn_toward_yaw_boundary_aware(desired_yaw)
                if abs(diff) <= float(self.config["navigation"].get("turn_tolerance_deg", 20.0)):
                    self.handle_navigation_noop(
                        reason="navigate_screen",
                        waypoint=waypoint,
                        diff=diff,
                        extra={"screen_id": screen.screen_id},
                    )
                else:
                    self.clear_navigation_noop()
            if self.state.needs_relocalize():
                if not self.localize_scan():
                    self.recover_from_no_tag_if_needed("scheduled_relocalize")
                if self.collision_recovery_pending:
                    self.recover_toward_field_center("forward_no_progress_after_localize", backoff=True)
        self.debug.event("navigate_failed", screen_id=screen.screen_id)
        return False

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
        data = {
            "mode": self.args.mode,
            "target_flower": self.target_flower,
            "time_left_s": round(self.time_left_s(), 1),
            "completed_count": self.map.completed_count(),
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
            "initial_discovery_scan": self.last_initial_discovery_scan,
            "last_passby_plan": self.last_passby_plan,
            "active_passby_stop": self.active_passby_stop,
            "route_scan_stops": self.last_route_scan_stops,
            "localization_health": {
                "seconds_since_any_tag": round(now_s() - self.last_any_tag_seen_s, 2),
                "seconds_since_localize": None if self.last_localize_success_s <= 0 else round(now_s() - self.last_localize_success_s, 2),
                "consecutive_no_tag_scans": self.consecutive_no_tag_scans,
                "consecutive_localize_failures": self.consecutive_localize_failures,
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
            },
            "interaction_log_path": self.interaction_audit_path,
            "screens": {sid: screen.as_dict() for sid, screen in sorted(self.map.screens.items())},
        }
        self.debug.state(data)
        self.debug.render_map(self.map, pose=self.state.pose, path=path, target_screen=target_screen, scan_stops=self.last_route_scan_stops)

    def close(self):
        try:
            self.hardware.close()
        finally:
            if self.interaction_audit_file is not None:
                self.interaction_audit_file.close()
            self.camera.release()
            self.debug.close()
