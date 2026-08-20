from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.localizer import Localizer
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import ActionResult, Confidence, RobotPose, ScreenStatus
from robot_tonypi.motion import MotionController, RobotState
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.utils import now_s


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))


def adaptive_manager(confidence=Confidence.HIGH):
    manager = TaskManager.__new__(TaskManager)
    manager.config = load_config(None)
    manager.debug = DebugStub()
    manager.state = RobotState(manager.config)
    manager.state.set_pose(RobotPose(150.0, 150.0, 0.0, confidence, "VISION", now_s()))
    manager.last_localize_success_s = now_s()
    manager.consecutive_localize_failures = 0
    manager.current_target_screen_id = 1
    return manager


class MotionAccountingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(None)
        self.state = RobotState(self.config)
        self.state.set_pose(RobotPose(0.0, 0.0, 0.0, Confidence.HIGH, "VISION", now_s()))

    def test_times_override_counts_actual_cycles_not_calls(self):
        result = ActionResult("forward_fast", "forward", 5, 0.0, model_forward_cm=17.5, executed_times=5)
        self.state.apply_action_result(result)
        self.assertEqual(self.state.actions_since_localize, 5)
        self.assertEqual(self.state.motion_uncertainty, 3.0)

    def test_partial_failure_counts_and_models_only_completed_prefix(self):
        result = ActionResult(
            "forward_fast", "forward", 5, 0.0,
            model_forward_cm=17.5, ok=False, error="stopped", executed_times=2,
        )
        self.state.apply_action_result(result)
        self.assertEqual(self.state.actions_since_localize, 2)
        self.assertAlmostEqual(self.state.pose.x_cm, 7.0)
        self.assertEqual(self.state.pose.confidence, Confidence.LOW)

    def test_failed_unknown_batch_does_not_claim_requested_cycles(self):
        result = ActionResult("forward_fast", "forward", 5, 0.0, model_forward_cm=17.5, ok=False)
        self.state.apply_action_result(result)
        self.assertEqual(self.state.actions_since_localize, 0)
        self.assertEqual(self.state.pose.x_cm, 0.0)

    def test_strafe_and_turn_uncertainty_exceed_forward(self):
        forward = RobotState(self.config)
        strafe = RobotState(self.config)
        turn = RobotState(self.config)
        for state in (forward, strafe, turn):
            state.set_pose(RobotPose(0, 0, 0, Confidence.HIGH, "VISION", now_s()))
        forward.apply_action_result(ActionResult("f", "f", 1, 0, model_forward_cm=3.5, executed_times=1))
        strafe.apply_action_result(ActionResult("s", "s", 1, 0, model_lateral_cm=4.0, executed_times=1))
        turn.apply_action_result(ActionResult("t", "t", 1, 0, model_yaw_deg=7.5, executed_times=1))
        self.assertGreater(strafe.motion_uncertainty, forward.motion_uncertainty)
        self.assertGreater(turn.motion_uncertainty, strafe.motion_uncertainty)

    def test_large_turn_uncertainty_exceeds_regular_turn(self):
        regular = RobotState(self.config)
        large = RobotState(self.config)
        for state in (regular, large):
            state.set_pose(RobotPose(0, 0, 0, Confidence.HIGH, "VISION", now_s()))
        regular.apply_action_result(ActionResult("t", "t", 1, 0, model_yaw_deg=7.5, executed_times=1))
        large.apply_action_result(ActionResult("lt", "lt", 1, 0, model_yaw_deg=45.0, executed_times=1))
        self.assertGreater(large.motion_uncertainty, regular.motion_uncertainty)

    def test_visual_pose_resets_cycles_and_uncertainty(self):
        self.state.apply_action_result(ActionResult("f", "f", 3, 0, model_forward_cm=10.5, executed_times=3))
        self.state.set_pose(RobotPose(9.8, 0, 0, Confidence.HIGH, "VISION", now_s()))
        self.assertEqual(self.state.actions_since_localize, 0)
        self.assertEqual(self.state.motion_uncertainty, 0.0)

    def test_high_confidence_five_cycle_batch_does_not_hit_relocalize_threshold(self):
        self.state.apply_action_result(
            ActionResult("f", "f", 5, 0, model_forward_cm=17.5, executed_times=5)
        )
        self.assertEqual(self.state.pose.confidence, Confidence.HIGH)
        self.assertFalse(self.state.needs_relocalize())

    def test_confidence_specific_relocalize_limits_are_configured(self):
        nav = self.config["navigation"]
        self.assertEqual(nav["relocalize_after_actions_high"], 8)
        self.assertEqual(nav["relocalize_after_actions_medium"], 4)
        self.assertEqual(nav["relocalize_after_actions_low"], 1)

    def test_motion_controller_logs_requested_and_actual(self):
        hardware = SimpleNamespace(
            run_action=lambda key, times_override=None: ActionResult(
                key, key, int(times_override), 0.0, model_forward_cm=7.0,
                ok=False, executed_times=1,
            )
        )
        debug = DebugStub()
        MotionController(hardware, self.state, debug).run("forward_fast", times_override=2)
        action = debug.events[-1][1]
        self.assertEqual(action["requested_action_cycles"], 2)
        self.assertEqual(action["actual_action_cycles"], 1)


class AdaptiveBatchTests(unittest.TestCase):
    def test_high_confidence_open_space_uses_normal_six_action_budget(self):
        manager = adaptive_manager(Confidence.HIGH)
        cycles, _ = manager.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)
        self.assertEqual(cycles, 6)

    def test_medium_and_low_confidence_shorten_batch(self):
        medium = adaptive_manager(Confidence.MEDIUM)
        low = adaptive_manager(Confidence.LOW)
        self.assertEqual(medium.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0], 4)
        self.assertEqual(low.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0], 1)

    def test_strafe_and_turn_caps_are_lower_than_forward(self):
        manager = adaptive_manager()
        forward = manager.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0]
        strafe = manager.select_adaptive_action_batch("strafe", 8, 3.0, 100, 100)[0]
        turn = manager.select_adaptive_action_batch("turn", 8, 7.5, 100, 100)[0]
        self.assertEqual((forward, strafe, turn), (6, 4, 2))

    def test_reverse_batch_cap_and_uncertainty_are_configured(self):
        manager = adaptive_manager()
        reverse = manager.select_adaptive_action_batch("reverse", 8, 2.5, 100, 100)[0]
        self.assertEqual(reverse, 6)
        self.assertGreater(
            manager.config["navigation"]["reverse_uncertainty_per_cycle"],
            manager.config["navigation"]["forward_uncertainty_per_cycle"],
        )

    def test_near_wall_recovery_and_near_target_force_short_batches(self):
        manager = adaptive_manager()
        self.assertEqual(manager.select_adaptive_action_batch("forward", 6, 3.5, 100, 100, near_wall=True)[0], 1)
        self.assertEqual(manager.select_adaptive_action_batch("strafe", 3, 3.0, 20, 20)[0], 1)
        self.assertEqual(manager.select_adaptive_action_batch("turn", 2, 7.5, 100, 100, recovery=True)[0], 1)

    def test_batch_cannot_cross_waypoint_or_target(self):
        manager = adaptive_manager()
        cycles, _ = manager.select_adaptive_action_batch("forward", 6, 3.5, 8.0, 100.0)
        self.assertEqual(cycles, 2)

    def test_localization_failure_forces_one_cycle(self):
        manager = adaptive_manager()
        manager.consecutive_localize_failures = 1
        cycles, _ = manager.select_adaptive_action_batch("forward", 6, 3.5, 100, 100)
        self.assertEqual(cycles, 1)

    def test_completed_batch_recenters_relocalizes_and_requests_replan(self):
        manager = adaptive_manager()
        centered = []
        manager.args = SimpleNamespace(dry_run=False)
        manager.hardware = SimpleNamespace(center_head=lambda: centered.append(True))

        def localize(*args, **kwargs):
            manager.state.set_pose(RobotPose(160.0, 150.0, 0.0, Confidence.HIGH, "VISION", now_s()))
            return True

        manager.localize_scan = localize
        before = TaskManager.copy_pose(manager.state.pose)
        result = ActionResult("forward_fast", "forward", 6, 0.0, model_forward_cm=21.0, executed_times=6)
        manager.state.apply_action_result(result)
        self.assertTrue(manager.post_action_relocalize("test", before, result, (200.0, 150.0)))
        self.assertEqual(len(centered), 1)
        self.assertTrue(manager.pending_post_action_replan)
        names = [name for name, _ in manager.debug.events]
        self.assertIn("post_action_relocalize", names)
        self.assertIn("post_action_replan", names)

    def test_small_normal_batch_skips_localization_but_still_replans(self):
        manager = adaptive_manager()
        centered = []
        localized = []
        manager.args = SimpleNamespace(dry_run=False)
        manager.hardware = SimpleNamespace(center_head=lambda: centered.append(True))
        manager.localize_scan = lambda *args, **kwargs: localized.append(True) or True
        before = TaskManager.copy_pose(manager.state.pose)
        result = ActionResult("forward_fast", "forward", 2, 0.0, model_forward_cm=7.0, executed_times=2)
        manager.state.apply_action_result(result)
        self.assertTrue(manager.post_action_relocalize("test", before, result, (200.0, 150.0)))
        self.assertEqual(centered, [])
        self.assertEqual(localized, [])
        self.assertEqual(manager.state.actions_since_localize, 2)
        decision = [data for name, data in manager.debug.events if name == "relocalization_decision"][-1]
        self.assertEqual(decision["decision"], "continue_dead_reckoning")
        self.assertTrue(manager.pending_post_action_replan)

    def test_phase_specific_action_budgets(self):
        manager = adaptive_manager()
        manager.state.actions_since_localize = 3
        manager.state.motion_uncertainty = 1.0
        direct = manager.adaptive_relocalization_decision("target_direct_approach", emit=False)
        staging = manager.adaptive_relocalization_decision("staging", emit=False)
        recovery = manager.adaptive_relocalization_decision("recovery", recovery=True, emit=False)
        self.assertEqual(direct["action_budget"], 3)
        self.assertEqual(direct["decision"], "relocalize_now")
        self.assertEqual(staging["action_budget"], 5)
        self.assertEqual(staging["decision"], "continue_dead_reckoning")
        self.assertEqual(recovery["action_budget"], 4)

    def test_uncertainty_limit_forces_early_localization(self):
        manager = adaptive_manager()
        manager.state.actions_since_localize = 1
        manager.state.motion_uncertainty = 6.0
        decision = manager.adaptive_relocalization_decision("normal", emit=False)
        self.assertEqual(decision["decision"], "relocalize_now")
        self.assertEqual(decision["reason"], "motion_uncertainty_limit")

    def test_low_confidence_and_obstacle_tight_force_localization(self):
        manager = adaptive_manager()
        manager.state.pose.confidence = Confidence.LOW
        low = manager.adaptive_relocalization_decision("normal", emit=False)
        manager.state.pose.confidence = Confidence.HIGH
        tight = manager.adaptive_relocalization_decision(
            "normal", obstacle_tight=True, emit=False
        )
        self.assertEqual(low["reason"], "pose_confidence_low")
        self.assertEqual(tight["reason"], "obstacle_tight_navigation")

    def test_pose_conflict_is_diagnostic_but_large_turn_forces_localization(self):
        manager = adaptive_manager()
        manager.last_localization_pose_conflict = True
        conflict = manager.adaptive_relocalization_decision("normal", emit=False)
        manager.last_localization_pose_conflict = False
        manager.state.actions_since_localize = 1
        large = manager.adaptive_relocalization_decision(
            "normal",
            last_action="turn_left_large",
            action_result=ActionResult(
                "turn_left_large", "turn", 1, 0.0,
                model_yaw_deg=45.0, executed_times=1,
            ),
            emit=False,
        )
        self.assertEqual(conflict["decision"], "continue_dead_reckoning")
        self.assertEqual(large["reason"], "large_turn")

    def test_new_large_turn_requires_one_relocalization(self):
        manager = adaptive_manager()
        manager.state.actions_since_localize = 1
        decision = manager.adaptive_relocalization_decision(
            "normal", last_action="turn_left_large", emit=False
        )
        self.assertEqual(decision["decision"], "relocalize_now")
        self.assertEqual(decision["reason"], "large_turn")
        self.assertTrue(decision["large_turn_relocalization_pending"])

    def test_successful_relocalization_consumes_historical_large_turn(self):
        manager = adaptive_manager()
        manager.last_motion_action = "turn_left_large"
        manager.state.set_pose(RobotPose(
            150.0, 150.0, 0.0, Confidence.HIGH, "VISION", now_s()
        ))
        decision = manager.adaptive_relocalization_decision("normal", emit=False)
        self.assertEqual(manager.state.actions_since_localize, 0)
        self.assertEqual(manager.state.motion_uncertainty, 0.0)
        self.assertEqual(decision["decision"], "continue_dead_reckoning")
        self.assertNotEqual(decision["reason"], "large_turn")
        self.assertFalse(decision["large_turn_relocalization_pending"])

    def test_later_large_turn_can_trigger_relocalization_again(self):
        manager = adaptive_manager()
        manager.last_motion_action = "turn_left_large"
        manager.state.set_pose(RobotPose(
            150.0, 150.0, 0.0, Confidence.HIGH, "VISION", now_s()
        ))
        first_consumed = manager.adaptive_relocalization_decision("normal", emit=False)
        self.assertNotEqual(first_consumed["reason"], "large_turn")

        new_turn = ActionResult(
            "turn_left_large", "turn", 1, 0.0,
            model_yaw_deg=45.0, executed_times=1,
        )
        manager.state.apply_action_result(new_turn)
        next_decision = manager.adaptive_relocalization_decision(
            "normal",
            last_action="turn_left_large",
            action_result=new_turn,
            emit=False,
        )
        self.assertGreater(manager.state.actions_since_localize, 0)
        self.assertEqual(next_decision["decision"], "relocalize_now")
        self.assertEqual(next_decision["reason"], "large_turn")

    def test_visual_odometry_conflict_is_diagnostic_only(self):
        manager = adaptive_manager()
        manager.localizer = SimpleNamespace(tag_area=lambda tag: tag.area)
        manager.last_localization_tag_count = 0
        manager.last_localization_quality = "NONE"
        manager.last_localization_pose_conflict = False
        pose = RobotPose(180.0, 150.0, 35.0, Confidence.HIGH, "VISION_TAG_1", now_s())
        prior = RobotPose(150.0, 150.0, 0.0, Confidence.MEDIUM, "DEAD_RECKONING", now_s())
        detail = manager.assess_visual_localization(
            pose,
            [SimpleNamespace(tag_id=1, area=800.0)],
            prior,
        )
        self.assertTrue(detail["visual_odometry_conflict"])
        self.assertEqual(pose.confidence, Confidence.HIGH)
        self.assertEqual(manager.last_localization_tag_count, 1)


class LocalizationScanBudgetTests(unittest.TestCase):
    def manager(self, outcomes):
        manager = adaptive_manager()
        manager.args = SimpleNamespace(dry_run=False)
        manager.hardware = SimpleNamespace(center_head=lambda: None)
        manager.publish_state = lambda *args, **kwargs: None
        manager.clear_turn_progress_watchdog = lambda *args, **kwargs: None
        manager.evaluate_pending_progress = lambda pose: None
        manager.consecutive_no_tag_scans = 0
        manager.localization_failures = 0
        manager.observe_transit_bindings = lambda frame, tags, annotated, pan, reason: annotated
        manager.debug.save_image = lambda *args, **kwargs: None
        pans = []
        manager.capture_with_tags = lambda pan: (pans.append(pan) or object(), [])
        queue = list(outcomes)
        manager.localizer = SimpleNamespace(
            estimate_from_frame=lambda *args, **kwargs: (
                queue.pop(0) if queue else None, object()
            ),
            tag_area=lambda tag: 0.0,
        )
        return manager, pans

    def test_first_localization_scans_configured_pans_until_pose(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([None, None, pose])
        self.assertTrue(manager.localize_scan(reason="post_action", allow_failure_escalation=False))
        self.assertEqual(pans, [100.0, 135.0, 65.0])

    def test_failed_localization_always_completes_configured_scan(self):
        manager, pans = self.manager([None])
        self.assertFalse(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(pans, [100.0, 135.0, 65.0, 145.0, 55.0])

    def test_pan_scan_stops_immediately_when_localized(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([None, pose])
        self.assertTrue(manager.localize_scan(allow_pan_search=True))
        self.assertEqual(pans, [100.0, 135.0])

    def test_required_target_search_continues_after_other_tag_localizes(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([pose, pose, pose])
        tags_by_pan = {
            100: [SimpleNamespace(tag_id=15, area=800.0)],
            135: [SimpleNamespace(tag_id=15, area=800.0)],
            65: [SimpleNamespace(tag_id=24, area=800.0)],
        }
        manager.capture_with_tags = lambda pan: (
            pans.append(pan) or object(), tags_by_pan[int(pan)]
        )
        manager.localizer.tag_area = lambda tag: float(tag.area)
        manager.update_dynamic_obstacles = lambda *args, **kwargs: None

        def observe(frame, tags, annotated, pan, reason):
            manager.last_transit_binding_screen_ids = {
                int(tag.tag_id) for tag in tags
            }
            return annotated

        manager.observe_transit_bindings = observe

        self.assertTrue(manager.localize_scan(
            reason="nfc_retry_visual_check",
            allow_pan_search=True,
            required_target_screen_id=24,
        ))
        self.assertEqual(pans, [100.0, 135.0, 65.0])
        stopped = [
            data for name, data in manager.debug.events
            if name == "pan_search_stopped_on_success"
        ]
        self.assertEqual(stopped[-1]["successful_pan"], 65.0)
        self.assertEqual(stopped[-1]["stop_condition"], "required_target_reacquired")

    def test_normal_localization_still_stops_on_any_valid_tag(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([pose, pose])
        tag = SimpleNamespace(tag_id=15, area=800.0)
        manager.capture_with_tags = lambda pan: (pans.append(pan) or object(), [tag])
        manager.localizer.tag_area = lambda item: float(item.area)
        manager.update_dynamic_obstacles = lambda *args, **kwargs: None

        self.assertTrue(manager.localize_scan(reason="navigation"))
        self.assertEqual(pans, [100.0])

    def test_scan_recenters_head_after_off_center_success(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, _ = self.manager([None, None, pose])
        recentered = []
        manager.hardware = SimpleNamespace(center_head=lambda: recentered.append(True))
        self.assertTrue(manager.localize_scan())
        self.assertEqual(recentered, [True])

    def test_visibility_recovery_never_calls_full_navigate(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        import ast
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "recover_target_visibility")
        calls = {getattr(node.func, "attr", "") for node in ast.walk(fn) if isinstance(node, ast.Call)}
        self.assertNotIn("navigate_to_screen", calls)


class NoTagRecoverySequenceTests(unittest.TestCase):
    def manager(self, localization_results):
        manager = adaptive_manager()
        manager.args = SimpleNamespace(dry_run=False)
        manager.current_target_screen_id = 17
        manager.current_target_goal = SimpleNamespace(generation_id=9)
        manager.consecutive_no_tag_scans = int(
            manager.config["navigation"]["no_tag_recovery_failures"]
        )
        manager.consecutive_localize_failures = manager.consecutive_no_tag_scans
        manager.last_no_tag_recovery_s = 0.0
        manager.last_any_tag_seen_s = now_s() - 10.0
        manager.no_tag_recovery_active = False
        manager.no_tag_recovery_exhausted = False
        manager.localization_recovery_exhausted = False
        manager.last_localization_attempt_result = "no_tag"
        manager.time_left_s = lambda: 100.0
        manager.is_facing_outside = lambda pose: False
        manager.state.pose = None
        calls = []
        queue = list(localization_results)

        def localize_scan(**kwargs):
            calls.append(kwargs)
            success = bool(queue.pop(0)) if queue else False
            manager.last_localization_attempt_result = (
                "accepted_visual_pose" if success else "no_tag"
            )
            if success:
                manager.consecutive_no_tag_scans = 0
                manager.consecutive_localize_failures = 0
            return success

        manager.localize_scan = localize_scan
        actions = []

        manager.hardware = SimpleNamespace(center_head=lambda: None)
        manager.motion = SimpleNamespace(
            run=lambda action, times_override=1: actions.append(
                (action, times_override)
            ) or True
        )
        return manager, calls, actions

    def test_threshold_enters_bounded_backoff_scan_recovery(self):
        manager, calls, actions = self.manager([False] * 8)
        self.assertFalse(manager.recover_from_no_tag_if_needed("test"))
        self.assertTrue(calls)
        self.assertEqual(len(calls), 3)
        self.assertEqual(actions, [("back_fast", 9)] * 3)
        names = [name for name, _ in manager.debug.events]
        self.assertIn("no_tag_recovery_triggered", names)
        self.assertIn("recovery_done", names)

    def test_successful_full_pan_skips_startup_actions(self):
        manager, calls, actions = self.manager([True])
        self.assertTrue(manager.recover_from_no_tag_if_needed("test"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(actions, [("back_fast", 9)])
        self.assertEqual(manager.consecutive_no_tag_scans, 0)

    def test_failed_full_pan_executes_configured_action_then_full_pan(self):
        manager, calls, actions = self.manager([False, True])
        self.assertTrue(manager.recover_from_no_tag_if_needed("test"))
        self.assertEqual(actions, [("back_fast", 9), ("back_fast", 9)])
        self.assertEqual(len(calls), 2)

    def test_success_after_nth_action_stops_and_preserves_target(self):
        manager, calls, actions = self.manager([False, False, True])
        original_goal = manager.current_target_goal
        self.assertTrue(manager.recover_from_no_tag_if_needed("test"))
        self.assertEqual(len(actions), 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(manager.current_target_screen_id, 17)
        self.assertIs(manager.current_target_goal, original_goal)

    def test_exhausted_sequence_enters_higher_recovery_and_explicit_failure(self):
        manager, calls, actions = self.manager([False] * 10)
        self.assertFalse(manager.recover_from_no_tag_if_needed("test"))
        self.assertEqual(len(actions), 3)
        self.assertFalse(manager.no_tag_recovery_exhausted)
        self.assertFalse(manager.localization_recovery_exhausted)

    def test_startup_and_runtime_use_same_search_sequence_helper(self):
        manager, _, _ = self.manager([])
        used = []
        manager.run_localization_search_sequence = lambda **kwargs: used.append(kwargs) or True
        self.assertTrue(manager.initial_localize())
        self.assertEqual(used[0]["reason"], "initial_localize")
        self.assertFalse(used[0]["runtime_safety"])

    def test_pose_unavailable_failures_use_same_full_pan_recovery(self):
        manager, calls, actions = self.manager([True])
        manager.consecutive_no_tag_scans = 0
        manager.consecutive_localize_failures = 2
        manager.last_localization_attempt_result = "pose_unavailable_with_tags"
        self.assertFalse(manager.recover_from_localization_failure_if_needed("test"))
        self.assertEqual(len(calls), 0)
        self.assertEqual(actions, [])

    def test_failure_threshold_counts_no_pose_even_when_tag_is_visible(self):
        manager, _, _ = self.manager([])
        manager.consecutive_localize_failures = 0
        manager.consecutive_no_tag_scans = 0
        manager.record_localization_failure(
            "pose_unavailable_with_tags", saw_any_tag=True, reason="first"
        )
        self.assertFalse(manager.localization_failure_recovery_needed())
        manager.record_localization_failure(
            "pose_unavailable_with_tags", saw_any_tag=True, reason="second"
        )
        self.assertFalse(manager.localization_failure_recovery_needed())
        self.assertEqual(manager.consecutive_no_tag_scans, 0)

    def test_all_tag_visible_full_pans_still_execute_body_recovery(self):
        manager, calls, actions = self.manager([False] * 10)
        manager.consecutive_no_tag_scans = 0
        manager.consecutive_localize_failures = 2
        manager.last_localization_attempt_result = "pose_unavailable_with_tags"

        def failed_with_tags(**kwargs):
            calls.append(kwargs)
            manager.last_localization_attempt_result = "pose_unavailable_with_tags"
            return False

        manager.localize_scan = failed_with_tags
        self.assertFalse(manager.recover_from_localization_failure_if_needed("test"))
        self.assertEqual(actions, [])

    def test_large_turn_pending_is_consumed_only_after_recovery_accepts_pose(self):
        manager, calls, actions = self.manager([False, True])
        manager.last_motion_action = "turn_right_large"
        manager.state.actions_since_localize = 2
        manager.state.motion_uncertainty = 5.2
        manager.consecutive_no_tag_scans = 2
        manager.consecutive_localize_failures = 2
        manager.last_localization_attempt_result = "pose_unavailable_with_tags"
        queue = [False, True]

        def localize_scan(**kwargs):
            calls.append(kwargs)
            if not queue.pop(0):
                manager.last_localization_attempt_result = "pose_unavailable_with_tags"
                return False
            manager.accept_visual_localization(RobotPose(
                151.0, 150.0, -90.0, Confidence.HIGH, "VISION_TAG_32", now_s()
            ), "test_recovery")
            return True

        manager.localize_scan = localize_scan
        self.assertTrue(manager.recover_from_localization_failure_if_needed("test"))
        decision = manager.adaptive_relocalization_decision(
            "normal", last_action="turn_right_large", emit=False
        )
        self.assertEqual(manager.state.actions_since_localize, 0)
        self.assertEqual(manager.state.motion_uncertainty, 0.0)
        self.assertFalse(decision["large_turn_relocalization_pending"])
        self.assertEqual(manager.current_target_screen_id, 17)
        self.assertEqual(manager.current_target_goal.generation_id, 9)


class LocalizationStateResetTests(unittest.TestCase):
    def scan_manager(self, outcomes, tags=None):
        helper = LocalizationScanBudgetTests()
        manager, pans = helper.manager(outcomes)
        if tags is not None:
            manager.capture_with_tags = lambda pan: (
                pans.append(pan) or object(), tags
            )
            manager.localizer.tag_area = lambda tag: float(tag.area)
            manager.update_dynamic_obstacles = lambda *args, **kwargs: None
        manager.last_localization_attempt_result = "unknown"
        manager.no_tag_recovery_exhausted = False
        manager.localization_recovery_exhausted = False
        return manager, pans

    def test_failed_localization_preserves_motion_accounting(self):
        manager, _ = self.scan_manager([None])
        manager.state.actions_since_localize = 3
        manager.state.motion_uncertainty = 2.5
        self.assertFalse(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(manager.state.actions_since_localize, 3)
        self.assertEqual(manager.state.motion_uncertainty, 2.5)
        names = [name for name, _ in manager.debug.events]
        self.assertNotIn("localization_state_reset", names)

    def test_conflicting_visual_pose_is_accepted_and_resets_dead_reckoning(self):
        visual = RobotPose(
            190.0, 150.0, 45.0, Confidence.HIGH, "VISION_TAG_1", now_s()
        )
        tag = SimpleNamespace(tag_id=1, area=800.0)
        manager, _ = self.scan_manager([visual], tags=[tag])
        manager.state.pose.source = "DEAD_RECKONING"
        manager.state.actions_since_localize = 4
        manager.state.motion_uncertainty = 3.5
        self.assertTrue(manager.localize_scan(allow_failure_escalation=False))
        self.assertIs(manager.state.pose, visual)
        self.assertEqual(manager.state.actions_since_localize, 0)
        self.assertEqual(manager.state.motion_uncertainty, 0.0)
        self.assertEqual(
            manager.last_localization_attempt_result,
            "accepted_visual_pose",
        )
        names = [name for name, _ in manager.debug.events]
        self.assertIn("visual_dead_reckoning_conflict_observed", names)

    def test_pose_unavailable_with_tags_preserves_motion_accounting(self):
        tag = SimpleNamespace(tag_id=26, area=800.0, center=(320, 220))
        manager, _ = self.scan_manager([None], tags=[tag])
        manager.state.actions_since_localize = 3
        manager.state.motion_uncertainty = 2.5
        self.assertFalse(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(
            manager.last_localization_attempt_result,
            "pose_unavailable_with_tags",
        )
        self.assertEqual(manager.state.actions_since_localize, 3)
        self.assertEqual(manager.state.motion_uncertainty, 2.5)

    def test_accepted_visual_pose_resets_motion_accounting(self):
        visual = RobotPose(
            151.0, 150.0, 1.0, Confidence.HIGH, "VISION_TAG_1", now_s()
        )
        tag = SimpleNamespace(tag_id=1, area=800.0)
        manager, _ = self.scan_manager([visual], tags=[tag])
        manager.state.pose.source = "DEAD_RECKONING"
        manager.state.actions_since_localize = 2
        manager.state.motion_uncertainty = 1.5
        self.assertTrue(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(manager.state.actions_since_localize, 0)
        self.assertEqual(manager.state.motion_uncertainty, 0.0)
        reset = next(
            data for name, data in manager.debug.events
            if name == "localization_state_reset"
        )
        self.assertEqual(reset["actions_before_reset"], 2)

    def test_no_motion_after_success_legitimately_keeps_zero_on_failure(self):
        visual = RobotPose(
            151.0, 150.0, 1.0, Confidence.HIGH, "VISION_TAG_1", now_s()
        )
        manager, _ = self.scan_manager([visual, None])
        self.assertTrue(manager.localize_scan(allow_failure_escalation=False))
        self.assertFalse(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(manager.state.actions_since_localize, 0)
        self.assertEqual(manager.state.motion_uncertainty, 0.0)

    def test_full_pan_pose_unavailable_then_accepts_next_angle(self):
        unavailable_tag = SimpleNamespace(tag_id=26, area=800.0, center=(320, 220))
        accepted = RobotPose(
            151.0, 150.0, 0.0, Confidence.HIGH, "VISION_TAG_26", now_s()
        )
        manager, pans = self.scan_manager(
            [None, accepted], tags=[unavailable_tag]
        )
        self.assertTrue(manager.localize_scan(allow_pan_search=True))
        self.assertEqual(pans, [100.0, 135.0])
        self.assertEqual(manager.last_localization_attempt_result, "accepted_visual_pose")


class LocalizerDiagnosticTests(unittest.TestCase):
    def localizer(self, outcomes):
        localizer = Localizer.__new__(Localizer)
        localizer.min_id = 1
        localizer.max_id = 36
        localizer.last_estimation_diagnostics = {}
        localizer.tag_area = lambda tag: float(tag.area)
        queue = dict(outcomes)

        def solve(tag, frame):
            outcome = queue[int(tag.tag_id)]
            if isinstance(outcome, RobotPose):
                return outcome, "accepted", "accepted_visual_pose"
            return None, outcome[0], outcome[1]

        localizer._solve_tag_pose_detailed = solve
        return localizer

    @staticmethod
    def tag(tag_id, area):
        return SimpleNamespace(
            tag_id=tag_id,
            area=area,
            center=np.array([100.0 + tag_id, 120.0]),
            corners=np.zeros((4, 2), dtype=np.float64),
        )

    def test_failed_tag_does_not_prevent_second_tag_success(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION_TAG_2", now_s())
        localizer = self.localizer({
            1: ("solve_pnp", "pnp_failed"),
            2: pose,
        })
        result, _ = localizer.estimate_from_frame(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [self.tag(1, 900), self.tag(2, 800)],
        )
        self.assertIs(result, pose)
        self.assertEqual(localizer.last_estimation_diagnostics["accepted_tag_id"], 2)
        self.assertEqual(
            localizer.last_estimation_diagnostics["rejected_tags"][0]["reason"],
            "pnp_failed",
        )

    def test_all_failed_tags_report_structured_rejection_summary(self):
        localizer = self.localizer({
            1: ("quality_gate", "edge_margin"),
            2: ("map_bounds", "pose_out_of_bounds"),
        })
        result, _ = localizer.estimate_from_frame(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [self.tag(1, 900), self.tag(2, 800)],
        )
        self.assertIsNone(result)
        detail = localizer.last_estimation_diagnostics
        self.assertEqual(detail["result"], "pose_unavailable_with_tags")
        self.assertEqual(detail["detected_tag_ids"], [1, 2])
        self.assertEqual(detail["candidate_localization_tag_ids"], [1, 2])
        self.assertEqual(
            [item["reason"] for item in detail["rejected_tags"]],
            ["edge_margin", "pose_out_of_bounds"],
        )


class PlannerPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(None)
        self.model = MapModel(load_tag_pos(), self.config)

    def test_configured_turn_cost_is_in_transition_g_cost(self):
        actions = self.model.action_planner_actions(self.config["navigation"], self.config["motion"])
        turn = next(item for item in actions if item["name"] == "turn_left_small")
        state = (*self.model.grid_pos((150.0, 150.0)), 0, 0, 0)
        _, cost = self.model.action_planner_transition(state, turn, 15.0, 24, 85.0, 1.0)
        nav = self.config["navigation"]
        expected = (
            nav["action_planner_turn_fixed_cost_cm"]
            + 7.5 * nav["action_planner_turn_cost_cm_per_deg"]
            + nav["action_planner_in_place_turn_penalty_cm"]
        )
        self.assertEqual(cost, expected)

    def translation_manager(self, confidence=Confidence.HIGH):
        manager = adaptive_manager(confidence)
        manager.map = MapModel(load_tag_pos(), manager.config)
        manager.motion = SimpleNamespace(
            forward_cycles_for_distance=lambda distance: min(8, max(1, int(abs(distance) / 3.5))),
            lateral_cycles_for_distance=lambda distance: min(4, max(1, int(abs(distance) / 4.0))),
            reverse_cycles_for_distance=lambda distance: min(6, max(1, int(abs(distance) / 2.5))),
        )
        return manager

    def test_target_directly_behind_prefers_reverse_without_turn(self):
        manager = self.translation_manager()
        action = manager.choose_translation_action(manager.state.pose, (100.0, 150.0))
        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "reverse")
        self.assertLess(action["planned_cm"], 0.0)
        event = next(data for name, data in manager.debug.events if name == "translation_preferred")
        self.assertTrue(event["reverse_preferred"])
        self.assertEqual(event["turn_penalty"], 20.0)

    def test_reverse_batch_executes_back_fast_then_requests_relocalize(self):
        manager = self.translation_manager()
        action = manager.choose_translation_action(manager.state.pose, (100.0, 150.0))
        calls = []
        manager.motion = SimpleNamespace(
            reverse_cycles_for_distance=lambda distance: 6,
            run=lambda key, times_override=1: calls.append((key, times_override))
            or ActionResult(
                key,
                "back",
                times_override,
                0.0,
                model_forward_cm=-2.5 * times_override,
                executed_times=times_override,
            ),
        )
        manager.forward_map_block_count = 0
        manager.clear_turn_progress_watchdog = lambda reason: None
        relocalized = []
        manager.post_action_relocalize = (
            lambda reason, pose, result, waypoint, **kwargs:
            relocalized.append((reason, waypoint)) or True
        )
        status = manager.execute_translation_action(
            action,
            manager.state.pose,
            (100.0, 150.0),
            50.0,
            {"reason": "test"},
        )
        self.assertEqual(status, "moved")
        self.assertEqual(calls, [("back_fast", 6)])
        self.assertEqual(relocalized, [("translation_reverse", (100.0, 150.0))])

    def test_start_escape_reverse_trusts_authoritative_planner_corridor(self):
        manager = self.translation_manager()
        manager.active_navigation_plan = {"goal_type": "start_projection"}
        manager.escape_corridor_metrics = lambda *args, **kwargs: {
            "clear": True,
            "path_obstacle_cost": 0.0,
            "minimum_wall_clearance_cm": 12.0,
        }
        action = manager.choose_translation_action(
            manager.state.pose, (140.0, 150.0)
        )
        self.assertEqual(action["kind"], "reverse")
        self.assertTrue(action["corridor_metrics"]["clear"])

        # Reproduce the old contradiction: normal navigation rejects the
        # prefix, although the start-escape planner has already accepted it.
        manager.movement_corridor_metrics = lambda *args, **kwargs: {
            "clear": False,
            "path_obstacle_cost": 80.0,
            "minimum_wall_clearance_cm": 12.0,
        }
        calls = []
        manager.motion = SimpleNamespace(
            reverse_cycles_for_distance=lambda distance: 1,
            run=lambda key, times_override=1: calls.append((key, times_override))
            or ActionResult(
                key,
                "back",
                times_override,
                0.0,
                model_forward_cm=-2.5 * times_override,
                executed_times=times_override,
            ),
        )
        manager.forward_map_block_count = 0
        manager.near_wall_now = lambda pose: False
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.post_action_relocalize = lambda *args, **kwargs: True

        status = manager.execute_translation_action(
            action,
            manager.state.pose,
            (140.0, 150.0),
            10.0,
            {"reason": "start_escape_test"},
        )

        self.assertEqual(status, "moved")
        self.assertEqual(calls, [("back_fast", 1)])
        self.assertFalse(any(
            name == "reverse_rejected"
            and data.get("reverse_rejected_reason")
            == "rear_corridor_blocked_before_execute"
            for name, data in manager.debug.events
        ))
        started = next(
            data for name, data in manager.debug.events
            if name == "action_batch_started"
        )
        self.assertTrue(started["movement_corridor_clear"])

    def test_identical_executor_veto_twice_emits_decision_stall(self):
        manager = self.translation_manager()
        pose = manager.state.pose
        target = (170.0, 150.0)

        self.assertFalse(manager.register_decision_stall(
            pose, target, "strafe", "translation_corridor_blocked_before_execute"
        ))
        self.assertTrue(manager.register_decision_stall(
            pose, target, "strafe", "translation_corridor_blocked_before_execute"
        ))

        event = next(
            data for name, data in manager.debug.events
            if name == "decision_stall_detected"
        )
        self.assertEqual(event["selected_action"], "strafe")
        self.assertFalse(event["executed"])
        self.assertEqual(event["count"], 2)

    def test_second_identical_executor_veto_enters_recovery(self):
        manager = self.translation_manager()
        manager.forward_map_block_count = 0
        manager.motion = SimpleNamespace(lateral_cycles_for_distance=lambda distance: 1)
        manager.near_wall_now = lambda pose: False
        manager.movement_corridor_metrics = lambda *args, **kwargs: {
            "clear": False,
            "path_obstacle_cost": 80.0,
            "minimum_wall_clearance_cm": 10.0,
        }
        localizations = []
        recoveries = []
        manager.localize_scan = lambda *args, **kwargs: localizations.append(True) or False
        manager.recover_from_near_wall = (
            lambda reason: recoveries.append(reason) or None
        )
        action = {
            "kind": "strafe",
            "distance_cm": 8.0,
            "planned_cm": 4.0,
            "progress_cm": 4.0,
            "forward_cm": 0.0,
            "lateral_cm": 8.0,
        }

        for _ in range(2):
            status = manager.execute_translation_action(
                action,
                manager.state.pose,
                (150.0, 170.0),
                20.0,
                {"reason": "identical_veto_test"},
            )
            self.assertEqual(status, "recovered")

        self.assertEqual(len(localizations), 1)
        self.assertEqual(len(recoveries), 1)
        self.assertIn("decision_stall", recoveries[0])

    def test_task_target_strafe_bypass_executes_despite_blocked_corridor(self):
        manager = self.translation_manager()
        calls = []
        manager.forward_map_block_count = 0
        manager.near_wall_now = lambda pose: True
        manager.movement_corridor_metrics = lambda *args, **kwargs: {
            "clear": False,
            "path_obstacle_cost": 80.0,
            "minimum_wall_clearance_cm": 0.0,
        }
        manager.motion = SimpleNamespace(
            lateral_cycles_for_distance=lambda distance: 1,
            run=lambda key, times_override=1: calls.append((key, times_override))
            or ActionResult(
                key,
                "strafe",
                times_override,
                0.0,
                model_lateral_cm=4.0,
                executed_times=times_override,
            ),
        )
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.post_action_relocalize = lambda *args, **kwargs: True
        action = {
            "kind": "strafe",
            "distance_cm": 8.0,
            "planned_cm": 4.0,
            "progress_cm": 4.0,
            "forward_cm": 0.0,
            "lateral_cm": 8.0,
        }

        status = manager.execute_translation_action(
            action,
            manager.state.pose,
            (150.0, 170.0),
            20.0,
            {"reason": "task_target"},
            bypass_action_safety=True,
        )

        self.assertEqual(status, "moved")
        self.assertEqual(calls, [("strafe_left_fast", 1)])

    def test_low_confidence_rejects_direct_reverse(self):
        manager = self.translation_manager(Confidence.LOW)
        action = manager.choose_translation_action(manager.state.pose, (100.0, 150.0))
        self.assertTrue(action is None or action["kind"] != "reverse")
        event = next(data for name, data in manager.debug.events if name == "reverse_preference_evaluated")
        self.assertEqual(event["reverse_rejected_reason"], "localization_confidence_low")

    def test_action_planner_uses_reverse_without_turn_for_rear_goal(self):
        pose = RobotPose(150.0, 150.0, 0.0, Confidence.HIGH, "VISION", now_s())
        path = self.model.plan_action_path(
            pose,
            (100.0, 150.0),
            self.config["navigation"],
            self.config["motion"],
        )
        self.assertTrue(path)
        actions = self.model.last_action_plan_metrics["selected_actions"]
        self.assertTrue(actions)
        self.assertEqual(set(actions), {"reverse"})
        self.assertEqual(self.model.last_action_plan_metrics["turn_cost"], 0.0)

    def test_rear_wall_rejects_reverse(self):
        manager = self.translation_manager()
        manager.map.add_dynamic_obstacle((140.0, 150.0), size_cm=8.0)
        action = manager.choose_translation_action(manager.state.pose, (100.0, 150.0))
        self.assertTrue(action is None or action["kind"] != "reverse")
        event = next(data for name, data in manager.debug.events if name == "reverse_preference_evaluated")
        self.assertEqual(event["reverse_rejected_reason"], "rear_corridor_blocked")

    def test_large_rear_lateral_error_rejects_blind_reverse(self):
        manager = self.translation_manager()
        action = manager.choose_translation_action(manager.state.pose, (100.0, 170.0))
        self.assertTrue(action is None or action["kind"] != "reverse")
        event = next(data for name, data in manager.debug.events if name == "reverse_preference_evaluated")
        self.assertEqual(event["reverse_rejected_reason"], "lateral_error_too_large")

    def test_reverse_batch_does_not_cross_rear_target(self):
        manager = self.translation_manager()
        action = manager.choose_translation_action(manager.state.pose, (140.0, 150.0))
        self.assertEqual(action["kind"], "reverse")
        self.assertGreater(manager.state.pose.x_cm + action["planned_cm"], 140.0)
        self.assertLessEqual(action["next_distance_cm"], manager.config["navigation"]["target_arrival_radius_cm"])

    def test_short_target_directly_behind_prefers_one_reverse_step(self):
        manager = self.translation_manager()
        pose = RobotPose(248.67, 233.79, 101.98, Confidence.HIGH, "VISION", now_s())
        manager.state.set_pose(pose)
        manager.last_localize_success_s = now_s()
        manager.movement_corridor_metrics = lambda *args, **kwargs: {
            "clear": True,
            "path_obstacle_cost": 0.0,
            "minimum_wall_clearance_cm": 20.0,
        }
        goal = (249.0, 227.5)
        action = manager.choose_translation_action(pose, goal)
        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "reverse")
        self.assertEqual(action["planned_cm"], -2.5)
        self.assertLessEqual(action["next_distance_cm"], manager.config["navigation"]["target_arrival_radius_cm"])

    def test_safe_lateral_target_selects_strafe(self):
        manager = self.translation_manager()
        # Stay on the open side of the nearby building so the complete strafe
        # corridor remains above the configured 25 cm navigation clearance.
        action = manager.choose_translation_action(manager.state.pose, (150.0, 138.0))
        self.assertEqual(action["kind"], "strafe")

    def test_side_wall_blocks_lateral_corridor(self):
        manager = self.translation_manager()
        manager.map.add_dynamic_obstacle((150.0, 160.0), size_cm=6.0)
        action = manager.choose_translation_action(manager.state.pose, (150.0, 180.0))
        self.assertIsNone(action)

    def test_slightly_longer_path_wins_when_wall_clearance_is_larger(self):
        manager = self.translation_manager()
        manager.map.grid[:] = 0
        manager.map.cost[:] = 0.0
        manager.map.dynamic_obstacles = []
        manager.map.building_bounds = {999: {
            "x_min": 170.0,
            "x_max": 190.0,
            "y_min": 130.0,
            # The direct route has only 15 cm clearance; its endpoints and the
            # detour remain just above the new 25 cm hard boundary.
            "y_max": 135.0,
        }}
        short = [(150.0, 150.0), (210.0, 150.0)]
        safe = [(150.0, 150.0), (150.0, 180.0), (210.0, 180.0), (210.0, 150.0)]
        short_metrics = manager.normal_path_metrics(manager.state.pose, short, translation_only=True)
        safe_metrics = manager.normal_path_metrics(manager.state.pose, safe, translation_only=True)
        self.assertGreater(
            safe_metrics["minimum_wall_clearance_cm"],
            short_metrics["minimum_wall_clearance_cm"],
        )
        self.assertLess(safe_metrics["total_cost"], short_metrics["total_cost"])
        selected_path = manager.plan_navigation_path(manager.state.pose, (210.0, 150.0))
        selected_metrics = manager.normal_path_metrics(manager.state.pose, selected_path)
        self.assertGreater(
            selected_metrics["minimum_wall_clearance_cm"],
            short_metrics["minimum_wall_clearance_cm"],
        )
        event = next(
            data for name, data in reversed(manager.debug.events)
            if name == "navigation_path_selected"
        )
        self.assertEqual(event["selected_path_type"], "action_planner")

    def test_segment_threshold_is_below_map_maximum_cost(self):
        self.assertLess(
            self.config["navigation"]["action_planner_segment_max_cost"],
            self.config["map"]["obstacle_cost_max"],
        )

    def test_consecutive_reverse_and_strafe_reversal_penalties_accumulate(self):
        actions = self.model.action_planner_actions(self.config["navigation"], self.config["motion"])
        left_turn = next(item for item in actions if item["name"] == "turn_left_small")
        right_strafe = next(item for item in actions if item["name"] == "strafe_right")
        neutral = (*self.model.grid_pos((150.0, 150.0)), 0, 0, 0)
        _, base = self.model.action_planner_transition(neutral, left_turn, 15.0, 24, 85.0, 1.0)
        _, same = self.model.action_planner_transition((*neutral[:3], 1, 0), left_turn, 15.0, 24, 85.0, 1.0)
        _, reverse = self.model.action_planner_transition((*neutral[:3], -1, 0), left_turn, 15.0, 24, 85.0, 1.0)
        self.assertGreater(same, base)
        self.assertGreater(reverse, same)
        _, normal_strafe = self.model.action_planner_transition(neutral, right_strafe, 15.0, 24, 85.0, 0.0)
        _, reversed_strafe = self.model.action_planner_transition((*neutral[:3], 0, 2), right_strafe, 15.0, 24, 85.0, 0.0)
        self.assertGreater(reversed_strafe, normal_strafe)

    def test_moving_away_from_goal_costs_more(self):
        forward = next(
            item for item in self.model.action_planner_actions(self.config["navigation"], self.config["motion"])
            if item["name"] == "forward"
        )
        grid = self.model.grid_pos((150.0, 150.0))
        goal = self.model.grid_pos((220.0, 150.0))
        toward = (*grid, 0, 0, 0)
        away = (*grid, 12, 0, 0)
        _, toward_cost = self.model.action_planner_transition(toward, forward, 15.0, 24, 85.0, 0.0, goal_node=goal)
        _, away_cost = self.model.action_planner_transition(away, forward, 15.0, 24, 85.0, 0.0, goal_node=goal)
        self.assertGreater(away_cost, toward_cost)

    def test_forward_is_selected_before_lateral_when_both_are_safe(self):
        manager = adaptive_manager()
        manager.map = SimpleNamespace(
            line_clear=lambda *args, **kwargs: True,
            is_free_xy=lambda xy: True,
            is_dangerously_close_to_wall=lambda *args: False,
        )
        manager.motion = SimpleNamespace(
            forward_cycles_for_distance=lambda distance: 3,
            lateral_cycles_for_distance=lambda distance: 2,
        )
        manager.forward_clear_for_distance = lambda *args, **kwargs: True
        manager.path_segments_clear = lambda *args, **kwargs: True
        action = manager.choose_translation_action(manager.state.pose, (170.0, 170.0))
        self.assertEqual(action["kind"], "forward")

    def test_transient_navigation_failures_do_not_consume_target_attempts(self):
        manager = adaptive_manager()
        target = self.model.screens[1]
        target.status = ScreenStatus.UNKNOWN
        target.attempts = 0
        manager.current_target_screen_id = target.screen_id
        manager.preserve_current_target(target, "near_wall_recovery_exhausted")
        self.assertEqual(target.attempts, 0)
        self.assertEqual(manager.current_target_screen_id, target.screen_id)
        self.assertTrue(manager.is_retryable_target_failure("RECOVERY_NO_PROGRESS"))


if __name__ == "__main__":
    unittest.main()
