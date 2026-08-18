from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.load_pos import load_tag_pos
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
    def test_high_confidence_open_space_allows_eight_forward_cycles(self):
        manager = adaptive_manager(Confidence.HIGH)
        cycles, _ = manager.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)
        self.assertEqual(cycles, 8)

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
        self.assertEqual((forward, strafe, turn), (8, 4, 2))

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
        result = ActionResult("forward_fast", "forward", 3, 0.0, model_forward_cm=10.5, executed_times=3)
        self.assertTrue(manager.post_action_relocalize("test", before, result, (200.0, 150.0)))
        self.assertEqual(len(centered), 1)
        self.assertTrue(manager.pending_post_action_replan)
        names = [name for name, _ in manager.debug.events]
        self.assertIn("post_action_relocalize", names)
        self.assertIn("post_action_replan", names)

    def test_tag_quality_and_visual_odometry_conflict_affect_confidence(self):
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
        self.assertEqual(pose.confidence, Confidence.LOW)
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
            estimate_from_frame=lambda *args, **kwargs: (queue.pop(0), object()),
            tag_area=lambda tag: 0.0,
        )
        return manager, pans

    def test_routine_localization_center_only(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([pose])
        self.assertTrue(manager.localize_scan(reason="post_action", allow_failure_escalation=False))
        self.assertEqual(pans, [100.0])

    def test_pan_search_only_after_explicit_escalation(self):
        manager, pans = self.manager([None])
        self.assertFalse(manager.localize_scan(allow_failure_escalation=False))
        self.assertEqual(pans, [100.0])
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([None, pose])
        self.assertTrue(manager.localize_scan(allow_pan_search=True))
        self.assertEqual(pans, [100.0, 135.0])

    def test_pan_scan_stops_immediately_when_localized(self):
        pose = RobotPose(10, 10, 0, Confidence.HIGH, "VISION", now_s())
        manager, pans = self.manager([None, pose])
        self.assertTrue(manager.localize_scan(allow_pan_search=True))
        self.assertEqual(pans, [100.0, 135.0])

    def test_visibility_recovery_never_calls_full_navigate(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        import ast
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "recover_target_visibility")
        calls = {getattr(node.func, "attr", "") for node in ast.walk(fn) if isinstance(node, ast.Call)}
        self.assertNotIn("navigate_to_screen", calls)


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
            lambda reason, pose, result, waypoint: relocalized.append((reason, waypoint)) or True
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
        action = manager.choose_translation_action(manager.state.pose, (150.0, 180.0))
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
            "y_max": 136.0,
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
