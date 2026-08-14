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
        self.assertEqual(self.state.motion_uncertainty, 5.0)

    def test_partial_failure_counts_and_models_only_completed_prefix(self):
        result = ActionResult(
            "forward_fast", "forward", 5, 0.0,
            model_forward_cm=17.5, ok=False, error="stopped", executed_times=2,
        )
        self.state.apply_action_result(result)
        self.assertEqual(self.state.actions_since_localize, 2)
        self.assertAlmostEqual(self.state.pose.x_cm, 7.0)

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
    def test_high_confidence_open_space_allows_six_forward_cycles(self):
        manager = adaptive_manager(Confidence.HIGH)
        cycles, _ = manager.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)
        self.assertEqual(cycles, 6)

    def test_medium_and_low_confidence_shorten_batch(self):
        medium = adaptive_manager(Confidence.MEDIUM)
        low = adaptive_manager(Confidence.LOW)
        self.assertEqual(medium.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0], 3)
        self.assertEqual(low.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0], 1)

    def test_strafe_and_turn_caps_are_lower_than_forward(self):
        manager = adaptive_manager()
        forward = manager.select_adaptive_action_batch("forward", 8, 3.5, 100, 100)[0]
        strafe = manager.select_adaptive_action_batch("strafe", 8, 3.0, 100, 100)[0]
        turn = manager.select_adaptive_action_batch("turn", 8, 7.5, 100, 100)[0]
        self.assertEqual((forward, strafe, turn), (6, 3, 2))

    def test_near_wall_recovery_and_near_target_force_short_batches(self):
        manager = adaptive_manager()
        self.assertEqual(manager.select_adaptive_action_batch("forward", 6, 3.5, 100, 100, near_wall=True)[0], 1)
        self.assertEqual(manager.select_adaptive_action_batch("strafe", 3, 3.0, 20, 20)[0], 1)
        self.assertEqual(manager.select_adaptive_action_batch("turn", 2, 7.5, 100, 100, recovery=True)[0], 1)

    def test_batch_cannot_cross_waypoint_or_target(self):
        manager = adaptive_manager()
        cycles, _ = manager.select_adaptive_action_batch("forward", 6, 3.5, 8.0, 8.0)
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

        def localize():
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


class PlannerPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(None)
        self.model = MapModel(load_tag_pos(), self.config)

    def test_configured_turn_cost_is_in_transition_g_cost(self):
        actions = self.model.action_planner_actions(self.config["navigation"], self.config["motion"])
        turn = next(item for item in actions if item["name"] == "turn_left_small")
        state = (*self.model.grid_pos((150.0, 150.0)), 0, 0, 0)
        _, cost = self.model.action_planner_transition(state, turn, 15.0, 24, 85.0, 1.0)
        expected = 60.0 + 7.5 * 5.0 + 50.0
        self.assertEqual(cost, expected)

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
