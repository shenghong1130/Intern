from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import (
    ActionResult,
    Confidence,
    NavigationPlan,
    PlannedNavigationAction,
    RobotPose,
    Screen,
    ScreenStatus,
)
from robot_tonypi.motion import RobotState
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.utils import now_s


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))

    def render_map(self, *args, **kwargs):
        pass


def bare_manager(config=None):
    manager = TaskManager.__new__(TaskManager)
    manager.config = config or load_config(None)
    manager.map = MapModel(load_tag_pos(), manager.config)
    manager.debug = DebugStub()
    manager.state = RobotState(manager.config)
    manager.last_localize_success_s = 0.0
    manager.consecutive_localize_failures = 0
    manager.consecutive_no_tag_scans = 0
    manager.localization_failures = 0
    manager.last_localization_tag_count = 0
    manager.last_localization_quality = "NONE"
    manager.last_localization_pose_conflict = False
    return manager


def fake_screen(screen_id, xy, yaw=0.0):
    return Screen(
        screen_id=screen_id,
        tag_corners_3d=None,
        center_xy=xy,
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=xy,
        interaction_xy=xy,
        interaction_yaw_deg=yaw,
        reader_xy=xy,
        screen_left_tangent_xy=(0.0, -1.0),
        navigation_staging_xy=(999.0, 999.0),
        interaction_target_xy=xy,
        task_target_xy=xy,
        task_target_yaw_deg=yaw,
    )


class LocalizationPhysicalGateTests(unittest.TestCase):
    def test_field_and_building_physics_only(self):
        manager = bare_manager()
        self.assertIsNone(manager.visual_pose_physical_rejection_reason(
            RobotPose(150.0, 150.0, 0.0)
        ))
        self.assertEqual(manager.visual_pose_physical_rejection_reason(
            RobotPose(-0.01, 150.0, 0.0)
        ), "pose_outside_field")
        self.assertEqual(manager.visual_pose_physical_rejection_reason(
            RobotPose(300.01, 150.0, 0.0)
        ), "pose_outside_field")
        bounds = next(iter(manager.map.building_bounds.values()))
        inside = RobotPose(
            (bounds["x_min"] + bounds["x_max"]) / 2.0,
            (bounds["y_min"] + bounds["y_max"]) / 2.0,
            0.0,
        )
        self.assertEqual(
            manager.visual_pose_physical_rejection_reason(inside),
            "pose_inside_building",
        )

    def test_soft_inflation_outside_real_rectangle_is_legal(self):
        manager = bare_manager()
        bounds = manager.map.building_bounds[0]
        pose = RobotPose(bounds["x_min"] - 25.0, bounds["y_min"] + 10.0, 0.0)
        self.assertGreater(float(manager.map.cost[manager.map.grid_pos(pose.xy())]), 0.0)
        self.assertIsNone(manager.visual_pose_physical_rejection_reason(pose))

    def test_hard_jump_rejected_without_confirmation_and_prior_retained(self):
        manager = bare_manager()
        prior = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "PRIOR", 1.0)
        manager.state.set_pose(prior)
        manager.capture_visual_pose_once = lambda *args, **kwargs: self.fail(
            "hard jump must not request confirmation"
        )
        result = manager.evaluate_and_accept_visual_pose(
            RobotPose(141.0, 100.0, 0.0, Confidence.HIGH, "VISION", 2.0),
            [], 100.0, "test", prior,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "pose_jump_rejected")
        self.assertIs(manager.state.pose, prior)
        self.assertIn("pose_jump_rejected", [name for name, _ in manager.debug.events])

    def test_moderate_jump_still_uses_confirmation(self):
        manager = bare_manager()
        prior = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "PRIOR", 1.0)
        manager.state.set_pose(prior)
        confirmed = RobotPose(126.0, 101.0, 2.0, Confidence.HIGH, "VISION", 3.0)
        manager.capture_visual_pose_once = lambda *args, **kwargs: {
            "pose": confirmed, "tags": [], "frame": object(), "annotated": object()
        }
        result = manager.evaluate_and_accept_visual_pose(
            RobotPose(125.0, 100.0, 0.0, Confidence.HIGH, "VISION", 2.0),
            [], 100.0, "test", prior,
        )
        self.assertTrue(result["accepted"])
        self.assertIs(manager.state.pose, confirmed)

    def test_physical_and_jump_rejections_are_not_no_tag(self):
        manager = bare_manager()
        manager.record_localization_failure(
            "pose_unavailable_with_tags", saw_any_tag=True, reason="physical_gate"
        )
        manager.record_localization_failure(
            "pose_unavailable_with_tags", saw_any_tag=True, reason="hard_jump"
        )
        self.assertEqual(manager.consecutive_localize_failures, 2)
        self.assertEqual(manager.consecutive_no_tag_scans, 0)


class TargetSelectionScoreTests(unittest.TestCase):
    def manager(self, screens, yaw=0.0):
        manager = TaskManager.__new__(TaskManager)
        manager.config = load_config(None)
        manager.state = SimpleNamespace(
            pose=RobotPose(0.0, 0.0, yaw, Confidence.HIGH, "TEST", 1.0)
        )
        manager.map = SimpleNamespace(screens={item.screen_id: item for item in screens})
        manager.debug = DebugStub()
        manager.last_target_plan = {}
        manager.temporarily_failed_targets = {}
        manager.nfc_gave_up_screen_ids = set()
        manager.current_target_screen_id = None
        return manager

    def test_uses_interaction_target_not_legacy_staging(self):
        near = fake_screen(1, (10.0, 0.0))
        far = fake_screen(2, (30.0, 0.0))
        near.navigation_staging_xy = (200.0, 0.0)
        far.navigation_staging_xy = (1.0, 0.0)
        manager = self.manager([near, far])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 1)

    def test_distance_window_prevents_far_orientation_win(self):
        near_behind = fake_screen(1, (-10.0, 0.0), yaw=180.0)
        far_front = fake_screen(2, (40.0, 0.0), yaw=0.0)
        manager = self.manager([near_behind, far_front], yaw=0.0)
        self.assertEqual(manager.choose_nearest_screen().screen_id, 1)

    def test_orientation_breaks_close_distance_choice(self):
        behind = fake_screen(1, (-10.0, 0.0), yaw=180.0)
        side = fake_screen(2, (0.0, 12.0), yaw=90.0)
        manager = self.manager([behind, side], yaw=0.0)
        self.assertEqual(manager.choose_nearest_screen().screen_id, 2)
        candidate = next(
            data for name, data in manager.debug.events
            if name == "target_selection_candidate" and data["screen_id"] == 2
        )
        self.assertTrue(candidate["selected"])
        self.assertEqual(candidate["behind_turn_deg"], 0.0)

    def test_locked_and_exclusion_rules_remain(self):
        locked = fake_screen(1, (30.0, 0.0))
        complete = fake_screen(2, (1.0, 0.0)); complete.status = ScreenStatus.CHANGED
        failed = fake_screen(3, (2.0, 0.0))
        gave_up = fake_screen(4, (3.0, 0.0))
        manager = self.manager([locked, complete, failed, gave_up])
        manager.current_target_screen_id = 1
        manager.temporarily_failed_targets = {3: {}}
        manager.nfc_gave_up_screen_ids = {4}
        self.assertIs(manager.choose_nearest_screen(), locked)


class MotionAStarTests(unittest.TestCase):
    def open_model(self):
        config = load_config(None)
        model = MapModel.__new__(MapModel)
        model.cfg = config
        model.width_cm = model.height_cm = 200.0
        model.res = 5.0
        model.rows = model.cols = 40
        model.grid = np.zeros((40, 40), dtype=np.uint8)
        model.cost = np.zeros((40, 40), dtype=np.float32)
        model.building_bounds = {}
        model.dynamic_obstacles = []
        model.last_action_plan_metrics = {}
        return model, config

    def test_position_goal_does_not_require_final_screen_yaw(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, start.xy(), 90.0, config["navigation"], config["motion"],
            goal_position_tolerance_cm=1.0, goal_yaw_tolerance_deg=10.0,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.actions, [])
        self.assertFalse(plan.metrics["require_goal_yaw"])

    def test_normal_reverse_is_only_expanded_within_fifteen_cm(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        at_limit = model.plan_motion_actions(
            start, (87.5, 102.5), 0.0, config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=10.0,
        )
        self.assertIsNotNone(at_limit)
        self.assertTrue(at_limit.actions)
        self.assertEqual(set(item.kind for item in at_limit.actions), {"reverse"})
        over_limit = model.plan_motion_actions(
            start, (86.5, 102.5), 0.0, config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=10.0,
        )
        self.assertIsNotNone(over_limit)
        self.assertNotIn("reverse", [item.kind for item in over_limit.actions])
        self.assertEqual(
            over_limit.metrics["reverse_start_evaluation"]["reason"],
            "goal_too_far_for_reverse",
        )

    def test_position_action_preferences_front_far_rear_and_near_rear(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)

        front = model.plan_motion_actions(
            start, (162.5, 102.5), 90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        far_rear = model.plan_motion_actions(
            start, (42.5, 102.5), 90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        near_rear = model.plan_motion_actions(
            start, (90.5, 102.5), 90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )

        self.assertEqual([item.kind for item in front.actions], ["forward", "forward"])
        self.assertEqual(
            [item.kind for item in far_rear.actions],
            ["turn_left_90", "turn_left_90", "forward", "forward"],
        )
        self.assertEqual(
            [item.kind for item in near_rear.actions],
            ["reverse", "reverse"],
        )
        self.assertEqual(far_rear.metrics["turn_cost"], 0.0)
        self.assertFalse(far_rear.metrics["turn_primary_cost_enabled"])
        self.assertEqual(far_rear.metrics["yaw_search_mode"], "quarter_turn_position")

    def test_position_free_turn_prefers_shorter_translation(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, (42.5, 102.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        kinds = [item.kind for item in plan.actions]
        self.assertEqual(kinds[:2], ["turn_left_90", "turn_left_90"])
        self.assertEqual(kinds[2:], ["forward", "forward"])
        self.assertAlmostEqual(plan.total_cost, 56.0)

    def test_position_quarter_turn_can_enable_long_forward_segment(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, (102.5, 162.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        self.assertEqual(
            [item.kind for item in plan.actions],
            ["turn_left_90", "forward", "forward"],
        )

    def test_reverse_rejects_bad_rear_angle_and_lateral_error(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        angled = model.plan_motion_actions(
            start, (94.5, 108.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        self.assertNotIn("reverse", [item.kind for item in angled.actions])
        self.assertEqual(
            angled.metrics["reverse_start_evaluation"]["reason"],
            "rear_angle_exceeds_tolerance",
        )

        lateral_cfg = copy.deepcopy(config["navigation"])
        lateral_cfg["reverse_prefer_rear_angle_tolerance_deg"] = 90.0
        lateral = model.plan_motion_actions(
            start, (90.5, 111.5), 0.0,
            lateral_cfg, config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        self.assertNotIn("reverse", [item.kind for item in lateral.actions])
        self.assertEqual(
            lateral.metrics["reverse_start_evaluation"]["reason"],
            "lateral_error_too_large",
        )

    def test_reverse_corridor_blocked_is_rejected(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        blocked = model.grid_pos((97.5, 102.5))
        model.grid[blocked[0], blocked[1]] = 255
        plan = model.plan_motion_actions(
            start, (90.5, 102.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        self.assertIsNotNone(plan)
        self.assertNotIn("reverse", [item.kind for item in plan.actions])
        self.assertEqual(
            plan.metrics["reverse_start_evaluation"]["reason"],
            "rear_corridor_blocked",
        )

    def test_position_plan_has_no_zero_cost_turn_cycle(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, (42.5, 102.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        kinds = [item.kind for item in plan.actions]
        cancelling_pairs = {
            ("turn_left_90", "turn_right_90"),
            ("turn_right_90", "turn_left_90"),
        }
        self.assertFalse(any(pair in cancelling_pairs for pair in zip(kinds, kinds[1:])))
        self.assertEqual(plan.metrics["turn_count"], 2)

    def test_position_free_turn_still_requires_rotation_sweep_clear(self):
        model, config = self.open_model()
        model.rotation_sweep_clear = lambda *args, **kwargs: False
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, (42.5, 102.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0,
        )
        self.assertIsNone(plan)

    def test_final_yaw_does_not_change_position_plan(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        east_facing = model.plan_motion_actions(
            start, (130.0, 102.5), 90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=10.0,
        )
        west_facing = model.plan_motion_actions(
            start, (130.0, 102.5), -90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=10.0,
        )
        self.assertIsNotNone(east_facing)
        self.assertIsNotNone(west_facing)
        self.assertEqual(
            [item.kind for item in east_facing.actions],
            [item.kind for item in west_facing.actions],
        )
        self.assertEqual(east_facing.path_xy, west_facing.path_xy)
        self.assertTrue(all(not item.kind.startswith("turn_") for item in east_facing.actions))

    def test_position_planner_keeps_yaw_state_for_world_motion(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, (102.5, 114.5), 90.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=10.0,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan.actions)
        self.assertTrue(all(item.kind.startswith("strafe_") for item in plan.actions))

    def test_physical_turn_angle_is_not_distorted_by_requested_yaw_bin(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        plan = model.plan_motion_actions(
            start, start.xy(), -18.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=1.0,
            goal_yaw_tolerance_deg=0.5,
            require_goal_yaw=True,
        )
        self.assertIsNotNone(plan)
        self.assertEqual([item.kind for item in plan.actions], ["turn_right_large"])
        action = plan.actions[0]
        self.assertAlmostEqual(action.predicted_yaw_deg, -18.0)
        self.assertAlmostEqual(action.configured_yaw_deg, -18.0)
        self.assertAlmostEqual(action.predicted_end_pose.yaw_deg, -18.0)
        self.assertEqual(plan.metrics["requested_yaw_bin_deg"], 15.0)
        self.assertEqual(plan.metrics["physical_yaw_lattice_deg"], 1.5)
        self.assertTrue(plan.metrics["turn_primary_cost_enabled"])
        self.assertGreater(plan.metrics["turn_cost"], 0.0)

    def test_screen_navigation_aligns_yaw_only_after_position_arrival(self):
        manager = bare_manager()
        screen = manager.map.screens[3]
        goal = manager.lock_target_goal(screen)
        stamp = now_s()
        manager.state.set_pose(RobotPose(
            goal.interaction_target_xy[0],
            goal.interaction_target_xy[1],
            0.0,
            Confidence.HIGH,
            "VISION",
            stamp,
        ))
        manager.last_localize_success_s = stamp
        manager.time_left_s = lambda: 100.0
        manager.localize_scan = lambda *args, **kwargs: self.fail(
            "fresh arrival pose should be used"
        )
        manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
            "position A* must not run after XY arrival"
        )
        manager.map.target_rotation_sweep_clear = lambda *args, **kwargs: True
        turns = []

        def align(target_yaw):
            turns.append(target_yaw)
            manager.state.set_pose(RobotPose(
                goal.interaction_target_xy[0],
                goal.interaction_target_xy[1],
                target_yaw,
                Confidence.HIGH,
                "VISION",
                now_s(),
            ))
            manager.last_localize_success_s = now_s()
            return True

        manager.turn_toward_yaw_boundary_aware = align
        self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))
        self.assertEqual(turns, [goal.desired_yaw_deg])
        names = [name for name, _ in manager.debug.events]
        self.assertLess(
            names.index("position_navigation_arrived"),
            names.index("final_yaw_alignment_started"),
        )
        self.assertIn("final_yaw_alignment_complete", names)

    def test_position_planner_avoids_gratuitous_turns_for_lateral_goal(self):
        model, config = self.open_model()
        start = RobotPose(102.5, 102.5, 0.0, Confidence.HIGH, "TEST", 1.0)
        low = copy.deepcopy(config["navigation"])
        low.update({
            "action_planner_turn_cost_cm_per_deg": 0.0,
            "action_planner_turn_fixed_cost_cm": 0.0,
            "action_planner_in_place_turn_penalty_cm": 0.0,
            "action_planner_consecutive_turn_penalty_cm": 0.0,
            "action_planner_reverse_turn_penalty_cm": 0.0,
        })
        low_plan = model.plan_motion_actions(
            start, (102.5, 114.5), 0.0, low, config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=5.0,
        )
        high_plan = model.plan_motion_actions(
            start, (102.5, 114.5), 0.0,
            config["navigation"], config["motion"],
            goal_position_tolerance_cm=4.0, goal_yaw_tolerance_deg=5.0,
        )
        self.assertIsNotNone(low_plan)
        self.assertIsNotNone(high_plan)
        self.assertTrue(all(item.kind.startswith("strafe_") for item in low_plan.actions))
        self.assertTrue(all(item.kind.startswith("strafe_") for item in high_plan.actions))

    def test_executor_runs_planned_action_key(self):
        manager = bare_manager()
        screen = manager.map.screens[1]
        manager.current_target_screen_id = 1
        manager.target_generation_counter = 1
        goal = manager.target_goal_from_screen(screen, 1)
        manager.current_target_goal = goal
        start = RobotPose(150.0, 50.0, 0.0, Confidence.HIGH, "TEST", 1.0)
        manager.state.set_pose(start)
        end = RobotPose(153.5, 50.0, 0.0, Confidence.HIGH, "MOTION_ASTAR", 1.0)
        action = PlannedNavigationAction(
            "forward", "forward_fast", 1, start, end, 3.5, 0.0, 0.0, 3.5
        )
        plan = NavigationPlan(goal.interaction_target_xy, goal.desired_yaw_deg, 3.5, [start.xy(), end.xy()], [action])
        calls = []
        manager.motion = SimpleNamespace(run=lambda key, times_override=1: calls.append((key, times_override)) or ActionResult(
            key, key, times_override, 0.0, model_forward_cm=3.5 * times_override,
            ok=True, executed_times=times_override,
        ))
        manager.select_adaptive_action_batch = lambda *args, **kwargs: (1, "test")
        manager.post_action_relocalize = lambda *args, **kwargs: True
        manager.set_pending_forward_progress = lambda *args, **kwargs: None
        manager.current_target_screen_id = 1
        manager.last_navigation_failure_reason = ""
        self.assertTrue(manager.execute_motion_astar_action(plan, screen, goal))
        self.assertEqual(calls, [("forward_fast", 1)])


if __name__ == "__main__":
    unittest.main()
