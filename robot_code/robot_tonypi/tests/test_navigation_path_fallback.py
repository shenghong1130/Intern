from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import default_config_path, load_config
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import Confidence, MissionState, RobotPose
from robot_tonypi.motion import MotionController, RobotState
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.utils import distance_xy, now_s


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))


def competition_manager(screen_id, pose):
    manager = TaskManager.__new__(TaskManager)
    manager.config = load_config(str(default_config_path()))
    manager.tag_poses = load_tag_pos()
    manager.map = MapModel(manager.tag_poses, manager.config)
    manager.configure_cardinal_task_targets()
    manager.debug = DebugStub()
    manager.current_target_screen_id = int(screen_id)
    manager.current_target_goal = None
    manager.target_generation_counter = 0
    manager.active_navigation_plan = None
    manager.local_replan_failures = 0
    manager.plan_failure_signature = None
    manager.identical_plan_failure_count = 0
    manager.state = RobotState(manager.config)
    stamped_pose = RobotPose(
        pose.x_cm, pose.y_cm, pose.yaw_deg, pose.confidence, pose.source, now_s()
    )
    manager.state.set_pose(stamped_pose)
    manager.motion = MotionController(SimpleNamespace(), manager.state, manager.debug)
    manager.consecutive_localize_failures = 0
    manager.last_localize_success_s = now_s()
    manager.lock_target_goal(manager.map.screens[int(screen_id)])
    return manager


class RealFailureRegressionTests(unittest.TestCase):
    def assert_real_case_moves(self, screen_id, pose_tuple, expected_goal):
        pose = RobotPose(*pose_tuple, Confidence.HIGH, "TEST", now_s())
        manager = competition_manager(screen_id, pose)
        screen = manager.map.screens[screen_id]
        self.assertEqual(screen.task_target_xy, expected_goal)
        path = manager.plan_navigation_path(pose, screen.task_target_xy, target_screen=screen)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(manager.active_navigation_plan["goal_type"], "start_projection")
        action = manager.choose_translation_action(pose, path[1])
        self.assertIsNotNone(action)
        self.assertIn(action["kind"], ("forward", "reverse", "strafe"))
        self.assertTrue(action["corridor_metrics"]["clear"])

    def test_normal_navigation_clearance_accepts_25cm_boundary(self):
        manager = competition_manager(
            35, RobotPose(150.0, 150.0, 0.0, Confidence.HIGH, "TEST", now_s())
        )
        self.assertFalse(manager.normal_navigation_clearance_ok(20.0))
        self.assertTrue(manager.normal_navigation_clearance_ok(25.0))
        self.assertTrue(manager.normal_navigation_clearance_ok(30.0))

    def test_footprint_traversable_uses_configured_25cm_clearance(self):
        manager = competition_manager(
            35, RobotPose(150.0, 150.0, 0.0, Confidence.HIGH, "TEST", now_s())
        )
        manager.map.grid_pos = lambda xy: (0, 0)
        manager.map.in_bounds_xy = lambda xy: True
        manager.map.is_free_xy = lambda xy: True
        manager.map.cost = {(0, 0): 0.0}
        manager.map._neighbors = lambda *args, **kwargs: []
        manager.map.is_free_grid = lambda node: True
        manager.map.robot_clearance_cm = lambda xy: float(xy[0])
        self.assertFalse(manager.navigation_point_diagnostics((20.0, 0.0))["footprint_traversable"])
        at_boundary = manager.navigation_point_diagnostics((25.0, 0.0))
        self.assertTrue(at_boundary["clearance_traversable"])
        self.assertTrue(at_boundary["footprint_traversable"])
        self.assertTrue(manager.navigation_point_diagnostics((30.0, 0.0))["footprint_traversable"])

    def test_screen_35_high_cost_start_has_safe_escape_action(self):
        self.assert_real_case_moves(
            35, (245.43, 125.23, 99.4), (203.0, 98.5)
        )

    def test_screen_4_high_cost_start_has_safe_escape_action(self):
        self.assert_real_case_moves(
            4, (172.50, 64.96, 106.5), (207.5, 50.0)
        )

    def test_screen_17_astar_successful_safe_candidate_is_reachable(self):
        pose = RobotPose(
            100.90, 234.88, 0.0, Confidence.HIGH, "TEST", now_s()
        )
        manager = competition_manager(17, pose)
        screen = manager.map.screens[17]
        path = manager.plan_navigation_path(
            pose,
            screen.task_target_xy,
            allow_goal_high_cost=True,
            target_screen=screen,
        )
        self.assertTrue(path)
        self.assertNotEqual(manager.active_navigation_plan["goal_type"], "none")
        candidate = next(
            data for name, data in manager.debug.events
            if name == "staging_candidate_generated"
            and tuple(data["candidate_xy"]) == (62.0, 274.5)
        )
        self.assertFalse(candidate["blocked"])
        self.assertTrue(candidate["footprint_traversable"])
        self.assertEqual(candidate["astar_reason"], "success")
        self.assertTrue(candidate["astar_path_found"])
        self.assertTrue(candidate["reachable"])
        self.assertIsNone(candidate["reachability_rejection_reason"])

    def test_ordinary_navigation_still_rejects_soft_high_cost_goal(self):
        pose = RobotPose(
            100.90, 234.88, 0.0, Confidence.HIGH, "TEST", now_s()
        )
        manager = competition_manager(17, pose)
        goal = manager.map.screens[17].task_target_xy
        self.assertFalse(manager.navigation_point_diagnostics(goal)["footprint_traversable"])
        path = manager.plan_navigation_path(
            pose, goal, allow_goal_high_cost=False, target_screen=None
        )
        self.assertEqual(path, [])
        event = next(
            data for name, data in manager.debug.events
            if name == "staging_candidate_generated"
        )
        self.assertFalse(event["reachable"])
        self.assertIn("goal_soft_cost_rejected", event["reachability_rejection_reason"])

    def test_screen_18_final_target_uses_target_owned_soft_cost_exemption(self):
        pose = RobotPose(140.0, 243.0, 0.0, Confidence.HIGH, "TEST", now_s())
        manager = competition_manager(18, pose)
        screen = manager.map.screens[18]
        detail = manager.navigation_point_diagnostics(screen.task_target_xy)
        self.assertFalse(detail["blocked"])
        self.assertFalse(detail["footprint_traversable"])
        path = manager.target_direct_approach_path(
            pose, screen, screen.task_target_xy
        )
        self.assertEqual(path[-1], screen.task_target_xy)

    def test_target_owned_approach_never_ignores_unrelated_obstacle(self):
        pose = RobotPose(140.0, 243.0, 0.0, Confidence.HIGH, "TEST", now_s())
        manager = competition_manager(18, pose)
        screen = manager.map.screens[18]
        manager.map.add_dynamic_obstacle((122.0, 243.0), size_cm=8.0)
        metrics = manager.target_owned_approach_metrics(
            pose, screen, screen.task_target_xy
        )
        self.assertFalse(metrics["clear"])
        self.assertEqual(
            metrics["reachability_rejection_reason"],
            "target_approach_physical_collision",
        )
        self.assertEqual(
            manager.target_direct_approach_path(pose, screen, screen.task_target_xy),
            [],
        )


class ApproachAndStagingTests(unittest.TestCase):
    def setUp(self):
        self.pose = RobotPose(250.0, 90.0, 90.0, Confidence.HIGH, "TEST", now_s())
        self.manager = competition_manager(35, self.pose)
        self.screen = self.manager.map.screens[35]

    def test_blocked_anchor_is_not_used_as_robot_center_goal(self):
        anchor = self.screen.center_xy
        self.assertTrue(self.manager.navigation_point_diagnostics(anchor)["blocked"])
        path = self.manager.plan_navigation_path(
            self.pose, self.screen.task_target_xy, target_screen=self.screen
        )
        self.assertTrue(path)
        self.assertNotEqual(tuple(self.manager.active_navigation_plan["goal_xy"]), anchor)

    def test_staging_is_the_actual_planned_endpoint(self):
        calls = []
        original_plan = self.manager.map.plan

        def recording_plan(start, goal, allow_goal_high_cost=False):
            calls.append(tuple(goal))
            return original_plan(start, goal, allow_goal_high_cost)

        self.manager.map.plan = recording_plan
        path = self.manager.plan_navigation_path(
            self.pose, self.screen.task_target_xy, target_screen=self.screen
        )
        selected = tuple(self.manager.active_navigation_plan["goal_xy"])
        self.assertIn(
            self.manager.active_navigation_plan["goal_type"],
            ("staging", "approach"),
        )
        self.assertIn(selected, calls)
        self.assertLess(distance_xy(path[-1], selected), 0.1)
        self.assertNotEqual(selected, self.screen.task_target_xy)

    def test_blocked_first_staging_selects_alternate_approach(self):
        candidates = self.manager.reachable_navigation_goal_candidates(
            self.pose, self.screen, self.screen.task_target_xy, False
        )
        first_staging = candidates[0]["xy"]
        original_plan = self.manager.map.plan

        def reject_first(start, goal, allow_goal_high_cost=False):
            if distance_xy(goal, first_staging) < 0.1:
                self.manager.map.last_astar_metrics = {
                    "reason": "forced_first_staging_blocked", "expanded_nodes": 0
                }
                return []
            return original_plan(start, goal, allow_goal_high_cost)

        self.manager.map.plan = reject_first
        path = self.manager.plan_navigation_path(
            self.pose, self.screen.task_target_xy, target_screen=self.screen
        )
        self.assertTrue(path)
        self.assertEqual(self.manager.active_navigation_plan["goal_type"], "approach")
        self.assertGreater(
            distance_xy(self.manager.active_navigation_plan["goal_xy"], first_staging), 0.1
        )


class RepeatedPlanningFailureTests(unittest.TestCase):
    def setUp(self):
        pose = RobotPose(150.0, 150.0, 0.0, Confidence.HIGH, "TEST", now_s())
        self.manager = competition_manager(35, pose)
        self.pose = self.manager.state.pose
        self.goal = self.manager.current_target_goal.goal_xy
        self.manager.active_navigation_plan = {
            "goal_type": "none", "goal_xy": None, "staging_xy": None
        }

    def test_identical_failure_counter_is_bounded_at_escalation_threshold(self):
        threshold = self.manager.config["navigation"][
            "identical_local_replan_failure_threshold"
        ]
        counts = [
            self.manager.register_plan_failure(
                self.pose, self.goal, "no_reachable_approach_or_staging"
            )[0]
            for _ in range(threshold + 4)
        ]
        self.assertEqual(counts[threshold - 1], threshold)
        self.assertEqual(self.manager.local_replan_failures, threshold)
        self.assertGreater(self.manager.identical_plan_failure_count, threshold)

    def test_robot_grid_change_starts_a_new_failure_signature(self):
        self.manager.register_plan_failure(
            self.pose, self.goal, "no_reachable_approach_or_staging"
        )
        moved = RobotPose(160.0, 150.0, 0.0, Confidence.HIGH, "TEST", now_s())
        count, _ = self.manager.register_plan_failure(
            moved, self.goal, "no_reachable_approach_or_staging"
        )
        self.assertEqual(count, 1)

    def test_dynamic_map_change_starts_a_new_failure_signature(self):
        self.manager.register_plan_failure(
            self.pose, self.goal, "no_reachable_approach_or_staging"
        )
        self.manager.map.add_dynamic_obstacle((120.0, 120.0), size_cm=10.0)
        count, _ = self.manager.register_plan_failure(
            self.pose, self.goal, "no_reachable_approach_or_staging"
        )
        self.assertEqual(count, 1)

    def test_interior_recovery_selection_preserves_original_target(self):
        original = self.manager.current_target_goal
        edge_pose = RobotPose(15.0, 150.0, 90.0, Confidence.LOW, "TEST", now_s())
        recovery = self.manager.choose_boundary_recovery_target(edge_pose)
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery["kind"], "interior_safe")
        self.assertIs(self.manager.current_target_goal, original)
        self.assertEqual(self.manager.current_target_screen_id, 35)

    def test_three_identical_failures_enter_navigation_blocked_without_step_limit(self):
        manager = self.manager
        manager.mission_state = MissionState.NAVIGATE_TO_TARGET
        manager.navigation_plan_episode = None
        manager.navigation_stall_signature = None
        manager.navigation_stall_count = 0
        manager.near_wall_recovery_no_progress_count = 0
        manager.near_wall_recovery_actions = 0
        manager.turn_navigation_abort = False
        manager.collision_recovery_pending = False
        manager.pending_post_action_replan = False
        manager.last_navigation_failure_reason = ""
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.clear_navigation_noop = lambda: None
        manager.time_left_s = lambda: 100.0
        manager.near_wall_now = lambda pose: False
        manager.target_direct_approach_path = lambda *args, **kwargs: []
        manager.plan_navigation_path = lambda *args, **kwargs: []
        manager.localize_scan = lambda *args, **kwargs: True
        manager.recover_via_indoor_waypoint = lambda reason: False
        manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)

        ok = manager.navigate_to_xy(
            self.goal,
            max_steps=20,
            target_screen=manager.map.screens[35],
            target_goal=manager.current_target_goal,
        )
        self.assertFalse(ok)
        self.assertEqual(manager.last_navigation_failure_reason, "navigation_blocked")
        self.assertEqual(manager.mission_state, MissionState.NAVIGATION_BLOCKED)
        self.assertEqual(manager.local_replan_failures, 3)

    def test_all_planning_candidates_unreachable_use_interior_recovery_then_resume(self):
        manager = self.manager
        original_goal = manager.current_target_goal
        manager.mission_state = MissionState.NAVIGATE_TO_TARGET
        manager.navigation_plan_episode = None
        manager.navigation_stall_signature = None
        manager.navigation_stall_count = 0
        manager.near_wall_recovery_no_progress_count = 0
        manager.near_wall_recovery_actions = 0
        manager.turn_navigation_abort = False
        manager.collision_recovery_pending = False
        manager.pending_post_action_replan = False
        manager.last_navigation_failure_reason = ""
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.clear_navigation_noop = lambda: None
        manager.time_left_s = lambda: 100.0
        manager.near_wall_now = lambda pose: False
        manager.target_direct_approach_path = lambda *args, **kwargs: []
        manager.plan_navigation_path = lambda *args, **kwargs: []
        manager.localize_scan = lambda *args, **kwargs: True
        manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)
        recovery_calls = []

        def recover(reason):
            recovery_calls.append(reason)
            manager.state.set_pose(RobotPose(
                self.goal[0], self.goal[1], self.pose.yaw_deg,
                Confidence.HIGH, "RECOVERY_TEST", now_s()
            ))
            return True

        manager.recover_via_indoor_waypoint = recover
        ok = manager.navigate_to_xy(
            self.goal,
            max_steps=20,
            target_screen=manager.map.screens[35],
            target_goal=original_goal,
        )
        self.assertTrue(ok)
        self.assertEqual(len(recovery_calls), 1)
        self.assertIs(manager.current_target_goal, original_goal)
        self.assertEqual(manager.current_target_screen_id, 35)

    def test_same_deterministic_failure_does_not_repeat_recovery_episode(self):
        manager = self.manager
        manager.mission_state = MissionState.NAVIGATE_TO_TARGET
        manager.navigation_plan_episode = None
        manager.navigation_stall_signature = None
        manager.navigation_stall_count = 0
        manager.near_wall_recovery_no_progress_count = 0
        manager.near_wall_recovery_actions = 0
        manager.turn_navigation_abort = False
        manager.collision_recovery_pending = False
        manager.pending_post_action_replan = False
        manager.last_navigation_failure_reason = ""
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.clear_navigation_noop = lambda: None
        manager.time_left_s = lambda: 100.0
        manager.near_wall_now = lambda pose: False
        manager.target_direct_approach_path = lambda *args, **kwargs: []
        manager.plan_navigation_path = lambda *args, **kwargs: []
        manager.localize_scan = lambda *args, **kwargs: True
        manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)
        recovery_calls = []

        def recover(reason):
            recovery_calls.append(reason)
            return True

        manager.recover_via_indoor_waypoint = recover
        ok = manager.navigate_to_xy(
            self.goal,
            max_steps=12,
            target_screen=manager.map.screens[35],
            target_goal=manager.current_target_goal,
        )
        self.assertFalse(ok)
        self.assertEqual(len(recovery_calls), 1)
        self.assertEqual(manager.mission_state, MissionState.NAVIGATION_BLOCKED)
        self.assertEqual(manager.last_navigation_failure_reason, "navigation_blocked")
        events = [
            data for name, data in manager.debug.events
            if name == "deterministic_recovery_repeat_blocked"
        ]
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["recovery_repeated"])

    def test_arrival_requires_fresh_visual_pose_after_dead_reckoning(self):
        manager = self.manager
        manager.mission_state = MissionState.NAVIGATE_TO_TARGET
        manager.navigation_plan_episode = None
        manager.navigation_stall_signature = None
        manager.navigation_stall_count = 0
        manager.near_wall_recovery_no_progress_count = 0
        manager.near_wall_recovery_actions = 0
        manager.turn_navigation_abort = False
        manager.collision_recovery_pending = False
        manager.pending_post_action_replan = False
        manager.last_navigation_failure_reason = ""
        manager.last_navigation_mode = "normal"
        manager.last_motion_action = "forward_fast"
        manager.last_localization_pose_conflict = False
        manager.clear_turn_progress_watchdog = lambda reason: None
        manager.clear_navigation_noop = lambda: None
        manager.time_left_s = lambda: 100.0
        manager.near_wall_now = lambda pose: False
        manager.state.set_pose(RobotPose(
            self.goal[0], self.goal[1], self.pose.yaw_deg,
            Confidence.HIGH, "DEAD_RECKONING", now_s()
        ))
        manager.state.actions_since_localize = 1
        manager.state.motion_uncertainty = 0.6
        localized = []

        def localize(*args, **kwargs):
            localized.append(kwargs.get("reason"))
            manager.state.set_pose(RobotPose(
                self.goal[0], self.goal[1], self.pose.yaw_deg,
                Confidence.HIGH, "VISION_TEST", now_s()
            ))
            manager.last_localize_success_s = now_s()
            return True

        manager.localize_scan = localize
        ok = manager.navigate_to_xy(
            self.goal,
            max_steps=4,
            target_screen=manager.map.screens[35],
            target_goal=manager.current_target_goal,
        )
        self.assertTrue(ok)
        self.assertEqual(localized, ["before_arrived_at_target"])
        decisions = [
            data for name, data in manager.debug.events
            if name == "relocalization_decision"
        ]
        self.assertTrue(any(
            data["reason"] == "before_arrived_at_target" for data in decisions
        ))


if __name__ == "__main__":
    unittest.main()
