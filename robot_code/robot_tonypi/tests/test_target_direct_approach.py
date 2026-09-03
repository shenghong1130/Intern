from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import Confidence, RobotPose
from robot_tonypi.task_manager import TaskManager


class TargetDirectApproachTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(None)
        self.model = MapModel(load_tag_pos(), self.config)
        self.screen = self.model.screens[1]
        self.goal = self.screen.task_target_xy
        # 155 cm is outside the neighbouring building's hard inflation band;
        # from here to screen 1 only the locked building raises target cost.
        self.pose = RobotPose(155.0, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)

    def manager(self):
        manager = TaskManager.__new__(TaskManager)
        manager.config = self.config
        manager.map = self.model
        manager.current_target_screen_id = self.screen.screen_id
        manager.max_forward_cycles_for_pose = lambda pose: 20
        manager.debug = SimpleNamespace(event=lambda *args, **kwargs: None)
        return manager

    def test_target_inflation_is_ignored_only_in_clear_final_corridor(self):
        # Emulate the high target-building inflation observed on the real cost
        # map.  The corridor decision deliberately recomputes costs with only
        # the locked target building excluded.
        self.model.cost[self.model.grid_pos(self.goal)] = 80.0
        self.assertGreaterEqual(
            float(self.model.cost[self.model.grid_pos(self.goal)]),
            float(self.config["navigation"]["target_direct_non_target_max_cost"]),
        )
        self.assertTrue(
            self.model.target_direct_corridor_clear(
                self.pose.xy(), self.goal, self.screen.screen_id, 6.0, 60.0
            )
        )
        self.assertEqual(
            self.manager().target_direct_approach_path(self.pose, self.screen, self.goal),
            [self.pose.xy(), self.goal],
        )

    def test_other_building_inflation_is_not_ignored(self):
        self.model.building_bounds[99] = {
            "x_min": 160.0,
            "x_max": 165.0,
            "y_min": self.goal[1] - 2.0,
            "y_max": self.goal[1] + 2.0,
        }
        self.assertFalse(
            self.model.target_direct_corridor_clear(
                self.pose.xy(), self.goal, self.screen.screen_id, 6.0, 60.0
            )
        )

    def test_physical_obstacle_always_blocks_direct_corridor(self):
        self.model.add_dynamic_obstacle((164.0, self.goal[1]), size_cm=6.0)
        self.assertFalse(
            self.model.target_direct_corridor_clear(
                self.pose.xy(), self.goal, self.screen.screen_id, 6.0, 60.0
            )
        )

    def test_direct_mode_requires_locked_target_and_range(self):
        manager = self.manager()
        manager.current_target_screen_id = 2
        self.assertEqual(manager.target_direct_approach_path(self.pose, self.screen, self.goal), [])
        manager.current_target_screen_id = 1
        far = RobotPose(100.0, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)
        self.assertEqual(manager.target_direct_approach_path(far, self.screen, self.goal), [])

    def test_near_target_uses_short_forward_action_not_coarse_28cm_step(self):
        manager = self.manager()
        pose = RobotPose(self.goal[0] - 7.0, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)
        action = manager.choose_target_direct_action(pose, self.goal, self.screen)
        self.assertIsNotNone(action)
        self.assertEqual(action["key"], "forward_micro")
        self.assertEqual(action["times"], 1)
        self.assertLess(action["planned_cm"], 7.0)

    def test_forward_is_preferred_before_lateral(self):
        manager = self.manager()
        pose = RobotPose(self.goal[0] - 8.0, self.goal[1] - 4.0, 0.0, Confidence.HIGH, "TEST", 1.0)
        action = manager.choose_target_direct_action(pose, self.goal, self.screen)
        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "forward")

    def test_task_target_bypass_does_not_veto_selected_forward(self):
        manager = self.manager()
        pose = RobotPose(self.goal[0] - 7.0, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)
        manager.map.target_direct_corridor_clear = lambda *args, **kwargs: False

        self.assertIsNone(
            manager.choose_target_direct_action(pose, self.goal, self.screen)
        )
        action = manager.choose_target_direct_action(
            pose,
            self.goal,
            self.screen,
            bypass_action_safety=True,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "forward")

    def test_short_rear_target_uses_reverse_not_turn(self):
        manager = self.manager()
        pose = RobotPose(self.goal[0] + 4.8, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)
        action = manager.choose_target_direct_action(
            pose,
            self.goal,
            self.screen,
            final_goal_distance_cm=4.8,
        )
        self.assertIsNotNone(action)
        self.assertEqual(action["kind"], "reverse")
        self.assertEqual(action["key"], "back_fast")
        self.assertEqual(action["times"], 1)

    def test_far_rear_target_does_not_use_reverse(self):
        manager = self.manager()
        pose = RobotPose(self.goal[0] + 20.0, self.goal[1], 0.0, Confidence.HIGH, "TEST", 1.0)
        action = manager.choose_target_direct_action(
            pose,
            self.goal,
            self.screen,
            final_goal_distance_cm=20.0,
        )
        self.assertTrue(action is None or action["kind"] != "reverse")

    def test_turn_costs_penalize_consecutive_and_reverse_turns(self):
        actions = self.model.action_planner_actions(
            self.config["navigation"], self.config["motion"]
        )
        turn = next(item for item in actions if item["name"] == "turn_left_small")
        yaw_bin = 15.0
        yaw_bins = 24
        state = (*self.model.grid_pos((100.0, 100.0)), 0, 0)
        _, base = self.model.action_planner_transition(state, turn, yaw_bin, yaw_bins, 85.0, 1.0)
        _, same = self.model.action_planner_transition((*state[:3], 1), turn, yaw_bin, yaw_bins, 85.0, 1.0)
        _, reverse = self.model.action_planner_transition((*state[:3], -1), turn, yaw_bin, yaw_bins, 85.0, 1.0)
        self.assertGreater(base, 25.0)
        self.assertGreater(same, base)
        self.assertGreater(reverse, same)

    def test_turning_remains_available_when_needed(self):
        actions = self.model.action_planner_actions(
            self.config["navigation"], self.config["motion"]
        )
        names = {item["name"] for item in actions}
        self.assertIn("turn_left_small", names)
        self.assertIn("turn_right_large", names)

    def test_large_turn_keeps_physical_yaw_distinct_from_planner_bins(self):
        actions = self.model.action_planner_actions(
            self.config["navigation"], self.config["motion"]
        )
        turn = next(item for item in actions if item["name"] == "turn_right_large")
        self.assertEqual(turn["yaw_deg"], -18.0)
        state = (*self.model.grid_pos((100.0, 100.0)), 0, 0)
        nxt, _ = self.model.action_planner_transition(
            state, turn, 15.0, 24, 85.0, 1.0
        )
        self.assertEqual(nxt[2], 22)
        self.assertEqual(
            self.model.yaw_from_action_bin(nxt[2], 15.0), -30.0
        )


if __name__ == "__main__":
    unittest.main()
