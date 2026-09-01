from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import ActionResult, Confidence, RobotPose
from robot_tonypi.motion import RobotState
from robot_tonypi.task_manager import TaskManager


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))


def geometry_manager():
    manager = TaskManager.__new__(TaskManager)
    manager.config = load_config(None)
    manager.map = MapModel(load_tag_pos(), manager.config)
    manager.debug = DebugStub()
    manager.current_target_screen_id = None
    manager.current_target_goal = None
    manager.target_generation_counter = 0
    return manager


class TargetConsistencyTests(unittest.TestCase):
    def test_all_configured_targets_resolve_atomically(self):
        manager = geometry_manager()
        for screen_id, screen in manager.map.screens.items():
            goal = manager.lock_target_goal(screen)
            self.assertEqual(goal.screen_id, screen_id)
            self.assertEqual(goal.tag_id, screen_id)
            self.assertEqual(goal.anchor_xy, screen.center_xy)
            self.assertEqual(goal.goal_xy, screen.task_target_xy)
            self.assertEqual(goal.navigation_staging_xy, screen.navigation_staging_xy)
            self.assertEqual(goal.interaction_target_xy, screen.interaction_target_xy)
            self.assertEqual(goal.desired_yaw_deg, screen.task_target_yaw_deg)
            self.assertTrue(manager.validate_target_goal(goal))

    def test_screen_26_coordinate_is_goal_not_anchor(self):
        manager = geometry_manager()
        goal = manager.lock_target_goal(manager.map.screens[26])
        self.assertEqual(goal.anchor_xy, (237.5, 246.5))
        self.assertEqual(goal.goal_xy, (248.5, 221.5))
        self.assertEqual(goal.desired_yaw_deg, 95.0)

    def test_stale_goal_for_same_screen_is_rejected(self):
        manager = geometry_manager()
        goal = manager.lock_target_goal(manager.map.screens[26])
        stale = replace(goal, goal_xy=manager.map.screens[23].task_target_xy)
        self.assertFalse(manager.validate_target_goal(stale, requested_xy=stale.goal_xy))
        self.assertTrue(any(name == "target_pose_mismatch" for name, _ in manager.debug.events))

    def test_stale_staging_point_for_same_screen_is_rejected(self):
        manager = geometry_manager()
        goal = manager.lock_target_goal(manager.map.screens[26])
        stale = replace(
            goal,
            navigation_staging_xy=manager.map.screens[23].navigation_staging_xy,
        )
        self.assertFalse(
            manager.validate_target_goal(
                stale, requested_xy=stale.navigation_staging_xy
            )
        )


class InteriorRecoveryTests(unittest.TestCase):
    def test_edge_pose_selects_interior_clear_waypoint_not_screen_target(self):
        manager = geometry_manager()
        pose = RobotPose(15.0, 150.0, 90.0, Confidence.LOW, "TEST", 1.0)
        target = manager.choose_boundary_recovery_target(pose)
        self.assertIsNotNone(target)
        margin = manager.config["navigation"]["interior_recovery_margin_cm"]
        self.assertGreaterEqual(target["xy"][0], margin)
        self.assertGreaterEqual(target["xy"][1], margin)
        self.assertLessEqual(target["xy"][0], manager.map.width_cm - margin)
        self.assertLessEqual(target["xy"][1], manager.map.height_cm - margin)
        self.assertEqual(target["kind"], "interior_safe")
        self.assertNotIn("screen_id", target)
        self.assertGreaterEqual(
            manager.map.robot_clearance_cm(target["xy"]),
            manager.config["navigation"]["interior_recovery_min_clearance_cm"],
        )

    def test_recovery_uses_strafe_and_reverse_without_turning(self):
        manager = geometry_manager()
        manager.state = RobotState(manager.config)
        manager.state.set_pose(RobotPose(100.0, 100.0, 90.0, Confidence.HIGH, "TEST", 1.0))
        manager.time_left_s = lambda: 100.0
        manager.recovery_translation_clear = lambda *args, **kwargs: True
        manager.publish_state = lambda *args, **kwargs: None
        manager.hardware = SimpleNamespace(center_head=lambda: None)
        manager.last_localize_success_s = 1.0
        manager.last_localization_pose_conflict = False
        localized = []

        def localize(*args, **kwargs):
            localized.append(True)
            manager.state.set_pose(TaskManager.copy_pose(manager.state.pose))
            return True

        manager.localize_scan = localize
        actions = []

        def run(key, times_override=1):
            spec = manager.config["motion"]["actions"][key]
            result = ActionResult(
                key=key,
                group=spec.get("group", key),
                times=times_override,
                elapsed_s=0.0,
                model_forward_cm=float(spec.get("forward_cm", 0.0)),
                model_lateral_cm=float(spec.get("lateral_cm", 0.0)),
                model_yaw_deg=float(spec.get("yaw_deg", 0.0)),
                ok=True,
                executed_times=times_override,
            )
            actions.append(key)
            manager.state.apply_action_result(result)
            return result

        manager.motion = SimpleNamespace(run=run)
        self.assertTrue(manager.blind_navigate_to_xy((110.0, 90.0), "test_left_back"))
        self.assertIn("strafe_right_fast", actions)
        self.assertIn("back_fast", actions)
        self.assertFalse(any(key.startswith("turn_") for key in actions))
        self.assertEqual(manager.state.pose.yaw_deg, 90.0)
        self.assertEqual(len(localized), 1)


if __name__ == "__main__":
    unittest.main()
