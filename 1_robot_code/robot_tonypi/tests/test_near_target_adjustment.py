from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.models import (
    ActionResult,
    Confidence,
    MissionState,
    RobotPose,
    Screen,
    TargetGoal,
)
from robot_tonypi.motion import RobotState
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.utils import distance_xy, now_s


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))

    def render_map(self, *args, **kwargs):
        return None


def screen_and_goal(goal_xy=(100.0, 100.0), yaw=0.0):
    screen = Screen(
        screen_id=1,
        tag_corners_3d=None,
        center_xy=goal_xy,
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=goal_xy,
        interaction_xy=goal_xy,
        interaction_yaw_deg=yaw,
        reader_xy=goal_xy,
        screen_left_tangent_xy=(0.0, 1.0),
        navigation_staging_xy=goal_xy,
        interaction_target_xy=goal_xy,
        task_target_xy=goal_xy,
        task_target_yaw_deg=yaw,
    )
    goal = TargetGoal(
        screen_id=1,
        tag_id=1,
        anchor_xy=goal_xy,
        goal_xy=goal_xy,
        navigation_staging_xy=goal_xy,
        interaction_target_xy=goal_xy,
        desired_yaw_deg=yaw,
        source="TEST",
        generation_id=1,
    )
    return screen, goal


def navigation_manager(pose, goal_xy=(100.0, 100.0), goal_yaw=0.0):
    manager = TaskManager.__new__(TaskManager)
    manager.config = load_config(None)
    manager.debug = DebugStub()
    manager.state = RobotState(manager.config)
    manager.state.set_pose(pose)
    manager.mission_state = MissionState.POSITION_NAVIGATION
    manager.active_navigation_phase = "POSITION_NAVIGATION"
    manager.active_navigation_plan = None
    manager.last_navigation_failure_reason = ""
    manager.last_motion_action = ""
    manager.last_localize_success_s = now_s()
    manager.current_target_screen_id = 1
    manager.current_target_goal = None
    manager.time_left_s = lambda: 100.0
    manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)
    manager.clear_plan_failure_watchdog = lambda *args, **kwargs: None
    manager.validate_target_goal = lambda *args, **kwargs: True
    manager.adaptive_relocalization_decision = lambda *args, **kwargs: {
        "decision": "continue", "reason": "test"
    }
    manager.visual_pose_is_fresh = lambda *args, **kwargs: True
    manager.near_wall_now = lambda *args, **kwargs: False
    manager.publish_state = lambda *args, **kwargs: None
    manager.recover_via_indoor_waypoint = lambda *args, **kwargs: False
    manager.map = SimpleNamespace(
        last_action_plan_metrics={},
        target_rotation_sweep_clear=lambda *args, **kwargs: True,
        plan_motion_actions=lambda *args, **kwargs: None,
    )
    screen, goal = screen_and_goal(goal_xy, goal_yaw)
    return manager, screen, goal


class SafeContinuousMap:
    def __init__(self, reject=None):
        self.reject = reject

    def in_bounds_xy(self, xy):
        return 0.0 <= xy[0] <= 300.0 and 0.0 <= xy[1] <= 300.0

    def target_direct_corridor_metrics(self, start, end, *args, **kwargs):
        reason = None if self.reject is None else self.reject(start, end)
        return {
            "clear": reason is None,
            "physical_collision": reason == "physical_collision",
            "soft_cost_rejected": reason == "soft_cost_rejected",
            "clearance_rejected": reason == "clearance_rejected",
        }


def adjustment_manager(pose, goal_xy, reject=None):
    manager, screen, goal = navigation_manager(pose, goal_xy, 0.0)
    manager.map = SafeContinuousMap(reject=reject)
    manager.publish_state = lambda *args, **kwargs: None
    calls = []

    def run(key, times_override=1):
        calls.append((key, times_override))
        spec = manager.config["motion"]["actions"][key]
        return ActionResult(
            key=key,
            group=key,
            times=times_override,
            elapsed_s=0.0,
            model_forward_cm=float(spec.get("forward_cm", 0.0)),
            model_lateral_cm=float(spec.get("lateral_cm", 0.0)),
            model_yaw_deg=0.0,
            ok=True,
            executed_times=times_override,
        )

    manager.motion = SimpleNamespace(run=run)
    localization_calls = []

    def localize_scan(**kwargs):
        localization_calls.append(kwargs)
        if calls:
            key = calls[-1][0]
            spec = manager.config["motion"]["actions"][key]
            xy = manager.translated_pose_xy(
                pose,
                forward_cm=float(spec.get("forward_cm", 0.0)),
                lateral_cm=float(spec.get("lateral_cm", 0.0)),
            )
            manager.state.set_pose(RobotPose(
                xy[0], xy[1], pose.yaw_deg,
                Confidence.HIGH, "VISION", now_s(),
            ))
        return True

    manager.localize_scan = localize_scan
    return manager, screen, goal, calls, localization_calls


class NearTargetAdjustmentTests(unittest.TestCase):
    def test_real_4_3cm_arrives_without_motion_astar(self):
        goal_xy = (248.5, 221.5)
        pose = RobotPose(248.13, 225.80, 0.0, Confidence.HIGH, "VISION", now_s())
        self.assertAlmostEqual(distance_xy(pose.xy(), goal_xy), 4.316, places=3)
        manager, screen, goal = navigation_manager(pose, goal_xy, 0.0)
        manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
            "real distance inside 5 cm must not enter Motion A*"
        )
        manager.localize_scan = lambda *args, **kwargs: self.fail(
            "fresh real pose should complete position immediately"
        )
        self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))
        self.assertEqual(manager.active_navigation_phase, "ARRIVED")
        self.assertNotIn("motion_astar_failed", [name for name, _ in manager.debug.events])

    def test_real_5cm_arrives(self):
        pose = RobotPose(105.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal = navigation_manager(pose)
        manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
            "exact arrival boundary must not enter Motion A*"
        )
        self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))

    def test_5_1cm_and_exact_10cm_use_near_target_not_astar(self):
        for distance in (5.1, 10.0):
            with self.subTest(distance=distance):
                pose = RobotPose(100.0 + distance, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
                manager, screen, goal = navigation_manager(pose)
                calls = []

                def adjust(*args):
                    calls.append(args)
                    manager.state.set_pose(RobotPose(
                        100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s()
                    ))
                    return "moved"

                manager.perform_near_target_adjustment = adjust
                manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
                    "5-10 cm must not enter Motion A*"
                )
                self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))
                self.assertEqual(len(calls), 1)

    def test_over_10cm_still_uses_motion_astar(self):
        pose = RobotPose(111.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal = navigation_manager(pose)
        planner_calls = []

        def plan(*args, **kwargs):
            planner_calls.append((args, kwargs))
            return SimpleNamespace(actions=[], path_xy=[pose.xy(), goal.interaction_target_xy], total_cost=6.0)

        manager.map.plan_motion_actions = plan
        manager.execute_motion_astar_action = lambda *args: (
            manager.state.set_pose(RobotPose(
                100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s()
            )) or True
        )
        manager.perform_near_target_adjustment = lambda *args: self.fail(
            "distance over 10 cm must use Motion A*"
        )
        self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))
        self.assertEqual(len(planner_calls), 1)
        self.assertFalse(planner_calls[0][1]["require_goal_yaw"])

    def test_strafe_left_is_best_and_only_one_cycle_then_localize(self):
        pose = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal, calls, localizations = adjustment_manager(
            pose, (100.0, 108.0)
        )
        self.assertEqual(manager.perform_near_target_adjustment(screen, goal, 1), "moved")
        self.assertEqual(calls, [("strafe_left_fast", 1)])
        self.assertEqual(localizations, [{"reason": "near_target_adjustment"}])
        candidates = {
            data["action"]: data
            for name, data in manager.debug.events
            if name == "near_target_candidate_evaluated"
        }
        self.assertEqual(candidates["forward_fast"]["rejection_reason"], "does_not_reduce_goal_distance")
        self.assertLess(
            candidates["strafe_left_fast"]["predicted_distance_cm"],
            candidates["strafe_right_fast"]["predicted_distance_cm"],
        )

    def test_near_rear_goal_selects_back_with_reverse_gates(self):
        pose = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal, calls, _ = adjustment_manager(pose, (92.0, 100.0))
        self.assertEqual(manager.perform_near_target_adjustment(screen, goal, 1), "moved")
        self.assertEqual(calls, [("back_fast", 1)])

    def test_bad_rear_angle_rejects_back(self):
        pose = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        goal_xy = (94.0, 100.0 + math.sqrt(28.0))
        manager, screen, goal, calls, _ = adjustment_manager(pose, goal_xy)
        manager.perform_near_target_adjustment(screen, goal, 1)
        back = next(
            data for name, data in manager.debug.events
            if name == "near_target_candidate_evaluated" and data["action"] == "back_fast"
        )
        self.assertEqual(back["rejection_reason"], "rear_angle_exceeds_tolerance")
        self.assertNotEqual(calls[0][0], "back_fast")

    def test_low_localization_confidence_rejects_back(self):
        pose = RobotPose(100.0, 100.0, 0.0, Confidence.LOW, "VISION", now_s())
        manager, screen, goal, calls, _ = adjustment_manager(pose, (92.0, 100.0))
        manager.perform_near_target_adjustment(screen, goal, 1)
        back = next(
            data for name, data in manager.debug.events
            if name == "near_target_candidate_evaluated" and data["action"] == "back_fast"
        )
        self.assertEqual(back["rejection_reason"], "localization_confidence_low")
        self.assertFalse(any(key == "back_fast" for key, _ in calls))

    def test_hard_collision_candidate_is_rejected(self):
        pose = RobotPose(100.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())

        def reject(start, end):
            if end[1] > start[1]:
                return "physical_collision"
            return None

        manager, screen, goal, calls, localizations = adjustment_manager(
            pose, (100.0, 108.0), reject=reject
        )
        self.assertEqual(manager.perform_near_target_adjustment(screen, goal, 1), "stalled")
        self.assertEqual(calls, [])
        self.assertEqual(localizations, [{"reason": "near_target_adjustment_stalled"}])
        left = next(
            data for name, data in manager.debug.events
            if name == "near_target_candidate_evaluated" and data["action"] == "strafe_left_fast"
        )
        self.assertFalse(left["safety_ok"])
        self.assertEqual(left["rejection_reason"], "physical_collision")

    def test_four_stalls_exhaust_to_existing_recovery(self):
        pose = RobotPose(108.0, 100.0, 0.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal = navigation_manager(pose)
        attempts = []
        manager.perform_near_target_adjustment = lambda *args: attempts.append(args[2]) or "stalled"
        manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
            "near target stalls must not enter Motion A*"
        )
        recoveries = []
        manager.recover_via_indoor_waypoint = lambda reason: recoveries.append(reason) or False
        self.assertFalse(manager.navigate_motion_plan_to_target(screen, goal))
        self.assertEqual(attempts, [1, 2, 3, 4])
        self.assertEqual(recoveries, ["near_target_adjustment_exhausted"])
        exhausted = [
            data for name, data in manager.debug.events
            if name == "near_target_adjustment_exhausted"
        ]
        self.assertEqual(exhausted[0]["attempts"], 4)

    def test_arrival_with_wrong_yaw_goes_to_final_yaw_not_position_planner(self):
        pose = RobotPose(104.0, 100.0, -60.0, Confidence.HIGH, "VISION", now_s())
        manager, screen, goal = navigation_manager(pose, goal_yaw=0.0)
        manager.map.plan_motion_actions = lambda *args, **kwargs: self.fail(
            "arrived XY must not re-enter Position A*"
        )
        turns = []

        def turn(yaw):
            turns.append(yaw)
            manager.state.set_pose(RobotPose(
                104.0, 100.0, yaw, Confidence.HIGH, "VISION", now_s()
            ))
            return True

        manager.turn_toward_yaw_boundary_aware = turn
        self.assertTrue(manager.navigate_motion_plan_to_target(screen, goal))
        self.assertEqual(turns, [0.0])


if __name__ == "__main__":
    unittest.main()
