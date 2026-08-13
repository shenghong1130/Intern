import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.models import Confidence, MissionState, RobotPose, Screen, ScreenStatus
from robot_tonypi.task_manager import TaskManager, evaluate_turn_progress
from robot_tonypi.config import load_config
from robot_tonypi.main import parse_args
from robot_tonypi.interaction_logic import store_flower_observation


def screen(screen_id, xy):
    return Screen(
        screen_id=screen_id,
        tag_corners_3d=None,
        center_xy=xy,
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=xy,
        interaction_xy=xy,
        interaction_yaw_deg=180.0,
        reader_xy=xy,
        screen_left_tangent_xy=(0.0, -1.0),
        interaction_staging_xy=xy,
    )


class FakeMap:
    def __init__(self, screens):
        self.screens = {item.screen_id: item for item in screens}

    def unfinished_screens(self):
        return (item for item in self.screens.values() if not item.done())


def manager_at(x, y, screens):
    manager = TaskManager.__new__(TaskManager)
    manager.state = SimpleNamespace(pose=RobotPose(x, y, 180.0, Confidence.HIGH, "TEST", 1.0))
    manager.map = FakeMap(screens)
    manager.last_target_plan = {}
    manager.config = {"navigation": {"arrival_radius_cm": 23.0, "arrival_yaw_tolerance_deg": 30.0}}
    return manager


class MissionSchedulerTests(unittest.TestCase):
    def test_turn_progress_accepts_clear_yaw_change(self):
        result = evaluate_turn_progress(
            RobotPose(10, 10, 0, Confidence.HIGH, "BEFORE", 1),
            RobotPose(10, 10, 8, Confidence.HIGH, "AFTER", 2),
            expected_delta=7.5,
            target_yaw=30,
        )
        self.assertFalse(result["turn_no_progress"])
        self.assertFalse(result["direction_conflict"])
        self.assertFalse(result["reject_visual_pose"])

    def test_turn_progress_uses_wrapped_yaw_delta(self):
        result = evaluate_turn_progress(
            RobotPose(10, 10, 179, Confidence.HIGH, "BEFORE", 1),
            RobotPose(10, 10, -173, Confidence.HIGH, "AFTER", 2),
            expected_delta=7.5,
            target_yaw=-160,
        )
        self.assertAlmostEqual(result["actual_delta"], 8.0)
        self.assertFalse(result["reject_visual_pose"])

    def test_stale_visual_turn_pose_is_rejected(self):
        result = evaluate_turn_progress(
            RobotPose(10, 10, 20, Confidence.HIGH, "BEFORE", 1),
            RobotPose(10.2, 10.1, 20.4, Confidence.HIGH, "AFTER", 2),
            expected_delta=15,
            target_yaw=80,
        )
        self.assertTrue(result["suspect_stale_pose"])
        self.assertTrue(result["turn_no_progress"])
        self.assertTrue(result["reject_visual_pose"])

    def test_scan_after_turn_stale_pose_keeps_dead_reckoning(self):
        manager = TaskManager.__new__(TaskManager)
        before = RobotPose(10, 10, 0, Confidence.HIGH, "BEFORE", 1)
        dead_reckoning = RobotPose(10, 10, 15, Confidence.HIGH, "DEAD_RECKONING", 2)
        stale_visual = RobotPose(10.1, 10.1, 0.2, Confidence.HIGH, "VISION", 3)
        manager.args = SimpleNamespace(dry_run=False)
        manager.config = {
            "vision": {"scan_after_turn_enabled": True, "scan_after_turn_min_interval_s": 0},
            "camera": {"head_center_angle": 100},
        }
        manager.start_time = 0
        manager.time_left_s = lambda: 100
        manager.last_scan_after_turn_s = 0
        manager.capture_with_tags = lambda center: (SimpleNamespace(), [])
        manager.localizer = SimpleNamespace(
            estimate_from_frame=lambda *args, **kwargs: (stale_visual, SimpleNamespace())
        )
        manager.state = SimpleNamespace(pose=dead_reckoning, set_pose=lambda pose: setattr(manager.state, "pose", pose))
        manager.debug = SimpleNamespace(
            event=lambda *args, **kwargs: None,
            save_image=lambda *args, **kwargs: None,
        )
        manager.observe_transit_bindings = lambda frame, tags, annotated, center, reason: annotated
        manager.transit_bindings = {}
        manager.publish_state = lambda *args, **kwargs: None
        manager.last_localize_success_s = 0
        manager.consecutive_localize_failures = 0
        manager.consecutive_no_tag_scans = 0
        manager.evaluate_pending_progress = lambda pose: None
        result = manager.scan_after_turn(
            "test",
            "turn_left_large",
            SimpleNamespace(model_yaw_deg=15.0),
            before_pose=before,
            target_yaw=60.0,
        )
        self.assertTrue(result["suspect_stale_pose"])
        self.assertFalse(result["accepted"])
        self.assertEqual(manager.state.pose.yaw_deg, 15.0)
        self.assertEqual(manager.state.pose.source, "DEAD_RECKONING")
        self.assertEqual(manager.state.pose.confidence, Confidence.LOW)

    def test_wrong_direction_turn_is_rejected(self):
        result = evaluate_turn_progress(
            RobotPose(10, 10, 20, Confidence.HIGH, "BEFORE", 1),
            RobotPose(10, 10, 12, Confidence.HIGH, "AFTER", 2),
            expected_delta=7.5,
            target_yaw=80,
        )
        self.assertTrue(result["direction_conflict"])
        self.assertTrue(result["reject_visual_pose"])

    def test_successful_turn_clears_counter(self):
        manager = TaskManager.__new__(TaskManager)
        manager.turn_no_progress_count = 2
        manager.turn_progress_failure_start_diff = 40.0
        manager.debug = SimpleNamespace(event=lambda *args, **kwargs: None)
        manager.clear_turn_progress_watchdog("test")
        self.assertEqual(manager.turn_no_progress_count, 0)
        self.assertIsNone(manager.turn_progress_failure_start_diff)

    def test_two_failed_turns_abort_navigation(self):
        manager = TaskManager.__new__(TaskManager)
        manager.turn_no_progress_count = 1
        manager.turn_progress_failure_start_diff = 60.0
        manager.turn_navigation_abort = False
        manager.debug = SimpleNamespace(event=lambda *args, **kwargs: None)
        manager.state = SimpleNamespace(pose=RobotPose(0, 0, 1, Confidence.HIGH, "RELOCALIZED", 3))
        manager.scan_after_turn = lambda *args, **kwargs: {
            "accepted": False,
            "turn_no_progress": True,
            "direction_conflict": False,
            "before_yaw": 0.0,
            "after_yaw": 0.2,
            "expected_delta": 15.0,
            "actual_delta": 0.2,
            "diff_before": 60.0,
            "target_improvement_deg": 0.2,
        }
        manager.localize_scan = lambda reset_turn_watchdog=True: True
        action_result = SimpleNamespace(key="turn_left_large", model_yaw_deg=15.0)
        ok = manager.monitor_turn_result(
            RobotPose(0, 0, 0, Confidence.HIGH, "BEFORE", 1),
            60.0,
            action_result,
            "test",
        )
        self.assertFalse(ok)
        self.assertTrue(manager.turn_navigation_abort)
        self.assertEqual(manager.turn_no_progress_count, 2)

    def test_nearest_target_uses_current_pose_to_target_xy(self):
        manager = manager_at(0.0, 0.0, [screen(3, (30.0, 0.0)), screen(2, (10.0, 0.0))])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 2)

    def test_reselects_from_latest_pose_after_completion(self):
        first = screen(1, (10.0, 0.0))
        second = screen(2, (80.0, 0.0))
        third = screen(3, (20.0, 0.0))
        manager = manager_at(0.0, 0.0, [first, second, third])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 1)
        first.status = ScreenStatus.ALREADY_TARGET
        manager.state.pose.x_cm = 75.0
        self.assertEqual(manager.choose_nearest_screen().screen_id, 2)

    def test_equal_distance_tie_breaks_by_tag_id(self):
        manager = manager_at(0.0, 0.0, [screen(9, (-10.0, 0.0)), screen(4, (10.0, 0.0))])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 4)

    def test_completed_and_failed_targets_are_not_reselected(self):
        complete = screen(1, (1.0, 0.0))
        complete.status = ScreenStatus.CHANGED
        invalid = screen(2, (2.0, 0.0))
        invalid.status = ScreenStatus.FAILED
        available = screen(3, (30.0, 0.0))
        manager = manager_at(0.0, 0.0, [complete, invalid, available])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 3)

    def test_classifier_gate_requires_arrival_and_locked_target(self):
        target = screen(2, (10.0, 0.0))
        manager = TaskManager.__new__(TaskManager)
        manager.current_target_screen_id = 2
        manager.arrived_at_target = False
        manager.mission_state = MissionState.NAVIGATE_TO_TARGET
        manager.state = SimpleNamespace(pose=RobotPose(10.0, 0.0, 180.0, Confidence.HIGH, "TEST", 1.0))
        manager.config = {"navigation": {"arrival_radius_cm": 23.0, "arrival_yaw_tolerance_deg": 30.0}}
        self.assertFalse(manager.classifier_gate_open(target))
        manager.arrived_at_target = True
        manager.mission_state = MissionState.ARRIVED_AT_TARGET
        self.assertTrue(manager.classifier_gate_open(target))
        self.assertFalse(manager.classifier_gate_open(screen(3, (10.0, 0.0))))

    def test_only_arrived_target_function_calls_classifier(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        callers = []
        for fn in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            if any(
                isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "classify_crop"
                for call in ast.walk(fn)
            ):
                callers.append(fn.name)
        self.assertEqual(callers, ["classify_arrived_target"])

    def test_navigation_has_no_task_level_observation_or_passby_branch(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        self.assertNotIn("observation_xy", source)
        self.assertNotIn("execute_passby_scan", source)
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "navigate_to_screen")
        calls = {getattr(call.func, "attr", "") for call in ast.walk(fn) if isinstance(call, ast.Call)}
        self.assertNotIn("classify_crop", calls)
        self.assertNotIn("change_flower", calls)

    def test_initial_localization_configuration_is_preserved(self):
        config = load_config(None)
        self.assertEqual(config["localization"]["scan_pan_angles"], [100, 135, 65, 155, 45])
        self.assertEqual(config["localization"]["startup_attempts"], 14)
        self.assertEqual(
            config["localization"]["startup_search_actions"],
            ["turn_left_fast", "turn_left_fast", "turn_left_fast", "turn_left_fast", "back_fast"],
        )
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "localize_scan")
        calls = {getattr(call.func, "attr", "") for call in ast.walk(fn) if isinstance(call, ast.Call)}
        self.assertIn("capture_with_tags", calls)
        self.assertIn("estimate_from_frame", calls)
        self.assertIn("observe_transit_bindings", calls)
        self.assertNotIn("classify_crop", calls)

    def test_cli_dry_run_and_skip_alias_semantics_are_preserved(self):
        dry = parse_args(["--target-flower", "hehua", "--dry-run"])
        alias = parse_args(["--target-flower", "hehua", "--skip-api"])
        self.assertTrue(dry.dry_run)
        self.assertFalse(dry.skip_change)
        self.assertTrue(alias.skip_change)

    def test_already_target_is_processed_without_physical_interaction(self):
        target = screen(2, (10.0, 0.0))
        decision = store_flower_observation(target, "hehua", "hehua", 0.95)
        self.assertEqual(decision, "already_target_observed")
        self.assertEqual(target.status, ScreenStatus.ALREADY_TARGET)
        self.assertTrue(target.done())
        self.assertFalse(target.needs_interaction())

    def test_transit_geometry_updates_no_flower_dependent_state(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "observe_transit_bindings")
        attrs = {node.attr for node in ast.walk(fn) if isinstance(node, ast.Attribute)}
        self.assertNotIn("last_classification", attrs)
        self.assertNotIn("last_confidence", attrs)
        self.assertNotIn("status", attrs)


if __name__ == "__main__":
    unittest.main()
