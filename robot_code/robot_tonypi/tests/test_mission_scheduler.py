import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.models import Confidence, MissionState, RobotPose, Screen, ScreenStatus
from robot_tonypi.task_manager import TaskManager, evaluate_turn_progress
from robot_tonypi.config import load_config
from robot_tonypi.main import parse_args
from robot_tonypi.interaction_logic import store_flower_observation
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.tests.test_capture_fpga_change import worker_id_for_screen as capture_worker_id_for_screen


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
        task_target_xy=xy,
        task_target_yaw_deg=180.0,
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
    manager.config = load_config(None)
    return manager


class MissionSchedulerTests(unittest.TestCase):
    def test_screen_and_worker_ids_are_identical_without_manual_mapping(self):
        config = load_config(None)
        self.assertNotIn("worker_mapping", config["interaction"])
        model = MapModel(load_tag_pos(), config)
        for item in model.screens.values():
            self.assertEqual(item.worker_id, item.screen_id)

        target = screen(25, (10.0, 0.0))
        target.worker_id = None  # A stale object cannot override the competition rule.
        manager = TaskManager.__new__(TaskManager)
        self.assertEqual(manager.worker_id_for_screen(target), 25)
        self.assertEqual(capture_worker_id_for_screen(25), 25)

    def test_map_and_tag_reference_coordinates_are_unchanged(self):
        config = load_config(None)
        tags = load_tag_pos()
        self.assertEqual(config["map"]["width_cm"], 300.0)
        self.assertEqual(config["map"]["height_cm"], 300.0)
        self.assertEqual(tags["1"].tolist(), [[196.0, 30.0, 39.8], [196.0, 25.0, 39.8], [196.0, 25.0, 34.8], [196.0, 30.0, 34.8]])
        self.assertEqual(tags["2"].tolist(), [[196.0, 5.0, 39.8], [201.0, 5.0, 39.8], [201.0, 5.0, 34.8], [196.0, 5.0, 34.8]])

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

    def test_nearest_target_uses_current_pose_to_15cm_task_target(self):
        old_near_new_far = screen(3, (30.0, 0.0))
        old_near_new_far.target_xy = (2.0, 0.0)
        old_near_new_far.task_target_xy = (30.0, 0.0)
        old_far_new_near = screen(2, (10.0, 0.0))
        old_far_new_near.target_xy = (80.0, 0.0)
        old_far_new_near.task_target_xy = (10.0, 0.0)
        manager = manager_at(0.0, 0.0, [old_near_new_far, old_far_new_near])
        self.assertEqual(manager.choose_nearest_screen().screen_id, 2)
        self.assertEqual(
            manager.last_target_plan["selection_rule"],
            "euclidean_current_pose_to_15cm_task_target_then_tag_id",
        )

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
        manager.state = SimpleNamespace(pose=RobotPose(10.0, 0.0, 180.0, Confidence.HIGH, "TEST", 100.0))
        manager.config = load_config(None)
        manager.arrival_geometry_check = lambda target: SimpleNamespace(ready=True)
        self.assertFalse(manager.classifier_gate_open(target))
        manager.arrived_at_target = True
        manager.mission_state = MissionState.ARRIVED_AT_TARGET
        self.assertTrue(manager.classifier_gate_open(target))
        self.assertFalse(manager.classifier_gate_open(screen(3, (10.0, 0.0))))

    def test_old_34cm_approach_cannot_open_classifier_gate(self):
        target = screen(2, (15.0, -5.0))
        target.target_xy = (34.0, -5.0)
        target.center_xy = (0.0, 0.0)
        target.normal_xy = (1.0, 0.0)
        target.screen_left_tangent_xy = (0.0, -1.0)
        target.interaction_xy = (15.0, -5.0)
        manager = TaskManager.__new__(TaskManager)
        manager.current_target_screen_id = 2
        manager.arrived_at_target = True
        manager.mission_state = MissionState.ARRIVED_AT_TARGET
        manager.state = SimpleNamespace(pose=RobotPose(34.0, -5.0, 180.0, Confidence.HIGH, "TEST", 100.0))
        manager.config = load_config(None)
        with unittest.mock.patch("robot_tonypi.interaction_logic.now_s", return_value=100.0):
            self.assertFalse(manager.classifier_gate_open(target))

    def test_non_target_flower_interaction_does_not_navigate_back_to_staging(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "process_screen_interaction")
        calls = {getattr(call.func, "attr", "") for call in ast.walk(fn) if isinstance(call, ast.Call)}
        self.assertNotIn("navigate_to_interaction_pose", function_names)
        self.assertNotIn("navigate_to_interaction_pose", calls)
        self.assertIn("arrival_geometry_check", calls)

    def test_full_interaction_gate_requires_locked_arrived_target(self):
        target = screen(2, (15.0, -5.0))
        target.center_xy = (0.0, 0.0)
        target.normal_xy = (1.0, 0.0)
        target.screen_left_tangent_xy = (0.0, -1.0)
        target.interaction_xy = (15.0, -5.0)
        target.worker_id = 12
        target.last_classification = "mudan"
        manager = TaskManager.__new__(TaskManager)
        manager.target_flower = "hehua"
        manager.config = load_config(None)
        manager.current_target_screen_id = 3
        manager.arrived_at_target = True
        pose = RobotPose(15.0, -5.0, 180.0, Confidence.HIGH, "TEST", 100.0)
        with unittest.mock.patch("robot_tonypi.interaction_logic.now_s", return_value=100.0):
            self.assertIn("target_lock_mismatch", manager.interaction_pose_check(target, pose).reasons)
            manager.current_target_screen_id = 2
            manager.arrived_at_target = False
            self.assertIn("target_not_arrived", manager.interaction_pose_check(target, pose).reasons)

    def test_navigation_approach_is_not_arrival_and_final_alignment_is_required(self):
        target = screen(2, (15.0, -5.0))
        target.target_xy = (34.0, -5.0)
        manager = TaskManager.__new__(TaskManager)
        manager.config = load_config(None)
        manager.arrived_at_target = True
        states = []
        manager.set_mission_state = lambda state: states.append(state)
        manager.navigate_to_xy = lambda *args, **kwargs: True
        manager.align_for_screen_interaction = lambda target, geometry_only=False: geometry_only
        manager.arrival_geometry_check = lambda target: SimpleNamespace(
            ready=True, as_dict=lambda: {"ready": True}
        )
        manager.debug = SimpleNamespace(event=lambda *args, **kwargs: None)
        manager.last_interaction_check = None
        self.assertTrue(manager.navigate_to_task_pose(target))
        self.assertFalse(manager.arrived_at_target)
        self.assertEqual(states, [MissionState.NAVIGATE_TO_APPROACH, MissionState.FINAL_ALIGN_15CM])

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
