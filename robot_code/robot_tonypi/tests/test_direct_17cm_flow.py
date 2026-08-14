from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.hardware import TonyPiHardware
from robot_tonypi.load_pos import load_tag_pos
from robot_tonypi.map_model import MapModel
from robot_tonypi.models import (
    ActionResult,
    ClassificationResult,
    Confidence,
    MissionState,
    RobotPose,
    Screen,
    ScreenStatus,
    TargetVisualConfirmation,
    VisualAuthorization,
    WorkerChangeResult,
)
from robot_tonypi.task_manager import TaskManager


def make_screen(screen_id=1, xy=(25.0, -2.0)):
    return Screen(
        screen_id=screen_id,
        tag_corners_3d=None,
        center_xy=(0.0, 0.0),
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=xy,
        interaction_xy=xy,
        interaction_yaw_deg=180.0,
        reader_xy=(0.0, -5.0),
        screen_left_tangent_xy=(0.0, -1.0),
        task_target_xy=xy,
        task_target_yaw_deg=180.0,
        worker_id=screen_id,
    )


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))

    def save_image(self, *args, **kwargs):
        return None

    def save_crop(self, *args, **kwargs):
        return None


def classification_manager(tag_ids, candidates, classification):
    manager = TaskManager.__new__(TaskManager)
    manager.config = load_config(None)
    manager.config["vision"]["vote_frames"] = 1
    manager.config["vision"]["min_votes"] = 1
    manager.config["vision"]["harvest_pan_angles"] = [100]
    manager.config["interaction"]["target_confirmation_retry_interval_s"] = 0.0
    manager.args = SimpleNamespace(dry_run=False, skip_change=False)
    manager.target_flower = "hehua"
    manager.current_target_screen_id = 1
    manager.arrived_at_target = True
    manager.mission_state = MissionState.ARRIVED_AT_TARGET
    manager.classifier_allowed = False
    manager.target_visual_confirmation = None
    manager.visual_authorization = None
    manager.final_forward_executed = False
    manager.target_confirmation_retry_count = 0
    manager.target_confirmation_recovery_cycle = 0
    manager.last_target_confirmation_diagnostics = {}
    manager.last_vote_summary = {}
    manager.state = SimpleNamespace(
        pose=RobotPose(25.0, -2.0, 180.0, Confidence.HIGH, "TEST", 1.0)
    )
    manager.debug = DebugStub()
    tags = [SimpleNamespace(tag_id=value) for value in tag_ids]
    manager.capture_with_tags = lambda pan: (np.zeros((20, 20, 3), dtype=np.uint8), tags)
    manager.screen_detector = SimpleNamespace(
        detect=lambda *args, **kwargs: candidates,
        annotate=lambda frame, *args, **kwargs: frame,
    )
    manager.classifier = SimpleNamespace(classify_crop=lambda crop: classification)
    manager.pan_angles_for_screen = lambda *args, **kwargs: [100]
    manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)
    manager.center_head_after_scan = lambda *args, **kwargs: None
    manager.publish_state = lambda *args, **kwargs: None
    return manager


def candidate(screen_id=1, tag_id=1):
    return SimpleNamespace(
        screen_id=screen_id,
        tag=SimpleNamespace(tag_id=tag_id),
        crop_28x28=np.zeros((28, 28, 3), dtype=np.uint8),
    )


class Direct25cmFlowTests(unittest.TestCase):
    def test_map_screens_have_one_25cm_target_with_lateral_offset(self):
        config = load_config(None)
        model = MapModel(load_tag_pos(), config)
        desired_lateral = config["interaction"]["target_lateral_offset_cm"]
        for item in model.screens.values():
            rel = (
                item.task_target_xy[0] - item.face_center_xy[0],
                item.task_target_xy[1] - item.face_center_xy[1],
            )
            normal_distance = rel[0] * item.cardinal_normal_xy[0] + rel[1] * item.cardinal_normal_xy[1]
            lateral_distance = rel[0] * item.screen_left_tangent_xy[0] + rel[1] * item.screen_left_tangent_xy[1]
            self.assertAlmostEqual(normal_distance, 25.0)
            self.assertAlmostEqual(lateral_distance, desired_lateral)
            self.assertEqual(item.target_xy, item.task_target_xy)
            self.assertEqual(item.interaction_xy, item.task_target_xy)
        west = model.screens[1]
        self.assertEqual(west.face_center_xy, (196.0, 17.5))
        self.assertEqual(west.tag_front_xy, (171.0, 17.5))
        self.assertEqual(west.task_target_xy, (171.0, 16.0))
        self.assertEqual(west.target_xy, west.task_target_xy)
        self.assertEqual(west.interaction_xy, west.task_target_xy)

    def test_high_cost_exact_target_is_not_moved_but_obstacle_is_rejected(self):
        model = MapModel(load_tag_pos(), load_config(None))
        target = model.screens[1].task_target_xy
        self.assertTrue(model.is_free_xy(target))
        model.cost[model.grid_pos(target)] = 80.0
        self.assertGreaterEqual(model.cost[model.grid_pos(target)], 60.0)
        path = model.plan((150.0, 17.5), target, allow_goal_high_cost=True)
        self.assertTrue(path)
        self.assertEqual(path[-1], target)
        action_path = model.plan_action_path(
            RobotPose(150.0, 22.5, 0.0, Confidence.HIGH, "TEST", 1.0),
            target,
            load_config(None)["navigation"],
            load_config(None)["motion"],
            allow_goal_high_cost=True,
        )
        self.assertTrue(action_path)
        self.assertEqual(action_path[-1], target)
        inside_building = (200.0, 17.5)
        self.assertEqual(model.plan((150.0, 17.5), inside_building, allow_goal_high_cost=True), [])

    def test_high_cost_exception_does_not_apply_to_ordinary_goal(self):
        model = MapModel(load_tag_pos(), load_config(None))
        target = model.screens[1].task_target_xy
        ordinary = model.plan((150.0, 17.5), target, allow_goal_high_cost=False)
        self.assertTrue(ordinary)
        self.assertNotEqual(ordinary[-1], target)

    def test_correct_tag_and_bound_screen_authorize_fpga(self):
        manager = classification_manager(
            [1], [candidate()], ClassificationResult(True, "chuju", confidence=0.95)
        )
        target = make_screen()
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        manager.final_forward_executed = True
        self.assertEqual(manager.classify_after_final_forward(target), 1)
        self.assertEqual(target.status, ScreenStatus.NEEDS_CHANGE)
        self.assertEqual(manager.visual_authorization.screen_id, 1)
        self.assertEqual(manager.visual_authorization.tag_id, 1)

    def test_tag_without_screen_blocks_fpga_and_authorization(self):
        calls = []
        result = ClassificationResult(True, "chuju", confidence=0.95)
        manager = classification_manager([1], [], result)
        manager.classifier.classify_crop = lambda crop: calls.append(crop) or result
        self.assertFalse(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertEqual(calls, [])
        self.assertIsNone(manager.visual_authorization)

    def test_screen_without_target_tag_blocks_fpga(self):
        manager = classification_manager([], [candidate()], ClassificationResult(True, "chuju", confidence=0.95))
        self.assertFalse(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertIsNone(manager.visual_authorization)

    def test_wrong_tag_binding_blocks_fpga(self):
        manager = classification_manager([1, 2], [candidate(2, 2)], ClassificationResult(True, "chuju", confidence=0.95))
        self.assertFalse(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertIsNone(manager.visual_authorization)

    def test_confirmation_retries_use_fresh_frames_and_configured_interval(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "chuju", confidence=0.95))
        manager.config["interaction"]["target_confirmation_retry_interval_s"] = 0.125
        calls = []
        outcomes = [None, None, candidate()]

        def capture(*args, **kwargs):
            calls.append(len(calls) + 1)
            manager.last_target_confirmation_diagnostics = {
                "target_tag_detected": True,
                "detected_tag_ids": [1],
                "screen_candidate_count": 0,
                "matched_screen_count": 0,
                "failure_reason": "target_screen_binding_missing",
            }
            return outcomes.pop(0)

        manager.capture_locked_target_candidate = capture
        with mock.patch("robot_tonypi.task_manager.time.sleep") as sleep:
            self.assertTrue(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.125)
        self.assertEqual(manager.target_confirmation_retry_count, 0)

    def test_confirmation_round_exhaustion_counts_failures_without_selecting_target(self):
        manager = classification_manager([1], [], ClassificationResult(True, "chuju", confidence=0.95))
        manager.config["interaction"]["target_confirmation_retry_interval_s"] = 0.0
        self.assertFalse(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertEqual(manager.target_confirmation_retry_count, 3)
        states = [data.get("state") for name, data in manager.debug.events if name == "mission_state"]
        self.assertNotIn(MissionState.SELECT_NEAREST_TARGET.value, states)

    def test_visibility_recovery_success_keeps_target_and_reconfirms(self):
        manager = classification_manager([1], [], ClassificationResult(True, "chuju", confidence=0.95))
        target = make_screen()
        confirmations = [False, True]
        recoveries = []
        manager.confirm_target_tag_and_screen = lambda screen: confirmations.pop(0)
        manager.recover_target_visibility = lambda screen, cycle: recoveries.append((screen.screen_id, cycle)) or True
        self.assertTrue(manager.confirm_target_with_visibility_recovery(target))
        self.assertEqual(recoveries, [(1, 1)])
        self.assertEqual(manager.current_target_screen_id, 1)

    def test_unresolved_confirmation_stops_finitely_and_preserves_target(self):
        manager = classification_manager([1], [], ClassificationResult(True, "chuju", confidence=0.95))
        target = make_screen()
        manager.confirm_target_tag_and_screen = lambda screen: False
        manager.recover_target_visibility = lambda screen, cycle: False
        manager.preserve_current_target = lambda screen, reason: setattr(manager, "current_target_screen_id", screen.screen_id)
        manager.last_navigation_failure_reason = ""
        self.assertFalse(manager.confirm_target_with_visibility_recovery(target))
        self.assertEqual(manager.mission_state, MissionState.MISSION_FAILED)
        self.assertEqual(manager.current_target_screen_id, 1)
        self.assertEqual(target.attempts, 1)
        self.assertFalse(manager.final_forward_executed)
        unresolved = [data for name, data in manager.debug.events if name == "target_screen_confirmation_unresolved"]
        self.assertEqual(len(unresolved), 1)
        self.assertTrue(unresolved[0]["target_preserved"])

    def test_fpga_failure_blocks_authorization(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(False, error="fpga_down"))
        target = make_screen()
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        manager.final_forward_executed = True
        self.assertEqual(manager.classify_after_final_forward(target), 0)
        self.assertIsNone(manager.visual_authorization)

    def test_low_confidence_blocks_authorization(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "chuju", confidence=0.10))
        target = make_screen()
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        manager.final_forward_executed = True
        self.assertEqual(manager.classify_after_final_forward(target), 0)
        self.assertIsNone(manager.visual_authorization)

    def test_already_target_is_known_only_after_single_forward_and_does_not_change(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "hehua", confidence=0.96))
        target = make_screen()
        manager.motion = SimpleNamespace(
            run=lambda key, times_override=1: ActionResult(
                key=key, group="go_forward_one_step", times=times_override,
                elapsed_s=0.0, model_forward_cm=15.0, ok=True,
            )
        )
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        self.assertTrue(manager.execute_final_forward(target))
        self.assertEqual(manager.classify_after_final_forward(target), 1)
        self.assertEqual(target.status, ScreenStatus.ALREADY_TARGET)
        self.assertTrue(manager.final_forward_executed)

    def interaction_manager(self, *, motion_ok=True, skip_change=False):
        target = make_screen()
        target.last_classification = "chuju"
        target.status = ScreenStatus.NEEDS_CHANGE
        manager = TaskManager.__new__(TaskManager)
        manager.config = load_config(None)
        manager.args = SimpleNamespace(dry_run=False, skip_change=skip_change)
        manager.target_flower = "hehua"
        manager.current_target_screen_id = 1
        manager.arrived_at_target = True
        manager.target_visual_confirmation = TargetVisualConfirmation(1, 1, True, 90.0)
        manager.visual_authorization = VisualAuthorization(1, 1, True, "chuju", 0.95, 100.0)
        manager.final_forward_executed = True
        manager.last_interaction_check = None
        manager.latest_interaction_result = None
        manager.recent_interaction_results = []
        manager.state = SimpleNamespace(pose=None)
        manager.debug = DebugStub()
        manager.sequence = []
        manager.motion = SimpleNamespace(
            run=lambda key, times_override=1: manager.sequence.append(("motion", key, times_override))
            or ActionResult(
                key=key,
                group="go_forward_one_step",
                times=1,
                elapsed_s=0.0,
                model_forward_cm=15.0,
                ok=motion_ok,
                error="" if motion_ok else "failed",
            )
        )

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["screen_id"], kwargs["worker_id"]))
            return WorkerChangeResult(True, simulated=skip_change, worker_id=1, response={"ok": True})

        manager.interaction = SimpleNamespace(change_flower=change_flower)
        manager.set_mission_state = lambda state: setattr(manager, "mission_state", state)
        manager.write_interaction_audit = lambda record: None
        return manager, target

    def test_post_forward_needs_change_runs_transaction_without_extra_motion(self):
        manager, target = self.interaction_manager()
        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(manager.sequence, [("change", 1, 1)])
        self.assertTrue(manager.final_forward_executed)
        self.assertEqual(target.status, ScreenStatus.CHANGED)

    def test_15cm_path_contains_no_sensing_or_extra_navigation_calls(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {
            "localize_scan",
            "capture_with_tags",
            "navigate_to_xy",
            "turn_toward_yaw_boundary_aware",
            "choose_translation_action",
            "classify_crop",
        }
        for name in ("execute_final_forward", "process_screen_interaction"):
            fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
            calls = {getattr(call.func, "attr", "") for call in ast.walk(fn) if isinstance(call, ast.Call)}
            self.assertTrue(forbidden.isdisjoint(calls), (name, calls & forbidden))

    def test_configuration_has_no_old_two_stage_distance(self):
        config = load_config(None)
        self.assertEqual(config["interaction"]["target_distance_cm"], 25.0)
        self.assertEqual(config["interaction"]["target_lateral_offset_cm"], -1.5)
        self.assertEqual(config["interaction"]["target_final_forward_cm"], 15.0)
        self.assertEqual(config["navigation"]["target_arrival_radius_cm"], 3.0)
        self.assertNotIn("target_arrival_distance_cm", config["map"])
        self.assertNotIn("interaction_staging_distance_cm", config["interaction"])
        self.assertNotIn("interaction_staging_arrival_radius_cm", config["interaction"])
        self.assertEqual(
            config["motion"]["actions"]["interaction_forward_15cm"]["forward_cm"],
            15.0,
        )
        final_action = config["motion"]["actions"]["interaction_forward_15cm"]
        self.assertEqual(final_action["times"], 1)
        self.assertEqual(
            [(step["group"], step["times"]) for step in final_action["sequence"]],
            [("go_forward_one_step", 3)],
        )
        self.assertEqual(config["vision"]["max_screen_area_ratio"], 0.98)

    def test_final_15cm_hardware_sequence_is_one_logical_action(self):
        hardware = TonyPiHardware(load_config(None), dry_run=True)
        with mock.patch("robot_tonypi.hardware.time.sleep"), mock.patch("builtins.print"):
            result = hardware.run_action("interaction_forward_15cm", times_override=1)
        self.assertTrue(result.ok)
        self.assertEqual(result.times, 1)
        self.assertEqual(result.executed_times, 1)
        self.assertEqual(result.model_forward_cm, 15.0)
        self.assertEqual(result.group, "go_forward_one_step")

    def test_flow_order_is_confirm_then_forward_then_classify_then_change(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_mission"
        )
        line = {}
        for call in (node for node in ast.walk(fn) if isinstance(node, ast.Call)):
            name = getattr(call.func, "attr", "")
            if name in {
                "confirm_target_with_visibility_recovery",
                "execute_final_forward",
                "classify_after_final_forward",
                "process_screen_interaction",
            }:
                line.setdefault(name, call.lineno)
        self.assertLess(line["confirm_target_with_visibility_recovery"], line["execute_final_forward"])
        self.assertLess(line["execute_final_forward"], line["classify_after_final_forward"])
        self.assertLess(line["classify_after_final_forward"], line["process_screen_interaction"])

    def test_15cm_failure_blocks_change(self):
        manager, target = self.interaction_manager(motion_ok=False)
        manager.final_forward_executed = False
        self.assertFalse(manager.execute_final_forward(target))
        self.assertEqual(manager.sequence, [("motion", "interaction_forward_15cm", 1)])
        self.assertEqual(target.status, ScreenStatus.NEEDS_CHANGE)

    def test_final_15cm_is_executed_only_once(self):
        manager, target = self.interaction_manager()
        manager.final_forward_executed = False
        self.assertTrue(manager.execute_final_forward(target))
        self.assertFalse(manager.execute_final_forward(target))
        self.assertEqual(manager.sequence, [("motion", "interaction_forward_15cm", 1)])

    def test_skip_change_runs_no_15cm_hardware_action(self):
        manager, target = self.interaction_manager(skip_change=True)
        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(manager.sequence, [("change", 1, 1)])

    def test_old_authorization_cannot_cross_target_lock(self):
        manager, target = self.interaction_manager()
        manager.current_target_screen_id = 2
        self.assertFalse(manager.process_screen_interaction(target))
        self.assertEqual(manager.sequence, [])


if __name__ == "__main__":
    unittest.main()
