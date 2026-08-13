import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.interaction_client import RobotInteractionClient
from robot_tonypi.interaction_logic import (
    apply_worker_change_result,
    build_interaction_geometry,
    building_bounds_from_tags,
    building_centers_from_tags,
    cardinal_surface_from_tag,
    evaluate_arrival_geometry,
    evaluate_interaction_pose,
    face_center_from_bounds,
    store_flower_observation,
)
from robot_tonypi.models import (
    Confidence,
    InteractionPoseCheck,
    RobotPose,
    Screen,
    ScreenStatus,
    WorkerChangeResult,
)
from robot_tonypi.utils import now_s
from robot_tonypi.load_pos import load_tag_pos


def make_screen(worker_id=12):
    # normal=(+x) means the visible/front side is +x. A robot in front faces
    # yaw=180. Viewer-left is (0,-1), hence the 5 cm reader/robot offset.
    return Screen(
        screen_id=2,
        tag_corners_3d=None,
        center_xy=(0.0, 0.0),
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=(48.0, 0.0),
        interaction_xy=(15.0, -5.0),
        interaction_yaw_deg=180.0,
        reader_xy=(0.0, -5.0),
        screen_left_tangent_xy=(0.0, -1.0),
        interaction_staging_xy=(34.0, -5.0),
        worker_id=worker_id,
    )


def fresh_pose(x, y, yaw):
    return RobotPose(x, y, yaw, Confidence.HIGH, "TEST", now_s())


def ready_check():
    return InteractionPoseCheck(ready=True, pose_valid=True, distance_cm=15.0)


def make_client(actions, response=None, error=None):
    config = load_config(None)
    config["interaction"]["left_hand_settle_s"] = 0.0

    def act(name, **kwargs):
        actions.append(("act", name, kwargs))

    def send(**kwargs):
        actions.append(("send_request", kwargs))
        if error is not None:
            raise error
        return response

    return RobotInteractionClient(
        "red",
        "1234",
        "red-1",
        config,
        act_fn=act,
        send_request_fn=send,
    )


def run_client(client):
    return client.change_flower(
        screen_id=2,
        worker_id=12,
        from_flower="chuju",
        to_flower="hehua",
        safety_gate=ready_check,
    )


class InteractionFlowTests(unittest.TestCase):
    def test_skip_change_simulates_without_real_act_or_send(self):
        actions = []
        config = load_config(None)
        client = RobotInteractionClient(
            "red",
            "1234",
            "red-1",
            config,
            skip_change=True,
            act_fn=lambda *args, **kwargs: actions.append(("act", args, kwargs)),
            send_request_fn=lambda **kwargs: actions.append(("send", kwargs)),
        )
        result = run_client(client)
        self.assertTrue(result.success)
        self.assertTrue(result.simulated)
        self.assertEqual(actions, [])

    def test_dry_run_simulates_without_real_act_or_send(self):
        actions = []
        config = load_config(None)
        client = RobotInteractionClient(
            None,
            None,
            None,
            config,
            dry_run=True,
            act_fn=lambda *args, **kwargs: actions.append(("act", args, kwargs)),
            send_request_fn=lambda **kwargs: actions.append(("send", kwargs)),
        )
        result = run_client(client)
        self.assertTrue(result.success)
        self.assertTrue(result.simulated)
        self.assertEqual(actions, [])

    def test_four_tag_planes_quantize_to_cardinal_faces_and_15cm_targets(self):
        tag_poses = load_tag_pos()
        centers = building_centers_from_tags(tag_poses)
        bounds = building_bounds_from_tags(tag_poses)
        expected = {
            1: ("WEST", (-1.0, 0.0), (196.0, 17.5), (181.0, 17.5), (181.0, 22.5), 0.0),
            2: ("SOUTH", (0.0, -1.0), (208.5, 5.0), (208.5, -10.0), (203.5, -10.0), 90.0),
            3: ("EAST", (1.0, 0.0), (221.0, 17.5), (236.0, 17.5), (236.0, 12.5), -180.0),
            4: ("NORTH", (0.0, 1.0), (208.5, 30.0), (208.5, 45.0), (213.5, 45.0), -90.0),
        }
        cfg = load_config(None)["interaction"]
        for tag_id, (face, normal, face_center, tag_front, body_target, yaw) in expected.items():
            surface = cardinal_surface_from_tag(tag_poses[str(tag_id)], centers[0])
            actual_face_center = face_center_from_bounds(bounds[0], surface["face"])
            geometry = build_interaction_geometry(actual_face_center, surface["normal_xy"], cfg)
            self.assertEqual(surface["face"], face)
            self.assertEqual(surface["normal_xy"], normal)
            self.assertEqual(actual_face_center, face_center)
            base_target = (
                actual_face_center[0] + normal[0] * cfg["interaction_distance_cm"],
                actual_face_center[1] + normal[1] * cfg["interaction_distance_cm"],
            )
            self.assertEqual(base_target, tag_front)
            self.assertEqual(geometry["interaction_xy"], body_target)
            self.assertEqual(geometry["interaction_yaw_deg"], yaw)
            self.assertIn(yaw, (0.0, -180.0, 90.0, -90.0))

    def test_arrival_geometry_gate_does_not_require_flower_or_worker(self):
        screen = make_screen(worker_id=None)
        screen.last_classification = None
        check = evaluate_arrival_geometry(
            screen,
            fresh_pose(15.0, -5.0, 180.0),
            load_config(None)["interaction"],
        )
        self.assertTrue(check.ready)
        self.assertNotIn("flower_unknown", check.reasons)
        self.assertNotIn("worker_id_missing", check.reasons)

    def test_arrival_geometry_uses_face_center_instead_of_tag_center(self):
        screen = make_screen()
        screen.center_xy = (0.0, 8.0)  # Deliberately off the full-face center.
        screen.face_center_xy = (0.0, 0.0)
        check = evaluate_arrival_geometry(
            screen,
            fresh_pose(15.0, -5.0, 180.0),
            load_config(None)["interaction"],
        )
        self.assertTrue(check.ready)
        self.assertEqual(check.distance_error_cm, 0.0)
        self.assertEqual(check.lateral_error_cm, 0.0)

    def test_arrival_geometry_gate_rejects_distance_yaw_confidence_and_stale_pose(self):
        cfg = load_config(None)["interaction"]
        screen = make_screen()
        self.assertIn("distance", evaluate_arrival_geometry(screen, fresh_pose(34, -5, 180), cfg).reasons)
        self.assertIn("yaw", evaluate_arrival_geometry(screen, fresh_pose(15, -5, 90), cfg).reasons)
        low = RobotPose(15, -5, 180, Confidence.LOW, "TEST", now_s())
        self.assertFalse(evaluate_arrival_geometry(screen, low, cfg).pose_valid)
        stale = RobotPose(15, -5, 180, Confidence.HIGH, "TEST", 1.0)
        self.assertIn("pose_stale", evaluate_arrival_geometry(screen, stale, cfg, check_time_s=10.0).reasons)

    def test_interaction_geometry_uses_viewer_left_and_faces_screen(self):
        geometry = build_interaction_geometry((0.0, 0.0), (1.0, 0.0), load_config(None)["interaction"])

        self.assertEqual(geometry["screen_left_tangent_xy"], (0.0, -1.0))
        self.assertEqual(geometry["reader_xy"], (0.0, -5.0))
        self.assertEqual(geometry["interaction_xy"], (15.0, -5.0))
        self.assertEqual(abs(geometry["interaction_yaw_deg"]), 180.0)

    def test_case_1_far_classification_records_only_and_never_interacts(self):
        screen = make_screen()

        decision = store_flower_observation(screen, "hehua", "chuju", 0.91)
        gate = evaluate_interaction_pose(
            screen,
            fresh_pose(80.0, -5.0, 180.0),
            "hehua",
            load_config(None)["interaction"],
            12,
        )

        self.assertEqual(decision, "needs_physical_interaction")
        self.assertEqual(screen.last_classification, "chuju")
        self.assertEqual(screen.status, ScreenStatus.NEEDS_CHANGE)
        self.assertFalse(gate.ready)
        self.assertIn("distance", gate.reasons)

    def test_case_2_close_but_wrong_yaw_is_blocked(self):
        screen = make_screen()
        screen.last_classification = "chuju"

        check = evaluate_interaction_pose(screen, fresh_pose(15.0, -5.0, 30.0), "hehua", load_config(None)["interaction"], 12)

        self.assertFalse(check.ready)
        self.assertIn("yaw", check.reasons)

    def test_case_3_wrong_reader_lateral_alignment_is_blocked(self):
        screen = make_screen()
        screen.last_classification = "chuju"

        check = evaluate_interaction_pose(screen, fresh_pose(15.0, 2.0, 180.0), "hehua", load_config(None)["interaction"], 12)

        self.assertFalse(check.ready)
        self.assertIn("lateral", check.reasons)

    def test_case_4_valid_pose_runs_notebook_order_and_arguments(self):
        actions = []
        result = run_client(make_client(actions, response={"ok": True, "worker_id": 12}))

        self.assertTrue(result.success)
        self.assertEqual(actions[0], ("act", "stand", {}))
        self.assertEqual(actions[1], ("act", "lift_left_hand", {"stand": False}))
        self.assertEqual(actions[2][0], "send_request")
        self.assertTrue(actions[2][1]["clear_first"])
        self.assertTrue(actions[2][1]["read_response"])
        self.assertTrue(actions[2][1]["wait_response"])
        self.assertTrue(actions[2][1]["verbose_wait"])
        self.assertEqual(
            set(actions[2][1]),
            {
                "team",
                "secret",
                "robot_id",
                "worker_id",
                "from_flower",
                "to_flower",
                "clear_first",
                "read_response",
                "wait_response",
                "scan_timeout_s",
                "verbose_wait",
            },
        )
        self.assertEqual(actions[-1], ("act", "stand", {}))
        screen = make_screen()
        self.assertTrue(apply_worker_change_result(screen, result))
        self.assertEqual(screen.status, ScreenStatus.CHANGED)

    def test_case_5_ok_false_does_not_report_success_and_stands(self):
        actions = []
        result = run_client(make_client(actions, response={"ok": False, "reason": "no worker"}))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "worker_response_not_ok")
        self.assertEqual(actions[-1], ("act", "stand", {}))

    def test_case_5_task_manager_keeps_failed_screen_retryable(self):
        screen = make_screen()
        result = WorkerChangeResult(
            success=False,
            worker_id=12,
            response={"ok": False},
            error="worker_response_not_ok",
        )

        changed = apply_worker_change_result(screen, result)

        self.assertFalse(changed)
        self.assertEqual(screen.status, ScreenStatus.NEEDS_CHANGE)
        self.assertNotEqual(screen.status, ScreenStatus.CHANGED)

    def test_case_6_exception_always_stands(self):
        actions = []
        result = run_client(make_client(actions, error=TimeoutError("timeout")))

        self.assertFalse(result.success)
        self.assertIn("timeout", result.error)
        self.assertEqual(actions[-1], ("act", "stand", {}))

    def test_removed_passby_has_no_call_path(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("execute_passby_scan", functions)
        self.assertNotIn("harvest_opportunistic", functions)
        self.assertNotIn("harvest_visible", functions)

    def test_localize_geometry_path_has_no_classifier_or_interaction(self):
        source = (Path(__file__).resolve().parents[1] / "task_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "observe_transit_bindings")
        calls = {getattr(node.func, "attr", "") for node in ast.walk(fn) if isinstance(node, ast.Call)}
        self.assertNotIn("classify_crop", calls)
        self.assertNotIn("change_flower", calls)
        self.assertNotIn("send_request", calls)
        self.assertIn("detect", calls)

    def test_safety_gate_is_rechecked_and_blocks_before_send(self):
        actions = []
        checks = iter(
            [
                ready_check(),
                InteractionPoseCheck(ready=False, pose_valid=True, reasons=["pose_stale"]),
            ]
        )
        client = make_client(actions, response={"ok": True})
        result = client.change_flower(
            screen_id=2,
            worker_id=12,
            from_flower="chuju",
            to_flower="hehua",
            safety_gate=lambda: next(checks),
        )

        self.assertFalse(result.success)
        self.assertFalse(any(item[0] == "send_request" for item in actions))
        self.assertEqual(actions[-1], ("act", "stand", {}))

    def test_bound_from_flower_cannot_change_before_request(self):
        screen = make_screen()
        screen.last_classification = "chuju"
        check = evaluate_interaction_pose(
            screen,
            fresh_pose(15.0, -5.0, 180.0),
            "hehua",
            load_config(None)["interaction"],
            12,
            expected_from_flower="lamei",
        )

        self.assertFalse(check.ready)
        self.assertIn("flower_changed_since_alignment", check.reasons)


if __name__ == "__main__":
    unittest.main()
