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
    RecentBoundFlowerObservation,
    Screen,
    ScreenStatus,
    TargetVisualConfirmation,
    VisualAuthorization,
    WorkerChangeResult,
)
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.utils import now_s


def make_screen(screen_id=1, xy=(20.0, -1.0)):
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
    manager.recent_bound_flower_observations = {}
    manager.bound_classification_last_attempt_s = {}
    manager.state = SimpleNamespace(
        pose=RobotPose(20.0, -1.0, 180.0, Confidence.HIGH, "TEST", 1.0)
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


class TargetStandoffFlowTests(unittest.TestCase):
    def test_map_screens_have_one_configured_target_with_lateral_offset(self):
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
            self.assertAlmostEqual(normal_distance, 20.0)
            self.assertAlmostEqual(lateral_distance, desired_lateral)
            self.assertEqual(item.target_xy, item.task_target_xy)
            self.assertEqual(item.interaction_xy, item.task_target_xy)
        west = model.screens[1]
        self.assertEqual(west.face_center_xy, (196.0, 17.5))
        self.assertEqual(west.tag_front_xy, (176.0, 17.5))
        self.assertEqual(west.task_target_xy, (176.0, 16.5))
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
        self.assertEqual(action_path, [])
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

    def test_fresh_fallback_retries_use_configured_interval(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "chuju", confidence=0.95))
        manager.config["interaction"]["target_confirmation_retry_interval_s"] = 0.125
        calls = []
        outcomes = [[], [], [candidate()]]
        manager._last_target_live_frame = None
        manager.screen_detector.detect = lambda *args, **kwargs: outcomes.pop(0)
        manager.capture_with_tags = lambda pan: (
            calls.append(len(calls) + 1) or np.zeros((20, 20, 3), dtype=np.uint8),
            [SimpleNamespace(tag_id=1)],
        )
        with mock.patch("robot_tonypi.task_manager.time.sleep") as sleep:
            self.assertIsNotNone(manager.bounded_fresh_target_observation(make_screen()))
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(0.125)
        self.assertEqual(manager.target_confirmation_retry_count, 0)

    def test_confirmation_round_exhaustion_stays_bounded_without_selecting_target(self):
        manager = classification_manager([1], [], ClassificationResult(True, "chuju", confidence=0.95))
        manager.config["interaction"]["target_confirmation_retry_interval_s"] = 0.0
        self.assertFalse(manager.confirm_target_tag_and_screen(make_screen()))
        classifier_failures = [name for name, _ in manager.debug.events if name == "target_fresh_fallback_failed"]
        self.assertEqual(len(classifier_failures), 1)
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
        self.assertEqual(manager.mission_state, MissionState.MISSION_BLOCKED)
        self.assertEqual(manager.current_target_screen_id, 1)
        self.assertEqual(target.attempts, 1)
        self.assertFalse(manager.final_forward_executed)
        unresolved = [data for name, data in manager.debug.events if name == "target_screen_confirmation_unresolved"]
        self.assertEqual(len(unresolved), 1)
        self.assertTrue(unresolved[0]["target_preserved"])

    def test_fpga_failure_blocks_authorization(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(False, error="fpga_down"))
        target = make_screen()
        self.assertFalse(manager.confirm_target_tag_and_screen(target))
        self.assertIsNone(manager.visual_authorization)

    def test_classifier_offline_preserves_live_tag_and_never_mission_fails(self):
        result = ClassificationResult(
            False,
            error="connection refused",
            error_kind="service_unavailable",
            retryable=True,
        )
        manager = classification_manager([1], [candidate()], result)
        target = make_screen()
        recoveries = []
        manager.recover_target_visibility = lambda *args: recoveries.append(args) or False
        manager.preserve_current_target = lambda screen, reason: setattr(
            manager, "current_target_screen_id", screen.screen_id
        )
        self.assertFalse(manager.confirm_target_with_visibility_recovery(target))
        self.assertEqual(manager.mission_state, MissionState.TARGET_CLASSIFICATION_WAIT)
        self.assertEqual(manager.current_target_screen_id, 1)
        self.assertEqual(manager.target_tag_confirmation.tag_id, 1)
        self.assertEqual(recoveries, [])

    def test_pan_frame_is_consumed_before_recenter_and_retry_uses_last_seen_pan(self):
        manager = classification_manager(
            [], [candidate()], ClassificationResult(True, "chuju", confidence=0.95)
        )
        sequence = []

        def capture(pan):
            sequence.append(("capture", float(pan)))
            tags = [] if float(pan) == 100.0 else [SimpleNamespace(tag_id=1)]
            return np.zeros((20, 20, 3), dtype=np.uint8), tags

        manager.capture_with_tags = capture
        detections = [[], [candidate()]]
        manager.screen_detector.detect = lambda *args, **kwargs: detections.pop(0)
        manager.classifier.classify_crop = lambda crop: sequence.append(("classify", 130.0)) or ClassificationResult(
            True, "chuju", confidence=0.95
        )
        manager.center_head_after_scan = lambda reason, pan: sequence.append(("recenter", float(pan)))
        self.assertTrue(manager.confirm_target_tag_and_screen(make_screen()))
        self.assertEqual(sequence[:5], [
            ("capture", 100.0),
            ("capture", 130.0),
            ("capture", 130.0),
            ("classify", 130.0),
            ("recenter", 130.0),
        ])

    def test_low_confidence_blocks_authorization(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "chuju", confidence=0.10))
        target = make_screen()
        self.assertFalse(manager.confirm_target_tag_and_screen(target))
        self.assertIsNone(manager.visual_authorization)

    def test_already_target_is_known_at_standoff_and_skips_forward(self):
        manager = classification_manager([1], [candidate()], ClassificationResult(True, "hehua", confidence=0.96))
        target = make_screen()
        manager.motion = SimpleNamespace(
            run=lambda key, times_override=1: ActionResult(
                key=key, group="go_forward_one_step", times=times_override,
                elapsed_s=0.0, model_forward_cm=10.0, ok=True,
            )
        )
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        self.assertEqual(target.status, ScreenStatus.ALREADY_TARGET)
        self.assertFalse(manager.final_forward_executed)

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
        manager.nfc_interaction_status = {}
        manager.nfc_interaction_stopped_for_mission_timeout = False
        manager.nfc_interaction_gave_up = False
        manager.nfc_gave_up_screen_ids = set()
        manager.recent_bound_flower_observations = {}
        manager.last_navigation_failure_reason = ""
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
                model_forward_cm=10.0,
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
        manager.time_left_s = lambda: 100.0
        manager.mission_retry_pause = lambda *args, **kwargs: None
        manager.lock_target_goal = lambda screen: (
            manager.sequence.append(("lock_target", screen.screen_id))
            or SimpleNamespace(
                screen_id=screen.screen_id,
                as_dict=lambda: {
                    "screen_id": screen.screen_id,
                    "goal_xy": list(screen.task_target_xy or screen.target_xy),
                    "desired_yaw_deg": screen.task_target_yaw_deg,
                },
            )
        )
        manager.navigate_to_screen = lambda screen: (
            manager.sequence.append(("navigate_target", screen.screen_id)) or True
        )
        manager.confirm_target_tag_now = lambda screen: (
            manager.sequence.append(("confirm_tag", screen.screen_id)) or True
        )
        return manager, target

    def test_post_forward_needs_change_runs_transaction_without_extra_motion(self):
        manager, target = self.interaction_manager()
        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(manager.sequence, [("change", 1, 1)])
        self.assertTrue(manager.final_forward_executed)
        self.assertEqual(target.status, ScreenStatus.CHANGED)

    def test_initial_target_flower_performs_zero_nfc_attempts(self):
        manager, target = self.interaction_manager()
        target.last_classification = manager.target_flower
        target.status = ScreenStatus.ALREADY_TARGET
        manager.final_forward_executed = False

        if target.needs_interaction():
            manager.process_screen_interaction(target)

        self.assertEqual(
            [item for item in manager.sequence if item[0] == "change"],
            [],
        )
        self.assertFalse(manager.final_forward_executed)

    def test_nfc_timeout_retreats_reapproaches_then_second_attempt_succeeds(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        def localize_scan(**kwargs):
            manager.recent_bound_flower_observations[target.screen_id] = (
                RecentBoundFlowerObservation(
                    target.screen_id,
                    target.screen_id,
                    True,
                    "chuju",
                    0.96,
                    now_s() + 1.0,
                    100.0,
                    "nfc_retry",
                )
            )
            return True

        manager.localize_scan = localize_scan
        results = iter((
            WorkerChangeResult(
                False,
                worker_id=1,
                response={"seq": 10, "response": None},
                error="nfc_timeout",
            ),
            WorkerChangeResult(
                True,
                worker_id=1,
                response={"seq": 11, "ok": True},
            ),
        ))

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["attempt"]))
            return next(results)

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(target.status, ScreenStatus.CHANGED)
        self.assertEqual(
            [item for item in manager.sequence if item[0] == "change"],
            [("change", 1), ("change", 2)],
        )
        self.assertIn(("motion", "back_fast", 4), manager.sequence)
        self.assertIn(("motion", "interaction_forward_10cm", 1), manager.sequence)
        retry_order = [
            item[0] for item in manager.sequence
            if item[0] in (
                "lock_target",
                "navigate_target",
                "confirm_tag",
                "motion",
                "change",
            )
        ]
        self.assertLess(retry_order.index("lock_target"), retry_order.index("navigate_target"))
        self.assertLess(retry_order.index("navigate_target"), retry_order.index("confirm_tag"))
        final_forward_index = next(
            index for index, item in enumerate(manager.sequence)
            if item == ("motion", "interaction_forward_10cm", 1)
        )
        self.assertLess(manager.sequence.index(("confirm_tag", target.screen_id)), final_forward_index)
        event_names = [name for name, _ in manager.debug.events]
        self.assertIn("nfc_interaction_retry_started", event_names)
        self.assertIn("nfc_retry_retreat", event_names)
        self.assertIn("nfc_retry_relocalize", event_names)
        self.assertIn("nfc_retry_reapproach", event_names)

    def test_two_nfc_timeouts_give_up_without_attempt_three(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        def localize_scan(**kwargs):
            manager.recent_bound_flower_observations[target.screen_id] = (
                RecentBoundFlowerObservation(
                    target.screen_id,
                    target.screen_id,
                    True,
                    "chuju",
                    0.96,
                    now_s() + 1.0,
                    100.0,
                    "nfc_retry",
                )
            )
            return True

        manager.localize_scan = localize_scan
        results = iter((
            WorkerChangeResult(False, worker_id=1, response={"seq": 20}, error="nfc_timeout"),
            WorkerChangeResult(False, worker_id=1, response={"seq": 21}, error="nfc_timeout"),
        ))

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["attempt"]))
            return next(results)

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertFalse(manager.process_screen_interaction(target))
        compact = [
            item for item in manager.sequence
            if item[0] == "change"
            or (item[0] == "motion" and item[1] in (
                "back_fast", "interaction_forward_10cm"
            ))
        ]
        self.assertEqual(compact, [
            ("change", 1),
            ("motion", "back_fast", 4),
            ("motion", "interaction_forward_10cm", 1),
            ("change", 2),
        ])
        self.assertEqual(target.status, ScreenStatus.FAILED)
        self.assertTrue(manager.nfc_interaction_gave_up)
        self.assertIn(target.screen_id, manager.nfc_gave_up_screen_ids)
        give_up = [
            data for name, data in manager.debug.events
            if name == "nfc_change_give_up"
        ]
        self.assertEqual(len(give_up), 1)
        self.assertEqual(give_up[0]["attempts"], 2)

    def test_timeout_then_retry_visual_target_skips_second_nfc_attempt(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id

        localization_calls = []

        def localize_scan(**kwargs):
            localization_calls.append(kwargs)
            observed_screen = (
                target.screen_id
                if kwargs.get("required_target_screen_id") == target.screen_id
                else 15
            )
            manager.recent_bound_flower_observations[observed_screen] = (
                RecentBoundFlowerObservation(
                    observed_screen,
                    observed_screen,
                    True,
                    manager.target_flower,
                    0.98,
                    now_s() + 1.0,
                    100.0,
                    "nfc_retry",
                )
            )
            return True

        manager.localize_scan = localize_scan

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["attempt"]))
            return WorkerChangeResult(
                False,
                worker_id=1,
                response={"seq": 30},
                error="nfc_timeout",
            )

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(
            [item for item in manager.sequence if item[0] == "change"],
            [("change", 1)],
        )
        self.assertIn(("motion", "back_fast", 4), manager.sequence)
        self.assertNotIn(("motion", "interaction_forward_10cm", 1), manager.sequence)
        self.assertNotIn(("lock_target", target.screen_id), manager.sequence)
        self.assertNotIn(("navigate_target", target.screen_id), manager.sequence)
        self.assertNotIn(("confirm_tag", target.screen_id), manager.sequence)
        self.assertEqual(
            [
                call.get("required_target_screen_id")
                for call in localization_calls
            ],
            [None, target.screen_id],
        )
        self.assertEqual(target.status, ScreenStatus.CHANGED)
        self.assertEqual(manager.visual_authorization.flower, manager.target_flower)
        checks = [
            data for name, data in manager.debug.events
            if name == "nfc_retry_visual_check"
        ]
        self.assertEqual(checks[-1]["decision"], "already_changed_skip_retry")

    def test_changed_status_after_retry_recovery_blocks_attempt_two(self):
        manager, target = self.interaction_manager()
        calls = []

        def change_flower(**kwargs):
            calls.append(kwargs["attempt"])
            return WorkerChangeResult(
                False,
                worker_id=1,
                response={"seq": 35},
                error="nfc_timeout",
            )

        def restore_contact(*args, **kwargs):
            target.status = ScreenStatus.CHANGED
            return "reapproached"

        manager.interaction = SimpleNamespace(change_flower=change_flower)
        manager.restore_nfc_physical_contact = restore_contact
        manager.recalibrate_target_for_nfc_retry = lambda *args, **kwargs: self.fail(
            "CHANGED must not enter target recalibration"
        )

        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(calls, [1])
        self.assertEqual(target.status, ScreenStatus.CHANGED)
        terminal = [
            data for name, data in manager.debug.events
            if name == "nfc_change_terminal_success"
        ]
        self.assertTrue(terminal)
        self.assertEqual(terminal[-1]["source"], "status_after_retry_recovery")

    def test_retry_visual_check_ignores_other_screen_classification(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        localize_calls = {"count": 0}

        def localize_scan(**kwargs):
            localize_calls["count"] += 1
            observed_screen = 2 if localize_calls["count"] == 1 else target.screen_id
            manager.recent_bound_flower_observations[observed_screen] = (
                RecentBoundFlowerObservation(
                    observed_screen,
                    observed_screen,
                    True,
                    manager.target_flower if observed_screen == 2 else "chuju",
                    0.97,
                    now_s() + 1.0,
                    100.0,
                    "nfc_retry",
                )
            )
            return True

        manager.localize_scan = localize_scan
        results = iter((
            WorkerChangeResult(False, worker_id=1, response={"seq": 40}, error="nfc_timeout"),
            WorkerChangeResult(True, worker_id=1, response={"seq": 41, "ok": True}),
        ))

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["attempt"]))
            return next(results)

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertTrue(manager.process_screen_interaction(target))
        self.assertEqual(localize_calls["count"], 2)
        self.assertEqual(
            [item for item in manager.sequence if item[0] == "change"],
            [("change", 1), ("change", 2)],
        )

    def test_only_wrong_target_tags_exhausts_bounded_reacquire(self):
        manager, target = self.interaction_manager()
        manager.config["interaction"]["nfc_retry_target_reacquire_max_cycles"] = 3
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        localization_calls = []

        def localize_scan(**kwargs):
            localization_calls.append(kwargs)
            manager.recent_bound_flower_observations[15] = (
                RecentBoundFlowerObservation(
                    15,
                    15,
                    True,
                    manager.target_flower,
                    0.99,
                    now_s() + 1.0,
                    100.0,
                    "wrong_target",
                )
            )
            return kwargs.get("required_target_screen_id") is None

        manager.localize_scan = localize_scan

        def change_flower(**kwargs):
            manager.sequence.append(("change", kwargs["attempt"]))
            return WorkerChangeResult(
                False,
                worker_id=1,
                response={"seq": 50},
                error="nfc_timeout",
            )

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertFalse(manager.process_screen_interaction(target))
        required_calls = [
            call for call in localization_calls
            if call.get("required_target_screen_id") == target.screen_id
        ]
        self.assertEqual(len(required_calls), 3)
        self.assertEqual(
            [item for item in manager.sequence if item[0] == "change"],
            [("change", 1)],
        )
        self.assertEqual(target.status, ScreenStatus.FAILED)
        self.assertTrue(manager.nfc_interaction_gave_up)
        self.assertNotEqual(target.status, ScreenStatus.CHANGED)
        exhausted = [
            data for name, data in manager.debug.events
            if name == "nfc_retry_target_reacquire_exhausted"
        ]
        self.assertEqual(len(exhausted), 1)
        self.assertEqual(exhausted[0]["cycles"], 3)

    def test_nfc_gave_up_screen_is_not_selected_again(self):
        manager, target = self.interaction_manager()
        other = make_screen(screen_id=2, xy=(40.0, 10.0))
        target.status = ScreenStatus.FAILED
        manager.nfc_gave_up_screen_ids.add(target.screen_id)
        manager.current_target_screen_id = None
        manager.state.pose = RobotPose(0.0, 0.0, 0.0, Confidence.HIGH)
        manager.map = SimpleNamespace(screens={target.screen_id: target, other.screen_id: other})

        selected = manager.choose_nearest_screen()

        self.assertIsNotNone(selected)
        self.assertEqual(selected.screen_id, other.screen_id)

    def test_nfc_retry_stops_only_at_global_timeout_without_target_failure(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        timed_out = {"value": False}
        manager.time_left_s = lambda: 0.0 if timed_out["value"] else 100.0

        def change_flower(**kwargs):
            timed_out["value"] = True
            return WorkerChangeResult(
                False,
                worker_id=1,
                response={"seq": 30},
                error="nfc_timeout",
            )

        manager.interaction = SimpleNamespace(change_flower=change_flower)

        self.assertFalse(manager.process_screen_interaction(target))
        self.assertTrue(manager.nfc_interaction_stopped_for_mission_timeout)
        self.assertEqual(manager.current_target_screen_id, target.screen_id)
        self.assertEqual(target.status, ScreenStatus.NEEDS_CHANGE)
        self.assertEqual(target.attempts, 0)

    def test_close_interaction_retreats_10cm_then_forces_search_localization(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        calls = []

        def run(key, times_override=1):
            calls.append((key, times_override))
            per_cycle = -2.5 if key == "back_fast" else 0.0
            return ActionResult(
                key=key,
                group=key,
                times=times_override,
                elapsed_s=0.0,
                model_forward_cm=per_cycle * times_override,
                ok=True,
                executed_times=times_override,
            )

        manager.motion = SimpleNamespace(run=run)
        localize_calls = []
        manager.localize_scan = lambda **kwargs: localize_calls.append(kwargs) or True
        self.assertTrue(manager.complete_post_interaction_retreat(target))
        self.assertEqual(calls, [("stand", 1), ("back_fast", 4)])
        self.assertEqual(localize_calls, [{
            "reason": "post_interaction_retreat",
            "allow_pan_search": True,
            "allow_failure_escalation": False,
        }])
        self.assertFalse(manager.post_interaction_retreat_pending)
        self.assertFalse(manager.final_forward_executed)
        names = [name for name, _ in manager.debug.events]
        self.assertIn("interaction_retreat_started", names)
        self.assertIn("interaction_retreat_completed", names)
        self.assertIn("post_interaction_relocalize", names)

    def test_failed_post_retreat_localization_does_not_repeat_reverse(self):
        manager, target = self.interaction_manager()
        manager.post_interaction_retreat_pending = True
        manager.post_interaction_retreat_completed = False
        manager.post_interaction_retreat_blocked = False
        manager.post_interaction_screen_id = target.screen_id
        calls = []

        def run(key, times_override=1):
            calls.append((key, times_override))
            return ActionResult(
                key=key, group=key, times=times_override, elapsed_s=0.0,
                model_forward_cm=(-2.5 * times_override if key == "back_fast" else 0.0),
                ok=True, executed_times=times_override,
            )

        manager.motion = SimpleNamespace(run=run)
        localization_results = iter((False, True))
        manager.localize_scan = lambda **kwargs: next(localization_results)
        self.assertFalse(manager.complete_post_interaction_retreat(target))
        self.assertTrue(manager.post_interaction_retreat_completed)
        self.assertTrue(manager.complete_post_interaction_retreat(target))
        self.assertEqual(calls.count(("back_fast", 4)), 1)

    def test_no_close_pose_requires_no_retreat(self):
        manager, target = self.interaction_manager()
        manager.final_forward_executed = False
        manager.post_interaction_retreat_pending = False
        manager.motion = SimpleNamespace(run=lambda *args, **kwargs: self.fail("unexpected motion"))
        self.assertTrue(manager.complete_post_interaction_retreat(target))

    def test_10cm_path_contains_no_sensing_or_extra_navigation_calls(self):
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
        self.assertEqual(config["interaction"]["target_distance_cm"], 20.0)
        self.assertEqual(config["interaction"]["target_lateral_offset_cm"], -1.0)
        self.assertEqual(config["vision"]["bound_classification_cache_ttl_s"], 15.0)
        self.assertEqual(config["vision"]["bound_classification_min_interval_s"], 1.0)
        self.assertEqual(config["interaction"]["target_final_forward_cm"], 10.0)
        self.assertEqual(config["navigation"]["target_arrival_radius_cm"], 4.0)
        self.assertNotIn("target_arrival_distance_cm", config["map"])
        self.assertNotIn("interaction_staging_distance_cm", config["interaction"])
        self.assertNotIn("interaction_staging_arrival_radius_cm", config["interaction"])
        self.assertEqual(
            config["motion"]["actions"]["interaction_forward_10cm"]["forward_cm"],
            10.0,
        )
        final_action = config["motion"]["actions"]["interaction_forward_10cm"]
        self.assertEqual(final_action["times"], 1)
        self.assertEqual(
            [(step["group"], step["times"]) for step in final_action["sequence"]],
            [("go_forward_one_step", 2)],
        )
        self.assertEqual(config["vision"]["max_screen_area_ratio"], 0.98)

    def test_final_10cm_hardware_sequence_is_one_logical_action(self):
        hardware = TonyPiHardware(load_config(None), dry_run=True)
        with mock.patch("robot_tonypi.hardware.time.sleep"), mock.patch("builtins.print"):
            result = hardware.run_action("interaction_forward_10cm", times_override=1)
        self.assertTrue(result.ok)
        self.assertEqual(result.times, 1)
        self.assertEqual(result.executed_times, 1)
        self.assertEqual(result.model_forward_cm, 10.0)
        self.assertEqual(result.group, "go_forward_one_step")

    def test_flow_order_is_confirm_then_optional_forward_then_change(self):
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
        self.assertLess(line["execute_final_forward"], line["process_screen_interaction"])

    def test_10cm_failure_blocks_change(self):
        manager, target = self.interaction_manager(motion_ok=False)
        manager.final_forward_executed = False
        self.assertFalse(manager.execute_final_forward(target))
        self.assertEqual(manager.sequence, [("motion", "interaction_forward_10cm", 1)])
        self.assertEqual(target.status, ScreenStatus.NEEDS_CHANGE)
        self.assertFalse(manager.final_forward_executed)

    def test_final_10cm_is_executed_only_once(self):
        manager, target = self.interaction_manager()
        manager.final_forward_executed = False
        self.assertTrue(manager.execute_final_forward(target))
        self.assertFalse(manager.execute_final_forward(target))
        self.assertEqual(manager.sequence, [("motion", "interaction_forward_10cm", 1)])

    def test_skip_change_runs_no_10cm_hardware_action(self):
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
