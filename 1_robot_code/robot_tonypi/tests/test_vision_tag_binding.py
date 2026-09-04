from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.models import (
    ClassificationResult,
    Confidence,
    MissionState,
    RecentBoundFlowerObservation,
    RobotPose,
    Screen,
    ScreenStatus,
    TagDetection,
)
from robot_tonypi.task_manager import TaskManager
from robot_tonypi.vision import ScreenDetector


QUAD = np.array(
    [
        [250.0, 250.0],  # top-left
        [350.0, 250.0],  # top-right
        [350.0, 350.0],  # bottom-right
        [250.0, 350.0],  # bottom-left
    ],
    dtype=np.float64,
)


def make_detector(max_px=100.0, diag_ratio=0.5):
    return ScreenDetector(
        {
            "vision": {
                "tag_bind_max_px": max_px,
                "tag_bind_diag_ratio": diag_ratio,
            }
        }
    )


def make_tag(tag_id, x, y=250.0):
    return TagDetection(tag_id=tag_id, center=np.array([x, y]), corners=[])


class LeftTagBindingTests(unittest.TestCase):
    def test_case_1_nearby_left_tag_binds(self):
        tag = make_tag(5, 220.0)

        bound = make_detector()._bind_left_upper_tag(QUAD, [tag])

        self.assertIs(bound, tag)

    def test_case_2_tag_on_disallowed_side_never_binds(self):
        tag = make_tag(5, 380.0)

        bound = make_detector(max_px=200.0)._bind_left_upper_tag(QUAD, [tag])

        self.assertIsNone(bound)

    def test_case_3_nearest_tag_to_top_left_wins(self):
        near = make_tag(5, 220.0)
        far = make_tag(6, 190.0)
        self.assertIs(make_detector()._bind_left_upper_tag(QUAD, [far, near]), near)

    def test_case_4_left_tag_outside_combined_limit_does_not_bind(self):
        tag = make_tag(5, 100.0)
        self.assertIsNone(make_detector(max_px=100.0, diag_ratio=0.5)._bind_left_upper_tag(QUAD, [tag]))

    def test_case_5_non_screen_tag_id_does_not_bind(self):
        tag = make_tag(37, 220.0)
        self.assertIsNone(make_detector()._bind_left_upper_tag(QUAD, [tag]))


class DebugStub:
    def __init__(self):
        self.events = []

    def event(self, name, **data):
        self.events.append((name, data))

    def save_crop(self, *args, **kwargs):
        pass


def cache_screen(screen_id=26):
    return Screen(
        screen_id=screen_id,
        tag_corners_3d=None,
        center_xy=(0.0, 0.0),
        normal_xy=(1.0, 0.0),
        normal_yaw_deg=0.0,
        target_xy=(20.0, 0.0),
        interaction_xy=(20.0, 0.0),
        interaction_yaw_deg=180.0,
        reader_xy=(0.0, -5.0),
        screen_left_tangent_xy=(0.0, -1.0),
        task_target_xy=(20.0, 0.0),
        task_target_yaw_deg=180.0,
        worker_id=screen_id,
    )


class BoundClassificationCacheTests(unittest.TestCase):
    def manager(self, results=None):
        manager = TaskManager.__new__(TaskManager)
        manager.config = load_config(None)
        manager.config["vision"]["bound_classification_min_interval_s"] = 0.0
        manager.debug = DebugStub()
        target = cache_screen()
        manager.map = SimpleNamespace(screens={26: target})
        queue = list(results or [])
        manager.classifier = SimpleNamespace(classify_crop=lambda crop: queue.pop(0))
        manager.recent_bound_flower_observations = {}
        manager.bound_classification_last_attempt_s = {}
        return manager, target

    @staticmethod
    def candidate(screen_id=26):
        return SimpleNamespace(
            screen_id=screen_id,
            tag=SimpleNamespace(tag_id=screen_id),
            crop_28x28=object(),
        )

    def test_bound_observation_created_and_does_not_change_status(self):
        manager, target = self.manager([ClassificationResult(True, "juhua", confidence=0.95)])
        observation = manager.process_bound_screen_candidate(
            self.candidate(), pan=100, reason="localize", captured_s=1.0
        )
        self.assertEqual(observation.flower, "juhua")
        self.assertEqual(manager.recent_bound_flower_observations[26].tag_id, 26)
        self.assertEqual(target.status, ScreenStatus.UNKNOWN)

    def test_newest_valid_wins_and_failure_does_not_destroy_it(self):
        manager, _ = self.manager([
            ClassificationResult(True, "juhua", confidence=0.90),
            ClassificationResult(False, error="timeout"),
            ClassificationResult(True, "shuixianhua", confidence=0.95),
        ])
        manager.process_bound_screen_candidate(self.candidate(), pan=100, reason="a", captured_s=1.0)
        manager.process_bound_screen_candidate(self.candidate(), pan=100, reason="b", captured_s=3.0)
        self.assertEqual(manager.recent_bound_flower_observations[26].flower, "juhua")
        manager.process_bound_screen_candidate(self.candidate(), pan=100, reason="c", captured_s=5.0)
        self.assertEqual(manager.recent_bound_flower_observations[26].flower, "shuixianhua")
        self.assertEqual(manager.recent_bound_flower_observations[26].captured_s, 5.0)

    def test_per_screen_classifier_rate_limit(self):
        manager, _ = self.manager([
            ClassificationResult(True, "juhua", confidence=0.90),
            ClassificationResult(True, "shuixianhua", confidence=0.95),
        ])
        manager.config["vision"]["bound_classification_min_interval_s"] = 1.0
        self.assertIsNotNone(manager.process_bound_screen_candidate(
            self.candidate(), pan=100, reason="a", captured_s=1.0
        ))
        self.assertIsNone(manager.process_bound_screen_candidate(
            self.candidate(), pan=100, reason="b", captured_s=1.5
        ))
        self.assertEqual(manager.recent_bound_flower_observations[26].flower, "juhua")
        self.assertTrue(any(
            name == "bound_flower_observation_skipped_rate_limit"
            for name, _ in manager.debug.events
        ))

    def test_cache_ttl_boundary_and_wrong_screen_rejected(self):
        manager, _ = self.manager()
        manager.recent_bound_flower_observations[26] = RecentBoundFlowerObservation(
            26, 26, True, "juhua", 0.95, 10.0, 100.0, "test"
        )
        self.assertIsNotNone(manager.latest_valid_bound_flower_observation(26, current_s=24.9))
        self.assertIsNone(manager.latest_valid_bound_flower_observation(26, current_s=25.01))
        manager.recent_bound_flower_observations[25] = RecentBoundFlowerObservation(
            25, 25, True, "juhua", 0.95, 24.0, 100.0, "test"
        )
        manager.recent_bound_flower_observations[26] = manager.recent_bound_flower_observations[25]
        self.assertIsNone(manager.latest_valid_bound_flower_observation(26, current_s=25.0))

    def arrived_manager(self, flower):
        manager, target = self.manager()
        manager.args = SimpleNamespace(dry_run=False)
        manager.target_flower = "shuixianhua"
        manager.current_target_screen_id = 26
        manager.arrived_at_target = True
        manager.mission_state = MissionState.ARRIVED_AT_TARGET
        manager.state = SimpleNamespace(
            pose=RobotPose(20, 0, 180, Confidence.HIGH, "VISION", 100.0)
        )
        manager.publish_state = lambda *args, **kwargs: None
        manager.confirm_target_tag_now = lambda screen: True
        manager._last_target_tag_seen_s = 100.0
        manager.recent_bound_flower_observations[26] = RecentBoundFlowerObservation(
            26, 26, True, flower, 0.95, 87.0, 100.0, "navigation"
        )
        manager.classifier = SimpleNamespace(
            classify_crop=lambda crop: self.fail("cache reuse must not call classifier")
        )
        return manager, target

    def test_arrival_live_tag_reuses_cache_without_classifier(self):
        manager, target = self.arrived_manager("juhua")
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        self.assertEqual(target.status, ScreenStatus.NEEDS_CHANGE)
        self.assertEqual(manager.visual_authorization.source, "recent_bound_cache")
        self.assertEqual(manager.visual_authorization.cache_age_s, 13.0)

    def test_arrival_live_tag_cache_already_target_skips_interaction_state(self):
        manager, target = self.arrived_manager("shuixianhua")
        self.assertTrue(manager.confirm_target_tag_and_screen(target))
        self.assertEqual(target.status, ScreenStatus.ALREADY_TARGET)
        self.assertFalse(target.needs_interaction())

    def test_missing_live_tag_never_adopts_cache(self):
        manager, target = self.arrived_manager("juhua")
        manager.confirm_target_tag_now = lambda screen: False
        self.assertFalse(manager.confirm_target_tag_and_screen(target))
        self.assertEqual(target.status, ScreenStatus.UNKNOWN)
        self.assertIsNone(getattr(manager, "visual_authorization", None))

if __name__ == "__main__":
    unittest.main()
