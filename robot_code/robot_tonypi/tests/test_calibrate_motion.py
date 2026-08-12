#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from robot_tonypi.calibrate_motion import (
    build_recommendations,
    normalize_manual_measurement,
    physical_action_spec,
    summarize,
    write_recommended_config,
)


class ManualMotionCalibrationTests(unittest.TestCase):
    def test_direction_signs(self):
        self.assertEqual(normalize_manual_measurement("forward_fast", 3.8), (3.8, 3.8))
        self.assertEqual(normalize_manual_measurement("back_fast", 2.1), (-2.1, -2.1))
        self.assertEqual(normalize_manual_measurement("strafe_left_fast", 4), (4.0, 4.0))
        self.assertEqual(normalize_manual_measurement("strafe_right_fast", 4), (-4.0, -4.0))
        self.assertEqual(normalize_manual_measurement("turn_left_fast", 8.2), (8.2, 8.2))
        self.assertEqual(normalize_manual_measurement("turn_right_fast", 10.4), (-10.4, -10.4))

    def test_negative_input_is_treated_as_absolute_measurement(self):
        self.assertEqual(normalize_manual_measurement("turn_right_large", -42), (-42.0, -42.0))

    def test_times_normalizes_to_per_action(self):
        total, per_action = normalize_manual_measurement("forward_fast", 11.4, 3)
        self.assertEqual(total, 11.4)
        self.assertAlmostEqual(per_action, 3.8)

    def test_summary_uses_median_recommendation(self):
        stats = summarize([8.1, 7.8, 8.0])
        self.assertAlmostEqual(stats["mean"], 7.9666666667)
        self.assertEqual(stats["median"], 8.0)

    def test_large_turn_sequence_is_preserved(self):
        spec = {
            "sequence": [
                {
                    "group": "turn_left_small_step_s80",
                    "times": 4,
                    "repeat": True,
                    "with_stand": False,
                }
            ],
            "yaw_deg": 30.0,
        }
        physical = physical_action_spec(spec, times=2)
        self.assertEqual(physical["sequence"][0]["group"], "turn_left_small_step_s80")
        self.assertEqual(physical["sequence"][0]["times"], 8)

    def test_recommendations_only_include_measured_actions(self):
        report = build_recommendations(
            [
                {"action": "forward_fast", "metric": "forward_cm", "recommended_value": 3.8},
                {"action": "back_fast", "metric": "forward_cm", "recommended_value": None},
            ]
        )
        self.assertEqual(
            report,
            {"motion": {"actions": {"forward_fast": {"forward_cm": 3.8}}}},
        )

    def test_write_config_backs_up_and_changes_only_recommended_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "competition_config.json"
            original = {
                "motion": {
                    "actions": {
                        "turn_left_fast": {
                            "group": "turn_left_small_step_s80",
                            "yaw_deg": 7.5,
                            "settle_s": 0.12,
                        }
                    }
                }
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            report = {
                "status": "COMPLETE",
                "recommended_config": {
                    "motion": {"actions": {"turn_left_fast": {"yaw_deg": 8.0}}}
                },
            }
            backup = write_recommended_config(path, report, "20260812_120000")
            updated = json.loads(path.read_text(encoding="utf-8"))
            backed_up = json.loads(backup.read_text(encoding="utf-8"))
            self.assertEqual(updated["motion"]["actions"]["turn_left_fast"]["yaw_deg"], 8.0)
            self.assertEqual(updated["motion"]["actions"]["turn_left_fast"]["group"], "turn_left_small_step_s80")
            self.assertEqual(backed_up, original)


if __name__ == "__main__":
    unittest.main()
