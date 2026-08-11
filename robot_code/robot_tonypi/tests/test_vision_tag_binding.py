from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.models import TagDetection
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

        bound = make_detector()._bind_left_upper_tag(QUAD, [far, near])

        self.assertIs(bound, near)

    def test_case_4_left_tag_outside_combined_limit_does_not_bind(self):
        tag = make_tag(5, 100.0)

        bound = make_detector(max_px=100.0, diag_ratio=0.5)._bind_left_upper_tag(QUAD, [tag])

        self.assertIsNone(bound)

    def test_case_5_non_screen_tag_id_does_not_bind(self):
        tag = make_tag(37, 220.0)

        bound = make_detector()._bind_left_upper_tag(QUAD, [tag])

        self.assertIsNone(bound)


if __name__ == "__main__":
    unittest.main()
