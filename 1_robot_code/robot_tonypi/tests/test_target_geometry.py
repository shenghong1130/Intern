from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_tonypi.config import load_config
from robot_tonypi.interaction_logic import (
    building_centers_from_tags,
    build_interaction_geometry,
    cardinal_surface_from_tag,
)
from robot_tonypi.load_pos import load_tag_pos


class TargetGeometryTests(unittest.TestCase):
    def test_cardinal_surface_does_not_define_a_second_target_yaw(self):
        tag_poses = load_tag_pos()
        centers = building_centers_from_tags(tag_poses)
        surface = cardinal_surface_from_tag(tag_poses["1"], centers[0])
        self.assertEqual(surface["normal_xy"], (-1.0, 0.0))
        self.assertNotIn("target_yaw_deg", surface)

    def test_cardinal_screen_facing_goal_yaws(self):
        cfg = load_config(None)["interaction"]
        expected = {
            "WEST": ((-1.0, 0.0), 5.0),
            "EAST": ((1.0, 0.0), -175.0),
            "SOUTH": ((0.0, -1.0), 95.0),
            "NORTH": ((0.0, 1.0), -85.0),
        }
        for face, (normal, yaw) in expected.items():
            with self.subTest(face=face):
                geometry = build_interaction_geometry((100.0, 100.0), normal, cfg)
                self.assertAlmostEqual(geometry["interaction_yaw_deg"], yaw)

    def test_yaw_offset_never_changes_25cm_interaction_xy(self):
        cfg = load_config(None)["interaction"]
        for normal in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
            with self.subTest(normal=normal):
                base = build_interaction_geometry(
                    (100.0, 100.0), normal,
                    dict(cfg, target_yaw_offset_deg=0.0),
                )
                offset = build_interaction_geometry(
                    (100.0, 100.0), normal,
                    dict(cfg, target_yaw_offset_deg=5.0),
                )
                self.assertEqual(
                    offset["interaction_target_xy"],
                    base["interaction_target_xy"],
                )
                self.assertEqual(
                    offset["navigation_staging_xy"],
                    offset["interaction_target_xy"],
                )


if __name__ == "__main__":
    unittest.main()
