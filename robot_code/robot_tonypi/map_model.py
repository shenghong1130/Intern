#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Known field map, screen model, and A* path planning."""

import heapq
import importlib.util
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .interaction_logic import (
    build_interaction_geometry,
    building_bounds_from_tags,
    building_centers_from_tags,
    cardinal_surface_from_tag,
    face_center_from_bounds,
)
from .models import Screen
from .utils import clamp, distance_xy, normalize_angle_deg


def load_tag_positions(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    if path:
        source = Path(path)
        if source.suffix.lower() == ".json":
            import json

            with source.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return {str(k): np.array(v, dtype=np.float64) for k, v in raw.items()}
        spec = importlib.util.spec_from_file_location("external_load_pos", str(source))
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import load_pos from {}".format(source))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {str(k): np.array(v, dtype=np.float64) for k, v in module.load_tag_pos().items()}

    from .load_pos import load_tag_pos

    return {str(k): np.array(v, dtype=np.float64) for k, v in load_tag_pos().items()}


class MapModel:
    def __init__(self, tag_poses: Dict[str, np.ndarray], config):
        self.tag_poses = tag_poses
        self.cfg = config
        self.width_cm = float(config["map"]["width_cm"])
        self.height_cm = float(config["map"]["height_cm"])
        self.res = float(config["map"]["grid_resolution_cm"])
        self.rows = int(math.ceil(self.width_cm / self.res))
        self.cols = int(math.ceil(self.height_cm / self.res))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.cost = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.building_bounds: Dict[int, dict] = {}
        self.screens: Dict[int, Screen] = {}
        self._build_screens()
        self._build_obstacles()
        # Save static map layers so dynamic obstacles can be cleared
        self._static_grid = self.grid.copy()
        self._static_cost = self.cost.copy()
        self.dynamic_obstacles: List[dict] = []

    def _build_screens(self) -> None:
        building_centers = building_centers_from_tags(self.tag_poses)
        building_bounds = building_bounds_from_tags(self.tag_poses)
        excluded_ids = {int(item) for item in self.cfg["map"].get("excluded_screen_ids", [])}
        for key, corners in self.tag_poses.items():
            tag_id = int(key)
            if not (1 <= tag_id <= 36) or tag_id in excluded_ids:
                continue
            center = np.mean(np.array(corners, dtype=np.float64)[:4, :2], axis=0)
            group_id = (tag_id - 1) // 4
            surface = cardinal_surface_from_tag(
                self.tag_poses[str(tag_id)],
                building_centers[group_id],
            )
            face_center = face_center_from_bounds(building_bounds[group_id], surface["face"])
            geometry = build_interaction_geometry(
                face_center,
                surface["normal_xy"],
                self.cfg["interaction"],
            )
            target = geometry["target_xy"]
            target_distance = float(self.cfg["interaction"]["target_distance_cm"])
            tag_front = (
                face_center[0] + surface["normal_xy"][0] * target_distance,
                face_center[1] + surface["normal_xy"][1] * target_distance,
            )
            self.screens[tag_id] = Screen(
                screen_id=tag_id,
                tag_corners_3d=self.tag_poses[str(tag_id)],
                center_xy=(float(center[0]), float(center[1])),
                normal_xy=geometry["normal_xy"],
                normal_yaw_deg=geometry["normal_yaw_deg"],
                target_xy=target,
                interaction_xy=geometry["interaction_xy"],
                interaction_yaw_deg=geometry["interaction_yaw_deg"],
                reader_xy=geometry["reader_xy"],
                screen_left_tangent_xy=geometry["screen_left_tangent_xy"],
                surface_face=surface["face"],
                cardinal_normal_xy=surface["normal_xy"],
                face_center_xy=face_center,
                tag_front_xy=tag_front,
                task_target_xy=target,
                task_target_yaw_deg=geometry["interaction_yaw_deg"],
                worker_id=int(tag_id),
            )

    def _plane_normal_xy(self, corners: np.ndarray) -> np.ndarray:
        pts = np.array(corners, dtype=np.float64)
        edge = pts[1, :2] - pts[0, :2]
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([1.0, 0.0], dtype=np.float64)
        return normal

    def _build_obstacles(self) -> None:
        bounds = {}
        for key, corners_3d in self.tag_poses.items():
            tag_id = int(key)
            if not (1 <= tag_id <= 36):
                continue
            group_id = (tag_id - 1) // 4
            corners = np.array(corners_3d, dtype=np.float64)
            xy = corners[:4, :2]
            item = bounds.setdefault(
                group_id,
                {"x_min": 999.0, "x_max": -999.0, "y_min": 999.0, "y_max": -999.0},
            )
            item["x_min"] = min(item["x_min"], float(np.min(xy[:, 0])))
            item["x_max"] = max(item["x_max"], float(np.max(xy[:, 0])))
            item["y_min"] = min(item["y_min"], float(np.min(xy[:, 1])))
            item["y_max"] = max(item["y_max"], float(np.max(xy[:, 1])))

        self.building_bounds = {group_id: dict(item) for group_id, item in bounds.items()}
        inflation_cm = max(0.0, float(self.cfg["map"].get("obstacle_inflation_cm", 0.0)))
        inflation = int(math.ceil(inflation_cm / self.res))
        for item in bounds.values():
            x0 = int(item["x_min"] / self.res)
            x1 = int(item["x_max"] / self.res)
            y0 = int(item["y_min"] / self.res)
            y1 = int(item["y_max"] / self.res)
            gx0 = max(0, x0 - inflation)
            gx1 = min(self.rows - 1, x1 + inflation)
            gy0 = max(0, y0 - inflation)
            gy1 = min(self.cols - 1, y1 + inflation)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    cell_xy = self.xy_from_grid((gx, gy))
                    dist = self.distance_to_obstacle_rect_cm(cell_xy, item)
                    self.cost[gx, gy] = max(self.cost[gx, gy], self.obstacle_cost_for_distance(dist))
            self.grid[max(0, x0) : min(self.rows - 1, x1) + 1, max(0, y0) : min(self.cols - 1, y1) + 1] = 1

    def distance_to_obstacle_rect_cm(self, xy, bounds: dict) -> float:
        dx = max(float(bounds["x_min"]) - float(xy[0]), 0.0, float(xy[0]) - float(bounds["x_max"]))
        dy = max(float(bounds["y_min"]) - float(xy[1]), 0.0, float(xy[1]) - float(bounds["y_max"]))
        return math.hypot(dx, dy)

    def obstacle_cost_for_distance(self, distance_cm: float, hard_margin_override: Optional[float] = None) -> float:
        map_cfg = self.cfg["map"]
        inflation = max(0.0, float(map_cfg.get("obstacle_inflation_cm", 0.0)))
        if inflation <= 0.0 or distance_cm > inflation:
            return 0.0
        max_cost = max(0.0, float(map_cfg.get("obstacle_cost_max", 80.0)))
        if hard_margin_override is not None:
            hard_margin = clamp(float(hard_margin_override), 0.0, inflation)
        else:
            hard_margin = clamp(float(map_cfg.get("obstacle_hard_margin_cm", 0.0)), 0.0, inflation)
        if distance_cm <= hard_margin:
            return max_cost
        soft_span = max(1e-6, inflation - hard_margin)
        ratio = 1.0 - (float(distance_cm) - hard_margin) / soft_span
        power = max(0.1, float(map_cfg.get("obstacle_cost_power", 2.0)))
        return max_cost * max(0.0, ratio) ** power

    def clear_dynamic_obstacles(self) -> None:
        """Remove all dynamic obstacles, restoring the static map layers."""
        self.grid[:] = self._static_grid
        self.cost[:] = self._static_cost
        self.dynamic_obstacles.clear()

    def add_dynamic_obstacle(self, center_xy: Tuple[float, float], size_cm: float = 20.0) -> None:
        """Add a dynamic obstacle (e.g. another robot) centered at center_xy with given square size.

        Applies the same inflation and cost model as static obstacles.
        """
        half = size_cm / 2.0
        hard_margin = float(self.cfg["obstacle"].get("dynamic_hard_margin_cm", 10.0))
        inflation_cm = max(hard_margin, float(self.cfg["map"].get("obstacle_inflation_cm", 0.0)))
        bounds = {
            "x_min": float(center_xy[0]) - half,
            "x_max": float(center_xy[0]) + half,
            "y_min": float(center_xy[1]) - half,
            "y_max": float(center_xy[1]) + half,
        }
        inflation_cm = max(hard_margin, float(self.cfg["map"].get("obstacle_inflation_cm", 0.0)))
        inflation = int(math.ceil(inflation_cm / self.res))
        x0 = int(bounds["x_min"] / self.res)
        x1 = int(bounds["x_max"] / self.res)
        y0 = int(bounds["y_min"] / self.res)
        y1 = int(bounds["y_max"] / self.res)
        gx0 = max(0, x0 - inflation)
        gx1 = min(self.rows - 1, x1 + inflation)
        gy0 = max(0, y0 - inflation)
        gy1 = min(self.cols - 1, y1 + inflation)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                cell_xy = self.xy_from_grid((gx, gy))
                dist = self.distance_to_obstacle_rect_cm(cell_xy, bounds)
                cost = self.obstacle_cost_for_distance(dist, hard_margin_override=hard_margin)
                self.cost[gx, gy] = max(self.cost[gx, gy], cost)
        # Mark hard obstacle cells
        self.grid[max(0, x0):min(self.rows - 1, x1) + 1, max(0, y0):min(self.cols - 1, y1) + 1] = 1
        self.dynamic_obstacles.append({"center_xy": center_xy, "size_cm": size_cm, **bounds})

    def grid_pos(self, xy) -> Tuple[int, int]:
        return (
            int(clamp(float(xy[0]) / self.res, 0, self.rows - 1)),
            int(clamp(float(xy[1]) / self.res, 0, self.cols - 1)),
        )

    def xy_from_grid(self, node) -> Tuple[float, float]:
        return (node[0] * self.res + self.res / 2.0, node[1] * self.res + self.res / 2.0)

    def in_bounds_xy(self, xy) -> bool:
        return 0.0 <= float(xy[0]) <= self.width_cm and 0.0 <= float(xy[1]) <= self.height_cm

    def is_free_grid(self, node) -> bool:
        return 0 <= node[0] < self.rows and 0 <= node[1] < self.cols and self.grid[node[0], node[1]] == 0

    def is_free_xy(self, xy) -> bool:
        return self.in_bounds_xy(xy) and self.is_free_grid(self.grid_pos(xy))

    def is_traversable_xy(self, xy, max_cost: Optional[float] = None) -> bool:
        if not self.is_free_xy(xy):
            return False
        node = self.grid_pos(xy)
        if max_cost is None:
            max_cost = float(self.cfg["map"].get("obstacle_line_clear_max_cost", 60.0))
        return self.cost[node[0], node[1]] < float(max_cost)

    def non_target_obstacle_cost_xy(self, xy, target_screen_id: int) -> float:
        """Return inflation cost while excluding only the locked target building."""
        target_group = (int(target_screen_id) - 1) // 4
        maximum = 0.0
        for group_id, bounds in self.building_bounds.items():
            if int(group_id) == target_group:
                continue
            maximum = max(
                maximum,
                self.obstacle_cost_for_distance(self.distance_to_obstacle_rect_cm(xy, bounds)),
            )
        dynamic_hard = float(self.cfg.get("obstacle", {}).get("dynamic_hard_margin_cm", 10.0))
        for bounds in self.dynamic_obstacles:
            maximum = max(
                maximum,
                self.obstacle_cost_for_distance(
                    self.distance_to_obstacle_rect_cm(xy, bounds),
                    hard_margin_override=dynamic_hard,
                ),
            )
        return maximum

    def target_direct_corridor_clear(
        self,
        start_xy,
        goal_xy,
        target_screen_id: int,
        half_width_cm: float,
        max_non_target_cost: float,
        sample_step_cm: float = 2.0,
    ) -> bool:
        """Check a narrow final corridor, ignoring only target-building inflation.

        Physical occupancy is never ignored. Inflation from every other static
        building and every dynamic obstacle remains active.
        """
        if not self.in_bounds_xy(start_xy) or not self.in_bounds_xy(goal_xy):
            return False
        dx = float(goal_xy[0]) - float(start_xy[0])
        dy = float(goal_xy[1]) - float(start_xy[1])
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return self.is_free_xy(goal_xy)
        tangent = (dx / length, dy / length)
        lateral = (-tangent[1], tangent[0])
        longitudinal_steps = max(1, int(math.ceil(length / max(0.5, sample_step_cm))))
        half_width = max(0.0, float(half_width_cm))
        lateral_steps = max(1, int(math.ceil(half_width / max(0.5, sample_step_cm))))
        offsets = [0.0]
        for index in range(1, lateral_steps + 1):
            offset = min(half_width, index * max(0.5, sample_step_cm))
            offsets.extend((-offset, offset))
        for index in range(longitudinal_steps + 1):
            along = min(length, index * max(0.5, sample_step_cm))
            center = (
                float(start_xy[0]) + tangent[0] * along,
                float(start_xy[1]) + tangent[1] * along,
            )
            for offset in offsets:
                sample = (
                    center[0] + lateral[0] * offset,
                    center[1] + lateral[1] * offset,
                )
                if not self.is_free_xy(sample):
                    return False
                if self.non_target_obstacle_cost_xy(sample, target_screen_id) >= float(max_non_target_cost):
                    return False
        return True

    def nearest_free_xy(self, xy) -> Tuple[float, float]:
        start = self.grid_pos(xy)
        queue = [start]
        visited = {start}
        while queue:
            node = queue.pop(0)
            if self.is_free_grid(node):
                return self.xy_from_grid(node)
            for nx in self._neighbors(node, include_diagonal=True):
                if nx not in visited:
                    visited.add(nx)
                    queue.append(nx)
        return self.xy_from_grid(start)

    def nearest_traversable_xy(self, xy) -> Tuple[float, float]:
        start = self.grid_pos(xy)
        queue = [start]
        visited = {start}
        while queue:
            node = queue.pop(0)
            if self.is_free_grid(node) and self.cost[node[0], node[1]] < float(self.cfg["map"].get("obstacle_line_clear_max_cost", 60.0)):
                return self.xy_from_grid(node)
            for nx in self._neighbors(node, include_diagonal=True):
                if nx not in visited:
                    visited.add(nx)
                    queue.append(nx)
        return self.nearest_free_xy(xy)

    def _neighbors(self, node, include_diagonal=True):
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if include_diagonal:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in steps:
            nxt = (node[0] + dx, node[1] + dy)
            if 0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols:
                yield nxt

    def line_clear(
        self,
        start_xy,
        goal_xy,
        step_cm=2.0,
        max_cost: Optional[float] = None,
        allow_goal_high_cost: bool = False,
    ) -> bool:
        dist = distance_xy(start_xy, goal_xy)
        if dist < 1e-6:
            if allow_goal_high_cost:
                return self.is_free_xy(goal_xy)
            return self.is_traversable_xy(start_xy, max_cost=max_cost)
        if not self.in_bounds_xy(start_xy) or not self.in_bounds_xy(goal_xy):
            return False
        goal_node = self.grid_pos(goal_xy)
        steps = max(1, int(dist / step_cm))
        for i in range(1, steps):
            t = i / float(steps)
            x = start_xy[0] + (goal_xy[0] - start_xy[0]) * t
            y = start_xy[1] + (goal_xy[1] - start_xy[1]) * t
            sample = (x, y)
            # The only high-cost exception is the grid cell containing the
            # locked task endpoint.  Physical obstacle cells remain blocked.
            in_goal_cell = allow_goal_high_cost and self.grid_pos(sample) == goal_node
            if in_goal_cell:
                if not self.is_free_xy(sample):
                    return False
            elif not self.is_traversable_xy(sample, max_cost=max_cost):
                return False
        if allow_goal_high_cost:
            return self.is_free_xy(goal_xy)
        return self.is_traversable_xy(goal_xy, max_cost=max_cost)

    def is_dangerously_close_to_wall(self, xy, yaw_deg, safe_dist_cm) -> bool:
        rad = math.radians(yaw_deg)
        steps = max(1, int(safe_dist_cm / 2.0))
        for i in range(1, steps + 1):
            pt = (xy[0] + i * 2.0 * math.cos(rad), xy[1] + i * 2.0 * math.sin(rad))
            if not self.is_free_xy(pt):
                return True
        return False

    def plan(self, start_xy, goal_xy, allow_goal_high_cost: bool = False) -> List[Tuple[float, float]]:
        start = self.grid_pos(start_xy)
        goal = self.grid_pos(goal_xy)
        if not self.in_bounds_xy(start_xy) or not self.is_free_grid(start):
            start = self.grid_pos(self.nearest_free_xy(start_xy))
        if allow_goal_high_cost and (not self.in_bounds_xy(goal_xy) or not self.is_free_xy(goal_xy)):
            return []
        if not allow_goal_high_cost and (not self.in_bounds_xy(goal_xy) or not self.is_traversable_xy(goal_xy)):
            goal = self.grid_pos(self.nearest_traversable_xy(goal_xy))

        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came = {}
        g = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = self._reconstruct_path(came, current)
                if allow_goal_high_cost:
                    exact_goal = (float(goal_xy[0]), float(goal_xy[1]))
                    if self.grid_pos(path[-1]) == goal:
                        path[-1] = exact_goal
                    elif distance_xy(path[-1], exact_goal) > 0.1:
                        path.append(exact_goal)
                return path
            for nxt in self._neighbors(current, include_diagonal=True):
                if not self.is_free_grid(nxt):
                    continue
                step = math.hypot(nxt[0] - current[0], nxt[1] - current[1])
                tentative = g[current] + step + float(self.cost[nxt[0], nxt[1]])
                if tentative < g.get(nxt, float("inf")):
                    came[nxt] = current
                    g[nxt] = tentative
                    priority = tentative + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(open_heap, (priority, nxt))
        return []

    def yaw_to_action_bin(self, yaw_deg: float, yaw_bin_deg: float, yaw_bins: int) -> int:
        yaw = normalize_angle_deg(yaw_deg)
        return int(round((yaw % 360.0) / yaw_bin_deg)) % yaw_bins

    def yaw_from_action_bin(self, yaw_idx: int, yaw_bin_deg: float) -> float:
        return normalize_angle_deg(float(yaw_idx) * yaw_bin_deg)

    def yaw_delta_to_bins(self, yaw_delta_deg: float, yaw_bin_deg: float) -> int:
        sign = 1 if yaw_delta_deg >= 0.0 else -1
        return sign * max(1, int(math.ceil(abs(float(yaw_delta_deg)) / max(1.0, yaw_bin_deg))))

    def action_planner_actions(self, navigation_cfg: dict, motion_cfg: dict) -> List[dict]:
        motion_actions = motion_cfg.get("actions", {})
        forward_default = abs(float(motion_actions.get("forward_fast", {}).get("forward_cm", 3.5))) * 4.0
        strafe_default = abs(float(motion_actions.get("strafe_left_fast", {}).get("lateral_cm", 4.0))) * 3.0
        forward_step = max(1.0, float(navigation_cfg.get("action_planner_forward_step_cm", forward_default)))
        strafe_step = max(1.0, float(navigation_cfg.get("action_planner_strafe_step_cm", strafe_default)))
        forward_cost = max(0.1, float(navigation_cfg.get("action_planner_forward_cost", 1.0)))
        strafe_cost = max(0.1, float(navigation_cfg.get("action_planner_strafe_cost", 1.0)))
        turn_cost_per_deg = max(0.0, float(navigation_cfg.get("action_planner_turn_cost_cm_per_deg", 1.1)))
        turn_fixed = max(0.0, float(navigation_cfg.get("action_planner_turn_fixed_cost_cm", 8.0)))
        large_extra = max(0.0, float(navigation_cfg.get("action_planner_large_turn_extra_cost_cm", 4.0)))
        consecutive_turn_penalty = max(
            0.0, float(navigation_cfg.get("action_planner_consecutive_turn_penalty_cm", 12.0))
        )
        reverse_turn_penalty = max(
            0.0, float(navigation_cfg.get("action_planner_reverse_turn_penalty_cm", 35.0))
        )
        in_place_turn_penalty = max(
            0.0, float(navigation_cfg.get("action_planner_in_place_turn_penalty_cm", 10.0))
        )

        actions = [
            {
                "name": "forward",
                "forward_cm": forward_step,
                "lateral_cm": 0.0,
                "yaw_deg": 0.0,
                "base_cost": forward_step * forward_cost,
            },
            {
                "name": "strafe_left",
                "forward_cm": 0.0,
                "lateral_cm": strafe_step,
                "yaw_deg": 0.0,
                "base_cost": strafe_step * strafe_cost,
            },
            {
                "name": "strafe_right",
                "forward_cm": 0.0,
                "lateral_cm": -strafe_step,
                "yaw_deg": 0.0,
                "base_cost": strafe_step * strafe_cost,
            },
        ]
        turn_specs = [
            ("turn_left_small", motion_actions.get("turn_left_fast", {}).get("yaw_deg", 7.5), 0.0),
            ("turn_right_small", motion_actions.get("turn_right_fast", {}).get("yaw_deg", -11.25), 0.0),
            ("turn_left_large", motion_actions.get("turn_left_large", {}).get("yaw_deg", 30.0), large_extra),
            ("turn_right_large", motion_actions.get("turn_right_large", {}).get("yaw_deg", -45.0), large_extra),
        ]
        for name, yaw_deg, extra_cost in turn_specs:
            yaw_delta = float(yaw_deg)
            if abs(yaw_delta) < 1e-6:
                continue
            actions.append(
                {
                    "name": name,
                    "forward_cm": 0.0,
                    "lateral_cm": 0.0,
                    "yaw_deg": yaw_delta,
                    "base_cost": turn_fixed + abs(yaw_delta) * turn_cost_per_deg + extra_cost,
                    "consecutive_turn_penalty": consecutive_turn_penalty,
                    "reverse_turn_penalty": reverse_turn_penalty,
                    "in_place_turn_penalty": in_place_turn_penalty,
                }
            )
        return actions

    def segment_obstacle_cost(self, start_xy, goal_xy, step_cm: float = 2.0) -> float:
        dist = distance_xy(start_xy, goal_xy)
        steps = max(1, int(dist / max(1.0, step_cm)))
        total = 0.0
        count = 0
        for i in range(steps + 1):
            t = i / float(steps)
            x = start_xy[0] + (goal_xy[0] - start_xy[0]) * t
            y = start_xy[1] + (goal_xy[1] - start_xy[1]) * t
            if not self.in_bounds_xy((x, y)):
                continue
            node = self.grid_pos((x, y))
            total += float(self.cost[node[0], node[1]])
            count += 1
        return total / max(1, count)

    def action_planner_transition(
        self,
        state,
        action: dict,
        yaw_bin_deg: float,
        yaw_bins: int,
        segment_max_cost: float,
        obstacle_cost_scale: float,
        goal_node=None,
        allow_goal_high_cost: bool = False,
    ):
        gx, gy, yaw_idx = state[:3]
        previous_turn_sign = int(state[3]) if len(state) >= 4 else 0
        current_xy = self.xy_from_grid((gx, gy))
        yaw_delta = float(action.get("yaw_deg", 0.0))
        if abs(yaw_delta) > 1e-6:
            bin_delta = self.yaw_delta_to_bins(yaw_delta, yaw_bin_deg)
            turn_sign = 1 if yaw_delta > 0.0 else -1
            turn_cost = float(action["base_cost"]) + float(action.get("in_place_turn_penalty", 0.0))
            if previous_turn_sign == turn_sign:
                turn_cost += float(action.get("consecutive_turn_penalty", 0.0))
            elif previous_turn_sign == -turn_sign:
                turn_cost += float(action.get("reverse_turn_penalty", 0.0))
            return (gx, gy, (yaw_idx + bin_delta) % yaw_bins, turn_sign), turn_cost

        yaw = math.radians(self.yaw_from_action_bin(yaw_idx, yaw_bin_deg))
        left_yaw = yaw + math.pi / 2.0
        next_xy = (
            current_xy[0]
            + float(action.get("forward_cm", 0.0)) * math.cos(yaw)
            + float(action.get("lateral_cm", 0.0)) * math.cos(left_yaw),
            current_xy[1]
            + float(action.get("forward_cm", 0.0)) * math.sin(yaw)
            + float(action.get("lateral_cm", 0.0)) * math.sin(left_yaw),
        )
        if not self.is_free_xy(next_xy):
            return None
        nx, ny = self.grid_pos(next_xy)
        is_goal = bool(allow_goal_high_cost and goal_node is not None and (nx, ny) == tuple(goal_node))
        if not self.line_clear(
            current_xy,
            next_xy,
            max_cost=segment_max_cost,
            allow_goal_high_cost=is_goal,
        ):
            return None
        if (nx, ny) == (gx, gy):
            return None
        move_cost = float(action["base_cost"])
        move_cost += obstacle_cost_scale * self.segment_obstacle_cost(current_xy, next_xy)
        return (nx, ny, yaw_idx, 0), move_cost

    def _reconstruct_action_path(self, came, current, start_xy) -> List[Tuple[float, float]]:
        states = [current]
        while current in came:
            current = came[current]
            states.append(current)
        states.reverse()
        points = [(float(start_xy[0]), float(start_xy[1]))]
        for state in states[1:]:
            gx, gy = state[:2]
            xy = self.xy_from_grid((gx, gy))
            if distance_xy(points[-1], xy) > 1.0:
                points.append(xy)
        return points

    def plan_action_path(
        self,
        start_pose,
        goal_xy,
        navigation_cfg: dict,
        motion_cfg: dict,
        allow_goal_high_cost: bool = False,
    ) -> List[Tuple[float, float]]:
        start_xy = start_pose.xy()
        if not self.in_bounds_xy(start_xy) or not self.is_free_xy(start_xy):
            start_xy = self.nearest_free_xy(start_xy)
        if allow_goal_high_cost and (not self.in_bounds_xy(goal_xy) or not self.is_free_xy(goal_xy)):
            return []
        if not allow_goal_high_cost and (not self.in_bounds_xy(goal_xy) or not self.is_traversable_xy(goal_xy)):
            goal_xy = self.nearest_traversable_xy(goal_xy)

        yaw_bin_deg = max(5.0, float(navigation_cfg.get("action_planner_yaw_bin_deg", 15.0)))
        yaw_bins = max(8, int(round(360.0 / yaw_bin_deg)))
        yaw_bin_deg = 360.0 / float(yaw_bins)
        start_grid = self.grid_pos(start_xy)
        start_state = (
            start_grid[0],
            start_grid[1],
            self.yaw_to_action_bin(start_pose.yaw_deg, yaw_bin_deg, yaw_bins),
            0,
        )
        goal_node = self.grid_pos(goal_xy)
        goal_tolerance = max(self.res, float(navigation_cfg.get("action_planner_goal_tolerance_cm", 8.0)))
        max_expansions = max(100, int(navigation_cfg.get("action_planner_max_expansions", 45000)))
        segment_max_cost = float(navigation_cfg.get("action_planner_segment_max_cost", 85.0))
        obstacle_cost_scale = max(0.0, float(navigation_cfg.get("action_planner_obstacle_cost_scale", 1.0)))
        actions = self.action_planner_actions(navigation_cfg, motion_cfg)
        if not actions:
            return []

        open_heap = []
        counter = 0
        start_h = distance_xy(self.xy_from_grid(start_grid), goal_xy)
        heapq.heappush(open_heap, (start_h, counter, start_state))
        came = {}
        g = {start_state: 0.0}
        closed = set()
        expansions = 0

        while open_heap and expansions < max_expansions:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            current_g = g.get(current, float("inf"))
            current_xy = self.xy_from_grid((current[0], current[1]))
            goal_reached = (current[0], current[1]) == goal_node
            if not allow_goal_high_cost:
                goal_reached = goal_reached or distance_xy(current_xy, goal_xy) <= goal_tolerance
            if goal_reached:
                path = self._reconstruct_action_path(came, current, start_xy)
                if allow_goal_high_cost:
                    exact_goal = (float(goal_xy[0]), float(goal_xy[1]))
                    if self.grid_pos(path[-1]) == goal_node:
                        path[-1] = exact_goal
                    elif distance_xy(path[-1], exact_goal) > 0.1:
                        path.append(exact_goal)
                return path
            expansions += 1
            for action in actions:
                transition = self.action_planner_transition(
                    current,
                    action,
                    yaw_bin_deg,
                    yaw_bins,
                    segment_max_cost,
                    obstacle_cost_scale,
                    goal_node=goal_node,
                    allow_goal_high_cost=allow_goal_high_cost,
                )
                if transition is None:
                    continue
                nxt, action_cost = transition
                tentative = current_g + action_cost
                if tentative >= g.get(nxt, float("inf")):
                    continue
                came[nxt] = current
                g[nxt] = tentative
                next_xy = self.xy_from_grid((nxt[0], nxt[1]))
                priority = tentative + distance_xy(next_xy, goal_xy)
                counter += 1
                heapq.heappush(open_heap, (priority, counter, nxt))
        return []

    def _reconstruct_path(self, came, current) -> List[Tuple[float, float]]:
        nodes = [current]
        while current in came:
            current = came[current]
            nodes.append(current)
        nodes.reverse()
        points = [self.xy_from_grid(node) for node in nodes]
        return self.smooth_path(points)

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_clear(path[i], path[j]):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def unfinished_screens(self) -> Iterable[Screen]:
        return (screen for screen in self.screens.values() if not screen.terminal())

    def completed_count(self) -> int:
        return sum(1 for screen in self.screens.values() if screen.successful())

    def processed_count(self) -> int:
        return sum(1 for screen in self.screens.values() if screen.done())

    def failed_count(self) -> int:
        return sum(1 for screen in self.screens.values() if screen.status.value == "FAILED")
