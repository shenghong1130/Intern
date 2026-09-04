#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small utility helpers."""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict


def now_s() -> float:
    return time.monotonic()


def normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle_diff_deg(target: float, current: float) -> float:
    return normalize_angle_deg(target - current)


def distance_xy(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ensure_dir(path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data: Dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
