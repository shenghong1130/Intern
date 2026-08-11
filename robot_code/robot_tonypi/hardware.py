#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TonyPi hardware adapters."""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .models import ActionResult


def add_tonypi_paths(config) -> None:
    for key in ("tonypi_root", "tonypi_sdk"):
        path = config["paths"].get(key)
        if path and path not in sys.path:
            sys.path.insert(0, path)


class RealtimeCamera:
    def __init__(self, config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.ret = False
        self.frame = None
        self.running = False
        self.thread = None
        self.cam = None
        if dry_run:
            return
        add_tonypi_paths(config)
        import hiwonder.Camera as Camera

        self.cam = Camera.Camera()
        self.cam.camera_open()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5.0
        while self.frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError("Camera opened but no frame arrived within 5 seconds")

    def _loop(self):
        while self.running:
            ret, frame = self.cam.read()
            if ret and frame is not None:
                self.ret = True
                self.frame = frame.copy()
            time.sleep(0.01)

    def read(self):
        if self.dry_run:
            return False, None
        if self.frame is None:
            return False, None
        return self.ret, self.frame.copy()

    def capture_settled(self, discard_frames: Optional[int] = None, frame_gap_s: Optional[float] = None):
        discard = self.config["camera"]["discard_frames"] if discard_frames is None else discard_frames
        gap = self.config["camera"]["frame_gap_s"] if frame_gap_s is None else frame_gap_s
        frame = None
        for _ in range(max(0, int(discard))):
            self.read()
            time.sleep(float(gap))
        ret, frame = self.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self.dry_run:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cam is not None:
            self.cam.camera_close()


class TonyPiHardware:
    def __init__(self, config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.board = None
        self.ctl = None
        self.servo_data = {"servo2": 1500}
        self.AGC = None
        self.gyro_bias = 0.0
        self.interaction_active = False
        if dry_run:
            return
        add_tonypi_paths(config)
        import hiwonder.ActionGroupControl as AGC
        import hiwonder.ros_robot_controller_sdk as rrc
        from hiwonder.Controller import Controller
        import hiwonder.yaml_handle as yaml_handle

        self.AGC = AGC
        self.board = rrc.Board()
        self.ctl = Controller(self.board)
        if bool(config["motion"].get("enable_imu", True)):
            try:
                self.ctl.enable_recv()
            except Exception:
                pass
        self.servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

    def require_action_group(self, group: str) -> None:
        action_dir = Path(self.config["paths"]["action_group_dir"])
        path = action_dir / (group + ".d6a")
        if not self.dry_run and not path.exists():
            raise FileNotFoundError("Action group not found: {}".format(path))

    def center_head(self) -> None:
        self.set_head_pan_angle(float(self.config["camera"]["head_center_angle"]))
        self.set_head_tilt_pulse(int(self.config["camera"]["head_tilt_center_pulse"]), use_time_ms=300)

    def set_head_tilt_pulse(self, pulse: int, use_time_ms: int = 500) -> None:
        pulse = max(500, min(2500, int(pulse)))
        if self.dry_run:
            print("[dry-run] head tilt pulse={} time={}ms".format(pulse, use_time_ms))
            return
        self.ctl.set_pwm_servo_pulse(1, pulse, use_time_ms)
        time.sleep(use_time_ms / 1000.0 + 0.05)

    def set_head_pan_angle(self, angle: float, use_time_ms: Optional[int] = None) -> None:
        if use_time_ms is None:
            use_time_ms = int(self.config["camera"]["head_move_ms"])
        pulse = int(1500 + (float(angle) - 100.0) * 10.0)
        pulse = max(500, min(2500, pulse))
        adjusted = pulse + int(self.servo_data.get("servo2", 1500)) - 1500
        adjusted = max(500, min(2500, adjusted))
        if self.dry_run:
            print("[dry-run] head pan angle={} pulse={} time={}ms".format(angle, adjusted, use_time_ms))
            return
        self.ctl.set_pwm_servo_pulse(2, adjusted, use_time_ms)
        time.sleep(use_time_ms / 1000.0 + float(self.config["camera"]["settle_s"]))

    def run_action(self, key: str, times_override: Optional[int] = None) -> ActionResult:
        if self.interaction_active and key != "stand":
            raise RuntimeError("navigation action '{}' blocked during left-hand interaction".format(key))
        actions = self.config["motion"]["actions"]
        if key not in actions:
            raise KeyError("Unknown motion action key: {}".format(key))
        spec = actions[key]
        times = int(times_override if times_override is not None else spec.get("times", 1))
        steps = spec.get("sequence")
        if steps:
            run_steps = []
            for step in steps:
                step_group = step["group"]
                step_times = int(step.get("times", 1))
                if bool(step.get("repeat", False)):
                    step_times *= times
                step_with_stand = bool(step.get("with_stand", False))
                run_steps.append((step_group, step_times, step_with_stand))
        else:
            group = spec["group"]
            with_stand = bool(spec.get("with_stand", False))
            run_steps = [(group, times, with_stand)]

        for step_group, _, _ in run_steps:
            self.require_action_group(step_group)
        group_label = "+".join(step_group for step_group, _, _ in run_steps)
        start = time.monotonic()
        error = ""
        ok = True
        print("[action] {} -> {}".format(key, ", ".join(
            "{} x{} with_stand={}".format(step_group, step_times, step_with_stand)
            for step_group, step_times, step_with_stand in run_steps
        )))
        try:
            for step_group, step_times, step_with_stand in run_steps:
                if step_times <= 0:
                    continue
                if self.dry_run:
                    time.sleep(0.05 * max(1, step_times))
                else:
                    self.AGC.runActionGroup(step_group, times=step_times, with_stand=step_with_stand)
        except Exception as exc:
            ok = False
            error = str(exc)
        time.sleep(float(spec.get("settle_s", 0.25)))
        elapsed = time.monotonic() - start
        return ActionResult(
            key=key,
            group=group_label,
            times=times,
            elapsed_s=elapsed,
            model_forward_cm=float(spec.get("forward_cm", 0.0)) * times,
            model_lateral_cm=float(spec.get("lateral_cm", 0.0)) * times,
            model_yaw_deg=float(spec.get("yaw_deg", 0.0)) * times,
            ok=ok,
            error=error,
        )

    def set_interaction_active(self, active: bool) -> None:
        """Block ordinary navigation actions while the NFC hand transaction runs."""
        self.interaction_active = bool(active)

    def read_imu(self):
        if self.dry_run or self.ctl is None:
            return None
        try:
            return self.ctl.get_imu()
        except Exception:
            return None

    def stop(self) -> None:
        if self.dry_run:
            return
        try:
            self.AGC.stopActionGroup()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.center_head()
        except Exception:
            pass
