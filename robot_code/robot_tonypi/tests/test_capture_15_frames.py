#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone TonyPi camera check that saves 15 fresh frames.

The module intentionally contains no ``unittest.TestCase`` and opens the
camera only from ``main()``, so normal unit-test discovery remains hardware
safe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from robot_tonypi.config import default_config_path, load_config
    from robot_tonypi.hardware import RealtimeCamera
else:
    from ..config import default_config_path, load_config
    from ..hardware import RealtimeCamera


FRAME_COUNT = 15


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Capture and save 15 TonyPi camera frames")
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.2,
        help="minimum delay between saved frames (default: 0.2 seconds)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory; default: capture_15_frames_runs/<timestamp>",
    )
    return parser.parse_args(argv)


def make_output_dir(value=None) -> Path:
    if value:
        output_dir = Path(value).expanduser().resolve()
    else:
        output_dir = Path.cwd() / "capture_15_frames_runs" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_frame(path: Path, frame) -> None:
    import cv2

    if frame is None or not cv2.imwrite(str(path), frame):
        raise RuntimeError("failed to save camera frame: {}".format(path))


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be >= 0")

    config = load_config(args.config)
    output_dir = make_output_dir(args.output_dir)
    camera = None
    captures = []
    try:
        camera = RealtimeCamera(config, dry_run=False)
        for index in range(1, FRAME_COUNT + 1):
            started = time.monotonic()
            frame = camera.capture_settled()
            if frame is None:
                raise RuntimeError("camera returned no frame at capture {}/{}".format(index, FRAME_COUNT))

            filename = "frame_{:02d}.jpg".format(index)
            save_frame(output_dir / filename, frame)
            captured_at = time.time()
            captures.append(
                {
                    "index": index,
                    "file": filename,
                    "captured_at_unix_s": captured_at,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            )
            print("[capture] {}/{} {}".format(index, FRAME_COUNT, output_dir / filename), flush=True)

            if index < FRAME_COUNT:
                remaining = float(args.interval_s) - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

        manifest = {
            "frame_count": len(captures),
            "requested_frame_count": FRAME_COUNT,
            "interval_s": float(args.interval_s),
            "captures": captures,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[done] saved {} frames to {}".format(len(captures), output_dir), flush=True)
        return 0
    finally:
        if camera is not None:
            camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
