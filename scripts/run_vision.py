#!/usr/bin/env python
"""Week 1 standalone demo: live camera feed with a diagnostics overlay.

Usage:
    python scripts/run_vision.py

Controls:
    Q  quit
    R  start/stop recording  (data/recordings/<timestamp>.mp4)
    S  screenshot            (data/recordings/<timestamp>.png)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

import cv2

from app.config import DATA_DIR
from app.logging_config import setup_logging
from app.vision.camera import Camera, CameraError
from app.vision.visualization import draw_diagnostics

logger = logging.getLogger("run_vision")


def main() -> int:
    setup_logging()
    try:
        with Camera() as cam:
            logger.info("Camera live. Q=quit, R=record, S=screenshot")
            while True:
                try:
                    frame, diag = cam.read()
                except CameraError as e:
                    logger.error("Camera error: %s", e)
                    break

                draw_diagnostics(frame, diag, recording=cam.is_recording)
                cv2.imshow("VisionStrike — Vision Monitor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    if cam.is_recording:
                        cam.stop_recording()
                    else:
                        cam.start_recording(DATA_DIR / "recordings" / f"{int(time.time())}.mp4")
                elif key == ord("s"):
                    cam.screenshot(frame, DATA_DIR / "recordings" / f"{int(time.time())}.png")
    except CameraError as e:
        logger.error("Could not start camera: %s", e)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
