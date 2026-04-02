"""Small, pure frame-preprocessing helpers shared by the camera diagnostics
overlay, the detector, and recordings. Kept separate from camera.py so
"how do we get a frame" and "what do we do to a frame" stay independent."""

from __future__ import annotations

import cv2
import numpy as np


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """OpenCV reads/writes BGR; anything handed to a non-OpenCV consumer
    (e.g. MediaPipe, which expects RGB) needs this conversion first."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def mirror(frame: np.ndarray) -> np.ndarray:
    """Horizontal flip — a webcam feed reads more naturally to the player
    as a mirror (their right hand appears on their screen-right)."""
    return cv2.flip(frame, 1)
