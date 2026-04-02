"""
Body pose landmark detection — Week 4.

Same Tasks-API pattern as hands.py, using `PoseLandmarker` instead of
`HandLandmarker`. 33 (x, y, z) points per detected person — shoulders,
hips, wrists, etc. Index constants below match MediaPipe's documented pose
landmark ordering (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import settings
from .hands import GestureModelError, Landmark

logger = logging.getLogger(__name__)

# The landmarks features.py actually needs — not the full 33.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14  # V2.0 Part D — arm_aim_vector's elbow->wrist direction
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


class PoseLandmarkerWrapper:
    """Loads a PoseLandmarker model bundle and runs per-frame inference.

    Usage:
        pose = PoseLandmarkerWrapper()
        landmarks = pose.detect(frame_rgb)   # list[Landmark] (33) or None
        pose.close()
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or settings.pose_model_path)
        self._landmarker = self._load()

    def _load(self):
        if not self.model_path.exists():
            raise GestureModelError(
                f"Pose landmarker model not found at {self.model_path}. "
                f"Download it from "
                f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                f"pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=settings.gesture_min_confidence,
            )
            self._mp = mp
            return vision.PoseLandmarker.create_from_options(options)
        except GestureModelError:
            raise
        except Exception as e:
            raise GestureModelError(f"Failed to load pose landmarker: {e}") from e

    def detect(self, frame_rgb: np.ndarray) -> list[Landmark] | None:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        # Only the first detected person — this project is single-player.
        return [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseLandmarkerWrapper:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
