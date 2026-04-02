"""
Hand landmark detection — Week 4.

Uses MediaPipe's Tasks API (`HandLandmarker`), not the older `mp.solutions`
API — current MediaPipe releases moved detection behind Tasks, which reads
a `.task` model bundle from disk rather than shipping weights in the pip
package (the same "pretrained weights, not shipped in the wheel" situation
as YOLO's `.pt` file). See `docs/cv_pipeline.md` for the download step.

WHAT'S A LANDMARK: 21 (x, y, z) points per hand — wrist, then 4 joints per
finger — in normalized image coordinates (0-1, origin top-left). This
module returns that raw geometry; `features.py` is where geometry becomes
meaning ("is the index finger extended").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class GestureModelError(Exception):
    """Raised when a MediaPipe .task model bundle is missing or fails to load."""


Landmark = tuple[float, float, float]  # (x, y, z), normalized


@dataclass
class HandResult:
    landmarks: list[Landmark]  # 21 points
    handedness: str  # "Left" | "Right" (as reported by MediaPipe, mirrored view)
    confidence: float


class HandLandmarkerWrapper:
    """Loads a HandLandmarker model bundle and runs per-frame inference.

    Usage:
        hands = HandLandmarkerWrapper()
        results = hands.detect(frame_rgb)   # list[HandResult], 0-2 entries
        hands.close()
    """

    def __init__(self, model_path: str | None = None, num_hands: int = 2):
        self.model_path = Path(model_path or settings.hand_model_path)
        self._landmarker = self._load(num_hands)

    def _load(self, num_hands: int):
        if not self.model_path.exists():
            raise GestureModelError(
                f"Hand landmarker model not found at {self.model_path}. "
                f"Download it from "
                f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                f"hand_landmarker/float16/1/hand_landmarker.task"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=settings.gesture_min_confidence,
            )
            self._mp = mp
            return vision.HandLandmarker.create_from_options(options)
        except GestureModelError:
            raise
        except Exception as e:
            raise GestureModelError(f"Failed to load hand landmarker: {e}") from e

    def detect(self, frame_rgb: np.ndarray) -> list[HandResult]:
        """`frame_rgb` must be RGB (see vision/preprocessing.to_rgb) — MediaPipe,
        unlike OpenCV, expects RGB, not BGR."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        hands: list[HandResult] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=False):
            points = [(lm.x, lm.y, lm.z) for lm in landmarks]
            top = handedness[0]
            hands.append(HandResult(landmarks=points, handedness=top.category_name, confidence=top.score))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandLandmarkerWrapper:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
