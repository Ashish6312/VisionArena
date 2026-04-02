"""
YOLO object detection — Week 2.

DETECTION vs CLASSIFICATION: classification answers "what is this image?"
(one label). Detection answers "what objects, and where?" — a list of
boxes, each with a label and confidence.

WHY YOLO: single forward pass through a CNN produces all boxes at once
("You Only Look Once"), which is what makes it fast enough for real-time
video. This project loads Ultralytics' pretrained YOLOv8n weights
(COCO-trained, 80 classes including "person") — it does not train or
implement a detector from scratch.

DECOUPLING FROM THE GAME: `Detector.detect()` returns plain `Detection`
dataclasses, not anything Ultralytics-specific. Nothing outside this file
imports `ultralytics` — swapping the detection backend later only touches
this one module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when the YOLO model weights cannot be loaded."""


@dataclass
class Detection:
    """One detected object in a single frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict:
        return {
            "bbox": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


_DEFAULT_TARGET_CLASSES = object()  # sentinel: distinguishes "not passed" from explicit None ("no filter")


class Detector:
    """Loads a pretrained YOLO model and runs inference on frames.

    `target_classes` defaults to `["person"]` (this project only ever needs
    to find players). Pass `target_classes=None` explicitly to keep every
    detected COCO class instead.

    Usage:
        detector = Detector()
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        target_classes=_DEFAULT_TARGET_CLASSES,
        device: str | None = None,
    ):
        if target_classes is _DEFAULT_TARGET_CLASSES:
            target_classes = ["person"]
        self.confidence_threshold = confidence_threshold or settings.yolo_confidence
        self.target_classes = set(target_classes) if target_classes else None
        self.device = device or settings.yolo_device
        self.model = self._load_model(model_path or settings.yolo_model)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError("ultralytics is not installed. Run: pip install ultralytics") from e

        try:
            logger.info("Loading YOLO model: %s (device=%s)", model_path, self.device)
            return YOLO(model_path)
        except Exception as e:  # ultralytics raises broad exceptions on bad weights/URLs
            raise ModelLoadError(f"Failed to load YOLO model '{model_path}': {e}") from e

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO inference on one BGR frame. Returns filtered detections."""
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names

        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, str(class_id))
            if self.target_classes is not None and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections
