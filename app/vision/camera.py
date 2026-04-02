"""
Camera capture — Week 1.

Wraps cv2.VideoCapture so nothing downstream (detector, game, scripts)
touches OpenCV's raw capture API directly. `cap.read()` returns `(ret,
frame)`: `ret=False` means end-of-stream for a file, or camera failure for
a live device — this class turns that into either a clean frame or a
raised `CameraError`, never a silent `None`.

Diagnostics (FPS, per-frame latency) are computed here, once, so every
consumer (the Week 1 diagnostics overlay, the Week 2 detector loop, the
Week 6 API) reads the same numbers instead of recomputing them differently.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

_FPS_WINDOW = 30  # frames averaged for the rolling FPS figure


class CameraError(Exception):
    """Raised when a camera can't be opened or stops producing frames."""


@dataclass
class FrameDiagnostics:
    """Per-frame timing info, recomputed every `Camera.read()` call."""

    frame_index: int
    timestamp: float  # time.time() when this frame was read
    fps: float  # rolling average over the last _FPS_WINDOW frames
    latency_ms: float  # how long this single cap.read() call took
    resolution: tuple[int, int]  # (width, height) actually negotiated with the device


class Camera:
    """Opens a webcam (or video file) and yields frames with diagnostics.

    Usage:
        with Camera() as cam:
            for frame, diag in cam.frames():
                ...  # frame is a (H, W, 3) BGR uint8 array
    """

    def __init__(
        self,
        index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        target_fps: int | None = None,
    ):
        self.index = index if index is not None else settings.camera_index
        self.width = width or settings.camera_width
        self.height = height or settings.camera_height
        self.target_fps = target_fps or settings.target_fps

        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._frame_index = 0
        self.resolution: tuple[int, int] = (self.width, self.height)

    # ---- lifecycle -----------------------------------------------------

    def open(self) -> None:
        logger.info(
            "Opening camera index=%s requested=%dx%d@%dfps",
            self.index,
            self.width,
            self.height,
            self.target_fps,
        )
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.index}. Check that it's connected "
                f"and not in use by another application."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w <= 0 or actual_h <= 0:
            cap.release()
            raise CameraError(f"Camera index {self.index} reported invalid resolution.")

        self.resolution = (actual_w, actual_h)
        self._cap = cap
        logger.info("Camera opened: negotiated resolution %dx%d", actual_w, actual_h)

    def release(self) -> None:
        if self._writer is not None:
            self.stop_recording()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released (%d frames read)", self._frame_index)

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    # ---- reading ---------------------------------------------------------

    def read(self) -> tuple[np.ndarray, FrameDiagnostics]:
        """Read one frame. Raises CameraError on failure (never returns None)."""
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() or use a `with Camera() as cam:` block.")

        t0 = time.perf_counter()
        ret, frame = self._cap.read()
        latency_ms = (time.perf_counter() - t0) * 1000

        if not ret or frame is None:
            raise CameraError(f"Camera index {self.index} stopped returning frames (disconnected?).")

        now = time.time()
        self._frame_times.append(now)
        fps = self._rolling_fps()

        diag = FrameDiagnostics(
            frame_index=self._frame_index,
            timestamp=now,
            fps=fps,
            latency_ms=round(latency_ms, 2),
            resolution=self.resolution,
        )
        self._frame_index += 1

        if self._writer is not None:
            self._writer.write(frame)

        return frame, diag

    def frames(self):
        """Generator form of read() — stops cleanly (StopIteration) rather than
        raising once the caller decides to break, but still raises CameraError
        for genuine failures encountered while reading."""
        while True:
            yield self.read()

    def _rolling_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return round((len(self._frame_times) - 1) / span, 1)

    # ---- recording / screenshots ------------------------------------------

    def start_recording(self, path: str | Path) -> None:
        if self._cap is None:
            raise CameraError("Cannot record before the camera is open.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.target_fps, self.resolution)
        if not writer.isOpened():
            raise CameraError(f"Could not open VideoWriter for '{path}'.")
        self._writer = writer
        logger.info("Recording started: %s", path)

    def stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Recording stopped.")

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @staticmethod
    def screenshot(frame: np.ndarray, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        logger.info("Screenshot saved: %s", path)
        return path
