"""
CameraWorker — Phase 2 (V2): camera capture on its own thread.

WHY CAMERA CAPTURE IS A SEPARATE THREAD FROM CV INFERENCE (not just "read
a frame right before running YOLO"):

A webcam's OS/driver-level buffer keeps filling while nobody calls
`cap.read()`. YOLO inference takes 150-370ms on this project's hardware
(see docs/cv_pipeline.md); if the SAME thread that's blocked inside YOLO
is also the one responsible for draining that driver buffer, the buffer
backs up, and the next `read()` returns a frame that's already stale by
the time CV even starts processing it — the opposite of the low-latency
goal. A dedicated thread whose only job is `camera.read()` in a tight loop
keeps the driver buffer drained and `LatestSlot` always holding the
frame closest to "right now", independent of how slow CV inference is.

The loop needs no `sleep()`: `Camera.read()` itself blocks until the
device produces a frame (~33ms at 30 FPS), which is what paces this
thread — an artificial delay would either fight that natural pacing or,
worse, be tuned to a frame rate that doesn't match the actual device.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .camera import Camera, CameraError, FrameDiagnostics
from .metrics import RollingRate
from .sync import LatestSlot

logger = logging.getLogger(__name__)

FrameAndDiagnostics = tuple[np.ndarray, FrameDiagnostics]

# V2 Phase 3: matches CVWorker.DEFAULT_STOP_TIMEOUT_S — camera.open() can
# itself take a moment (driver initialization), so this worker gets the same
# realistic margin. See docs/cv_pipeline.md Technical Challenges.
DEFAULT_STOP_TIMEOUT_S = 5.0


class CameraWorker:
    """Usage:
    worker = CameraWorker(camera)   # camera not yet open()ed
    worker.start()
    ...
    latest = worker.slot.get()      # (frame, diagnostics) | None
    ...
    worker.stop()
    """

    def __init__(self, camera: Camera):
        self.camera = camera
        self.slot: LatestSlot[FrameAndDiagnostics] = LatestSlot()
        self.rate = RollingRate()
        self.error: str | None = None
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="visionstrike-camera")
        self._thread.start()

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # camera.read() has no cancellation hook — if the driver
                # itself is hung, this thread can outlive `timeout`. It's a
                # daemon thread, so it won't block process exit; logged
                # rather than silently accepted. See docs/cv_pipeline.md
                # Technical Challenges. Deliberately NOT clearing
                # `self._thread` here — `is_alive` must keep reporting the
                # truth (still running) rather than lying "stopped" just
                # because we stopped waiting.
                logger.warning(
                    "Camera worker did not stop within %.1fs (daemon thread, process exit is safe).", timeout
                )
            else:
                self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.camera.open()
        except CameraError as e:
            self.error = str(e)
            logger.error("Camera worker could not open camera: %s", e)
            return

        try:
            while not self._stop_flag.is_set():
                try:
                    frame, diag = self.camera.read()
                except CameraError as e:
                    self.error = str(e)
                    logger.error("Camera worker stopping (read failed): %s", e)
                    return
                self.slot.put((frame, diag))
                self.rate.tick()
        finally:
            self.camera.release()
