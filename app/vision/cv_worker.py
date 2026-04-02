"""
CVWorker — Phase 2 (V2): the full detect -> track -> gesture pipeline on
its own thread, decoupled from both the camera thread and the game loop.

CONSUMES: `CameraWorker.slot` (blocks efficiently via `wait_for_update`
until a genuinely new frame exists — never busy-polls, never processes
the same frame twice, never falls behind into a backlog: if a new frame
arrives while this worker is still mid-YOLO on the previous one, the next
`get()` simply returns the newer one, not a queue of stale ones).

PUBLISHES: `self.state_slot` — a `VisionState` per completed pass. The
game loop (`GameRunner`, `scripts/run_full_system.py`) calls
`state_slot.get()`, which never blocks and never waits for CV — it either
has a fresh-enough `VisionState` or it doesn't, and either way the game
keeps ticking.

WHY WALL-CLOCK TIME FOR TRAJECTORY TRACKING (a deliberate change from
before Phase 2): `TrajectoryStore.update()` previously received
`engine.state.elapsed_seconds` — a value owned by `GameEngine`, itself
only advanced by the game loop's own thread. Now that CV runs on an
independent thread, reading that value here would reintroduce exactly the
cross-thread coupling this phase removes. `time.time()` is monotonic
enough for velocity/distance math (only the DIFFERENCE between samples
matters, not the absolute epoch), and makes `CVWorker` fully self-
contained — it needs nothing from `GameEngine` at all.

FAILURE GRANULARITY: an exception during detect/track/gesture stops this
worker entirely (logs the full traceback, sets `self.error`, thread
exits) rather than skipping the one bad frame and silently continuing.
This is a deliberate choice, not the only valid one — "skip and retry"
would be more resilient to a single transient bad frame, but "stop and
surface the error" is simpler to reason about and matches this project's
rule against hiding worker failures behind a broad except that just
carries on. The game itself is unaffected either way — see
`GameRunner`'s handling of `cv_worker.error`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from ..config import settings
from ..gestures.classifier import GestureRecognizer
from ..gestures.hands import GestureModelError
from .camera_worker import CameraWorker
from .detector import Detector, ModelLoadError
from .metrics import RollingAverage, RollingRate
from .state import VisionState
from .sync import LatestSlot
from .tracker import Tracker
from .trajectory import TrajectoryStore

logger = logging.getLogger(__name__)

FRAME_WAIT_TIMEOUT_S = 0.5  # how often this worker wakes to re-check its own stop flag

# V2 Phase 3: was 2.0s, raised after real-hardware testing showed cold model
# loading (YOLO + 2 MediaPipe models) can itself take 2-3s — a stop() called
# during that window couldn't possibly succeed within the old timeout, since
# construction happens before the stop-flag-checking loop even starts (see
# `_run`). This does not make startup interruptible; it just stops the
# timeout from being shorter than realistic startup time. See
# docs/cv_pipeline.md Technical Challenges for the residual (documented, not
# "fixed") risk on a cold first run where YOLO weights are still downloading.
DEFAULT_STOP_TIMEOUT_S = 5.0


class CVWorker:
    """Usage:
    worker = CVWorker(camera_worker)
    worker.start()
    ...
    state = worker.state_slot.get()   # VisionState | None, never blocks
    ...
    worker.stop()
    """

    def __init__(
        self,
        camera_worker: CameraWorker,
        detector_factory: Callable[[], Detector] = Detector,
        tracker_factory: Callable[[], Tracker] | None = None,
        gestures_factory: Callable[[], GestureRecognizer] = GestureRecognizer,
    ):
        self._camera_worker = camera_worker
        self._detector_factory = detector_factory
        self._tracker_factory = tracker_factory or (lambda: Tracker(frame_rate=settings.target_fps))
        self._gestures_factory = gestures_factory

        self.state_slot: LatestSlot[VisionState] = LatestSlot()
        self.track_store = TrajectoryStore()  # session-long, queried by GameRunner._finish
        self.rate = RollingRate()
        self.latency_ms = RollingAverage()
        self.error: str | None = None

        self._primary_track_id: int | None = None
        self._frame_id = 0
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="visionstrike-cv")
        self._thread.start()

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Model construction (Detector()/Tracker()/GestureRecognizer())
                # happens before the stop-flag-checking loop even starts, so
                # it can't be interrupted — stop() called during that window
                # can legitimately take longer than `timeout`. Deliberately
                # NOT clearing `self._thread` here — `is_alive` must keep
                # reporting the truth rather than lying "stopped". See
                # docs/cv_pipeline.md Technical Challenges.
                logger.warning(
                    "CV worker did not stop within %.1fs (daemon thread, process exit is safe).", timeout
                )
            else:
                self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            detector = self._detector_factory()
            tracker = self._tracker_factory()
            gestures = self._gestures_factory()
        except (ModelLoadError, GestureModelError) as e:
            self.error = str(e)
            logger.error("CV worker could not start (model load failed): %s", e)
            return

        try:
            while not self._stop_flag.is_set():
                if not self._camera_worker.slot.wait_for_update(timeout=FRAME_WAIT_TIMEOUT_S):
                    continue  # no new frame yet — normal, just recheck the stop flag
                latest = self._camera_worker.slot.get()
                if latest is None:
                    continue
                frame, diag = latest
                self._process_frame(frame, diag.timestamp, detector, tracker, gestures)
        except Exception as e:  # noqa: BLE001 — logged with traceback, never hidden (see module docstring)
            self.error = f"CV pipeline error: {e}"
            logger.exception("CV worker stopping: unexpected error")
        finally:
            gestures.close()

    def _process_frame(self, frame, frame_timestamp: float, detector, tracker, gestures) -> None:
        t0 = time.perf_counter()

        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        self.track_store.update(tracked, timestamp=time.time())
        gesture_results = gestures.process(frame)

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_ms.add(latency_ms)
        self.rate.tick()

        if self._primary_track_id is None and tracked:
            self._primary_track_id = tracked[0].track_id  # first ID this session ever sees, then fixed

        self._frame_id += 1
        self.state_slot.put(
            VisionState(
                frame_id=self._frame_id,
                timestamp=time.time(),
                frame_timestamp=frame_timestamp,
                detections=detections,
                tracked_objects=tracked,
                primary_track_id=self._primary_track_id,
                gestures=gesture_results,
                game_events=gestures.to_game_events(gesture_results),
                processing_latency_ms=round(latency_ms, 1),
            )
        )
