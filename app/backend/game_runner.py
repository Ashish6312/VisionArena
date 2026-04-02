"""
Background game loop — Week 6, decoupled from CV in V2 Phase 2.

Runs GameEngine + EnemyController every tick, broadcasting state over the
WebSocket manager. Runs on its own thread rather than as an asyncio task —
game physics is blocking, synchronous work — and hands WebSocket
broadcasts to the server's event loop with `run_coroutine_threadsafe`
because a worker thread has no event loop of its own to send a frame on.

CV/GAME LOOP DECOUPLING (V2 Phase 2): before this phase, `use_camera=True`
ran the whole camera-read -> YOLO -> ByteTrack -> MediaPipe pipeline
INLINE, in this same loop, once per game tick — meaning the effective
"game tick rate" was capped by YOLO's ~150-370ms inference time. Now this
loop starts a `CameraWorker` and a `CVWorker` (each on their own thread,
see app/vision/) and reads whatever `VisionState` they most recently
published via `CVWorker.state_slot.get()` — a call that NEVER blocks and
NEVER waits for CV. If CV falls behind, is slow, or fails outright, this
loop keeps ticking at `settings.game_tick_hz` regardless; see `_cv_status()` for how
that's surfaced (connected/stale/unavailable) rather than silently used.
Full reasoning: docs/architecture.md "CV/Game Loop Decoupling".
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path

from ..ai.enemy_controller import EnemyController
from ..analytics.metrics import zone_time as compute_zone_time
from ..analytics.performance import build_report
from ..analytics.reaction import ReactionTracker
from ..analytics.telemetry import TelemetryRecorder
from ..analytics.trajectory_analysis import summarize_movement
from ..config import settings
from ..database.repository import SessionRepository
from ..events import GameEvent
from ..game.engine import GameEngine
from ..game.state import Difficulty
from ..vision.camera import Camera
from ..vision.camera_worker import CameraWorker
from ..vision.cv_worker import CVWorker
from ..vision.metrics import RollingRate
from ..vision.state import classify_cv_status
from .websocket_manager import manager

logger = logging.getLogger(__name__)

# V2.0 Part I: was a hardcoded module constant, now `settings.game_tick_hz`
# (still 30 by default) — every other rate/threshold in this file already
# came from config; this was the one that didn't.


class GameRunner:
    """Usage:
    runner = GameRunner()
    runner.bind_loop(asyncio.get_running_loop())   # once, at server startup
    session_id = runner.start(Difficulty.MEDIUM, use_camera=True)
    ...
    runner.stop()
    """

    def __init__(self, repository: SessionRepository | None = None):
        self.repository = repository or SessionRepository()
        self.thread: threading.Thread | None = None
        self.running = False
        self.session_id: str | None = None
        self.error: str | None = None
        self.engine: GameEngine | None = None
        self._stop_flag = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gesture_counts: dict[str, int] = {}
        self._zone_log: list[tuple[float, float, float]] = []
        self._reaction_tracker = ReactionTracker()
        self._stimulus_seq: dict[int, int] = {}
        self._camera_worker: CameraWorker | None = None
        self._cv_worker: CVWorker | None = None
        self.game_fps = RollingRate()
        self.telemetry: TelemetryRecorder | None = None  # V2.0 Part G

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---- lifecycle -------------------------------------------------------------

    def start(self, difficulty: Difficulty, use_camera: bool) -> str:
        if self.running:
            raise RuntimeError("A game session is already running. Stop it first.")
        self.session_id = str(uuid.uuid4())
        self.error = None
        self._stop_flag.clear()
        self._gesture_counts = {}
        self._zone_log = []
        self._reaction_tracker = ReactionTracker()
        self._stimulus_seq = {}
        self.game_fps = RollingRate()
        self.engine = GameEngine(difficulty=difficulty)
        self.repository.create(self.session_id, difficulty.value)

        # Constructing these is cheap (no I/O, no model loading yet) — done
        # here, synchronously, on the caller's thread, so `self._cv_worker`
        # is visible to `state()` the instant `start()` returns. The actual
        # camera.open() / YOLO / MediaPipe loading happens inside each
        # worker's own thread once `.start()` is called on it, below.
        self._camera_worker = CameraWorker(Camera()) if use_camera else None
        self._cv_worker = CVWorker(self._camera_worker) if use_camera else None

        # V2.0 Part G: structured per-tick telemetry, not video — see
        # app/analytics/telemetry.py and scripts/replay_session.py.
        self.telemetry = TelemetryRecorder(Path(settings.telemetry_dir) / f"{self.session_id}.jsonl")
        self.telemetry.start()

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="visionstrike-runner")
        self.thread.start()
        return self.session_id

    def stop(self) -> None:
        self._stop_flag.set()

    def state(self) -> dict:
        return {
            "running": self.running,
            "session_id": self.session_id,
            "error": self.error,
            "game": self.engine.to_dict() if self.engine else None,
            "cv": self._cv_status(),
            # V2.0 Part G — where this session's replay telemetry is being
            # written; `python scripts/replay_session.py <this path>` after
            # the session ends.
            "telemetry_path": str(self.telemetry.path) if self.telemetry else None,
        }

    # ---- the loop ---------------------------------------------------------------

    def _run(self) -> None:
        controller = EnemyController()
        last_processed_frame_id = -1
        try:
            if self._camera_worker is not None and self._cv_worker is not None:
                self._camera_worker.start()
                self._cv_worker.start()

            last_tick = time.perf_counter()
            while not self._stop_flag.is_set() and self.engine.state.is_active:
                now = time.perf_counter()
                dt = now - last_tick
                last_tick = now
                self.game_fps.tick()

                events: list[GameEvent] = []
                if self._cv_worker is not None:
                    vision_state = self._cv_worker.state_slot.get()
                    if vision_state is not None and vision_state.frame_id != last_processed_frame_id:
                        last_processed_frame_id = vision_state.frame_id
                        events = vision_state.game_events
                        for event in events:
                            self._reaction_tracker.observe_event(event)
                        for gesture in vision_state.gestures:
                            self._gesture_counts[gesture.gesture.value] = (
                                self._gesture_counts.get(gesture.gesture.value, 0) + 1
                            )
                        for event in events:
                            self._emit({"type": "gesture", **event.to_dict()})

                self.engine.apply_events(events)
                actions = controller.decide(self.engine.enemies, self.engine.player)
                self._record_attack_stimuli(actions)
                self.engine.update(dt, enemy_actions=actions)
                self._zone_log.append(
                    (self.engine.state.elapsed_seconds, self.engine.player.x, self.engine.player.y)
                )
                if self.telemetry is not None:
                    latest_gesture = events[-1].type.value if events else None
                    self.telemetry.record(
                        self.engine, gesture=latest_gesture, events=[e.type.value for e in events]
                    )

                self._emit({"type": "game_state", "cv": self._cv_status(), **self.engine.to_dict()})
                budget = (1.0 / settings.game_tick_hz) - (time.perf_counter() - now)
                if budget > 0:
                    time.sleep(budget)
        except Exception as e:  # last-resort guard — the API must never crash the process
            self.error = f"Unexpected error: {e}"
            logger.exception("Unexpected game session error")
            self._emit({"type": "error", "message": self.error})
        finally:
            # `running` must flip to False no matter what happens during
            # cleanup — a broadcast failing (e.g. the event loop it was
            # bound to is already gone) must never leave the singleton
            # runner permanently stuck "running" (see docs/architecture.md
            # Week 6 decision log — the same rule now also covers the CV
            # workers below: their .stop() calls are best-effort, never
            # allowed to prevent `running` from settling to False).
            try:
                self._finish()
            except Exception:
                logger.exception("Error finishing session %s", self.session_id)
            if self.telemetry is not None:
                try:
                    self.telemetry.stop()
                except Exception:
                    logger.exception("Error closing telemetry for session %s", self.session_id)
            if self._cv_worker is not None:
                self._cv_worker.stop()
            if self._camera_worker is not None:
                self._camera_worker.stop()
            self.running = False

    def _record_attack_stimuli(self, actions: dict[int, str]) -> None:
        """A real stimulus, not a fabricated one: `enemy.state` still holds
        LAST tick's FSM state here (GameEngine.update, called right after
        this, is what overwrites it from `actions`) — so comparing the two
        catches the exact moment an enemy's decided action transitions into
        ATTACK, using the same FSM decision the game already acts on.

        Each stimulus gets a unique ID (V2.0 Part C) — "enemy_{id}_attack_{n}" —
        so simultaneous attacks from different enemies are tracked as
        distinct, independently-expirable stimuli, not pooled into one
        shared FIFO bucket. See app/analytics/reaction.py for how a
        response later resolves which one it closes.
        """
        now = time.time()
        for enemy in self.engine.enemies:
            if actions.get(enemy.enemy_id) == "ATTACK" and enemy.state != "ATTACK":
                seq = self._stimulus_seq.get(enemy.enemy_id, 0) + 1
                self._stimulus_seq[enemy.enemy_id] = seq
                stimulus_id = f"enemy_{enemy.enemy_id}_attack_{seq:03d}"
                self._reaction_tracker.record_stimulus(stimulus_id, now)
        self._reaction_tracker.expire_stale(now)

    def _cv_status(self) -> dict:
        """Never lets the caller mistake old data for current: `status` is
        explicitly one of disabled/connecting/connected/stale/unavailable,
        not just "here's whatever we last saw". `game_fps` is included even
        when CV is disabled — the game loop's own rate is real, measured
        data independent of whether a camera is attached at all; a
        `/metrics` caller checking "is the game responsive" in keyboard
        mode must not see it hidden behind "no CV" (a real bug caught by
        tests/test_api.py::test_metrics_never_fabricates_values_when_cv_disabled)."""
        if self._cv_worker is None or self._camera_worker is None:
            return {"enabled": False, "status": "disabled", "game_fps": self.game_fps.rate}

        vision_state = self._cv_worker.state_slot.get()
        worker_error = self._camera_worker.error or self._cv_worker.error
        status = classify_cv_status(vision_state, worker_error, settings.vision_stale_seconds)

        return {
            "enabled": True,
            "status": status,
            "error": worker_error,
            "camera_fps": self._camera_worker.rate.rate,
            "cv_fps": self._cv_worker.rate.rate,
            "cv_latency_ms": self._cv_worker.latency_ms.average,
            "vision_age_seconds": round(vision_state.age_seconds, 3) if vision_state else None,
            "primary_track_id": vision_state.primary_track_id if vision_state else None,
            "game_fps": self.game_fps.rate,
            # `runner.running == False` means the GAME LOOP stopped — it does
            # NOT guarantee the workers have fully torn down (model loading
            # is uninterruptible, see cv_worker.py). A caller that needs to
            # know "is everything actually done" (a graceful server
            # shutdown, a test) should check these, not just `running`.
            "camera_worker_alive": self._camera_worker.is_alive,
            "cv_worker_alive": self._cv_worker.is_alive,
        }

    def _finish(self) -> None:
        if self.engine is None or self.session_id is None:
            return
        # Single-player: the physical track, if any camera track exists at
        # all, is whichever ID the tracker assigned first (CVWorker fixes
        # this the same way once and never changes it — see cv_worker.py).
        track = None
        if self._cv_worker is not None:
            track = next(iter(self._cv_worker.track_store.all_tracks().values()), None)

        report = build_report(
            session_id=self.session_id,
            state=self.engine.state,
            damage_taken=self.engine.player.max_health - self.engine.player.health,
            movement=summarize_movement(track),
            reaction_times=self._reaction_tracker.reaction_times,
            gesture_counts=self._gesture_counts,
            zone_time=compute_zone_time(self.engine.arena, self._zone_log),
        )
        self.repository.finish(self.session_id, report)
        self._emit({"type": "session_summary", **report.to_dict()})

    def _emit(self, message: dict) -> None:
        if self._loop is None:
            return
        coro = manager.broadcast(message)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # The bound event loop is already closed (e.g. the server is
            # shutting down mid-broadcast) — dropping the message is
            # correct here, there's no one left to deliver it to.
            # run_coroutine_threadsafe raised before scheduling `coro`, so
            # it was never awaited — close it explicitly or it leaks as an
            # unawaited-coroutine RuntimeWarning at GC time.
            coro.close()
            logger.debug("Dropped broadcast: event loop is closed.")


# Single shared instance — every route module imports this one, the same
# way VisionArena's SessionManager was a module-level singleton main.py's
# routes all shared.
runner = GameRunner()
