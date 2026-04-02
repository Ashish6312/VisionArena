#!/usr/bin/env python
"""The full VisionStrike pipeline in one process: webcam -> YOLO -> ByteTrack
-> MediaPipe gestures -> GameEvent -> GameEngine -> AI opponent -> Pygame
window. This is the "final demo" the project was built towards — no
FastAPI/WebSocket layer in between, camera and game share one process.

(The FastAPI path in scripts/run_server.py exists for a *different* demo:
a browser dashboard watching a session over the network, and doesn't
require a display on the machine running the CV pipeline. Both paths
share the same GameEngine/EnemyController/CameraWorker/CVWorker code —
only who's driving the loop and where it renders differs.)

DECOUPLED SINCE V2 PHASE 2: the render loop below targets 60 FPS and never
blocks on CV — it starts a CameraWorker + CVWorker (each on their own
thread) and reads whatever VisionState they most recently published,
non-blocking, every frame. See app/vision/cv_worker.py and
docs/architecture.md "CV/Game Loop Decoupling".

Usage:
    python scripts/run_full_system.py [--mode camera|keyboard] [--difficulty EASY|MEDIUM|HARD]

    --mode camera     (default) full CV pipeline — needs a real webcam and
                       the MediaPipe model bundles (docs/cv_pipeline.md).
    --mode keyboard    no camera at all — same GameEngine, same GameEvent
                       contract, just fed by keys instead of gestures. This
                       is not a "fake CV" stand-in; it's a genuinely
                       different, fully-supported input source (see
                       app/events.py) — the engine can't tell the two apart.

Controls (in addition to gestures):
    ESC   pause / quit prompt handled by window close
    F3    toggle the CV/performance diagnostics panel (V2 Phase 5)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from app.ai.enemy_controller import EnemyController
from app.config import settings
from app.game.engine import GameEngine
from app.game.state import Difficulty
from app.logging_config import setup_logging
from app.vision.camera import Camera
from app.vision.camera_worker import CameraWorker
from app.vision.cv_worker import CVWorker
from app.vision.metrics import RollingRate
from app.vision.state import classify_cv_status
from scripts.run_game import draw, keyboard_events

logger = logging.getLogger("run_full_system")

METRICS_LOG_INTERVAL_S = 2.0  # how often the CV/game FPS line below is logged


def _debug_snapshot(
    game_fps: RollingRate,
    camera_worker: CameraWorker | None,
    cv_worker: CVWorker | None,
    show_debug: bool,
) -> dict:
    """Every value here is read straight off a real object — see
    scripts/run_game.py::draw's docstring for why that matters."""
    if camera_worker is None or cv_worker is None:
        return {"show": show_debug, "cv_status": "disabled", "game_fps": game_fps.rate}

    vision_state = cv_worker.state_slot.get()
    worker_error = camera_worker.error or cv_worker.error
    status = classify_cv_status(vision_state, worker_error, settings.vision_stale_seconds)
    latest_gesture = vision_state.gestures[-1] if vision_state and vision_state.gestures else None

    return {
        "show": show_debug,
        "cv_status": status,
        "track_id": vision_state.primary_track_id if vision_state else None,
        "gesture": latest_gesture.gesture.value if latest_gesture else None,
        "gesture_confidence": latest_gesture.confidence if latest_gesture else None,
        "game_fps": game_fps.rate,
        "cv_fps": cv_worker.rate.rate,
        "cv_latency_ms": cv_worker.latency_ms.average,
        "vision_age_ms": (vision_state.age_seconds * 1000) if vision_state else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionStrike — full CV + game pipeline")
    parser.add_argument("--mode", choices=["camera", "keyboard"], default="camera", help="Input source")
    parser.add_argument("--no-camera", action="store_true", help="Deprecated alias for --mode keyboard")
    parser.add_argument("--difficulty", default="MEDIUM", choices=["EASY", "MEDIUM", "HARD"])
    args = parser.parse_args()

    setup_logging()
    pygame.init()

    engine = GameEngine(difficulty=Difficulty(args.difficulty))
    controller = EnemyController()
    screen = pygame.display.set_mode((engine.arena.width, engine.arena.height))
    pygame.display.set_caption("VisionStrike — Full System")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    game_fps = RollingRate()
    show_debug = True

    use_camera = args.mode == "camera" and not args.no_camera
    camera_worker: CameraWorker | None = None
    cv_worker: CVWorker | None = None
    if use_camera:
        camera_worker = CameraWorker(Camera())
        cv_worker = CVWorker(camera_worker)
        camera_worker.start()
        cv_worker.start()
        logger.info("CV pipeline starting in the background (camera + YOLO + ByteTrack + gestures).")

    logger.info("VisionStrike started. Difficulty=%s CV=%s", args.difficulty, use_camera)
    last_processed_frame_id = -1
    last_metrics_log = time.perf_counter()
    running = True
    latest_vision_state = None   # kept separately so draw() always has a reference
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            game_fps.tick()
            for pg_event in pygame.event.get():
                if pg_event.type == pygame.QUIT:
                    running = False
                elif pg_event.type == pygame.KEYDOWN and pg_event.key == pygame.K_F3:
                    show_debug = not show_debug

            events = list(keyboard_events())
            if cv_worker is not None:
                # Non-blocking: whatever the CV worker most recently
                # published, or nothing yet — this loop never waits on it.
                vision_state = cv_worker.state_slot.get()
                if vision_state is not None:
                    latest_vision_state = vision_state
                    if vision_state.frame_id != last_processed_frame_id:
                        last_processed_frame_id = vision_state.frame_id
                        events += vision_state.game_events

            engine.apply_events(events)
            actions = controller.decide(engine.enemies, engine.player)
            engine.update(dt, enemy_actions=actions)
            debug = _debug_snapshot(game_fps, camera_worker, cv_worker, show_debug)

            # Pull the latest raw frame for the camera background — non-blocking,
            # same "latest wins" pattern as VisionState.  None until camera opens.
            latest_cam = camera_worker.slot.get() if camera_worker is not None else None
            camera_frame = latest_cam[0] if latest_cam is not None else None

            draw(
                screen, engine, font,
                debug=debug,
                camera_frame=camera_frame,
                vision_state=latest_vision_state,
            )

            if cv_worker is not None and time.perf_counter() - last_metrics_log >= METRICS_LOG_INTERVAL_S:
                last_metrics_log = time.perf_counter()
                logger.info(
                    "GAME FPS: %.1f  CV FPS: %.1f  LATENCY: %.0fms  AGE: %s  CAM FPS: %.1f  STATUS: %s",
                    debug["game_fps"],
                    debug["cv_fps"],
                    debug["cv_latency_ms"],
                    f"{debug['vision_age_ms']:.0f}ms" if debug["vision_age_ms"] is not None else "n/a",
                    camera_worker.rate.rate if camera_worker else 0.0,
                    debug["cv_status"],
                )
    finally:
        if cv_worker is not None:
            cv_worker.stop()
        if camera_worker is not None:
            camera_worker.stop()
        pygame.quit()

    summary = engine.to_dict()["state"]
    logger.info("Session ended: %s", summary)
    print("\nFinal state:", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
