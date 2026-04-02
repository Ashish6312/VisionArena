"""Liveness (`/health`) and structured diagnostics (`/metrics`) — V2.0 Part H
added `/metrics`. Deliberately lightweight (no Prometheus, no metrics
library) — this project's whole diagnostics surface is a handful of
numbers already computed by `GameRunner`/`CVWorker`/`CameraWorker`; a
metrics framework would be solving a problem this project doesn't have."""

from __future__ import annotations

from fastapi import APIRouter

from ... import __version__
from ..game_runner import runner

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "visionstrike",
        "version": __version__,
        "game_session_running": runner.running,
    }


@router.get("/metrics")
def metrics() -> dict:
    """Structured JSON snapshot of the real-time performance numbers —
    every value here is read live from `GameRunner`/`CameraWorker`/
    `CVWorker`, never fabricated. `None` means "not applicable right now"
    (e.g. no session running, or a camera-less session), not zero."""
    status = runner.state()
    cv = status["cv"]

    return {
        "camera": {
            "fps": cv.get("camera_fps"),
            "status": cv.get("status", "disabled"),
        },
        "cv": {
            "fps": cv.get("cv_fps"),
            # Only total pipeline latency is measured in production (see
            # CVWorker.latency_ms) — per-stage YOLO/tracking/MediaPipe
            # breakdown is intentionally NOT tracked here; adding that
            # instrumentation to a hot path for a number nothing in this
            # project consumes wasn't worth it. That breakdown does exist,
            # measured for real, in scripts/benchmark.py.
            "latency_ms": cv.get("cv_latency_ms"),
        },
        "game": {
            "fps": cv.get("game_fps"),
        },
        "vision_state_age_seconds": cv.get("vision_age_seconds"),
        "worker_status": {
            "camera_worker_alive": cv.get("camera_worker_alive"),
            "cv_worker_alive": cv.get("cv_worker_alive"),
        },
        "session": {
            "running": status["running"],
            "session_id": status["session_id"],
        },
    }
