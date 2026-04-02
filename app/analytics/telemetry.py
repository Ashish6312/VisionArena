"""
Session telemetry recording — V2.0 Part G.

Structured per-tick records, not raw video — one JSON object per line
(JSONL), so a session can be inspected, diffed, or replayed
(`scripts/replay_session.py`) without ever storing a video file. This is
deliberately NOT a pixel-perfect input log: it captures derived state
(position, velocity, health, score, gestures, events) at each tick, not
every keystroke/frame needed to bit-for-bit re-simulate the original
physics — see `scripts/replay_session.py`'s docstring for what "replay"
actually means here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..game.engine import GameEngine

logger = logging.getLogger(__name__)


class TelemetryRecorder:
    """Usage (once per session):
    recorder = TelemetryRecorder(path)
    recorder.start()
    ...
    recorder.record(engine, gesture="SHOOT", events=["SHOOT"])   # once per tick
    ...
    recorder.stop()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._prev_pos: tuple[float, float] | None = None
        self._prev_time: float | None = None
        self.frame_count = 0

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._prev_pos = None
        self._prev_time = None
        self.frame_count = 0

    def record(self, engine: GameEngine, gesture: str | None = None, events: list[str] | None = None) -> None:
        if self._file is None:
            return

        now = time.time()
        vx = vy = 0.0
        if self._prev_pos is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                vx = (engine.player.x - self._prev_pos[0]) / dt
                vy = (engine.player.y - self._prev_pos[1]) / dt
        self._prev_pos = (engine.player.x, engine.player.y)
        self._prev_time = now

        frame = {
            "timestamp": round(now, 3),
            "elapsed_seconds": round(engine.state.elapsed_seconds, 3),
            "player": {
                "x": round(engine.player.x, 1),
                "y": round(engine.player.y, 1),
                "vx": round(vx, 1),
                "vy": round(vy, 1),
                "health": engine.player.health,
            },
            "score": engine.state.score,
            "shots_fired": engine.state.shots_fired,
            "shots_hit": engine.state.shots_hit,
            "gesture": gesture,
            "events": events or [],
            "enemies": [
                {
                    "id": e.enemy_id,
                    "x": round(e.x, 1),
                    "y": round(e.y, 1),
                    "state": e.state,
                    "health": e.health,
                }
                for e in engine.enemies
                if e.is_alive
            ],
        }
        self._file.write(json.dumps(frame) + "\n")
        self.frame_count += 1

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info("Telemetry recorded: %s (%d frames)", self.path, self.frame_count)


def load_frames(path: str | Path) -> list[dict]:
    """Reads a recorded .jsonl file back into a list of frame dicts —
    used by `scripts/replay_session.py` and by tests."""
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames
