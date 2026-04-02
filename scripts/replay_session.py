#!/usr/bin/env python
"""
Replay a recorded VisionStrike session — V2.0 Part G.

Reads structured per-tick telemetry (app/analytics/telemetry.py, written
by GameRunner during every session — see GameRunner.state()["telemetry_path"])
and plays it back visually, no webcam or live GameEngine required.

NOT a pixel-perfect deterministic replay, stated plainly: telemetry is a
coarse per-tick snapshot (position/health/score/gesture/events), not a
full input log capable of bit-for-bit re-simulating the original physics
(projectile paths, exact collision timing, etc. aren't recorded). This
reconstructs *what happened* — where the player and enemies were, what
they did, how the score changed — for debugging, analytics review, and
portfolio demonstration, not a byte-identical rerun.

Usage:
    python scripts/replay_session.py data/sessions/telemetry/<session_id>.jsonl
    python scripts/replay_session.py <path> --speed 2.0   # 2x playback
    python scripts/replay_session.py <path> --text-only   # no window, just a timeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analytics.telemetry import load_frames

_BG = (18, 20, 26)
_PLAYER_COLOR = (90, 200, 255)
_ENEMY_COLOR = (230, 90, 90)
_ARENA_SIZE = (1280, 720)


def print_timeline(frames: list[dict]) -> None:
    """Text-only fallback (also useful in a headless/CI environment with
    no display) — one line per tick that actually changed something
    worth narrating, not a dump of every single frame."""
    prev_score = None
    prev_health = None
    for frame in frames:
        score = frame["score"]
        health = frame["player"]["health"]
        notes = []
        if frame.get("gesture"):
            notes.append(f"gesture={frame['gesture']}")
        if prev_score is not None and score != prev_score:
            notes.append(f"score {prev_score}->{score}")
        if prev_health is not None and health != prev_health:
            notes.append(f"health {prev_health}->{health}")
        if notes:
            print(f"t={frame['elapsed_seconds']:6.2f}s  " + "  ".join(notes))
        prev_score, prev_health = score, health

    if frames:
        final = frames[-1]
        print("\n--- Final ---")
        print(
            f"Score: {final['score']}  Shots: {final['shots_fired']}  "
            f"Hits: {final['shots_hit']}  Health: {final['player']['health']}"
        )


def replay_visual(frames: list[dict], speed: float) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(_ARENA_SIZE)
    pygame.display.set_caption("VisionStrike Replay")
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    running = True
    last_ts = frames[0]["timestamp"]
    for frame in frames:
        if not running:
            break
        for pg_event in pygame.event.get():
            if pg_event.type == pygame.QUIT:
                running = False

        screen.fill(_BG)
        for enemy in frame.get("enemies", []):
            pos = (int(enemy["x"]), int(enemy["y"]))
            pygame.draw.circle(screen, _ENEMY_COLOR, pos, 16)
            label = font.render(enemy["state"], True, (230, 230, 230))
            screen.blit(label, label.get_rect(center=(pos[0], pos[1] - 28)))

        player = frame["player"]
        pygame.draw.circle(screen, _PLAYER_COLOR, (int(player["x"]), int(player["y"])), 18)

        hud = [
            f"HP: {player['health']}",
            f"Score: {frame['score']}   Shots: {frame['shots_fired']}   Hits: {frame['shots_hit']}",
            f"t = {frame['elapsed_seconds']:.1f}s",
        ]
        if frame.get("gesture"):
            hud.append(f"Gesture: {frame['gesture']}")
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (230, 230, 230)), (10, 10 + i * 22))

        pygame.display.flip()

        wait_s = max(0.0, (frame["timestamp"] - last_ts) / max(speed, 0.01))
        last_ts = frame["timestamp"]
        pygame.time.wait(int(wait_s * 1000))
        clock.tick(60)

    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded VisionStrike session")
    parser.add_argument("path", help="Path to a .jsonl telemetry file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1.0)")
    parser.add_argument(
        "--text-only", action="store_true", help="Print a timeline instead of opening a window"
    )
    args = parser.parse_args()

    frames = load_frames(args.path)
    if not frames:
        print(f"No telemetry frames found in {args.path}")
        return 1

    print(f"Loaded {len(frames)} frames from {args.path}")
    if args.text_only:
        print_timeline(frames)
    else:
        replay_visual(frames, args.speed)
        print_timeline(frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
