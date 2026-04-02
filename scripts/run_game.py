#!/usr/bin/env python
"""Week 5 standalone demo: VisionStrike, keyboard-controlled. Gameplay
feedback (hit markers, damage flash, aim line, enemy state labels) added
in V2.0 Part F.

Usage:
    python scripts/run_game.py

Controls:
    Arrows / WASD   move
    SPACE           shoot
    SHIFT           shield
    CTRL            crouch
    ESC             pause
    F3              toggle diagnostics panel (CV shows DISABLED here — no
                    camera in this script; see run_full_system.py for the
                    real CV-connected panel)

Keyboard presses become the exact same `GameEvent` objects a gesture would
produce (see app/events.py) — the engine below has no idea which one it's
getting, which is the point: Week 4's gesture recognizer is a drop-in
alternative input source, not a special case the engine has to know about.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import numpy as np

if TYPE_CHECKING:
    from app.vision.state import VisionState

import pygame

from app.events import GameEvent, GameEventType
from app.game.arena import ZoneKind
from app.game.engine import GameEngine
from app.game.state import GameStatus
from app.logging_config import setup_logging
from app.vision.metrics import RollingRate

logger = logging.getLogger("run_game")

_KEYMAP = {
    pygame.K_LEFT: GameEventType.MOVE_LEFT,
    pygame.K_a: GameEventType.MOVE_LEFT,
    pygame.K_RIGHT: GameEventType.MOVE_RIGHT,
    pygame.K_d: GameEventType.MOVE_RIGHT,
    pygame.K_UP: GameEventType.MOVE_FORWARD,
    pygame.K_w: GameEventType.MOVE_FORWARD,
    pygame.K_DOWN: GameEventType.MOVE_BACKWARD,
    pygame.K_s: GameEventType.MOVE_BACKWARD,
    pygame.K_SPACE: GameEventType.SHOOT,
    pygame.K_LSHIFT: GameEventType.SHIELD,
    pygame.K_RSHIFT: GameEventType.SHIELD,
    pygame.K_LCTRL: GameEventType.CROUCH,
    pygame.K_RCTRL: GameEventType.CROUCH,
    pygame.K_ESCAPE: GameEventType.PAUSE,
}

_ZONE_COLOR = {ZoneKind.SAFE: (40, 90, 50), ZoneKind.DANGER: (90, 40, 40)}
_BG = (18, 20, 26)
_PLAYER_COLOR = (90, 200, 255)
_ENEMY_COLOR = (230, 90, 90)
_PROJECTILE_COLOR = (255, 220, 90)
_AIM_LINE_COLOR = (255, 255, 255)
_TARGET_RING_COLOR = (255, 240, 90)
_HIT_MARKER_COLOR = (255, 255, 255)
_KILL_MARKER_COLOR = (255, 210, 60)
_DAMAGE_FLASH_COLOR = (200, 40, 40)
_DAMAGE_FLASH_DURATION_S = 0.25
_AIM_LINE_LENGTH_PX = 70.0

_CV_STATUS_COLOR = {
    "connected": (110, 230, 140),
    "connecting": (230, 200, 90),
    "stale": (230, 200, 90),
    "unavailable": (230, 90, 90),
    "disabled": (140, 140, 150),
}


def keyboard_events() -> list[GameEvent]:
    pressed = pygame.key.get_pressed()
    return [
        GameEvent.now(event_type, source="keyboard") for key, event_type in _KEYMAP.items() if pressed[key]
    ]


def draw(
    screen: pygame.Surface,
    engine: GameEngine,
    font: pygame.font.Font,
    *,
    debug: dict | None = None,
    camera_frame: np.ndarray | None = None,
    vision_state: VisionState | None = None,
) -> None:
    """`debug`, when provided, is a plain dict of already-computed values —
    this function has no idea what a CVWorker or RollingRate is, it just
    renders whatever numbers the caller measured (see `_debug_snapshot` in
    run_full_system.py / run_game.py's own `main()`). Every value shown is
    read from the real system; there is no code path that invents one.

    Expected keys (all optional, missing/None renders as "--"):
        show (bool), cv_status (str), track_id (int), gesture (str),
        gesture_confidence (float), game_fps (float), cv_fps (float),
        cv_latency_ms (float), vision_age_ms (float)

    camera_frame: optional latest BGR numpy frame from CameraWorker — when
        provided the live webcam feed is rendered as the background instead of
        the solid colour fill.  vision_state supplies the YOLO boxes drawn on
        top of that feed.
    """
    if camera_frame is not None:
        from app.game.camera_bg import draw_camera_background  # lazy import — only used in camera mode
        draw_camera_background(
            screen, camera_frame, vision_state,
            arena_w=engine.arena.width,
            arena_h=engine.arena.height,
        )
    else:
        screen.fill(_BG)

    for zone in engine.arena.zones:
        rect = pygame.Rect(zone.x1, zone.y1, zone.x2 - zone.x1, zone.y2 - zone.y1)
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        surface.fill((*_ZONE_COLOR[zone.kind], 60))
        screen.blit(surface, rect.topleft)
        pygame.draw.rect(screen, _ZONE_COLOR[zone.kind], rect, 2)

    for enemy in engine.enemies:
        if not enemy.is_alive:
            continue
        pos = (int(enemy.x), int(enemy.y))
        pygame.draw.circle(screen, _ENEMY_COLOR, pos, int(enemy.radius))
        if enemy.enemy_id == engine.player.target_enemy_id:  # V2.0 Part D/F — aim-target highlight
            pygame.draw.circle(screen, _TARGET_RING_COLOR, pos, int(enemy.radius) + 6, width=2)
        state_label = font.render(enemy.state, True, (235, 235, 235))  # V2.0 Part F — enemy state indicator
        screen.blit(state_label, state_label.get_rect(center=(pos[0], pos[1] - int(enemy.radius) - 12)))

    for p in engine.projectiles:
        pygame.draw.circle(screen, _PROJECTILE_COLOR, (int(p.x), int(p.y)), int(p.radius))

    _draw_hit_markers(screen, engine)  # V2.0 Part F

    player_color = (120, 255, 160) if engine.player.shielded else _PLAYER_COLOR
    player_pos = (int(engine.player.x), int(engine.player.y))
    pygame.draw.circle(screen, player_color, player_pos, int(engine.player.radius))
    if engine.player.aiming:  # V2.0 Part D/F — aim direction indicator
        vx, vy = engine.player.last_aim_vector
        end = (int(player_pos[0] + vx * _AIM_LINE_LENGTH_PX), int(player_pos[1] + vy * _AIM_LINE_LENGTH_PX))
        pygame.draw.line(screen, _AIM_LINE_COLOR, player_pos, end, width=2)

    _draw_damage_flash(screen, engine)  # V2.0 Part F

    hud = engine.state.to_dict()
    lines = [
        f"HP: {engine.player.health}/{engine.player.max_health}",
        f"Score: {hud['score']}   Kills: {hud['kills']}",
        f"Accuracy: {hud['accuracy']}%   Time: {hud['elapsed_seconds']:.0f}s",
        f"Status: {hud['status']}",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (230, 230, 230)), (10, 10 + i * 22))

    if engine.state.status in (GameStatus.WON, GameStatus.LOST):
        big = pygame.font.SysFont(None, 64)
        msg = "VICTORY" if engine.state.status == GameStatus.WON else "GAME OVER"
        text = big.render(msg, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(engine.arena.width // 2, engine.arena.height // 2)))

    if debug and debug.get("show", True):
        _draw_debug_panel(screen, font, engine, debug)

    pygame.display.flip()


def _draw_hit_markers(screen: pygame.Surface, engine: GameEngine) -> None:
    """A short-lived ring at each recent projectile impact
    (`GameEngine.recent_hits`, see app/game/engine.py) — gold for a kill,
    white for a non-lethal hit. Purely visual; the engine already expires
    these on its own after HIT_MARKER_LIFETIME_SECONDS."""
    for hit in engine.recent_hits:
        color = _KILL_MARKER_COLOR if hit["killed"] else _HIT_MARKER_COLOR
        pygame.draw.circle(screen, color, (int(hit["x"]), int(hit["y"])), 14, width=2)


def _draw_damage_flash(screen: pygame.Surface, engine: GameEngine) -> None:
    """A brief red vignette right after the player takes damage
    (`Player.last_damage_time`) — fades out over `_DAMAGE_FLASH_DURATION_S`."""
    if engine.player.last_damage_time <= 0.0:
        return
    age = time.time() - engine.player.last_damage_time
    if age >= _DAMAGE_FLASH_DURATION_S:
        return
    alpha = int(90 * (1.0 - age / _DAMAGE_FLASH_DURATION_S))
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((*_DAMAGE_FLASH_COLOR, alpha))
    screen.blit(overlay, (0, 0))


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    return f"{value:.{digits}f}{suffix}" if isinstance(value, int | float) else "--"


def _draw_debug_panel(
    screen: pygame.Surface, font: pygame.font.Font, engine: GameEngine, debug: dict
) -> None:
    """F3 diagnostics, grouped as VISION (CV connection health, tracking,
    gesture, aim) and PERFORMANCE (the real-time game/CV FPS + latency
    split that V2 Phase 2's decoupled architecture makes worth showing at
    all)."""
    status = debug.get("cv_status") or "disabled"
    status_color = _CV_STATUS_COLOR.get(status, (200, 200, 200))
    aim_target = engine.player.target_enemy_id
    header_color = (150, 160, 180)
    body_color = (210, 212, 220)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("VISION", header_color),
        (f"CV: {status.upper()}", status_color),
        (
            f"Track ID: {debug.get('track_id') if debug.get('track_id') is not None else '--'}",
            body_color,
        ),
        (
            f"Gesture: {debug.get('gesture') or '--'}"
            + (f" ({_fmt(debug.get('gesture_confidence'), digits=2)})" if debug.get("gesture") else ""),
            body_color,
        ),
        (f"Aim target: enemy {aim_target}" if aim_target is not None else "Aim target: --", body_color),
        ("", (0, 0, 0)),
        ("PERFORMANCE", header_color),
        (f"Game FPS: {_fmt(debug.get('game_fps'))}", (200, 220, 255)),
        (f"CV FPS: {_fmt(debug.get('cv_fps'))}", (200, 220, 255)),
        (f"CV latency: {_fmt(debug.get('cv_latency_ms'), 'ms', 0)}", (200, 220, 255)),
        (f"Vision age: {_fmt(debug.get('vision_age_ms'), 'ms', 0)}", (200, 220, 255)),
    ]

    panel_width = 220
    panel_height = 14 + sum(20 for _ in lines)
    x = screen.get_width() - panel_width - 10
    y = 10
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((10, 12, 16, 190))
    screen.blit(panel, (x, y))

    ty = y + 8
    for text, color in lines:
        if text:
            screen.blit(font.render(text, True, color), (x + 10, ty))
        ty += 20


def main() -> int:
    setup_logging()
    pygame.init()
    engine = GameEngine()
    screen = pygame.display.set_mode((engine.arena.width, engine.arena.height))
    pygame.display.set_caption("VisionStrike")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    logger.info("VisionStrike started (keyboard mode). Difficulty=%s", engine.state.difficulty.value)
    game_fps = RollingRate()
    show_debug = True
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        game_fps.tick()
        for pg_event in pygame.event.get():
            if pg_event.type == pygame.QUIT:
                running = False
            elif pg_event.type == pygame.KEYDOWN and pg_event.key == pygame.K_F3:
                show_debug = not show_debug

        engine.apply_events(keyboard_events())
        engine.update(dt)
        draw(
            screen,
            engine,
            font,
            debug={"show": show_debug, "cv_status": "disabled", "game_fps": game_fps.rate},
        )

    pygame.quit()
    logger.info("VisionStrike session ended: %s", engine.state.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
