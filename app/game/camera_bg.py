"""
camera_bg.py — live webcam feed as the VisionStrike game background.

"Intelligent" rendering pipeline (in order):
  1. BGR→RGB + horizontal mirror  — webcam output looks natural when mirrored
  2. Aspect-ratio-correct scale + centre-crop  — fills 1280×720 without distortion
  3. Sci-fi colour grade  — cool blue shift so the arena feels like a game, not
     a plain video feed; done in NumPy (vectorised, no per-pixel Python loops)
  4. Dark vignette overlay  — semi-transparent dark layer so game sprites / HUD
     text stay legible regardless of what the camera sees
  5. YOLO bounding-box overlays  — for every tracked person:
       • semi-transparent glow ring (expands beyond box)
       • solid corner-tick box (more readable than a full rectangle)
       • "YOU ▶" label in green on the primary player, "ID:N" in cyan on others

Call `draw_camera_background()` BEFORE `draw()` so all game sprites/HUD
render on top.  Returns immediately (never blocks) — the latest raw camera
frame from `CameraWorker.slot` may be a few ms old, which is fine.
"""

from __future__ import annotations

import numpy as np
import pygame

from ..vision.state import VisionState

# ── tuneable constants ─────────────────────────────────────────────────────
_DIM_ALPHA = 130  # 0-255: overlay darkness.  130 ≈ 50 % opaque dark layer
_BOX_COLOR = (40, 200, 240)  # cyan — any detected person
_GLOW_COLOR = (20, 80, 160, 65)  # dark cyan, semi-transparent glow ring
_PRIMARY_COLOR = (80, 255, 140)  # green — the primary tracked player
_BOX_THICKNESS = 2
_GLOW_EXPAND = 9  # px by which the glow ring grows beyond the detection box
_TICK_MAX_PX = 16  # max corner-tick length in pixels
_LABEL_FONT_SIZE = 20
# ──────────────────────────────────────────────────────────────────────────

_label_font: pygame.font.Font | None = None


def _get_font() -> pygame.font.Font:
    global _label_font
    if _label_font is None:
        _label_font = pygame.font.SysFont(None, _LABEL_FONT_SIZE)
    return _label_font


def _colour_grade(rgb: np.ndarray) -> None:
    """Sci-fi blue tint: reduce red channel 25 %, lift blue 15 %.
    Operates in-place on a (H, W, 3) uint8 RGB array — fully vectorised."""
    rgb[:, :, 0] = (rgb[:, :, 0].astype(np.uint16) * 75 // 100).astype(np.uint8)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2].astype(np.uint16) * 115 // 100, 0, 255).astype(np.uint8)


def _build_background_surface(frame: np.ndarray, arena_w: int, arena_h: int) -> pygame.Surface:
    """Convert a BGR numpy frame to a mirrored, colour-graded, scaled Pygame
    Surface sized exactly (arena_w × arena_h).

    Transform chain:
      BGR frame  →  flip x-axis + swap B↔R  →  colour grade  →
      scale (preserve aspect, fill arena)  →  centre-crop  →  copy
    Returns a fresh Surface; the caller can blit it immediately.
    """
    # BGR → RGB and mirror in one contiguous slice (no extra allocation)
    rgb: np.ndarray = np.ascontiguousarray(frame[:, ::-1, ::-1])
    _colour_grade(rgb)

    cam_h, cam_w = rgb.shape[:2]
    scale = max(arena_w / cam_w, arena_h / cam_h)
    new_w = int(cam_w * scale)
    new_h = int(cam_h * scale)

    # surfarray.make_surface expects (W, H, 3) — swap axes first
    raw_surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    scaled_surf = pygame.transform.smoothscale(raw_surf, (new_w, new_h))

    x_off = (new_w - arena_w) // 2
    y_off = (new_h - arena_h) // 2
    cropped = scaled_surf.subsurface((x_off, y_off, arena_w, arena_h))
    return cropped.copy()  # copy breaks the subsurface↔parent lifetime tie


def _box_to_arena(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cam_w: int,
    cam_h: int,
    arena_w: int,
    arena_h: int,
) -> tuple[int, int, int, int]:
    """Map a detection box from camera-pixel space → arena-display space,
    applying the identical mirror + scale + crop used in _build_background_surface."""
    # Mirror x (same as ::-1 on the frame's x-axis)
    mx1 = cam_w - x2
    mx2 = cam_w - x1

    scale = max(arena_w / cam_w, arena_h / cam_h)
    new_w = int(cam_w * scale)
    new_h = int(cam_h * scale)
    x_off = (new_w - arena_w) // 2
    y_off = (new_h - arena_h) // 2

    ax1 = int(mx1 * scale) - x_off
    ay1 = int(y1 * scale) - y_off
    ax2 = int(mx2 * scale) - x_off
    ay2 = int(y2 * scale) - y_off

    # Clamp to arena bounds
    ax1 = max(0, ax1)
    ay1 = max(0, ay1)
    ax2 = min(arena_w, ax2)
    ay2 = min(arena_h, ay2)
    return ax1, ay1, ax2, ay2


def _draw_detection_box(
    screen: pygame.Surface,
    ax1: int,
    ay1: int,
    ax2: int,
    ay2: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw glow ring + corner-tick box + label for one detection."""
    bw, bh = ax2 - ax1, ay2 - ay1
    if bw <= 0 or bh <= 0:
        return

    # ── glow ring ──────────────────────────────────────────────────────
    g = _GLOW_EXPAND
    glow_surf = pygame.Surface((bw + 2 * g, bh + 2 * g), pygame.SRCALPHA)
    glow_surf.fill((0, 0, 0, 0))
    pygame.draw.rect(glow_surf, _GLOW_COLOR, glow_surf.get_rect(), 5)
    screen.blit(glow_surf, (ax1 - g, ay1 - g))

    # ── corner ticks (instead of a full rectangle) ─────────────────────
    tick = min(_TICK_MAX_PX, bw // 4, bh // 4)
    corners = [
        (ax1, ay1, 1, 1),
        (ax2, ay1, -1, 1),
        (ax1, ay2, 1, -1),
        (ax2, ay2, -1, -1),
    ]
    for cx, cy, dx, dy in corners:
        pygame.draw.line(screen, color, (cx, cy), (cx + dx * tick, cy), 2)
        pygame.draw.line(screen, color, (cx, cy), (cx, cy + dy * tick), 2)

    # ── label ──────────────────────────────────────────────────────────
    font = _get_font()
    lbl = font.render(label, True, color)
    screen.blit(lbl, (ax1 + 4, max(0, ay1 - 22)))


def draw_camera_background(
    screen: pygame.Surface,
    frame: np.ndarray,
    vision_state: VisionState | None,
    *,
    arena_w: int,
    arena_h: int,
) -> None:
    """Render the webcam frame as the full game background with all intelligent
    overlays.  Call this BEFORE draw() so game elements render on top.

    Parameters
    ----------
    screen      : The pygame display surface to draw onto.
    frame       : Latest raw BGR numpy frame from CameraWorker.slot.
    vision_state: Latest VisionState from CVWorker (may be None while CV warms up).
    arena_w/h   : Pixel dimensions of the game arena (same as screen size).
    """
    # ── 1. camera frame ────────────────────────────────────────────────
    bg = _build_background_surface(frame, arena_w, arena_h)
    screen.blit(bg, (0, 0))

    # ── 2. dark vignette overlay ───────────────────────────────────────
    dim = pygame.Surface((arena_w, arena_h), pygame.SRCALPHA)
    dim.fill((6, 8, 18, _DIM_ALPHA))
    screen.blit(dim, (0, 0))

    # ── 3. YOLO detection boxes ────────────────────────────────────────
    if vision_state is None or not vision_state.tracked_objects:
        return

    cam_h, cam_w = frame.shape[:2]
    primary_id = vision_state.primary_track_id

    for obj in vision_state.tracked_objects:
        is_primary = obj.track_id == primary_id
        color = _PRIMARY_COLOR if is_primary else _BOX_COLOR
        label = "YOU ▶" if is_primary else f"ID:{obj.track_id}"

        ax1, ay1, ax2, ay2 = _box_to_arena(
            obj.x1, obj.y1, obj.x2, obj.y2,
            cam_w, cam_h, arena_w, arena_h,
        )
        _draw_detection_box(screen, ax1, ay1, ax2, ay2, color, label)
