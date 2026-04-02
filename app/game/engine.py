"""
Game engine — Week 5.

Drives one game tick: applies this tick's GameEvents (from keyboard or
gestures — identically, see app/events.py), moves entities, resolves
collisions, updates score/state. Rendering (the Pygame surface) lives in
scripts/run_game.py, not here — the engine has no `import pygame`, which
is what lets tests (and later, a headless server-side simulation) drive it
without a display.
"""

from __future__ import annotations

import logging
import time

from ..events import GameEvent, GameEventType
from . import aim as aim_module
from .arena import Arena, ZoneKind, default_arena
from .collision import circles_collide
from .enemy import Enemy
from .player import MOVE_STEP_PX, Player
from .scoring import register_hit, register_shot_fired
from .state import DIFFICULTY_PROFILES, Difficulty, GameState, GameStatus
from .weapons import SHOOT_COOLDOWN_SECONDS, Projectile, fire_laser

AIM_TARGET_MAX_DISTANCE_PX = 900.0  # V2.0 Part D — how far the aim-target cone reaches
AIM_TARGET_CONE_HALF_ANGLE_DEG = 12.0
HIT_MARKER_LIFETIME_SECONDS = 0.3  # V2.0 Part F — how long a hit marker stays visible

logger = logging.getLogger(__name__)

DANGER_ZONE_DPS = 5  # damage per second while standing in a DANGER zone
PAUSE_COOLDOWN_SECONDS = 0.5
ATTACK_RANGE_PX = 60.0
ATTACK_COOLDOWN_SECONDS = 1.0

EnemyAction = str  # "PATROL" | "SEARCH" | "CHASE" | "ATTACK" | "RETREAT" — set by ai/ (Week 7)


class GameEngine:
    """Usage (one tick):
    engine.apply_events(events)                       # from keyboard or gestures
    engine.update(dt, enemy_actions={1: "CHASE", ...}) # enemy_actions from ai/ (Week 7)
    """

    def __init__(self, arena: Arena | None = None, difficulty: Difficulty = Difficulty.MEDIUM):
        self.arena = arena or default_arena()
        self.state = GameState(difficulty=difficulty)
        self.player = Player(x=self.arena.width * 0.5, y=self.arena.height * 0.85)
        self.enemies: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.recent_hits: list[dict] = []  # V2.0 Part F — HUD hit-marker feedback, see _resolve_collisions
        self._last_shot_time = 0.0
        self._last_pause_time = 0.0
        self._next_enemy_id = 1
        self._spawn_enemies()

    def _spawn_enemies(self) -> None:
        profile = DIFFICULTY_PROFILES[self.state.difficulty]
        for i in range(profile.enemy_count):
            x = self.arena.width * (i + 1) / (profile.enemy_count + 1)
            y = self.arena.height * 0.15
            self.enemies.append(
                Enemy(
                    enemy_id=self._next_enemy_id,
                    x=x,
                    y=y,
                    health=profile.enemy_health,
                    max_health=profile.enemy_health,
                    speed=profile.enemy_speed,
                    damage=profile.enemy_damage,
                    waypoints=self._patrol_loop(x, y),
                )
            )
            self._next_enemy_id += 1

    def _patrol_loop(self, x: float, y: float) -> list[tuple[float, float]]:
        """A small rectangular patrol route around the spawn point (V2.0
        Part E) — deterministic, no randomness, clamped inside the arena
        so a spawn near an edge doesn't route an enemy out of bounds."""
        half = 60.0
        corners = [(x - half, y), (x + half, y), (x + half, y + half), (x - half, y + half)]
        return [self.arena.clamp(cx, cy, radius=half) for cx, cy in corners]

    # ---- input -----------------------------------------------------------------

    def apply_events(self, events: list[GameEvent]) -> None:
        """Consumes one tick's GameEvents. Movement/status events are
        level-triggered — they must arrive every tick to keep having an
        effect, same as a physically held key. SHOOT and PAUSE are
        edge-triggered with a cooldown, so a continuously-raised hand
        doesn't fire 30 shots/second."""
        self.player.reset_tick_status()
        if not self.state.is_active:
            for event in events:  # PAUSE is the only event allowed to act while paused/over
                if event.type == GameEventType.PAUSE:
                    self._toggle_pause()
            return

        for event in events:
            self._apply_event(event)

    def _apply_event(self, event: GameEvent) -> None:
        t = event.type
        if t == GameEventType.MOVE_LEFT:
            self.player.move(-MOVE_STEP_PX, 0)
        elif t == GameEventType.MOVE_RIGHT:
            self.player.move(MOVE_STEP_PX, 0)
        elif t == GameEventType.MOVE_FORWARD:
            self.player.move(0, -MOVE_STEP_PX)
        elif t == GameEventType.MOVE_BACKWARD:
            self.player.move(0, MOVE_STEP_PX)
        elif t == GameEventType.SHIELD:
            self.player.shielded = True
        elif t == GameEventType.CROUCH:
            self.player.crouching = True
        elif t == GameEventType.AIM:
            self.player.aiming = True
            if event.aim_vector is not None:
                self.player.last_aim_vector = event.aim_vector
        elif t == GameEventType.SHOOT:
            self._try_shoot()
        elif t == GameEventType.PAUSE:
            self._toggle_pause()

        self.player.x, self.player.y = self.arena.clamp(self.player.x, self.player.y, self.player.radius)

    def _toggle_pause(self) -> None:
        now = time.time()
        if now - self._last_pause_time < PAUSE_COOLDOWN_SECONDS:
            return
        self._last_pause_time = now
        if self.state.status == GameStatus.RUNNING:
            self.state.status = GameStatus.PAUSED
        elif self.state.status == GameStatus.PAUSED:
            self.state.status = GameStatus.RUNNING

    def _try_shoot(self) -> None:
        now = time.time()
        if now - self._last_shot_time < SHOOT_COOLDOWN_SECONDS:
            return
        self._last_shot_time = now
        register_shot_fired(self.state)
        # V2.0 Part D: fires along the player's last known aim direction
        # (default straight up, matching the pre-Part-D fixed behavior
        # exactly when no aim data exists — see Player.last_aim_vector).
        direction_deg = aim_module.direction_degrees(self.player.last_aim_vector)
        self.projectiles.append(
            fire_laser(self.player.x, self.player.y, direction_deg=direction_deg, owner="player", damage=25)
        )

    # ---- simulation --------------------------------------------------------------

    def update(self, dt: float, enemy_actions: dict[int, EnemyAction] | None = None) -> None:
        if not self.state.is_active:
            return
        self.state.elapsed_seconds += dt
        enemy_actions = enemy_actions or {}

        self._apply_zone_effects(dt)
        self._update_enemies(dt, enemy_actions)
        self._update_aim_target()
        self._update_projectiles(dt)
        self._resolve_collisions()
        self._check_win_lose()

    def _update_aim_target(self) -> None:
        """V2.0 Part D: which enemy (if any) the current aim direction
        covers — purely for HUD/analytics ("what is the reticle over"),
        does not fire anything or deal damage. Only computed while the
        player is actively aiming this tick (not a stale reticle left over
        from a gesture that ended ticks ago). The actual hit/miss mechanic
        stays `_resolve_collisions`'s real projectile-travel physics,
        direction-aware since `_try_shoot` now uses the same aim vector."""
        if not self.player.aiming:
            self.player.target_enemy_id = None
            return
        targets = [(e.enemy_id, e.x, e.y, e.radius) for e in self.enemies if e.is_alive]
        self.player.target_enemy_id = aim_module.find_aim_target(
            origin=(self.player.x, self.player.y),
            vector=self.player.last_aim_vector,
            targets=targets,
            max_distance=AIM_TARGET_MAX_DISTANCE_PX,
            cone_half_angle_deg=AIM_TARGET_CONE_HALF_ANGLE_DEG,
        )

    def _apply_zone_effects(self, dt: float) -> None:
        zone = self.arena.zone_at(self.player.x, self.player.y)
        if zone and zone.kind == ZoneKind.DANGER:
            self.player.take_damage(round(DANGER_ZONE_DPS * dt))

    def _update_enemies(self, dt: float, enemy_actions: dict[int, EnemyAction]) -> None:
        for enemy in self.enemies:
            if not enemy.is_alive:
                continue
            action = enemy_actions.get(enemy.enemy_id, enemy.state)
            enemy.state = action
            if action == "ATTACK":
                self._enemy_attack(enemy)
            elif action == "CHASE":
                enemy.step_towards(self.player.x, self.player.y, dt)
            elif action == "SEARCH":
                # V2.0 Part E: moves toward where the player was last
                # actually seen (EnemyController writes this only while
                # visible — see enemy_controller.py), not the player's true
                # live position, which the enemy has no way to know once
                # it's lost sight of them. Falls back to the live position
                # only if a SEARCH was somehow entered with no prior
                # sighting recorded (shouldn't happen via the FSM, but
                # fails safe rather than crashing on None).
                target = enemy.last_known_player_pos or (self.player.x, self.player.y)
                enemy.step_towards(target[0], target[1], dt * 0.4)  # cautious advance
            elif action == "RETREAT":
                # Flee directly away from the player along the same line,
                # not towards a fixed point.
                away_x = enemy.x + (enemy.x - self.player.x)
                away_y = enemy.y + (enemy.y - self.player.y)
                enemy.step_towards(away_x, away_y, dt)
            elif action == "PATROL":
                enemy.patrol_step(dt)  # V2.0 Part E — real waypoint loop, see app/game/enemy.py

    def _enemy_attack(self, enemy: Enemy) -> None:
        """Contact damage: only lands if the enemy is actually close enough
        and its own attack cooldown has elapsed — an enemy the AI marked
        ATTACK from far away (e.g. a stale decision) can't hit through walls
        of distance."""
        dx, dy = enemy.x - self.player.x, enemy.y - self.player.y
        if (dx * dx + dy * dy) ** 0.5 > ATTACK_RANGE_PX:
            return
        now = time.time()
        if now - enemy.last_attack_time < ATTACK_COOLDOWN_SECONDS:
            return
        enemy.last_attack_time = now
        self.player.take_damage(enemy.damage)

    def _update_projectiles(self, dt: float) -> None:
        for p in self.projectiles:
            p.update(dt)
            if p.out_of_bounds(self.arena.width, self.arena.height):
                p.alive = False
        self.projectiles = [p for p in self.projectiles if p.alive]

    def _resolve_collisions(self) -> None:
        for p in self.projectiles:
            if not p.alive:
                continue
            if p.owner == "player":
                for enemy in self.enemies:
                    if not enemy.is_alive:
                        continue
                    if circles_collide(p.x, p.y, p.radius, enemy.x, enemy.y, enemy.radius):
                        p.alive = False
                        enemy.take_damage(p.damage)
                        killed = not enemy.is_alive
                        register_hit(self.state, killed=killed)
                        self._record_hit_marker(p.x, p.y, killed=killed)
                        break
            else:
                if circles_collide(p.x, p.y, p.radius, self.player.x, self.player.y, self.player.radius):
                    p.alive = False
                    self.player.take_damage(p.damage)
                    self._record_hit_marker(p.x, p.y, killed=False)
        self.projectiles = [p for p in self.projectiles if p.alive]
        self._expire_hit_markers()

    def _record_hit_marker(self, x: float, y: float, killed: bool) -> None:
        """V2.0 Part F: purely visual feedback for the HUD — where a
        projectile just landed, and whether it was a kill. Not read by any
        game-logic path, only `scripts/run_game.py::draw`."""
        self.recent_hits.append({"x": x, "y": y, "timestamp": time.time(), "killed": killed})

    def _expire_hit_markers(self) -> None:
        now = time.time()
        self.recent_hits = [
            h for h in self.recent_hits if now - h["timestamp"] <= HIT_MARKER_LIFETIME_SECONDS
        ]

    def _check_win_lose(self) -> None:
        if not self.player.is_alive:
            self.state.status = GameStatus.LOST
        elif self.enemies and all(not e.is_alive for e in self.enemies):
            self.state.status = GameStatus.WON

    # ---- serialization ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "player": self.player.to_dict(),
            "enemies": [e.to_dict() for e in self.enemies if e.is_alive],
            "projectiles": [
                {"x": round(p.x, 1), "y": round(p.y, 1), "owner": p.owner} for p in self.projectiles
            ],
            "recent_hits": [
                {"x": round(h["x"], 1), "y": round(h["y"], 1), "killed": h["killed"]}
                for h in self.recent_hits
            ],
        }
