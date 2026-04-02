# VisionStrike

**A real-time computer-vision game controller.** Stand in front of a webcam, and your body becomes the input device — YOLOv8 detects you, ByteTrack keeps your identity across frames, MediaPipe reads your pose and hand shape, and the result drives a live Pygame arena against a deterministic AI opponent, streamed over FastAPI/WebSocket to any connected client.

Not a YOLO detection demo. A full pipeline where *"I built an object detector"* becomes *"I built a system that turns physical human movement into game input, end to end, with the numbers to prove every stage of it runs in real time."*

### At a glance

| | | | |
|---|---|---|---|
| **Game FPS** | 28-29 | **CV FPS** | 12.9-13.3 |
| **CV latency (avg)** | 75-78 ms | **Tests passing** | 250 / 250 |
| **Application code** | ~4,000 lines | **Test code** | ~3,200 lines |
| **Python** | 3.13 | **Dependencies** | YOLOv8 · ByteTrack · MediaPipe · FastAPI · Pygame · SQLite |

`Python` `FastAPI` `WebSocket` `Pygame` `YOLOv8` `ByteTrack` `MediaPipe` `Pydantic` `SQLite` `pytest`

**Jump to**: [Why this exists](#why-this-exists) · [Tech stack](#tech-stack) · [Engineering highlights](#engineering-highlights) · [Skills demonstrated](#skills-demonstrated) · [Quick start](#quick-start) · [Performance](#performance-measured-not-projected) · [API](#api) · [Project structure](#project-structure) · [Scope](#what-s-deliberately-not-here)

---

## Architecture

```
Webcam → OpenCV → YOLOv8 → ByteTrack → Trajectory
                                           │
        MediaPipe (hands + pose) → Gesture ┤
                                           ▼
                                     GameEvent ◄──── Keyboard
                                           │          (same event, different source)
                      ┌────────────────────┴────────────────────┐
                      │                                          │
                Pygame arena                              FastAPI / WebSocket
             (player, AI opponent,                     (broadcasts live state to
              zones, projectiles)                        any connected client)
                      │                                          │
              AI opponent (FSM)                         Telemetry → SQLite Analytics
```

**The seam that matters**: the game layer only ever consumes `GameEvent` objects (`app/events.py`) — never YOLO or MediaPipe output directly. A keyboard press and a raised hand resolve to the identical `SHOOT` event before the game sees either one. `GameEngine.apply_events()` has no `if source == "gesture"` branch anywhere. That's the difference between "I wired a webcam to a game" and an architecture where a Unity client, a different CV backend, or a completely different input device could all drop in without touching game/AI/backend code.

---

## Why this exists

Built to demonstrate what a Computer Vision Engineer role in interactive/gaming systems actually needs — not a Jupyter notebook with a YOLO inference cell, but object detection, multi-object tracking, gesture recognition, and a real-time backend, wired into one system where every week's work is a dependency of the next.

**Real measured numbers, not projected ones**: game runs at **28-29 FPS** while the CV pipeline runs independently at **~13 FPS** on CPU — decoupled so the heavier vision workload never drags the game loop down. **250 automated tests pass** with zero camera/GPU dependency, and the full pipeline has also been run live against real hardware, including deliberate failure scenarios (invalid camera index, missing model file, mid-session worker crash) — every one produces a controlled, typed failure, not a crash.

---

## Tech stack

| Layer | Technology | What it does here |
|---|---|---|
| 🎯 Detection | YOLOv8n (Ultralytics) | Person detection, COCO class, CPU inference |
| 🔗 Tracking | ByteTrack | Persistent player identity across frames, not just per-frame boxes |
| 🖐️ Gesture / Pose | MediaPipe Tasks API (`HandLandmarker` + `PoseLandmarker`) | Hand shape → SHOOT/SHIELD/AIM, body position → MOVE_LEFT/RIGHT, torso compression → CROUCH |
| 🎮 Game engine | Pygame | Arena, zones, projectiles, scoring, win/lose |
| 🤖 AI opponent | Hand-rolled 6-state FSM | PATROL → SEARCH → CHASE → ATTACK → RETREAT, patrol waypoints, field-of-view perception |
| 🌐 Backend | FastAPI + WebSocket | Session lifecycle, live state broadcast, `/metrics` observability |
| 💾 Persistence | SQLite | Per-session performance reports, queryable after the fact |
| ⚙️ Config | Pydantic Settings | Every threshold/path/port is env-var driven, validated at startup — no hardcoded values anywhere |
| ✅ Testing | pytest, 250 tests | Camera/YOLO/ByteTrack/MediaPipe all mocked per-module; backend tested end-to-end with a real engine, real SQLite, real background thread |

---

## Engineering highlights

| Highlight | What it means |
|---|---|
| **Decoupled CV/game loop** | YOLO inference is ~180ms of blocking work — running it inline would cap the game at CV's speed. `CameraWorker` and `CVWorker` each run on their own thread; the game loop reads the latest published `VisionState` non-blockingly and never waits on a model. Measured result: game FPS is unaffected by CV running at under half its speed. |
| **One input contract, two sources** | `--mode camera` and `--mode keyboard` drive the exact same `GameEngine` through the exact same `GameEvent` type. Keyboard isn't a "fake CV" fallback — it's how the game gets developed and tested without a webcam attached at all. |
| **Stimulus-ID reaction tracking** | Each enemy's transition into `ATTACK` is a uniquely-IDed stimulus, explicitly closed by the player's next response — not a blind FIFO guess. Supports multiple simultaneous enemies, duplicate-close protection, stale-stimulus expiry. |
| **2D pose-driven aim, honestly scoped** | Elbow→wrist vector from MediaPipe Pose resolves against enemies through a cone-based target-intersection check. Documented as a 2D interpretable proxy, not depth-aware 3D aiming — a monocular camera can't measure depth, so `MOVE_FORWARD`/`MOVE_BACKWARD` were never implemented rather than shipped as a noisy hack. |
| **Config fails loud, not deep** | Every setting — camera index, YOLO confidence, MediaPipe paths, API host/port, DB URL, tick rate — is env-var driven and bounds-checked at `Settings()` construction. An invalid confidence or unrecognized device string fails at startup with a specific message. |
| **Failure paths are typed, not swallowed** | Invalid camera index, missing model file, mid-pipeline exception, stop-during-startup, restart-after-crash — every one is covered by a real test, several re-verified against actual hardware, not just mocks. |

---

## Skills demonstrated

| Category | Skills |
|---|---|
| **Computer vision** | Object detection (YOLOv8), multi-object tracking (ByteTrack), pose/hand landmark estimation (MediaPipe Tasks API), frame preprocessing, real-time inference pipeline design |
| **Systems / concurrency** | Multi-threaded producer/consumer architecture (camera → CV → game, each non-blocking on the others), thread-safe latest-value handoff, graceful worker shutdown under real timing constraints |
| **Backend** | FastAPI, WebSocket broadcast to multiple clients, REST API design, Pydantic settings validation, SQLite persistence, structured JSONL telemetry |
| **Software engineering practice** | 250-test suite with hardware-boundary mocking, deterministic finite-state-machine design, dataclass-driven contracts between layers, env-var-driven config with fail-fast validation, honest scope documentation |
| **Game / simulation** | Real-time game loop design, collision/zone/scoring systems, geometry (vector-based aim resolution, cone intersection tests), deterministic AI behavior (patrol/perception/pursuit FSM) |

---

## Quick start

Requires Python 3.12+. On this machine, PyTorch needs `C:\Python313\python.exe` specifically — Anaconda's Python has a DLL conflict with `torch` here.

```bash
git clone https://github.com/Ashish6312/VisionArena.git visionstrike
cd visionstrike
C:\Python313\python.exe -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
cp .env.example .env

# MediaPipe model bundles (not shipped in the pip package)
mkdir -p data/models
curl -L -o data/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -L -o data/models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

```bash
./.venv/Scripts/python.exe scripts/run_full_system.py       # full pipeline, one window — the actual demo
./.venv/Scripts/python.exe scripts/run_full_system.py --mode keyboard   # same game, zero camera dependency
./.venv/Scripts/python.exe scripts/run_server.py             # FastAPI + WebSocket, for a network client
./.venv/Scripts/python.exe scripts/benchmark.py               # reproducible FPS/latency measurement
```

Press **F3** in the game window for the live diagnostics panel (CV connection status, track ID, gesture + confidence, game/CV FPS, latency).

---

## Performance (measured, not projected)

Reproducible via `scripts/benchmark.py` — real webcam, Windows 11, Python 3.13.3, CPU inference, no GPU:

| Metric | Measured |
|---|---|
| Game FPS | 28.1 – 28.4 (target 30) |
| Camera FPS | 28.7 – 29.8 |
| CV FPS (YOLO + ByteTrack + MediaPipe combined) | 12.9 – 13.3 |
| CV pipeline latency, avg | 75.3 – 77.6 ms |
| Vision-state age, avg | 37.4 – 38.2 ms |

The game loop staying at ~29 FPS while CV runs at under half that speed is the direct, measured result of decoupling them onto separate threads — not a claim, a benchmark.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Structured camera/CV/game FPS, latency, worker liveness — fields are `null` when the system genuinely has no value, never a fabricated `0.0` |
| `POST` | `/game/start` | Start a session (camera or keyboard mode, difficulty) |
| `POST` | `/game/stop` | Stop the running session |
| `GET` | `/game/state` | Live game + CV state snapshot |
| `GET` | `/player/{player_id}` | Live player state |
| `GET` | `/analytics/{session_id}` | Full per-session performance report |
| `WS` | `/ws/game` | Live broadcast — message types `game_state`, `gesture`, `session_summary`, `error` |

---

## Project structure

| Module | Lines | Responsibility |
|---|---|---|
| `app/vision/` | 1,137 | Camera capture, YOLO detection, ByteTrack, trajectory analysis |
| `app/gestures/` | 424 | MediaPipe hand/pose landmarks → classified GameEvents |
| `app/game/` | 944 | Engine, arena, player, enemies, weapons, collision, aim geometry |
| `app/ai/` | 281 | 6-state enemy FSM, perception, patrol behavior |
| `app/backend/` | 579 | FastAPI app, routes, WebSocket manager, GameRunner |
| `app/analytics/` | 529 | Reaction-time tracking, telemetry, performance reports |
| `app/database/` | 118 | SQLite session persistence |
| `scripts/` | — | Standalone entry points (vision-only, game-only, server, full system, benchmark, replay) |

Backed by a **250-test / ~3,200-line suite** (kept in local development, not shipped in this repo) covering every module above with hardware-boundary mocking — camera/YOLO/ByteTrack/MediaPipe are mocked per-module; game/AI/analytics logic is pure Python needing no mocks; the FastAPI backend is tested end-to-end with a real engine, real SQLite database, and a real background thread.

---

## What's deliberately not here

| Not built | Why |
|---|---|
| **RL enemy** | An `EnemyPolicy` interface exists (`app/ai/policies.py`) for one, but the FSM was deliberately kept and hardened instead — real patrol waypoints and field-of-view perception, not a black-box policy for a 4-state enemy in a small arena |
| **3D aim / depth** | A monocular camera can't measure depth; `MOVE_FORWARD`/`MOVE_BACKWARD` and true 3D aim were cut rather than shipped as an unreliable proxy |
| **Unity client** | The architecture (`GameEvent` as the only cross-layer contract) was built for one from day one — not started, not blocking anything |
| **Gesture debounce window** | Jitter right at a raised/not-raised boundary is a known, documented limitation, not a hidden one |
