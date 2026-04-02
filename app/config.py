"""
Application configuration.

WHY PYDANTIC SETTINGS INSTEAD OF PLAIN os.getenv:
Every setting here gets a declared type and a default. Pydantic validates
and coerces on load (e.g. "CAMERA_WIDTH=abc" fails loudly at startup
instead of crashing deep inside OpenCV later with a confusing error), and
the rest of the app gets real autocomplete/type-checking on `settings.x`
instead of stringly-typed `os.environ["X"]` scattered everywhere.

Values are read from, in order of precedence: real environment variables,
then a local `.env` file (see `.env.example`), then the defaults below.
Nothing here is a secret; DATABASE_URL is included for symmetry with a
real deployment where it would point at a credentialed Postgres instance.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_YOLO_DEVICES = {"cpu", "cuda"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    # --- Camera / video (Week 1) ---
    camera_index: int = Field(default=0, ge=0)
    camera_width: int = Field(default=1280, gt=0)
    camera_height: int = Field(default=720, gt=0)
    target_fps: int = Field(default=30, gt=0)

    # --- YOLO detection (Week 2) ---
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = Field(default=0.4, gt=0.0, le=1.0)
    yolo_device: str = "cpu"  # "cpu" or "cuda"

    # --- MediaPipe gestures (Week 4) ---
    # .task model bundles (MediaPipe's newer Tasks API, not the legacy
    # `solutions` API) — not bundled with the pip package, downloaded once
    # into data/models/. See app/gestures/hands.py / pose.py.
    hand_model_path: str = str(DATA_DIR / "models" / "hand_landmarker.task")
    pose_model_path: str = str(DATA_DIR / "models" / "pose_landmarker_lite.task")
    gesture_min_confidence: float = Field(default=0.5, gt=0.0, le=1.0)

    # --- CV/game loop decoupling (V2 Phase 2) ---
    # A VisionState older than this is treated as stale/disconnected rather
    # than silently reused as if it were current — see app/vision/cv_worker.py.
    vision_stale_seconds: float = Field(default=1.5, gt=0.0)

    # --- API / server (Week 6) ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # --- Persistence (Week 6/7) ---
    database_url: str = f"sqlite:///{DATA_DIR / 'sessions' / 'visionstrike.db'}"

    # --- Telemetry / replay (V2.0 Part G) ---
    telemetry_dir: str = str(DATA_DIR / "sessions" / "telemetry")

    # --- Game loop rate (V2.0 Part I) ---
    game_tick_hz: int = Field(default=30, gt=0)

    # --- Observability ---
    log_level: str = "INFO"

    # V2.0 Part I: config validation on startup. Pydantic's own type
    # coercion (see module docstring) already catches "CAMERA_WIDTH=abc"
    # style errors; these two validators catch values that ARE the right
    # type but are still nonsense (a device string that isn't cpu/cuda, a
    # log level logging.py won't recognize) — both would otherwise fail
    # confusingly deep inside YOLO or the logging module instead of here,
    # at `Settings()` construction, which runs at import time before any
    # camera/model/socket is touched (see app/backend/main.py `create_app`).
    @field_validator("yolo_device")
    @classmethod
    def _validate_yolo_device(cls, v: str) -> str:
        if v not in _VALID_YOLO_DEVICES:
            raise ValueError(f"YOLO_DEVICE must be one of {sorted(_VALID_YOLO_DEVICES)}, got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {v!r}")
        return upper


settings = Settings()
