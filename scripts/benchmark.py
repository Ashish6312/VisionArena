#!/usr/bin/env python
"""
Reproducible performance benchmark — V2 Phase 3.

Measures two separate things, both from REAL execution against REAL
hardware — nothing here is fabricated or estimated:

1. Per-stage CV pipeline cost (camera read, YOLO, ByteTrack, MediaPipe).
   Instrumented directly in this script rather than inside CVWorker —
   CVWorker only needs (and only measures) total pipeline latency for its
   actual job; adding per-stage timers to already-tested production code
   just for benchmarking isn't a measurable-enough reason to touch it.

2. The decoupled architecture's real end-to-end behavior: game FPS vs CV
   FPS vs vision-state age, using the actual `CameraWorker`/`CVWorker`
   classes `GameRunner` drives in production, with a simulated 30 Hz
   consumer loop standing in for the game tick.

Usage:
    python scripts/benchmark.py                  # 15s, both sections
    python scripts/benchmark.py --duration 30
    python scripts/benchmark.py --skip-decoupled  # per-stage timing only

Requires a webcam. If none is available, this script reports that and
exits — it does not fall back to synthetic numbers.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import __version__
from app.config import settings
from app.gestures.classifier import GestureRecognizer
from app.gestures.hands import GestureModelError
from app.logging_config import setup_logging
from app.vision.camera import Camera, CameraError
from app.vision.camera_worker import CameraWorker
from app.vision.cv_worker import CVWorker
from app.vision.detector import Detector, ModelLoadError
from app.vision.tracker import Tracker


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile — correct for small sample sizes,
    no dependency on `statistics.quantiles`' cut-point conventions."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


@dataclass
class StageStats:
    name: str
    samples_ms: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples_ms.append(ms)

    @property
    def avg(self) -> float:
        return round(sum(self.samples_ms) / len(self.samples_ms), 1) if self.samples_ms else 0.0

    @property
    def min(self) -> float:
        return round(min(self.samples_ms), 1) if self.samples_ms else 0.0

    @property
    def max(self) -> float:
        return round(max(self.samples_ms), 1) if self.samples_ms else 0.0

    @property
    def p95(self) -> float:
        return round(percentile(self.samples_ms, 95), 1)


def print_stage_table(stages: list[StageStats]) -> None:
    print(f"{'Component':<22}{'Avg':>10}{'Min':>10}{'Max':>10}{'P95':>10}   n")
    print("-" * 68)
    for s in stages:
        print(
            f"{s.name:<22}{s.avg:>8.1f}ms{s.min:>8.1f}ms{s.max:>8.1f}ms{s.p95:>8.1f}ms   {len(s.samples_ms)}"
        )


def benchmark_pipeline_stages(duration: float) -> list[StageStats] | None:
    """Runs the real pipeline stages directly (not through CVWorker) so
    each one's cost is measured individually."""
    camera_stage = StageStats("Camera read")
    yolo_stage = StageStats("YOLO (detect)")
    tracking_stage = StageStats("ByteTrack")
    mediapipe_stage = StageStats("MediaPipe")
    total_stage = StageStats("Total CV pipeline")

    print(f"\nOpening camera (index {settings.camera_index}) and loading models...")
    camera = Camera()
    try:
        camera.open()
    except CameraError as e:
        print(f"Could not open camera: {e}")
        print("Per-stage benchmark requires a real webcam — skipping this section.")
        return None

    try:
        detector = Detector()
        tracker = Tracker(frame_rate=settings.target_fps)
        gestures = GestureRecognizer()
    except (ModelLoadError, GestureModelError) as e:
        print(f"Could not load models: {e}")
        camera.release()
        return None

    print("Warming up (models JIT/cold-start on first call)...")
    for _ in range(3):
        frame, _ = camera.read()
        detector.detect(frame)

    print(f"Measuring for {duration:.0f}s (Ctrl+C to stop early)...")
    end_at = time.perf_counter() + duration
    try:
        while time.perf_counter() < end_at:
            t_total0 = time.perf_counter()

            t0 = time.perf_counter()
            frame, _diag = camera.read()
            camera_stage.record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            yolo_stage.record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            tracked = tracker.update(detections)
            tracking_stage.record((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            gestures.process(frame)
            mediapipe_stage.record((time.perf_counter() - t0) * 1000)

            total_stage.record((time.perf_counter() - t_total0) * 1000)
            _ = tracked  # not used further; kept so the tracker call doesn't read as dead code
    except KeyboardInterrupt:
        print("Stopped early by user.")
    finally:
        gestures.close()
        camera.release()

    return [camera_stage, yolo_stage, tracking_stage, mediapipe_stage, total_stage]


def benchmark_decoupled_loop(duration: float) -> dict | None:
    """Runs the real CameraWorker + CVWorker, with a simulated 30 Hz
    consumer loop standing in for GameRunner's game tick — measuring
    exactly what GameRunner experiences in production."""
    print(f"\nStarting decoupled CameraWorker + CVWorker for a {duration:.0f}s run...")
    camera_worker = CameraWorker(Camera())
    cv_worker = CVWorker(camera_worker)
    camera_worker.start()
    cv_worker.start()

    tick_count = 0
    last_frame_id = -1
    vision_ages_ms: list[float] = []
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < duration:
            tick_start = time.perf_counter()
            state = cv_worker.state_slot.get()
            if state is not None:
                vision_ages_ms.append(state.age_seconds * 1000)
                if state.frame_id != last_frame_id:
                    last_frame_id = state.frame_id
            tick_count += 1
            budget = (1.0 / 30) - (time.perf_counter() - tick_start)
            if budget > 0:
                time.sleep(budget)
    finally:
        cv_worker.stop()
        camera_worker.stop()

    elapsed = time.perf_counter() - start
    if camera_worker.error or cv_worker.error:
        print(f"Camera/CV worker error during run: {camera_worker.error or cv_worker.error}")
        return None

    return {
        "game_fps": round(tick_count / elapsed, 1),
        "camera_fps": camera_worker.rate.rate,
        "cv_fps": cv_worker.rate.rate,
        "cv_latency_avg_ms": cv_worker.latency_ms.average,
        "vision_age_avg_ms": round(sum(vision_ages_ms) / len(vision_ages_ms), 1) if vision_ages_ms else None,
        "vision_age_max_ms": round(max(vision_ages_ms), 1) if vision_ages_ms else None,
    }


def print_environment(duration: float) -> None:
    print("Environment")
    print("-" * 68)
    print(f"  VisionStrike version : {__version__}")
    print(f"  Python               : {platform.python_version()}")
    print(f"  Platform             : {platform.platform()}")
    print(f"  YOLO model           : {settings.yolo_model} (device={settings.yolo_device})")
    print(
        f"  Camera requested     : {settings.camera_width}x{settings.camera_height}@{settings.target_fps}fps"
    )
    print(f"  Benchmark duration   : {duration:.0f}s per section")


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionStrike performance benchmark")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds per benchmark section")
    parser.add_argument("--skip-decoupled", action="store_true", help="Only run the per-stage timing section")
    args = parser.parse_args()

    setup_logging()
    print("=" * 68)
    print("VISIONSTRIKE PERFORMANCE BENCHMARK")
    print("=" * 68)
    print_environment(args.duration)

    print("\n" + "=" * 68)
    print("PER-STAGE PIPELINE COST")
    print("=" * 68)
    stages = benchmark_pipeline_stages(args.duration)
    if stages:
        print()
        print_stage_table(stages)

    if not args.skip_decoupled:
        print("\n" + "=" * 68)
        print("DECOUPLED ARCHITECTURE — END-TO-END")
        print("=" * 68)
        result = benchmark_decoupled_loop(args.duration)
        if result:
            print(f"\n  Game FPS (simulated 30Hz consumer) : {result['game_fps']}")
            print(f"  Camera FPS                          : {result['camera_fps']}")
            print(f"  CV FPS                               : {result['cv_fps']}")
            print(f"  CV latency, avg                      : {result['cv_latency_avg_ms']} ms")
            age_avg, age_max = result["vision_age_avg_ms"], result["vision_age_max_ms"]
            print(f"  Vision-state age, avg / max         : {age_avg} / {age_max} ms")

    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
