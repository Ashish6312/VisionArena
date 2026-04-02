"""
Human-readable session report — Week 7.

Templated text from real, measured numbers — not an LLM call. The project
brief allows generating the summary sentence with an LLM later, but is
explicit that the underlying metrics must come from this project's own
CV/game system; this module is that underlying-metrics layer, so a future
LLM step would read `PerformanceReport.to_dict()` as its input rather than
inventing numbers.
"""

from __future__ import annotations

from .performance import PerformanceReport

_HIGH_ACCURACY = 60.0
_LOW_ACCURACY = 35.0
_HIGH_DIRECTION_CHANGES = 15


def _movement_note(report: PerformanceReport) -> str:
    m = report.movement
    if m.distance_traveled_px == 0:
        return "No physical movement data recorded for this session (keyboard-only, no camera)."
    if m.direction_changes >= _HIGH_DIRECTION_CHANGES:
        return "Frequent direction changes — an evasive, hard-to-hit playstyle."
    if m.stationary_seconds > report.survival_seconds * 0.5:
        return "Spent over half the session stationary — more movement reduces exposure to enemy fire."
    return "Steady, moderate movement throughout the session."


def _aim_note(report: PerformanceReport) -> str:
    if report.shots_fired == 0:
        return "No shots fired this session."
    if report.accuracy_pct >= _HIGH_ACCURACY:
        return f"Strong accuracy ({report.accuracy_pct}%) — shots were well-aimed."
    if report.accuracy_pct <= _LOW_ACCURACY:
        return f"Low accuracy ({report.accuracy_pct}%) — consider using AIM before SHOOT."
    return f"Moderate accuracy ({report.accuracy_pct}%)."


def _reaction_note(report: PerformanceReport) -> str:
    """One line summarizing avg/min/max/median — or an honest "no data"
    instead of a fabricated number when no stimulus->response pair was
    ever measured (see app/analytics/reaction.py)."""
    if report.reaction_time_samples == 0:
        return "no measured reactions this session"
    return (
        f"{report.reaction_time_avg_s:.2f}s avg "
        f"(min {report.reaction_time_min_s:.2f}s, max {report.reaction_time_max_s:.2f}s, "
        f"median {report.reaction_time_median_s:.2f}s, n={report.reaction_time_samples})"
    )


def generate_text_report(report: PerformanceReport) -> str:
    lines = [
        "VISIONSTRIKE PERFORMANCE REPORT",
        "",
        f"Session:        {report.session_id}",
        f"Score:          {report.score}",
        f"Enemies defeated: {report.kills}",
        f"Accuracy:       {report.accuracy_pct}%  ({report.shots_hit}/{report.shots_fired})",
        f"Damage taken:   {report.damage_taken}",
        f"Survival time:  {report.survival_seconds:.0f}s",
        f"Distance:       {report.movement.distance_traveled_px:.1f}px",
        f"Average speed:  {report.movement.average_speed_px_s:.1f}px/s",
        f"Reaction time:  {_reaction_note(report)}",
        "",
        "Movement Analysis:",
        f"- {_movement_note(report)}",
        "",
        "Aim Analysis:",
        f"- {_aim_note(report)}",
    ]
    if report.zone_time:
        lines += ["", "Zone Time:"]
        lines += [f"- {name}: {seconds:.1f}s" for name, seconds in report.zone_time.items()]
    if report.gesture_counts:
        lines += ["", "Gestures Used:"]
        lines += [f"- {name}: {count}" for name, count in report.gesture_counts.items()]
    return "\n".join(lines)
