"""Format times and paces for display."""

from __future__ import annotations


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_pace(min_per_mile: float) -> str:
    if min_per_mile <= 0 or min_per_mile > 60:
        return "—"
    minutes = int(min_per_mile)
    seconds = int(round((min_per_mile - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/mi"
