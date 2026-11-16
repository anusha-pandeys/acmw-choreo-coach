"""Turn measured mistakes into short coaching sentences (template-based, no LLM needed).

Later, the same ErrorEvent / TimingEvent data can be sent to an LLM to reword these, but the
numbers here stay the source of truth.
"""

from __future__ import annotations

from .joints import COACHING_PHRASES, side_and_part
from .types import ErrorEvent, TimingEvent


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def _beats(lag_s: float, beat_period_s: float | None) -> str:
    if not beat_period_s:
        return ""
    beats = abs(lag_s) / beat_period_s
    if beats < 0.375:
        label = "about a quarter beat"
    elif beats < 0.75:
        label = "about half a beat"
    elif beats < 1.5:
        label = "about a full beat"
    else:
        label = f"about {beats:.0f} beats"
    return f" ({label})"


def timing_sentence(ev: TimingEvent, video_end_s: float, beat_period_s: float | None = None) -> str:
    amount = f"{abs(ev.lag_s):.1f}s{_beats(ev.lag_s, beat_period_s)}"
    if ev.lag_s > 0:
        start = f"At {fmt_time(ev.start_s)} you fell about {amount} behind the music"
        fix = "Start that move earlier."
    else:
        start = f"At {fmt_time(ev.start_s)} you rushed about {amount} ahead of the music"
        fix = "Hold back and wait for the count."
    if video_end_s - ev.end_s < 0.5:
        return f"{start} and stayed off until the end. {fix}"
    return f"{start} until {fmt_time(ev.end_s)}. {fix}"


def error_sentence(ev: ErrorEvent) -> str:
    side, part = side_and_part(ev.joint)
    phrases = COACHING_PHRASES.get(part)
    deg = int(round(ev.mean_error_deg / 5.0) * 5) or int(round(ev.mean_error_deg))
    if phrases is None:
        body = f"your {ev.joint.replace('_', ' ')} is about {deg}° off from the reference."
    else:
        body = phrases[ev.direction].format(side=side or "", deg=deg)
    when = fmt_time(ev.start_s)
    if ev.end_s - ev.start_s >= 1.0:
        when = f"{when}–{fmt_time(ev.end_s)}"
    return f"At {when}, {body}"


def build_feedback(
    errors: list[ErrorEvent],
    timing_events: list[TimingEvent],
    mirrored: bool,
    video_end_s: float,
    beat_period_s: float | None = None,
    max_items: int = 3,
    tolerance_deg: float = 10.0,
) -> list[str]:
    items: list[str] = []
    # One timing slip usually explains a lot, so the biggest one goes first.
    if timing_events:
        biggest = max(timing_events, key=lambda e: abs(e.lag_s) * (e.end_s - e.start_s))
        items.append(timing_sentence(biggest, video_end_s, beat_period_s))
    for ev in errors:
        if len(items) >= max_items:
            break
        items.append(error_sentence(ev))
    if not items:
        items.append(
            f"Nice! Your joints stayed within about {tolerance_deg:.0f}° of the reference and you kept time."
        )
    if mirrored:
        items.append(
            "Note: you danced this mirrored (or the video is flipped), so we compared your left side "
            "to the reference's right."
        )
    return items
