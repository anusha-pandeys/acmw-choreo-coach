"""Compare a user's attempt against a reference move.

Two separate timelines (the key design rule):
1. Music time: the recordings are lined up by their music (no penalty for when record was pressed).
2. Motion-aligned time: DTW lines up matching moves, within max_shift_s, to judge pose accuracy.
   How far DTW had to shift becomes the timing score, so timing mistakes are measured, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from .align import dtw_align, fill_short_gaps, lag_per_ref_frame, resample, runs, signed_diff, smooth_lag
from .feedback import build_feedback
from .joints import DEFAULT_WEIGHTS, JOINT_NAMES, SIGNED_JOINTS, side_and_part
from .reference import Reference
from .types import (
    STATUS_OK,
    STATUS_UNSCORED,
    STATUS_WRONG,
    ErrorEvent,
    PoseSequence,
    ScoreResult,
    TimingEvent,
)


@dataclass
class ScoreConfig:
    mode: str = "dtw"                   # "dtw" (handles timing differences) or "direct" (same clock)
    fps: float = 20.0                   # comparison frame rate
    tolerance_deg: float = 10.0         # error at or below this scores 100%
    max_error_deg: float = 45.0         # error at or above this scores 0%
    wrong_threshold_deg: float = 30.0   # sustained error above this is reported as a mistake
    min_error_duration_s: float = 0.25  # shorter blips are ignored (likely tracking noise)
    max_shift_s: float = 1.0            # how far DTW may shift moves in time
    step_penalty: float = 2.0           # DTW preference for staying in sync during held poses
    timing_tolerance_s: float = 0.15    # lag at or below this counts as on time
    timing_max_s: float = 0.5           # lag at or above this scores 0% for timing
    min_timing_duration_s: float = 0.5  # lag must last this long to be reported
    timing_weight: float = 0.3          # share of the total score that is timing
    mirror: str = "auto"                # "auto", "never" or "always"
    mirror_margin: float = 5.0          # mirrored must win by this many points to be chosen in auto
    max_gap_s: float = 0.2              # fill tracking dropouts up to this long
    min_audio_confidence: float = 5.0   # below this, the music match is ignored
    min_visible_fraction: float = 0.5   # warn when a joint is visible less than this often
    max_feedback: int = 3
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        if self.mode not in ("dtw", "direct"):
            raise ValueError(f"mode must be 'dtw' or 'direct', got {self.mode!r}")
        if self.mirror not in ("auto", "never", "always"):
            raise ValueError(f"mirror must be 'auto', 'never' or 'always', got {self.mirror!r}")
        if not self.tolerance_deg < self.max_error_deg:
            raise ValueError("tolerance_deg must be smaller than max_error_deg")


def pretty_joint(joint: str) -> str:
    side, part = side_and_part(joint)
    part = part.replace("_", " ")
    return f"{side} {part}" if side else part


def _angle_score(err: np.ndarray, cfg: ScoreConfig) -> np.ndarray:
    return np.clip(1.0 - (err - cfg.tolerance_deg) / (cfg.max_error_deg - cfg.tolerance_deg), 0.0, 1.0)


def _mean_by_index(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """Mean of rows of `values` (P, J) grouped by `index` (P,) into n groups, ignoring NaN."""
    j = values.shape[1]
    sums = np.zeros((n, j))
    counts = np.zeros((n, j))
    ok = np.isfinite(values)
    np.add.at(sums, index, np.where(ok, values, 0.0))
    np.add.at(counts, index, ok.astype(float))
    with np.errstate(invalid="ignore"):
        out = sums / counts
    out[counts == 0] = np.nan
    return out


def _compare_once(user: PoseSequence, ref: PoseSequence, offset: float, cfg: ScoreConfig) -> dict:
    names = list(JOINT_NAMES)
    weights = np.array([cfg.weights.get(n, 1.0) for n in names])
    signed_mask = np.array([n in SIGNED_JOINTS for n in names])
    step = 1.0 / cfg.fps
    notes: list[str] = []

    # Put both recordings on the reference's music timeline.
    user_start, user_end = user.timestamps[0] - offset, user.timestamps[-1] - offset
    ref_start, ref_end = ref.timestamps[0], ref.timestamps[-1]
    t0 = max(ref_start, user_start)
    t1 = min(ref_end, user_end)
    if t1 - t0 < 1.0:
        raise ValueError(
            "The attempt and the reference overlap by less than 1 second. Check that the videos "
            "are of the same move, or pass the correct audio offset."
        )
    if user_start > ref_start + 0.5:
        notes.append(f"Your video starts {user_start - ref_start:.1f}s into the move; the beginning wasn't scored.")
    if user_end < ref_end - 0.5:
        notes.append(f"Your video ends {ref_end - user_end:.1f}s before the move does; the ending wasn't scored.")

    ref_times = np.arange(t0, t1 + 1e-9, step)
    user_last = min(user_end, t1 + cfg.max_shift_s) if cfg.mode == "dtw" else t1
    user_times = np.arange(t0, user_last + 1e-9, step)
    ref_grid = resample(ref, ref_times)
    user_grid = resample(user, user_times + offset)
    n, m = len(ref_times), len(user_times)

    if cfg.mode == "dtw":
        path = dtw_align(ref_grid, user_grid, weights, signed_mask,
                         radius=int(round(cfg.max_shift_s * cfg.fps)), step_penalty=cfg.step_penalty)
    else:
        k = min(n, m)
        path = np.stack([np.arange(k), np.arange(k)], axis=1)

    pi, pj = path[:, 0], path[:, 1]
    pair_diff = signed_diff(user_grid[pj], ref_grid[pi], signed_mask)
    pair_err = np.abs(pair_diff)

    # Accuracy: every reference moment counts once.
    ref_err = _mean_by_index(pair_err, pi, n)
    per_joint: dict[str, float] = {}
    for j, name in enumerate(names):
        col = ref_err[:, j]
        if np.isfinite(col).mean() >= 0.2:
            per_joint[name] = float(np.nanmean(_angle_score(col, cfg)) * 100.0)
    if per_joint:
        w = np.array([cfg.weights.get(k, 1.0) for k in per_joint])
        accuracy = float(np.dot(w, list(per_joint.values())) / w.sum())
    else:
        accuracy = 0.0
        notes.append("No joints were visible in both videos at the same time, so pose accuracy couldn't be measured.")

    # What the user's video shows at each of its moments (for the overlay and error events).
    user_err = _mean_by_index(pair_err, pj, m)
    user_sdiff = _mean_by_index(pair_diff, pj, m)
    user_ref_angle = _mean_by_index(ref_grid[pi], pj, m)

    timing = None
    lag_s = None
    timing_events: list[TimingEvent] = []
    if cfg.mode == "dtw":
        lag_s = smooth_lag(lag_per_ref_frame(path, ref_times, user_times), cfg.fps)
        finite = np.isfinite(lag_s)
        if finite.any():
            lag_scores = np.clip(
                1.0 - (np.abs(lag_s[finite]) - cfg.timing_tolerance_s) / (cfg.timing_max_s - cfg.timing_tolerance_s),
                0.0, 1.0,
            )
            timing = float(lag_scores.mean() * 100.0)
        min_len = int(np.ceil(cfg.min_timing_duration_s * cfg.fps))
        gap = int(round(0.2 * cfg.fps))
        for sign in (1, -1):
            mask = finite & (sign * np.nan_to_num(lag_s) > cfg.timing_tolerance_s)
            for a, b in runs(mask, max_gap=gap):
                if b - a < min_len:
                    continue
                seg = lag_s[a:b]
                # Times are when the move *should* have happened, on the user's video clock.
                timing_events.append(TimingEvent(
                    start_s=float(ref_times[a] + offset),
                    end_s=float(ref_times[b - 1] + offset + step),
                    lag_s=float(np.nanmedian(seg)),
                    ref_start_s=float(ref_times[a]),
                ))
        timing_events.sort(key=lambda e: e.start_s)

    # Error events: sustained, large errors on one joint.
    status = np.where(np.isfinite(user_err), STATUS_OK, STATUS_UNSCORED).astype(np.int8)
    errors: list[ErrorEvent] = []
    min_len = max(1, int(np.ceil(cfg.min_error_duration_s * cfg.fps)))
    for j, name in enumerate(names):
        col = user_err[:, j]
        mask = np.isfinite(col) & (col > cfg.wrong_threshold_deg)
        for a, b in runs(mask, max_gap=2):
            if b - a < min_len:
                continue
            seg = col[a:b]
            seg_ok = np.isfinite(seg)
            u_ang = float(np.nanmean(user_grid[a:b, j]))
            r_ang = float(np.nanmean(user_ref_angle[a:b, j]))
            if name in SIGNED_JOINTS:
                direction = "more" if abs(u_ang) > abs(r_ang) else "less"
            else:
                direction = "lower" if np.nanmean(user_sdiff[a:b, j]) < 0 else "higher"
            duration = (b - a) * step
            mean_err = float(np.nanmean(seg[seg_ok]))
            errors.append(ErrorEvent(
                joint=name,
                start_s=float(user_times[a] + offset),
                end_s=float(user_times[b - 1] + offset + step),
                mean_error_deg=mean_err,
                peak_error_deg=float(np.nanmax(seg)),
                user_angle_deg=u_ang,
                ref_angle_deg=r_ang,
                direction=direction,
                severity=float((mean_err - cfg.tolerance_deg) * duration * cfg.weights.get(name, 1.0)),
            ))
            run_status = status[a:b, j]
            run_status[np.isfinite(seg)] = STATUS_WRONG
    errors.sort(key=lambda e: e.severity, reverse=True)

    # Visibility warnings.
    for label, grid in (("your", user_grid[: max(pj) + 1]), ("the reference's", ref_grid)):
        hidden = [names[j] for j in range(len(names))
                  if np.isfinite(grid[:, j]).mean() < cfg.min_visible_fraction]
        if hidden:
            joined = ", ".join(pretty_joint(h) for h in hidden)
            tip = " Make sure your whole body is in frame." if label == "your" else ""
            notes.append(f"We couldn't see {label} {joined} most of the time, so those joints count less.{tip}")

    return {
        "accuracy": accuracy,
        "timing": timing,
        "per_joint": per_joint,
        "errors": errors,
        "timing_events": timing_events,
        "status_times": user_times + offset,
        "frame_status": status,
        "frame_error": user_err,
        "lag_times": ref_times + offset if lag_s is not None else None,
        "lag_s": lag_s,
        "notes": notes,
        "video_end_s": float(user.timestamps[-1]),
    }


def compare(
    user: PoseSequence,
    reference: Reference | PoseSequence,
    config: ScoreConfig | None = None,
    user_audio_envelope: np.ndarray | None = None,
    audio_offset_s: float | None = None,
) -> ScoreResult:
    """Score `user` against `reference`.

    Music sync: pass `audio_offset_s` if you already know it, or `user_audio_envelope`
    (audio.onset_envelope of the user's video) and a Reference that has an audio envelope.
    Otherwise both videos are assumed to start at the same point in the music.
    """
    cfg = config or ScoreConfig()
    ref_obj = reference if isinstance(reference, Reference) else Reference(name="reference", pose=reference)
    ref = fill_short_gaps(ref_obj.pose.to_standard_joints(), cfg.max_gap_s)
    user_std = fill_short_gaps(user.to_standard_joints(), cfg.max_gap_s)
    warnings: list[str] = []

    offset = 0.0
    if audio_offset_s is not None:
        offset = float(audio_offset_s)
    elif user_audio_envelope is not None and ref_obj.audio_envelope is not None:
        est, conf = audio_mod.estimate_offset(ref_obj.audio_envelope, user_audio_envelope, ref_obj.envelope_rate)
        if conf >= cfg.min_audio_confidence:
            offset = est
        else:
            warnings.append(
                "Couldn't confidently match the music between the two videos, so we assumed they start "
                "at the same point. Make sure the song is audible in your recording."
            )

    candidates: list[tuple[bool, dict]] = []
    if cfg.mirror in ("auto", "never"):
        candidates.append((False, _compare_once(user_std, ref, offset, cfg)))
    if cfg.mirror in ("auto", "always"):
        candidates.append((True, _compare_once(user_std.mirrored(), ref, offset, cfg)))
    if cfg.mirror == "auto" and candidates[1][1]["accuracy"] > candidates[0][1]["accuracy"] + cfg.mirror_margin:
        mirrored, r = candidates[1]
    else:
        mirrored, r = candidates[0]

    timing = r["timing"]
    accuracy = r["accuracy"]
    if r["per_joint"] and accuracy < 50:
        warnings.append(
            "This attempt looks very different from the reference. Check that you picked the right move."
        )
    total = accuracy if timing is None else (1 - cfg.timing_weight) * accuracy + cfg.timing_weight * timing
    feedback = build_feedback(
        r["errors"], r["timing_events"], mirrored, r["video_end_s"],
        beat_period_s=ref_obj.beat_period_s, max_items=cfg.max_feedback, tolerance_deg=cfg.tolerance_deg,
    )
    return ScoreResult(
        total=float(total),
        accuracy=float(accuracy),
        timing=timing,
        per_joint=r["per_joint"],
        errors=r["errors"],
        timing_events=r["timing_events"],
        feedback=feedback,
        warnings=warnings + r["notes"],
        mirrored=mirrored,
        audio_offset_s=offset,
        mode=cfg.mode,
        joint_names=list(JOINT_NAMES),
        status_times=r["status_times"],
        frame_status=r["frame_status"],
        frame_error=r["frame_error"],
        lag_times=r["lag_times"],
        lag_s=r["lag_s"],
    )


def score_attempt(
    user_pose: PoseSequence,
    reference: Reference | str | Path,
    user_video: str | Path | None = None,
    config: ScoreConfig | None = None,
) -> ScoreResult:
    """Convenience entry point for the UI: reference may be a Reference or a path to its .npz,
    and user_video (if given) is used to sync the music."""
    ref = reference if isinstance(reference, Reference) else Reference.load(reference)
    env = None
    if user_video is not None and ref.audio_envelope is not None:
        y, sr = audio_mod.load_audio(user_video)
        env = audio_mod.onset_envelope(y, sr, ref.envelope_rate)
    return compare(user_pose, ref, config, user_audio_envelope=env)
