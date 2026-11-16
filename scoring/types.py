"""Data types passed between team members.

PoseSequence   Member A -> Member B   (one per video)
ScoreResult    Member B -> Member C   (one per comparison)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .joints import JOINT_NAMES, MIRROR_MAP, SIGNED_JOINTS

# frame_status values
STATUS_OK = 1
STATUS_WRONG = 0
STATUS_UNSCORED = -1  # joint not visible in one of the videos at that moment


@dataclass
class PoseSequence:
    """Joint angles over time for one video.

    angles: (T, J) degrees, columns in `joint_names` order. NaN = landmark not visible enough.
    timestamps: (T,) seconds from the start of the video, strictly increasing.
    """

    timestamps: np.ndarray
    angles: np.ndarray
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    fps: float | None = None

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.angles = np.asarray(self.angles, dtype=float)
        if self.angles.ndim != 2:
            raise ValueError(f"angles must be (T, J), got shape {self.angles.shape}")
        if len(self.timestamps) != len(self.angles):
            raise ValueError(
                f"timestamps ({len(self.timestamps)}) and angles ({len(self.angles)}) differ in length"
            )
        if self.angles.shape[1] != len(self.joint_names):
            raise ValueError(
                f"angles has {self.angles.shape[1]} columns but {len(self.joint_names)} joint names"
            )
        if len(self.timestamps) > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if self.fps is None and len(self.timestamps) > 1:
            self.fps = float(1.0 / np.median(np.diff(self.timestamps)))

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self) else 0.0

    def to_standard_joints(self) -> "PoseSequence":
        """Reorder columns to JOINT_NAMES; joints this sequence lacks become all-NaN."""
        if self.joint_names == JOINT_NAMES:
            return self
        out = np.full((len(self), len(JOINT_NAMES)), np.nan)
        for j, name in enumerate(JOINT_NAMES):
            if name in self.joint_names:
                out[:, j] = self.angles[:, self.joint_names.index(name)]
        return PoseSequence(self.timestamps.copy(), out, list(JOINT_NAMES), self.fps)

    def mirrored(self) -> "PoseSequence":
        """Swap left/right joints and negate signed angles, as if the video were flipped."""
        out = np.empty_like(self.angles)
        for j, name in enumerate(self.joint_names):
            src = self.joint_names.index(MIRROR_MAP.get(name, name))
            out[:, j] = self.angles[:, src]
            if name in SIGNED_JOINTS:
                out[:, j] = -out[:, j]
        return PoseSequence(self.timestamps.copy(), out, list(self.joint_names), self.fps)

    def crop(self, start_s: float, end_s: float, rebase: bool = True) -> "PoseSequence":
        """Keep frames in [start_s, end_s]; with rebase, the first kept frame's time becomes start - start_s."""
        keep = (self.timestamps >= start_s) & (self.timestamps <= end_s)
        if not keep.any():
            raise ValueError(f"no frames between {start_s}s and {end_s}s")
        ts = self.timestamps[keep] - (start_s if rebase else 0.0)
        return PoseSequence(ts, self.angles[keep].copy(), list(self.joint_names), self.fps)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            angles=self.angles,
            joint_names=np.array(self.joint_names),
            fps=np.array(self.fps if self.fps is not None else np.nan),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        with np.load(path, allow_pickle=False) as d:
            if "angles" not in d:
                raise ValueError(
                    f"{path} has no 'angles' array. If it holds raw landmarks, use "
                    "scoring.pose_io.load_pose_file instead."
                )
            fps = float(d["fps"]) if "fps" in d and np.isfinite(d["fps"]) else None
            return cls(d["timestamps"], d["angles"], [str(n) for n in d["joint_names"]], fps)


@dataclass
class ErrorEvent:
    """One sustained pose mistake on one joint."""

    joint: str
    start_s: float          # in the user's video time
    end_s: float
    mean_error_deg: float
    peak_error_deg: float
    user_angle_deg: float   # mean over the event
    ref_angle_deg: float
    direction: str          # "lower" (user angle smaller than reference) or "higher"
    severity: float         # mean excess error x duration; used for ranking

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimingEvent:
    """A stretch where the dancer is ahead of or behind the reference."""

    start_s: float          # in the user's video time
    end_s: float
    lag_s: float            # median lag; positive = behind (late), negative = ahead (rushing)
    ref_start_s: float      # the same moment in the reference's timeline

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreResult:
    total: float                        # 0-100, combines accuracy and timing
    accuracy: float                     # 0-100, pose accuracy after alignment
    timing: float | None                # 0-100, None if timing wasn't analyzed (direct mode)
    per_joint: dict[str, float]         # 0-100 per joint (NaN joints omitted)
    errors: list[ErrorEvent]            # sorted most severe first
    timing_events: list[TimingEvent]
    feedback: list[str]                 # ready-to-display sentences, most important first
    warnings: list[str]                 # data-quality notes (joint not visible, etc.)
    mirrored: bool                      # True if the attempt was compared as a mirror image
    audio_offset_s: float               # user music start minus reference music start (0 if unknown)
    mode: str                           # "dtw" or "direct"
    joint_names: list[str]
    # For the UI's red/green overlay: status for each joint at each sampled user moment.
    status_times: np.ndarray            # (K,) seconds in the user's video
    frame_status: np.ndarray            # (K, J) int8: STATUS_OK / STATUS_WRONG / STATUS_UNSCORED
    frame_error: np.ndarray             # (K, J) degrees of error, NaN where unscored
    lag_times: np.ndarray | None = None  # (L,) seconds in the user's video
    lag_s: np.ndarray | None = None      # (L,) smoothed lag at each time (dtw mode only)

    def status_at(self, t_s: float) -> dict[str, bool | None]:
        """Joint status nearest to user-video time t_s. True = OK, False = wrong, None = not scored."""
        if len(self.status_times) == 0:
            return {name: None for name in self.joint_names}
        k = int(np.argmin(np.abs(self.status_times - t_s)))
        out: dict[str, bool | None] = {}
        for j, name in enumerate(self.joint_names):
            s = int(self.frame_status[k, j])
            out[name] = None if s == STATUS_UNSCORED else s == STATUS_OK
        return out

    def wrong_joints(self) -> list[str]:
        """Joints with at least one error event, most severe first."""
        seen: list[str] = []
        for e in self.errors:
            if e.joint not in seen:
                seen.append(e.joint)
        return seen

    def to_dict(self, include_arrays: bool = True) -> dict:
        """JSON-serialisable dict (NaN becomes None)."""
        clean = _to_json_array
        d = {
            "total": round(self.total, 1),
            "accuracy": round(self.accuracy, 1),
            "timing": None if self.timing is None else round(self.timing, 1),
            "per_joint": {k: round(v, 1) for k, v in self.per_joint.items()},
            "wrong_joints": self.wrong_joints(),
            "errors": [e.to_dict() for e in self.errors],
            "timing_events": [e.to_dict() for e in self.timing_events],
            "feedback": self.feedback,
            "warnings": self.warnings,
            "mirrored": self.mirrored,
            "audio_offset_s": round(self.audio_offset_s, 3),
            "mode": self.mode,
            "joint_names": self.joint_names,
        }
        if include_arrays:
            d["status_times"] = clean(self.status_times)
            d["frame_status"] = self.frame_status.astype(int).tolist()
            d["frame_error"] = clean(self.frame_error)
            d["lag_times"] = clean(self.lag_times)
            d["lag_s"] = clean(self.lag_s)
        return d


def _to_json_array(a: np.ndarray | None):
    if a is None:
        return None
    a = np.round(np.asarray(a, dtype=float), 3)
    obj = a.astype(object)
    obj[~np.isfinite(a)] = None
    return obj.tolist()
