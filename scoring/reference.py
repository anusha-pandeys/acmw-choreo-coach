"""Reference moves: the move list (moves.yaml) and the reference data files built from it.

A reference .npz is self-contained: the pose angles plus (optionally) the music's onset envelope and
beat times for the same time window, so scoring never needs the original reference video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import audio as audio_mod
from .pose_io import load_pose_file
from .types import PoseSequence


@dataclass
class Move:
    name: str
    pose_file: str
    start_s: float = 0.0
    end_s: float | None = None
    audio_file: str | None = None
    mirror: bool = False
    aist_view: int | None = 0
    description: str = ""


@dataclass
class Reference:
    name: str
    pose: PoseSequence
    audio_envelope: np.ndarray | None = None   # onset envelope at envelope_rate, same window as pose
    envelope_rate: float = audio_mod.ENVELOPE_RATE
    beat_times: np.ndarray | None = None       # seconds, relative to the start of the move
    description: str = ""
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def beat_period_s(self) -> float | None:
        if self.beat_times is None or len(self.beat_times) < 2:
            return None
        return float(np.median(np.diff(self.beat_times)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "name": np.array(self.name),
            "timestamps": self.pose.timestamps,
            "angles": self.pose.angles,
            "joint_names": np.array(self.pose.joint_names),
            "fps": np.array(self.pose.fps if self.pose.fps is not None else np.nan),
            "envelope_rate": np.array(self.envelope_rate),
            "description": np.array(self.description),
            "source": np.array(self.source),
        }
        if self.audio_envelope is not None:
            arrays["audio_envelope"] = self.audio_envelope
        if self.beat_times is not None:
            arrays["beat_times"] = self.beat_times
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        with np.load(path, allow_pickle=False) as d:
            fps = float(d["fps"]) if np.isfinite(d["fps"]) else None
            pose = PoseSequence(d["timestamps"], d["angles"], [str(n) for n in d["joint_names"]], fps)
            return cls(
                name=str(d["name"]),
                pose=pose.to_standard_joints(),
                audio_envelope=d["audio_envelope"] if "audio_envelope" in d.files else None,
                envelope_rate=float(d["envelope_rate"]),
                beat_times=d["beat_times"] if "beat_times" in d.files else None,
                description=str(d["description"]),
                source=str(d["source"]),
            )


def load_moves(path: str | Path) -> list[Move]:
    """Read the move list. Relative file paths are resolved against the YAML file's folder."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    moves = []
    for i, entry in enumerate(doc.get("moves", [])):
        if "name" not in entry or "pose_file" not in entry:
            raise ValueError(f"{path}: move #{i + 1} needs at least 'name' and 'pose_file'")
        move = Move(**entry)
        move.pose_file = str((path.parent / move.pose_file).resolve())
        if move.audio_file:
            move.audio_file = str((path.parent / move.audio_file).resolve())
        moves.append(move)
    names = [m.name for m in moves]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate move names {sorted(dupes)}")
    return moves


def build_reference(move: Move) -> Reference:
    """Crop the source pose (and music) to the move's window and package it as a Reference."""
    pose = load_pose_file(move.pose_file, aist_view=move.aist_view)
    end = move.end_s if move.end_s is not None else float(pose.timestamps[-1])
    if end <= move.start_s:
        raise ValueError(f"move {move.name}: end_s ({end}) must be after start_s ({move.start_s})")
    pose = pose.crop(move.start_s, end)
    if move.mirror:
        pose = pose.mirrored()

    envelope = beats = None
    if move.audio_file:
        y, sr = audio_mod.load_audio(move.audio_file)
        y = y[int(move.start_s * sr) : int(end * sr)]
        envelope = audio_mod.onset_envelope(y, sr)
        beats = audio_mod.beat_times(y, sr)

    visible = np.isfinite(pose.angles).mean(axis=0)
    return Reference(
        name=move.name,
        pose=pose,
        audio_envelope=envelope,
        beat_times=beats,
        description=move.description,
        source=f"{Path(move.pose_file).name} [{move.start_s:.2f}s-{end:.2f}s]{' mirrored' if move.mirror else ''}",
        meta={"visible_fraction": dict(zip(pose.joint_names, visible.round(3).tolist()))},
    )
