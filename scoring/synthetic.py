"""Fake dances for testing the scorer (and for Member C to build the UI) before real data exists."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .joints import JOINT_INDEX, JOINT_NAMES
from .types import PoseSequence

# Rough angle ranges for a dancer facing the camera: (centre, swing).
_RANGES = {
    "l_elbow": (130, 45), "r_elbow": (130, 45),
    "l_shoulder": (70, 60), "r_shoulder": (70, 60),
    "l_hip": (165, 15), "r_hip": (165, 15),
    "l_knee": (160, 20), "r_knee": (160, 20),
    "torso_lean": (0, 10),
}


def dance_curve(duration_s: float, fps: float = 30.0, seed: int = 0, smooth_s: float = 0.15):
    """A continuous, non-repeating 'choreography' as a function: t (seconds) -> (len(t), J) angles."""
    rng = np.random.default_rng(seed)
    hi_fps = 200.0
    n = int(np.ceil((duration_s + 4.0) * hi_fps))  # margin so shifted versions stay in range
    t_hi = np.arange(n) / hi_fps - 2.0
    curves = np.empty((n, len(JOINT_NAMES)))
    for j, name in enumerate(JOINT_NAMES):
        centre, swing = _RANGES[name]
        x = gaussian_filter1d(rng.standard_normal(n), sigma=smooth_s * hi_fps)
        x /= np.abs(x).max() + 1e-9
        curves[:, j] = centre + swing * x

    def at(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.stack([np.interp(t, t_hi, curves[:, j]) for j in range(len(JOINT_NAMES))], axis=1)

    return at


def sample(curve, duration_s: float, fps: float = 30.0, noise_deg: float = 0.0, seed: int = 1,
           warp=None, start_offset_s: float = 0.0) -> PoseSequence:
    """Record `curve` like a camera would.

    warp: function mapping video time -> choreography time (to simulate being late/early).
    start_offset_s: the move starts this many seconds into the video (simulates pressing record early).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_s * fps))) / fps
    choreo_t = t - start_offset_s
    if warp is not None:
        choreo_t = warp(choreo_t)
    angles = curve(choreo_t) + noise_deg * rng.standard_normal((len(t), len(JOINT_NAMES)))
    return PoseSequence(t, angles, list(JOINT_NAMES), fps)


def fall_behind(at_s: float, lag_s: float, ramp_s: float = 0.2):
    """Warp: on time until at_s, then lag_s behind for the rest (lateness builds up over ramp_s)."""

    def warp(t: np.ndarray) -> np.ndarray:
        ramp = np.clip((t - at_s) / ramp_s, 0.0, 1.0)
        return t - lag_s * ramp

    return warp


def inject_error(seq: PoseSequence, joint: str, start_s: float, end_s: float, delta_deg: float) -> PoseSequence:
    """Add delta_deg to one joint between start_s and end_s (e.g. -40 on an elbow = arm too bent)."""
    angles = seq.angles.copy()
    mask = (seq.timestamps >= start_s) & (seq.timestamps < end_s)
    angles[mask, JOINT_INDEX[joint]] += delta_deg
    return PoseSequence(seq.timestamps.copy(), angles, list(seq.joint_names), seq.fps)


def hide_joint(seq: PoseSequence, joint: str, start_s: float = 0.0, end_s: float = np.inf) -> PoseSequence:
    angles = seq.angles.copy()
    mask = (seq.timestamps >= start_s) & (seq.timestamps < end_s)
    angles[mask, JOINT_INDEX[joint]] = np.nan
    return PoseSequence(seq.timestamps.copy(), angles, list(seq.joint_names), seq.fps)


def click_track(duration_s: float, sr: int = 8000, seed: int = 0, lead_silence_s: float = 0.0) -> np.ndarray:
    """Irregular percussive 'music' so cross-correlation has one clear answer."""
    rng = np.random.default_rng(seed)
    y = 0.003 * rng.standard_normal(int((duration_s + lead_silence_s) * sr))
    t = 0.0
    while t < duration_s:
        start = int((t + lead_silence_s) * sr)
        burst = rng.uniform(0.3, 1.0) * np.exp(-np.arange(int(0.05 * sr)) / (0.01 * sr))
        burst = burst * rng.standard_normal(len(burst))
        y[start : start + len(burst)] += burst[: len(y) - start]
        t += rng.uniform(0.15, 0.6)
    return y
