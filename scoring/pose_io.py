"""Turn pose data from various sources into a PoseSequence.

Supported inputs
- .npz with `angles` (+ `timestamps`, `joint_names`): Member A's angle output. Preferred.
- .npz with `landmarks` (T, K, 2-4) + `timestamps` + `layout` ("mediapipe33" or "coco17"),
  optionally `visibility` (T, K) and `image_size` [width, height].
- AIST++ keypoint .pkl files (COCO-17, 60 fps) for developing before Member A's pipeline is ready.
  Final references should go through Member A's MediaPipe pipeline so that reference and
  user are measured the same way.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .joints import ANGLE_TRIPLETS, JOINT_NAMES, LANDMARK_LAYOUTS
from .types import PoseSequence

AIST_FPS = 60.0


def _three_point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of points shaped (T, D)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang[(n1 < 1e-9) | (n2 < 1e-9)] = np.nan
    return ang


def angles_from_landmarks(
    landmarks: np.ndarray,
    timestamps: np.ndarray,
    layout: str = "mediapipe33",
    visibility: np.ndarray | None = None,
    visibility_threshold: float = 0.5,
    image_size: tuple[float, float] | None = None,
    use_3d: bool = False,
) -> PoseSequence:
    """Compute the standard joint angles from keypoints.

    landmarks: (T, K, D). Columns are x, y[, z][, visibility]. If D >= 4 and `visibility` is None,
        column 3 is used as visibility (MediaPipe style).
    image_size: (width, height). Pass this when x/y are normalised to [0, 1] (MediaPipe), otherwise
        angles are distorted by the video's aspect ratio. Not needed for pixel coordinates.
    use_3d: use x, y, z for the three-point angles (only meaningful for true 3D data such as AIST++
        keypoints3d). torso_lean needs image "up" and is NaN in 3D mode.
    """
    if layout not in LANDMARK_LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {list(LANDMARK_LAYOUTS)}")
    lm = np.asarray(landmarks, dtype=float)
    if lm.ndim != 3:
        raise ValueError(f"landmarks must be (T, K, D), got {lm.shape}")
    idx = LANDMARK_LAYOUTS[layout]
    need = max(idx.values()) + 1
    if lm.shape[1] < need:
        raise ValueError(f"layout {layout} needs {need} landmarks, got {lm.shape[1]}")

    if visibility is None and lm.shape[2] >= 4:
        visibility = lm[:, :, 3]
    dims = 3 if use_3d else 2
    if lm.shape[2] < dims:
        raise ValueError(f"need at least {dims} coordinate columns, got {lm.shape[2]}")
    pts = lm[:, :, :dims].copy()
    if image_size is not None and not use_3d:
        pts[:, :, 0] *= float(image_size[0])
        pts[:, :, 1] *= float(image_size[1])

    def p(name: str) -> np.ndarray:
        return pts[:, idx[name], :]

    def visible(*names: str) -> np.ndarray:
        if visibility is None:
            return np.ones(len(lm), dtype=bool)
        vis = np.asarray(visibility, dtype=float)
        return np.all([vis[:, idx[n]] >= visibility_threshold for n in names], axis=0)

    angles = np.full((len(lm), len(JOINT_NAMES)), np.nan)
    for j, joint in enumerate(JOINT_NAMES):
        if joint in ANGLE_TRIPLETS:
            a, b, c = ANGLE_TRIPLETS[joint]
            ang = _three_point_angle(p(a), p(b), p(c))
            ang[~visible(a, b, c)] = np.nan
            angles[:, j] = ang
        elif joint == "torso_lean" and not use_3d:
            sh = (p("l_shoulder") + p("r_shoulder")) / 2
            hip = (p("l_hip") + p("r_hip")) / 2
            dx = sh[:, 0] - hip[:, 0]
            dy = sh[:, 1] - hip[:, 1]  # image y points down, so upright => dy < 0
            lean = np.degrees(np.arctan2(dx, -dy))
            lean[~visible("l_shoulder", "r_shoulder", "l_hip", "r_hip")] = np.nan
            angles[:, j] = lean
    angles[~np.isfinite(angles)] = np.nan
    return PoseSequence(np.asarray(timestamps, dtype=float), angles, list(JOINT_NAMES))


def load_aist_keypoints(
    path: str | Path,
    view: int | None = 0,
    confidence_threshold: float = 0.3,
) -> PoseSequence:
    """Load an AIST++ keypoints2d / keypoints3d .pkl (COCO-17, 60 fps).

    view: camera index for keypoints2d (0-8). Pick the camera that faces the dancer, since users
        will film themselves from the front. Ignored for keypoints3d.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "keypoints2d" in data:
        kp = np.asarray(data["keypoints2d"], dtype=float)
        if kp.ndim != 4 or kp.shape[2:] != (17, 3):
            raise ValueError(
                f"{path}: expected keypoints2d shaped (views, frames, 17, 3), got {kp.shape}"
            )
        if view is None or not 0 <= view < kp.shape[0]:
            raise ValueError(f"{path}: choose a view between 0 and {kp.shape[0] - 1}")
        kp = kp[view]
        ts = np.arange(len(kp)) / AIST_FPS
        return angles_from_landmarks(
            kp[:, :, :2], ts, layout="coco17", visibility=kp[:, :, 2],
            visibility_threshold=confidence_threshold,
        )
    for key in ("keypoints3d_optim", "keypoints3d"):
        if key in data:
            kp = np.asarray(data[key], dtype=float)
            if kp.ndim != 3 or kp.shape[1:] != (17, 3):
                raise ValueError(f"{path}: expected {key} shaped (frames, 17, 3), got {kp.shape}")
            ts = np.arange(len(kp)) / AIST_FPS
            return angles_from_landmarks(kp, ts, layout="coco17", use_3d=True)
    raise ValueError(f"{path}: no keypoints2d/keypoints3d found (keys: {sorted(data)})")


def load_pose_file(path: str | Path, aist_view: int | None = 0) -> PoseSequence:
    """Load any supported pose file and return it with the standard joint columns."""
    path = Path(path)
    if path.suffix == ".pkl":
        return load_aist_keypoints(path, view=aist_view).to_standard_joints()
    if path.suffix != ".npz":
        raise ValueError(f"unsupported pose file type: {path} (expected .npz or AIST++ .pkl)")
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        if "angles" in keys:
            pass  # handled below via PoseSequence.load
        elif "landmarks" in keys:
            layout = str(d["layout"]) if "layout" in keys else "mediapipe33"
            image_size = tuple(d["image_size"]) if "image_size" in keys else None
            vis = d["visibility"] if "visibility" in keys else None
            return angles_from_landmarks(
                d["landmarks"], d["timestamps"], layout=layout, visibility=vis, image_size=image_size
            ).to_standard_joints()
        else:
            raise ValueError(f"{path}: expected an 'angles' or 'landmarks' array, found {sorted(keys)}")
    return PoseSequence.load(path).to_standard_joints()
