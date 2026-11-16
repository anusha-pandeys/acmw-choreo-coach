"""Shared joint definitions: the contract between Member A (pose extraction) and Member B (scoring).

Every angle array in this project has its columns in JOINT_NAMES order. Member A can either produce
angles directly in this order, or hand over raw landmarks and let `pose_io.angles_from_landmarks`
compute them using the triplets below.
"""

from __future__ import annotations

# Column order for every (T, J) angle array.
JOINT_NAMES: list[str] = [
    "l_elbow",
    "r_elbow",
    "l_shoulder",
    "r_shoulder",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "torso_lean",
]

JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# Unsigned angles come from three points (A-B-C, angle at B) and live in [0, 180].
# Signed angles can be negative and are negated when the video is mirrored.
SIGNED_JOINTS: set[str] = {"torso_lean"}

# Which joint each joint becomes when the dancer is mirrored (or the video is flipped).
MIRROR_MAP: dict[str, str] = {
    "l_elbow": "r_elbow",
    "r_elbow": "l_elbow",
    "l_shoulder": "r_shoulder",
    "r_shoulder": "l_shoulder",
    "l_hip": "r_hip",
    "r_hip": "l_hip",
    "l_knee": "r_knee",
    "r_knee": "l_knee",
    "torso_lean": "torso_lean",
}

# How much each joint counts toward the total score.
DEFAULT_WEIGHTS: dict[str, float] = {
    "l_elbow": 1.0,
    "r_elbow": 1.0,
    "l_shoulder": 1.0,
    "r_shoulder": 1.0,
    "l_hip": 0.75,
    "r_hip": 0.75,
    "l_knee": 1.0,
    "r_knee": 1.0,
    "torso_lean": 0.5,
}

# Landmark indices for each named body point, per keypoint layout.
LANDMARK_LAYOUTS: dict[str, dict[str, int]] = {
    # MediaPipe Pose Landmarker, 33 landmarks.
    "mediapipe33": {
        "l_shoulder": 11, "r_shoulder": 12,
        "l_elbow": 13, "r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16,
        "l_hip": 23, "r_hip": 24,
        "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28,
    },
    # COCO 17 keypoints (AIST++, MoveNet, YOLO Pose).
    "coco17": {
        "l_shoulder": 5, "r_shoulder": 6,
        "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10,
        "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14,
        "l_ankle": 15, "r_ankle": 16,
    },
}

# Three-point angle definitions: (point A, vertex B, point C) -> angle at B.
ANGLE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),      # 180 = straight arm
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_shoulder": ("l_hip", "l_shoulder", "l_elbow"),     # 0 = arm down, 180 = arm straight up
    "r_shoulder": ("r_hip", "r_shoulder", "r_elbow"),
    "l_hip": ("l_shoulder", "l_hip", "l_knee"),           # 180 = standing straight
    "r_hip": ("r_shoulder", "r_hip", "r_knee"),
    "l_knee": ("l_hip", "l_knee", "l_ankle"),             # 180 = straight leg
    "r_knee": ("r_hip", "r_knee", "r_ankle"),
}
# torso_lean is computed separately: the angle between the hip-midpoint -> shoulder-midpoint
# vector and image vertical. 0 = upright, positive = shoulders toward the image's right.

# Coaching phrases. "lower" means the user's angle is smaller than the reference.
_SIDE = {"l": "left", "r": "right"}


def side_and_part(joint: str) -> tuple[str | None, str]:
    if joint[:2] in ("l_", "r_"):
        return _SIDE[joint[0]], joint[2:]
    return None, joint


COACHING_PHRASES: dict[str, dict[str, str]] = {
    "elbow": {
        "lower": "your {side} elbow is about {deg}° more bent than the reference. Straighten that arm more.",
        "higher": "your {side} arm is about {deg}° straighter than the reference. Bend that elbow more.",
    },
    "shoulder": {
        "lower": "your {side} arm is about {deg}° lower than the reference. Raise it higher.",
        "higher": "your {side} arm is about {deg}° higher than the reference. Bring it down a bit.",
    },
    "hip": {
        "lower": "you're folding about {deg}° more at your {side} hip than the reference. Open up that hip.",
        "higher": "your {side} hip is about {deg}° more open than the reference. Fold more there (lift the knee or hinge forward).",
    },
    "knee": {
        "lower": "your {side} knee is about {deg}° more bent than the reference. Straighten that leg more.",
        "higher": "your {side} leg is about {deg}° straighter than the reference. Bend that knee more and get lower.",
    },
    "torso_lean": {
        "more": "you're leaning about {deg}° more than the reference. Keep your torso more upright.",
        "less": "you're about {deg}° more upright than the reference. Lean further into the move.",
    },
}
