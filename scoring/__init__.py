"""Member B: reference motions and scoring logic for the choreo coach.

Typical use (Member C's UI):

    from scoring import PoseSequence, score_attempt
    result = score_attempt(user_pose, "data/references/my_move.npz", user_video="attempt.mp4")
    result.total, result.feedback, result.status_at(3.2)
"""

from .joints import JOINT_NAMES
from .reference import Move, Reference, build_reference, load_moves
from .score import ScoreConfig, compare, score_attempt
from .types import ErrorEvent, PoseSequence, ScoreResult, TimingEvent

__all__ = [
    "JOINT_NAMES",
    "ErrorEvent",
    "Move",
    "PoseSequence",
    "Reference",
    "ScoreConfig",
    "ScoreResult",
    "TimingEvent",
    "build_reference",
    "compare",
    "load_moves",
    "score_attempt",
]
