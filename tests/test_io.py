import pickle

import numpy as np
import pytest
from scipy.io import wavfile

from scoring import JOINT_NAMES, PoseSequence, build_reference, load_moves, score_attempt
from scoring import synthetic as syn
from scoring.joints import JOINT_INDEX
from scoring.pose_io import angles_from_landmarks, load_pose_file
from scoring.reference import Reference


def _mp_landmarks(frames: int = 3) -> np.ndarray:
    """MediaPipe-style landmarks (x, y, z, visibility) for a person standing, left arm bent 90°."""
    lm = np.zeros((frames, 33, 4))
    lm[:, :, 3] = 1.0
    pts = {
        11: (0.40, 0.30), 12: (0.60, 0.30),   # shoulders
        13: (0.40, 0.45), 14: (0.60, 0.45),   # elbows, straight below shoulders
        15: (0.55, 0.45), 16: (0.60, 0.60),   # left wrist out sideways (90°), right wrist straight down
        23: (0.42, 0.60), 24: (0.58, 0.60),   # hips
        25: (0.42, 0.80), 26: (0.58, 0.80),   # knees
        27: (0.42, 1.00), 28: (0.58, 1.00),   # ankles
    }
    for i, (x, y) in pts.items():
        lm[:, i, 0], lm[:, i, 1] = x, y
    return lm


def test_angles_from_landmarks():
    seq = angles_from_landmarks(_mp_landmarks(), np.arange(3) / 30)
    a = seq.angles[0]
    assert a[JOINT_INDEX["l_elbow"]] == pytest.approx(90, abs=0.5)
    assert a[JOINT_INDEX["r_elbow"]] == pytest.approx(180, abs=0.5)
    assert a[JOINT_INDEX["l_knee"]] == pytest.approx(180, abs=0.5)
    assert a[JOINT_INDEX["torso_lean"]] == pytest.approx(0, abs=0.5)


def test_normalised_coords_need_image_size():
    lm = _mp_landmarks()
    # In a 9:16 phone video, normalised x and y have different pixel scales.
    true = angles_from_landmarks(lm, np.arange(3) / 30, image_size=(1080, 1920))
    naive = angles_from_landmarks(lm, np.arange(3) / 30)
    i = JOINT_INDEX["l_elbow"]
    assert true.angles[0, i] == pytest.approx(90, abs=0.5)  # perpendicular stays perpendicular
    lm2 = lm.copy()
    lm2[:, 15, 1] = 0.55  # wrist moved diagonally: now the aspect ratio matters
    true = angles_from_landmarks(lm2, np.arange(3) / 30, image_size=(1080, 1920))
    naive = angles_from_landmarks(lm2, np.arange(3) / 30)
    assert abs(true.angles[0, i] - naive.angles[0, i]) > 5


def test_low_visibility_becomes_nan_and_lean_sign():
    lm = _mp_landmarks()
    lm[1, 15, 3] = 0.1  # left wrist barely visible in frame 1
    lm[2, 11, 0] += 0.1  # shift shoulders to the image's right in frame 2
    lm[2, 12, 0] += 0.1
    seq = angles_from_landmarks(lm, np.arange(3) / 30)
    assert np.isnan(seq.angles[1, JOINT_INDEX["l_elbow"]])
    assert np.isfinite(seq.angles[1, JOINT_INDEX["r_elbow"]])
    assert seq.angles[2, JOINT_INDEX["torso_lean"]] > 10
    assert seq.mirrored().angles[2, JOINT_INDEX["torso_lean"]] < -10


def test_pose_sequence_roundtrip_and_validation(tmp_path):
    seq = syn.sample(syn.dance_curve(3), 3)
    seq.save(tmp_path / "p.npz")
    back = PoseSequence.load(tmp_path / "p.npz")
    np.testing.assert_allclose(back.angles, seq.angles)
    assert back.joint_names == JOINT_NAMES
    with pytest.raises(ValueError):
        PoseSequence([0, 0.1, 0.1], np.zeros((3, len(JOINT_NAMES))))
    with pytest.raises(ValueError):
        PoseSequence([0, 0.1], np.zeros((2, 3)))


def test_partial_joint_set_is_padded():
    seq = PoseSequence([0, 0.1], [[120.0], [125.0]], joint_names=["r_elbow"]).to_standard_joints()
    assert seq.joint_names == JOINT_NAMES
    assert seq.angles[1, JOINT_INDEX["r_elbow"]] == 125.0
    assert np.isnan(seq.angles[0, JOINT_INDEX["l_elbow"]])


def test_landmark_npz_and_aist_pkl_loading(tmp_path):
    np.savez(tmp_path / "lm.npz", landmarks=_mp_landmarks(), timestamps=np.arange(3) / 30,
             layout="mediapipe33", image_size=[1080, 1920])
    seq = load_pose_file(tmp_path / "lm.npz")
    assert seq.angles[0, JOINT_INDEX["l_elbow"]] == pytest.approx(90, abs=0.5)

    coco = np.zeros((9, 5, 17, 3))
    coco[..., 2] = 0.9
    mp = _mp_landmarks(5)
    for name_mp, name_coco in [(11, 5), (12, 6), (13, 7), (14, 8), (15, 9), (16, 10),
                               (23, 11), (24, 12), (25, 13), (26, 14), (27, 15), (28, 16)]:
        coco[:, :, name_coco, :2] = mp[:, name_mp, :2] * 1000
    with open(tmp_path / "gBR_x.pkl", "wb") as f:
        pickle.dump({"keypoints2d": coco, "det_scores": np.ones((9, 5)), "timestamps": np.arange(5)}, f)
    seq = load_pose_file(tmp_path / "gBR_x.pkl", aist_view=0)
    assert seq.fps == pytest.approx(60)
    assert seq.angles[0, JOINT_INDEX["l_elbow"]] == pytest.approx(90, abs=0.5)

    with open(tmp_path / "k3d.pkl", "wb") as f:
        pickle.dump({"keypoints3d": np.concatenate([coco[0], np.zeros((5, 17, 1))], -1)[..., [0, 1, 3]]}, f)
    seq3 = load_pose_file(tmp_path / "k3d.pkl")
    assert seq3.angles[0, JOINT_INDEX["l_elbow"]] == pytest.approx(90, abs=0.5)
    assert np.isnan(seq3.angles[0, JOINT_INDEX["torso_lean"]])


def test_moves_yaml_to_reference_to_score(tmp_path):
    curve = syn.dance_curve(12, seed=4)
    full = syn.sample(curve, 12, noise_deg=1, seed=1)
    full.save(tmp_path / "source_pose.npz")
    sr = 8000
    music = syn.click_track(12, sr=sr, seed=2)
    wavfile.write(tmp_path / "source.wav", sr, (music * 20000).astype(np.int16))
    (tmp_path / "moves.yaml").write_text(
        "moves:\n"
        "  - name: test_move\n"
        "    pose_file: source_pose.npz\n"
        "    audio_file: source.wav\n"
        "    start_s: 2.0\n"
        "    end_s: 9.0\n"
        "    description: a test\n",
        encoding="utf-8",
    )
    moves = load_moves(tmp_path / "moves.yaml")
    ref = build_reference(moves[0])
    assert ref.pose.timestamps[0] == pytest.approx(0.0, abs=1e-6)
    assert ref.pose.duration == pytest.approx(7.0, abs=0.05)
    ref.save(tmp_path / "refs" / "test_move.npz")
    loaded = Reference.load(tmp_path / "refs" / "test_move.npz")
    assert loaded.audio_envelope is not None and loaded.name == "test_move"

    # The user recorded 1 s of setup, then danced the move; their music is the same song from 2.0 s.
    user_video_audio = np.concatenate([np.zeros(sr), music[2 * sr :]])
    wavfile.write(tmp_path / "attempt.wav", sr, (user_video_audio * 20000).astype(np.int16))
    user_t = np.arange(int(8 * 30)) / 30
    user = PoseSequence(user_t, curve(user_t - 1.0 + 2.0), list(JOINT_NAMES), 30)
    result = score_attempt(user, tmp_path / "refs" / "test_move.npz", user_video=tmp_path / "attempt.wav")
    assert result.audio_offset_s == pytest.approx(1.0, abs=0.03)
    assert result.total > 95


def test_duplicate_move_names_rejected(tmp_path):
    (tmp_path / "moves.yaml").write_text(
        "moves:\n  - {name: a, pose_file: x.npz}\n  - {name: a, pose_file: y.npz}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_moves(tmp_path / "moves.yaml")
