import json

import numpy as np
import pytest

from scoring import ScoreConfig, compare
from scoring import synthetic as syn
from scoring.audio import estimate_offset, onset_envelope
from scoring.reference import Reference

DUR = 8.0


@pytest.fixture(scope="module")
def curve():
    return syn.dance_curve(DUR + 4, seed=3)


@pytest.fixture(scope="module")
def ref(curve):
    return syn.sample(curve, DUR, noise_deg=2, seed=1)


@pytest.fixture(scope="module")
def good(curve):
    return syn.sample(curve, DUR, noise_deg=2, seed=2)


def test_identical_dance_scores_high_with_no_mistakes(ref, good):
    r = compare(good, ref)
    assert r.total > 95
    assert r.errors == []
    assert r.timing_events == []
    assert not r.mirrored
    assert "Nice" in r.feedback[0]


def test_bent_elbow_is_flagged_at_the_right_time_and_direction(ref, good):
    user = syn.inject_error(good, "r_elbow", 3.0, 4.0, -45)
    r = compare(user, ref)
    assert r.wrong_joints()[0] == "r_elbow"
    ev = r.errors[0]
    assert ev.start_s == pytest.approx(3.0, abs=0.15)
    assert ev.end_s == pytest.approx(4.0, abs=0.15)
    assert ev.direction == "lower"
    assert ev.mean_error_deg == pytest.approx(45, abs=6)
    assert "right elbow" in r.feedback[0] and "Straighten" in r.feedback[0]
    # The UI overlay agrees with the feedback.
    assert r.status_at(3.5)["r_elbow"] is False
    assert r.status_at(1.0)["r_elbow"] is True
    assert r.status_at(3.5)["l_elbow"] is True
    assert r.per_joint["r_elbow"] < r.per_joint["l_elbow"]


def test_straighter_knee_reports_higher_direction(ref, good):
    user = syn.inject_error(good, "l_knee", 5.0, 6.0, +40)
    r = compare(user, ref)
    ev = r.errors[0]
    assert ev.joint == "l_knee" and ev.direction == "higher"
    assert "Bend that knee" in r.feedback[0]


def test_short_blips_are_ignored(ref, good):
    user = syn.inject_error(good, "l_elbow", 2.0, 2.1, -60)  # 0.1 s, below min duration
    r = compare(user, ref)
    assert r.errors == []


def test_falling_behind_is_one_timing_event_not_constant_pose_errors(curve, ref):
    user = syn.sample(curve, DUR, noise_deg=2, seed=2, warp=syn.fall_behind(4.0, 0.3))
    r = compare(user, ref)
    assert len(r.timing_events) == 1
    ev = r.timing_events[0]
    assert ev.lag_s == pytest.approx(0.3, abs=0.08)
    assert ev.start_s == pytest.approx(4.0, abs=0.3)
    assert r.accuracy > 95          # the moves themselves were right
    assert r.timing < 90            # but the lateness is penalised
    assert "behind" in r.feedback[0] and "until the end" in r.feedback[0]

    direct = compare(user, ref, ScoreConfig(mode="direct"))
    assert direct.timing is None
    assert direct.accuracy < r.accuracy  # without DTW, lateness looks like wrong poses


def test_rushing_is_reported_as_ahead(curve, ref):
    user = syn.sample(curve, DUR, noise_deg=2, seed=2, warp=syn.fall_behind(3.0, -0.3))
    r = compare(user, ref)
    assert r.timing_events and r.timing_events[0].lag_s < -0.2
    assert "ahead" in r.feedback[0]


def test_recording_offset_is_removed_using_music(curve, ref):
    sr = 8000
    music = syn.click_track(DUR + 2, sr=sr, seed=7)
    ref_env = onset_envelope(music[: int(DUR * sr)], sr)
    user_audio = np.concatenate([0.003 * np.random.default_rng(0).standard_normal(2 * sr), music])
    user_env = onset_envelope(user_audio, sr)
    user = syn.sample(curve, DUR + 2, noise_deg=2, seed=2, start_offset_s=2.0)

    reference = Reference(name="move", pose=ref, audio_envelope=ref_env)
    r = compare(user, reference, user_audio_envelope=user_env)
    assert r.audio_offset_s == pytest.approx(2.0, abs=0.03)
    assert r.total > 95
    assert r.timing_events == []

    unsynced = compare(user, ref)
    assert unsynced.total < r.total - 10


def test_estimate_offset_sign_and_confidence():
    sr = 8000
    music = syn.click_track(10, sr=sr, seed=1)
    ref_env = onset_envelope(music, sr)
    late = onset_envelope(np.concatenate([np.zeros(int(1.5 * sr)), music]), sr)
    early = onset_envelope(music[int(1.0 * sr):], sr)
    off, conf = estimate_offset(ref_env, late)
    assert off == pytest.approx(1.5, abs=0.02) and conf > 5
    off, _ = estimate_offset(ref_env, early)
    assert off == pytest.approx(-1.0, abs=0.02)
    noise = onset_envelope(np.random.default_rng(5).standard_normal(10 * sr), sr)
    _, conf = estimate_offset(ref_env, noise)
    assert conf < 6


def test_mirrored_attempt_is_detected(ref, good):
    r = compare(good.mirrored(), ref)
    assert r.mirrored and r.total > 95
    assert any("mirrored" in f for f in r.feedback)
    never = compare(good.mirrored(), ref, ScoreConfig(mirror="never"))
    assert not never.mirrored and never.total < r.total


def test_ranking_good_over_sloppy_over_different(ref, good):
    sloppy = syn.inject_error(syn.inject_error(good, "r_elbow", 2, 4, -40), "l_shoulder", 5, 7, 40)
    different = syn.sample(syn.dance_curve(DUR, seed=123), DUR, noise_deg=2)
    scores = [compare(u, ref).total for u in (good, sloppy, different)]
    assert scores[0] > scores[1] > scores[2]


def test_hidden_joint_warns_and_is_left_out(ref, good):
    user = syn.hide_joint(good, "l_knee")
    r = compare(user, ref)
    assert "l_knee" not in r.per_joint
    assert any("left knee" in w and "in frame" in w for w in r.warnings)
    assert r.status_at(2.0)["l_knee"] is None
    assert r.total > 95


def test_short_tracking_dropouts_are_filled(ref, good):
    user = syn.hide_joint(good, "r_elbow", 2.0, 2.1)
    r = compare(user, ref)
    assert r.status_at(2.05)["r_elbow"] is True


def test_partial_overlap_warns(curve, ref):
    user = syn.sample(curve, DUR - 2, noise_deg=2, seed=2)
    r = compare(user, ref)
    assert any("ends" in w for w in r.warnings)
    assert r.total > 90


def test_no_overlap_raises(curve, ref):
    user = syn.sample(curve, 0.5, noise_deg=2)
    with pytest.raises(ValueError, match="overlap"):
        compare(user, ref)


def test_result_is_json_serialisable(ref, good):
    r = compare(syn.inject_error(good, "r_elbow", 3, 4, -45), ref)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["wrong_joints"] == ["r_elbow"]
    assert len(d["frame_status"]) == len(d["status_times"])
    assert len(d["frame_status"][0]) == len(d["joint_names"])


def test_beat_period_appears_in_timing_feedback(curve, ref):
    user = syn.sample(curve, DUR, noise_deg=2, seed=2, warp=syn.fall_behind(4.0, 0.3))
    reference = Reference(name="m", pose=ref, beat_times=np.arange(0, DUR, 0.5))  # 120 bpm
    r = compare(user, reference)
    assert "half a beat" in r.feedback[0]


def test_config_validation():
    with pytest.raises(ValueError):
        ScoreConfig(mode="fast")
    with pytest.raises(ValueError):
        ScoreConfig(tolerance_deg=50, max_error_deg=40)
