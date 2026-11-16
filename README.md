# Choreo Coach

Compare a dance attempt against a reference move and get a score plus coaching feedback.

| Part | Owner | Status |
|---|---|---|
| Pose detection & joint angles ("the eyes") | Member A | not started |
| Reference moves & scoring ("the brain"), in `scoring/` | Member B | **working, tested on synthetic data** |
| UI, test videos, demo ("the face") | Member C | not started |

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest              # 25 tests, ~3 s
python scripts/demo_synthetic.py
```

FFmpeg on PATH (or `pip install librosa`) is only needed to read music out of `.mp4` files.

---

## For Member A: what scoring needs from you

One **PoseSequence** per video, saved as `.npz`:

```python
from scoring import PoseSequence, JOINT_NAMES
PoseSequence(timestamps=ts, angles=angles).save("data/poses/my_video.npz")
```

| field | shape | meaning |
|---|---|---|
| `timestamps` | `(T,)` | seconds from the start of the video, increasing |
| `angles` | `(T, 9)` | degrees, columns in `JOINT_NAMES` order |

- `JOINT_NAMES = l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, torso_lean`.
  How each angle is measured is defined in [scoring/joints.py](scoring/joints.py) (`ANGLE_TRIPLETS`).
- **Use NaN, not 0, when a joint isn't visible enough.** 0° is a real angle.
- Smoothing is your call; scoring only fills dropouts shorter than 0.2 s.

**Or skip the angles:** save raw landmarks and scoring computes the angles for you:

```python
np.savez("my_video.npz", landmarks=lm,          # (T, 33, 4): x, y, z, visibility from MediaPipe
         timestamps=ts, layout="mediapipe33",
         image_size=[width, height])            # needed: MediaPipe x/y are normalised 0-1
```

`scoring.pose_io.angles_from_landmarks` is a reference implementation of the angle maths. If you
compute angles differently, tell Member B, because reference and attempt must be measured the same way.

## For Member C: what you get back

```python
from scoring import score_attempt
r = score_attempt(user_pose, "data/references/break_basic_01.npz", user_video="attempt.mp4")
```

| field | use |
|---|---|
| `r.total`, `r.accuracy`, `r.timing` | 0-100 scores (`timing` is None in direct mode) |
| `r.per_joint` | `{"r_elbow": 72.5, ...}` |
| `r.feedback` | list of ready-to-show sentences, most important first |
| `r.warnings` | data problems ("we couldn't see your left knee...") |
| `r.status_at(t)` | `{"r_elbow": False, "l_elbow": True, "l_knee": None, ...}` at user-video time `t` → red / green / grey joints |
| `r.errors` | each mistake: joint, start_s, end_s, mean/peak error, direction |
| `r.timing_events` | each stretch of being late (lag_s > 0) or early (lag_s < 0) |
| `r.lag_times`, `r.lag_s` | the lag curve, for a timing chart |
| `r.to_dict()` | everything as JSON |

`data/examples/*.json` are real results (correct / wrong arm / late / mirrored) you can build the UI
against now. Regenerate with `python scripts/demo_synthetic.py`.

---

## How scoring works

1. **Music sync.** The two recordings are lined up by their music (onset-envelope cross-correlation).
   Pressing record early isn't a mistake, so this shift is not penalised. We never align on when the
   dancer starts moving, because starting late *is* a mistake.
2. **Motion alignment (DTW).** Dynamic time warping lines up matching moments, but only within ±1 s.
   That way a dancer who is a bit late still gets credit for the right moves.
3. **Accuracy.** For each joint, the error between matched moments: ≤10° counts as perfect, ≥45° as 0,
   and in between is linear. The total is a weighted mean across joints.
4. **Timing.** How far DTW had to shift becomes the lag curve. A sustained lag is reported once
   ("at 0:04 you fell 0.3 s behind and stayed off until the end") instead of as many pose errors.
   Total = 70% accuracy + 30% timing.
5. **Mistakes & feedback.** Errors above 30° lasting at least 0.25 s become events. The sign of the
   difference picks the advice ("more bent" → "straighten that arm"). The top 3 are turned into sentences.
6. **Mirroring.** The attempt is also scored with left and right swapped. If that is clearly better,
   it's used and the user is told.

All thresholds live in `ScoreConfig` ([scoring/score.py](scoring/score.py)).

## Scripts

| command | does |
|---|---|
| `python scripts/build_reference.py` | `data/moves.yaml` → `data/references/<move>.npz` |
| `python scripts/score_attempt.py --reference R.npz --user U.npz [--user-video U.mp4] [--json out.json]` | score one attempt from the command line |
| `python scripts/calibrate.py data/test_attempts.csv` | tune thresholds on labelled real recordings |
| `python scripts/demo_synthetic.py` | fake data end to end, writes `data/examples/` |

## Reference data (AIST++)

Run AIST++ videos through **Member A's pipeline** and point `moves.yaml` at the output. Then
reference and attempt contain the same kind of pose-estimation noise. AIST++'s own keypoint
`.pkl` files also load (`aist_view` = camera index), which is useful before Member A's code exists.
Use the camera that faces the dancer, since users will film from the front.

## Not done yet / next steps

- **Calibrate on real videos.** The 10°/45°/30° thresholds were checked only on synthetic data. Once C
  records correct / wrong / late attempts, run `scripts/calibrate.py`.
- **Flow** (smoothness, sharp hits): not implemented. Plan: compare velocity/jerk profiles against
  the reference (Savitzky–Golay derivatives).
- **LLM-worded feedback** (Azure OpenAI): optional; feed it `r.errors` / `r.timing_events`.
