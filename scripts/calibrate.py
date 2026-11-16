"""Tune the scoring thresholds on real, labelled test recordings.

Make a CSV (paths relative to the CSV's folder) like:

    reference,attempt,label,video
    references/break_basic.npz,poses/cici_correct.npz,correct,videos/cici_correct.mp4
    references/break_basic.npz,poses/cici_wrong_arms.npz,wrong,
    references/break_basic.npz,poses/cici_late.npz,late,videos/cici_late.mp4

Then:  python scripts/calibrate.py data/test_attempts.csv

For each threshold setting it prints the average score per label. Pick the setting where
correct > late > wrong with the biggest gaps, then put those values in ScoreConfig.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scoring import audio as audio_mod  # noqa: E402
from scoring.pose_io import load_pose_file  # noqa: E402
from scoring.reference import Reference  # noqa: E402
from scoring.score import ScoreConfig, compare  # noqa: E402

GRID = [(tol, mx) for tol in (5, 10, 15) for mx in (35, 45, 60)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--mode", choices=["dtw", "direct"], default="dtw")
    args = ap.parse_args()

    base = Path(args.csv).resolve().parent
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref = Reference.load(base / row["reference"])
            user = load_pose_file(base / row["attempt"])
            env = None
            if row.get("video") and ref.audio_envelope is not None:
                y, sr = audio_mod.load_audio(base / row["video"])
                env = audio_mod.onset_envelope(y, sr, ref.envelope_rate)
            rows.append((row["label"], row["attempt"], ref, user, env))
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        return 1
    labels = sorted({r[0] for r in rows})

    print(f"{len(rows)} attempts, labels: {', '.join(labels)}\n")
    header = "tol/max  " + "".join(f"{lab:>12}" for lab in labels) + "   smallest gap"
    print(header)
    print("-" * len(header))
    best = None
    for tol, mx in GRID:
        cfg = ScoreConfig(mode=args.mode, tolerance_deg=tol, max_error_deg=mx)
        by_label = defaultdict(list)
        for label, _, ref, user, env in rows:
            by_label[label].append(compare(user, ref, cfg, user_audio_envelope=env).total)
        means = {lab: float(np.mean(v)) for lab, v in by_label.items()}
        ordered = sorted(means.values(), reverse=True)
        gap = min((a - b for a, b in zip(ordered, ordered[1:])), default=0.0)
        print(f"{tol:>3}/{mx:<4} " + "".join(f"{means[lab]:>12.1f}" for lab in labels) + f"   {gap:>8.1f}")
        if best is None or gap > best[0]:
            best = (gap, tol, mx)

    print(f"\nWidest separation: tolerance_deg={best[1]}, max_error_deg={best[2]}")
    print("Check the label order matches what you expect (e.g. correct > late > wrong) before adopting it.")

    cfg = ScoreConfig(mode=args.mode)
    print(f"\nPer-attempt detail at current defaults (tol {cfg.tolerance_deg}, max {cfg.max_error_deg}):")
    for label, name, ref, user, env in rows:
        r = compare(user, ref, cfg, user_audio_envelope=env)
        timing = "  n/a" if r.timing is None else f"{r.timing:5.0f}"
        print(f"  {label:>10}  total {r.total:5.1f}  acc {r.accuracy:5.1f}  timing {timing}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
