"""Score one attempt against a reference move.

    python scripts/score_attempt.py --reference data/references/break_basic.npz \
        --user data/poses/my_try.npz --user-video my_try.mp4 --json result.json

--user accepts anything scoring.pose_io.load_pose_file understands (Member A's angle .npz,
a landmark .npz, or an AIST++ .pkl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.pose_io import load_pose_file  # noqa: E402
from scoring.score import ScoreConfig, pretty_joint, score_attempt  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="reference .npz from build_reference.py")
    ap.add_argument("--user", required=True, help="the attempt's pose file")
    ap.add_argument("--user-video", help="the attempt's video/audio, used to sync the music")
    ap.add_argument("--mode", choices=["dtw", "direct"], default="dtw")
    ap.add_argument("--mirror", choices=["auto", "never", "always"], default="auto")
    ap.add_argument("--json", help="write the full result (including per-frame status) here")
    args = ap.parse_args()

    user = load_pose_file(args.user)
    cfg = ScoreConfig(mode=args.mode, mirror=args.mirror)
    r = score_attempt(user, args.reference, user_video=args.user_video, config=cfg)

    timing = "n/a" if r.timing is None else f"{r.timing:.0f}"
    print(f"Total {r.total:.0f}   (accuracy {r.accuracy:.0f}, timing {timing})")
    if r.audio_offset_s:
        print(f"Music offset: {r.audio_offset_s:+.2f}s (not penalised)")
    print("Per joint: " + ", ".join(f"{pretty_joint(k)} {v:.0f}" for k, v in r.per_joint.items()))
    print("\nFeedback:")
    for line in r.feedback:
        print(f"  - {line}")
    if r.warnings:
        print("\nWarnings:")
        for w in r.warnings:
            print(f"  - {w}")
    if args.json:
        Path(args.json).write_text(json.dumps(r.to_dict(), indent=1), encoding="utf-8")
        print(f"\nFull result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
