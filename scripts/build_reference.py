"""Build reference files from the move list.

    python scripts/build_reference.py                      # all moves in data/moves.yaml
    python scripts/build_reference.py --only break_basic   # one move
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.reference import build_reference, load_moves  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moves", default=str(ROOT / "data" / "moves.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "references"))
    ap.add_argument("--only", nargs="*", help="build only these move names")
    args = ap.parse_args()

    moves = load_moves(args.moves)
    if args.only:
        unknown = set(args.only) - {m.name for m in moves}
        if unknown:
            print(f"Unknown move(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        moves = [m for m in moves if m.name in args.only]

    failed = 0
    for move in moves:
        try:
            ref = build_reference(move)
        except Exception as exc:  # keep going so one bad entry doesn't block the rest
            print(f"FAILED  {move.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        out = Path(args.out) / f"{move.name}.npz"
        ref.save(out)
        vis = ref.meta["visible_fraction"]
        low = [j for j, v in vis.items() if v < 0.5]
        extras = []
        if ref.audio_envelope is not None:
            extras.append("music")
        if ref.beat_period_s:
            extras.append(f"{60 / ref.beat_period_s:.0f} bpm")
        print(f"ok      {move.name}: {ref.pose.duration:.1f}s, {len(ref.pose)} frames"
              f"{' + ' + ', '.join(extras) if extras else ''} -> {out}")
        if low:
            print(f"        warning: rarely visible joints: {', '.join(low)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
