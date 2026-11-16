"""Generate a fake reference plus four fake attempts, score them, and save example results.

    python scripts/demo_synthetic.py

Writes data/examples/*.json: real ScoreResult output that Member C can build the UI against
before Member A's pose pipeline is ready.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import compare  # noqa: E402
from scoring import synthetic as syn  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    out = ROOT / "data" / "examples"
    out.mkdir(parents=True, exist_ok=True)

    curve = syn.dance_curve(12, seed=3)
    ref = syn.sample(curve, 8, noise_deg=2, seed=1)
    good = syn.sample(curve, 8, noise_deg=3, seed=2)
    attempts = {
        "correct": good,
        "wrong_arm": syn.inject_error(syn.inject_error(good, "r_elbow", 2.0, 3.5, -45), "l_shoulder", 5.0, 6.0, -40),
        "late": syn.sample(curve, 8, noise_deg=3, seed=2, warp=syn.fall_behind(3.0, 0.35)),
        "mirrored": good.mirrored(),
    }
    for name, attempt in attempts.items():
        r = compare(attempt, ref)
        timing = "n/a" if r.timing is None else f"{r.timing:.0f}"
        print(f"\n== {name}: total {r.total:.0f} (accuracy {r.accuracy:.0f}, timing {timing})")
        for line in r.feedback:
            print(f"   - {line}")
        (out / f"{name}.json").write_text(json.dumps(r.to_dict(), indent=1), encoding="utf-8")
    print(f"\nExample results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
