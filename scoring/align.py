"""Resampling, gap filling and banded dynamic time warping (DTW).

DTW lines up corresponding moments of two performances even when one is a little faster or
slower. The band limits how far it may shift (max_shift_s), so it can't match unrelated moves.
The shift it does find is reported as timing lag rather than silently forgiven.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from .types import PoseSequence


def fill_short_gaps(seq: PoseSequence, max_gap_s: float = 0.2) -> PoseSequence:
    """Linearly fill NaN runs no longer than max_gap_s that have valid data on both sides."""
    angles = seq.angles.copy()
    t = seq.timestamps
    for j in range(angles.shape[1]):
        col = angles[:, j]
        nan = ~np.isfinite(col)
        if not nan.any() or nan.all():
            continue
        k = 0
        n = len(col)
        while k < n:
            if not nan[k]:
                k += 1
                continue
            start = k
            while k < n and nan[k]:
                k += 1
            end = k  # exclusive
            if start == 0 or end == n:
                continue  # can't interpolate at the edges
            if t[end] - t[start - 1] <= max_gap_s + 1e-9:
                col[start:end] = np.interp(t[start:end], [t[start - 1], t[end]], [col[start - 1], col[end]])
    return PoseSequence(t.copy(), angles, list(seq.joint_names), seq.fps)


def resample(seq: PoseSequence, times: np.ndarray) -> np.ndarray:
    """Angles at the requested times (linear interpolation). NaN where the nearest source frame is NaN
    or the time is outside the recording."""
    out = np.full((len(times), seq.angles.shape[1]), np.nan)
    t = seq.timestamps
    if len(t) == 0:
        return out
    inside = (times >= t[0] - 1e-9) & (times <= t[-1] + 1e-9)
    nearest = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    prev = np.clip(nearest - 1, 0, len(t) - 1)
    nearest = np.where(np.abs(t[prev] - times) < np.abs(t[nearest] - times), prev, nearest)
    for j in range(seq.angles.shape[1]):
        col = seq.angles[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        vals = np.interp(times, t[ok], col[ok])
        vals[~ok[nearest] | ~inside] = np.nan
        out[:, j] = vals
    return out


def signed_diff(user: np.ndarray, ref: np.ndarray, signed_mask: np.ndarray) -> np.ndarray:
    """user - ref in degrees; wrapped to [-180, 180) for signed joints."""
    d = user - ref
    if signed_mask.any():
        d[..., signed_mask] = (d[..., signed_mask] + 180.0) % 360.0 - 180.0
    return d


def frame_cost(
    ref_row: np.ndarray, user_rows: np.ndarray, weights: np.ndarray, signed_mask: np.ndarray
) -> np.ndarray:
    """Weighted mean absolute angle difference between one reference frame and several user frames.
    NaN where no joint is visible in both."""
    d = np.abs(signed_diff(user_rows, ref_row[None, :], signed_mask))
    w = np.where(np.isfinite(d), weights[None, :], 0.0)
    total_w = w.sum(axis=1)
    with np.errstate(invalid="ignore"):
        c = np.where(total_w > 0, np.nansum(d * w, axis=1) / total_w, np.nan)
    return c


def dtw_align(
    ref: np.ndarray,
    user: np.ndarray,
    weights: np.ndarray,
    signed_mask: np.ndarray,
    radius: int,
    step_penalty: float = 2.0,
) -> np.ndarray:
    """Banded DTW. Starts at (0, 0); ends on the last reference frame at whichever user frame
    fits best, so a dancer who is behind at the end isn't forced back into sync.

    radius: maximum |user index - ref index| in frames (both grids share the same frame rate).
    step_penalty: extra cost (degrees) for non-diagonal steps, so that during held poses the
        path stays on the diagonal instead of wandering.
    Returns path as an (P, 2) int array of (ref index, user index).
    """
    n, m = len(ref), len(user)
    if n == 0 or m == 0:
        return np.zeros((0, 2), dtype=int)
    lo = np.clip(np.arange(n) - radius, 0, m - 1)
    hi = np.clip(np.arange(n) + radius, 0, m - 1)

    cost = np.full((n, m), np.nan)
    for i in range(n):
        if lo[i] <= hi[i]:
            cost[i, lo[i] : hi[i] + 1] = frame_cost(ref[i], user[lo[i] : hi[i] + 1], weights, signed_mask)
    band = np.zeros((n, m), dtype=bool)
    for i in range(n):
        band[i, lo[i] : hi[i] + 1] = True
    # Moments where nothing comparable is visible: neutral cost so the path just passes through.
    finite = np.isfinite(cost) & band
    fill = float(np.median(cost[finite])) if finite.any() else 0.0
    cost[band & ~np.isfinite(cost)] = fill

    inf = np.inf
    acc = np.full((n, m), inf)
    back = np.zeros((n, m), dtype=np.int8)  # 0 diag, 1 from (i-1, j), 2 from (i, j-1)
    acc[0, 0] = cost[0, 0]
    for j in range(1, hi[0] + 1):
        acc[0, j] = acc[0, j - 1] + cost[0, j] + step_penalty
        back[0, j] = 2
    for i in range(1, n):
        for j in range(lo[i], hi[i] + 1):
            c = cost[i, j]
            best = acc[i - 1, j - 1] if j > 0 else inf
            move = 0
            up = acc[i - 1, j] + step_penalty
            if up < best:
                best, move = up, 1
            if j > 0:
                left = acc[i, j - 1] + step_penalty
                if left < best:
                    best, move = left, 2
            if best < inf:
                acc[i, j] = best + c
                back[i, j] = move

    # Open end: pick the best end column on the last reference row, normalised by path length.
    js = np.arange(lo[-1], hi[-1] + 1)
    ends = acc[n - 1, js] / (n + js)
    if not np.isfinite(ends).any():
        raise RuntimeError("DTW found no path; are the recordings far too different in length?")
    j = int(js[np.nanargmin(np.where(np.isfinite(ends), ends, np.nan))])
    i = n - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        move = back[i, j]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return np.array(path[::-1], dtype=int)


def lag_per_ref_frame(path: np.ndarray, ref_times: np.ndarray, user_times: np.ndarray) -> np.ndarray:
    """Mean (user time - ref time) of the user frames matched to each reference frame."""
    n = len(ref_times)
    lag = np.full(n, np.nan)
    sums = np.zeros(n)
    counts = np.zeros(n)
    np.add.at(sums, path[:, 0], user_times[path[:, 1]] - ref_times[path[:, 0]])
    np.add.at(counts, path[:, 0], 1)
    ok = counts > 0
    lag[ok] = sums[ok] / counts[ok]
    return lag


def smooth_lag(lag: np.ndarray, fps: float, window_s: float = 0.6) -> np.ndarray:
    k = max(3, int(round(window_s * fps)) | 1)  # odd kernel
    if len(lag) < k:
        return lag.copy()
    filled = lag.copy()
    bad = ~np.isfinite(filled)
    if bad.all():
        return filled
    if bad.any():
        idx = np.arange(len(filled))
        filled[bad] = np.interp(idx[bad], idx[~bad], filled[~bad])
    return medfilt(filled, kernel_size=k)


def runs(mask: np.ndarray, max_gap: int = 0) -> list[tuple[int, int]]:
    """(start, end_exclusive) of True runs, bridging False gaps of up to max_gap samples."""
    out: list[tuple[int, int]] = []
    n = len(mask)
    k = 0
    while k < n:
        if not mask[k]:
            k += 1
            continue
        start = k
        while k < n and mask[k]:
            k += 1
        if out and start - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], k)
        else:
            out.append((start, k))
    return out
