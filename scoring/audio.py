"""Music synchronisation: find how far apart the song is in two recordings.

The offset is a *recording* difference (someone pressed record earlier), so it carries no penalty.
We never align on when the dancers start moving, because starting late is a real timing mistake.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

ENVELOPE_RATE = 100.0  # envelope samples per second (10 ms resolution)
DEFAULT_SR = 22050


def load_audio(path: str | Path, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load mono audio from a video or audio file. Uses FFmpeg if installed, else librosa."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".wav":
        rate, data = wavfile.read(path)
        return _to_mono_float(data), int(rate)
    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "pipe:1"],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg could not read audio from {path}: {proc.stderr.decode(errors='replace')}")
        rate, data = wavfile.read(io.BytesIO(proc.stdout))
        return _to_mono_float(data), int(rate)
    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Reading audio from {path.suffix} files needs FFmpeg on PATH or `pip install librosa`."
        ) from exc
    y, rate = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(float), int(rate)


def _to_mono_float(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float) / np.iinfo(data.dtype).max
    data = data.astype(float)
    return data.mean(axis=1) if data.ndim == 2 else data


def onset_envelope(audio: np.ndarray, sr: int, rate: float = ENVELOPE_RATE) -> np.ndarray:
    """Rectified rise in log loudness, sampled `rate` times per second, z-normalised.

    Onsets (drum hits, note starts) survive different phone mics and volumes much better
    than the raw waveform does.
    """
    audio = np.asarray(audio, dtype=float)
    hop = max(1, int(round(sr / rate)))
    n = len(audio) // hop
    if n < 2:
        return np.zeros(max(n, 0))
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    log_rms = np.log(rms + 1e-6)
    env = np.maximum(np.diff(log_rms, prepend=log_rms[0]), 0.0)
    std = env.std()
    return (env - env.mean()) / std if std > 0 else env * 0.0


def estimate_offset(
    ref_env: np.ndarray,
    user_env: np.ndarray,
    rate: float = ENVELOPE_RATE,
    max_offset_s: float = 30.0,
) -> tuple[float, float]:
    """Return (offset_s, confidence).

    offset_s = how much later the music appears in the user's recording than in the reference.
    So reference time t corresponds to user-video time t + offset_s.
    confidence: how far the best match stands above the other candidate offsets
    (z-score). Values below ~5 mean the match is doubtful.
    """
    if len(ref_env) < 10 or len(user_env) < 10:
        return 0.0, 0.0
    corr = signal.correlate(user_env, ref_env, mode="full", method="fft")
    lags = signal.correlation_lags(len(user_env), len(ref_env), mode="full")
    # Normalise by overlap so partial overlaps at the edges aren't unfairly favoured or punished.
    # corr[k] = sum_n user[n + lag] * ref[n], over n with 0 <= n < len(ref) and 0 <= n + lag < len(user).
    overlap = np.minimum(len(ref_env), len(user_env) - lags) - np.maximum(0, -lags)
    overlap = np.clip(overlap, 1, None)
    min_overlap = 0.5 * min(len(ref_env), len(user_env))
    valid = (np.abs(lags) <= max_offset_s * rate) & (overlap >= min_overlap)
    if not valid.any():
        return 0.0, 0.0
    score = np.where(valid, corr / overlap, -np.inf)
    best = int(np.argmax(score))
    candidates = score[valid]
    spread = candidates.std()
    confidence = float((score[best] - np.median(candidates)) / spread) if spread > 0 else 0.0
    return float(lags[best] / rate), confidence


def beat_times(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Beat times in seconds using librosa, or None if librosa isn't installed."""
    try:
        import librosa  # type: ignore
    except ImportError:
        return None
    _, frames = librosa.beat.beat_track(y=np.asarray(audio, dtype=np.float32), sr=sr)
    return librosa.frames_to_time(frames, sr=sr)
