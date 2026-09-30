"""Reference-based mastering: match EQ, stereo width and loudness, then limit."""
from dataclasses import dataclass, asdict

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

MAX_EQ_DB = 9.0          # never boost/cut more than this at any frequency
CEILING_DB = -1.0        # output peak ceiling (dBFS)
LOUDNESS_RANGE = (-20.0, -6.0)  # clamp for the target LUFS taken from the reference
FFT = 8192


@dataclass
class Report:
    sample_rate: int
    input_lufs: float
    reference_lufs: float
    output_lufs: float
    input_peak_db: float
    output_peak_db: float
    stereo_width_ratio: float
    eq_freqs: list
    eq_gain_db: list
    input_spectrum_db: list
    reference_spectrum_db: list

    def to_dict(self):
        return asdict(self)


def load_audio(path, sr=None):
    """Load as float32 (samples, channels); optionally resample to `sr`."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[0] < file_sr // 2:
        raise ValueError("Audio is too short (need at least 0.5 s).")
    if data.shape[1] > 2:
        data = data[:, :2]
    if sr and sr != file_sr:
        g = np.gcd(int(sr), int(file_sr))
        data = signal.resample_poly(data, sr // g, file_sr // g, axis=0).astype("float32")
        file_sr = sr
    return data, file_sr


def _lufs(x, sr):
    return float(pyln.Meter(sr).integrated_loudness(x))


def _peak_db(x):
    return float(20 * np.log10(max(np.abs(x).max(), 1e-9)))


def _avg_spectrum(x, sr):
    """Long-term average power spectrum (dB) of the mono sum, ignoring silence."""
    mono = x.mean(axis=1)
    freqs, psd = signal.welch(mono, sr, nperseg=FFT, noverlap=FFT * 3 // 4)
    return freqs, 10 * np.log10(psd + 1e-14)


def _smooth_octave(freqs, db, fraction=3):
    """Smooth a dB spectrum on a log-frequency axis (1/`fraction` octave)."""
    out = np.empty_like(db)
    ratio = 2 ** (1 / (2 * fraction))
    power = 10 ** (db / 10)
    for i, f in enumerate(freqs):
        if f <= 0:
            out[i] = db[i]
            continue
        m = (freqs >= f / ratio) & (freqs <= f * ratio)
        out[i] = 10 * np.log10(power[m].mean() + 1e-20)
    return out


def _eq_curve(audio, ref, sr):
    freqs, a = _avg_spectrum(audio, sr)
    _, r = _avg_spectrum(ref, sr)
    a_s, r_s = _smooth_octave(freqs, a), _smooth_octave(freqs, r)
    diff = r_s - a_s
    # Overall level is handled by loudness matching, so centre the curve on the mids.
    mid = (freqs > 200) & (freqs < 4000)
    diff -= np.median(diff[mid])
    diff = np.clip(diff, -MAX_EQ_DB, MAX_EQ_DB)
    # Roll the correction off at the extremes where estimates are unreliable.
    diff[freqs < 30] *= 0.0
    top = min(20000, sr / 2 * 0.95)
    diff[freqs > top] = 0.0
    return freqs, diff, a_s, r_s


def _apply_eq(audio, freqs, gain_db, sr, taps=4095):
    nyq = sr / 2
    f = freqs / nyq
    g = 10 ** (gain_db / 20)
    fir = signal.firwin2(taps, f, g, window="hann")
    out = signal.fftconvolve(audio, fir[:, None], mode="same", axes=0)
    return out.astype("float32")


def _width(x):
    """Side/mid RMS ratio of a stereo signal."""
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    return float(np.sqrt((s ** 2).mean()) / (np.sqrt((m ** 2).mean()) + 1e-9))


def _match_width(audio, ref):
    if audio.shape[1] != 2 or ref.shape[1] != 2:
        return audio, 1.0
    wa, wr = _width(audio), _width(ref)
    k = float(np.clip(wr / (wa + 1e-9), 0.6, 1.6)) if wa > 1e-4 else 1.0
    m, s = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
    s = s * k
    return np.stack([m + s, m - s], axis=1).astype("float32"), k


def limit(x, sr, ceiling_db=CEILING_DB, lookahead_ms=6.0):
    """Look-ahead peak limiter with smooth gain reduction."""
    ceiling = 10 ** (ceiling_db / 20)
    n = max(int(sr * lookahead_ms / 1000), 1)
    peak = np.abs(x).max(axis=1)
    gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    gain = minimum_filter1d(gain, 2 * n + 1, mode="nearest")
    gain = uniform_filter1d(gain, n + 1, mode="nearest")  # attack ramp
    out = gain
    y = x * out[:, None]
    return np.clip(y, -ceiling, ceiling).astype("float32")


def _match_loudness(audio, sr, target_lufs):
    """Gain + limit iteratively until the limited output hits the target LUFS."""
    y = audio
    gain_db = target_lufs - _lufs(audio, sr)
    for _ in range(4):
        y = limit(audio * 10 ** (gain_db / 20), sr)
        err = target_lufs - _lufs(y, sr)
        if abs(err) < 0.3:
            break
        gain_db += err
    return y


def master(target_path, reference_path, output_path):
    audio, sr = load_audio(target_path)
    ref, _ = load_audio(reference_path, sr=sr)
    in_lufs, in_peak = _lufs(audio, sr), _peak_db(audio)
    ref_lufs = _lufs(ref, sr)

    freqs, eq, a_s, r_s = _eq_curve(audio, ref, sr)
    y = _apply_eq(audio, freqs, eq, sr)
    y, width_k = _match_width(y, ref)

    target = float(np.clip(ref_lufs, *LOUDNESS_RANGE))
    y = _match_loudness(y, sr, target)
    sf.write(output_path, y, sr, subtype="PCM_24")

    # decimate spectra to log-spaced points for the UI
    pts = np.geomspace(30, min(18000, sr / 2 * 0.9), 96)
    idx = np.searchsorted(freqs, pts)
    return Report(
        sample_rate=sr,
        input_lufs=round(in_lufs, 1),
        reference_lufs=round(ref_lufs, 1),
        output_lufs=round(_lufs(y, sr), 1),
        input_peak_db=round(in_peak, 1),
        output_peak_db=round(_peak_db(y), 1),
        stereo_width_ratio=round(width_k, 2),
        eq_freqs=[round(float(p)) for p in pts],
        eq_gain_db=[round(float(eq[i]), 2) for i in idx],
        input_spectrum_db=[round(float(a_s[i]), 1) for i in idx],
        reference_spectrum_db=[round(float(r_s[i]), 1) for i in idx],
    )
