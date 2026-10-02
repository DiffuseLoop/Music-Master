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


def load_audio(path, sr=None, max_seconds=None):
    """Load as float32 (samples, channels); optionally resample to `sr`."""
    info = sf.info(path)
    if max_seconds and info.duration > max_seconds:
        raise ValueError(f"Track is {info.duration / 60:.1f} min long; the limit is {max_seconds / 60:.0f} min.")
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
    """Integrated loudness (BS.1770-4), computed channel by channel to keep memory low."""
    meter = pyln.Meter(sr)
    step = int(round(0.1 * sr))             # 100 ms hop; a 400 ms gating block = 4 hops
    n_steps = x.shape[0] // step
    if n_steps < 4:
        return float(meter.integrated_loudness(x))
    energy = np.zeros(n_steps)
    for ch in range(x.shape[1]):
        y = x[:, ch].astype("float64")
        for name in ("high_shelf", "high_pass"):
            f = meter._filters[name]
            y = signal.lfilter(f.b, f.a, y)
        for i in range(0, n_steps, 1024):
            j = min(i + 1024, n_steps)
            energy[i:j] += (y[i * step:j * step].reshape(j - i, step) ** 2).mean(axis=1)
        del y
    z = np.convolve(energy, np.ones(4) / 4, mode="valid")  # per-block mean square, summed over channels
    loud = -0.691 + 10 * np.log10(z + 1e-20)
    keep = loud > -70
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(z[keep].mean()) - 10
    keep &= loud > rel
    if not keep.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(z[keep].mean()))


def _peak_db(x):
    return float(20 * np.log10(max(np.abs(x).max(), 1e-9)))


def _avg_spectrum(x, sr):
    """Long-term average power spectrum (dB) of the mono sum, ignoring silence."""
    chunk = sr * 20                       # average PSDs of 20 s pieces to bound memory
    total, weight = 0.0, 0
    for i in range(0, x.shape[0], chunk):
        mono = x[i:i + chunk].mean(axis=1, dtype="float64")
        if len(mono) < FFT:
            continue
        freqs, psd = signal.welch(mono, sr, nperseg=FFT, noverlap=FFT * 3 // 4)
        total = total + psd * len(mono)
        weight += len(mono)
    return freqs, 10 * np.log10(total / weight + 1e-14)


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
    half = taps // 2
    n = audio.shape[0]
    out = np.zeros_like(audio)
    block = 1 << 18                       # overlap-add in blocks to bound memory
    for ch in range(audio.shape[1]):
        for i in range(0, n, block):
            seg = signal.fftconvolve(audio[i:i + block, ch].astype("float64"), fir)
            lo, hi = i - half, i - half + len(seg)        # position in output of seg[0] / end
            a, b = max(lo, 0), min(hi, n)
            out[a:b, ch] += seg[a - lo:b - lo]
    return out


def _width(x):
    """Side/mid RMS ratio of a stereo signal."""
    m = (x[:, 0] + x[:, 1]) / 2
    sd = (x[:, 0] - x[:, 1]) / 2
    return float(np.sqrt((sd ** 2).mean(dtype="float64")) / (np.sqrt((m ** 2).mean(dtype="float64")) + 1e-9))


def _match_width(audio, ref):
    if audio.shape[1] != 2 or ref.shape[1] != 2:
        return audio, 1.0
    wa, wr = _width(audio), _width(ref)
    k = float(np.clip(wr / (wa + 1e-9), 0.6, 1.6)) if wa > 1e-4 else 1.0
    block = 1 << 20                       # in place, block by block
    for i in range(0, audio.shape[0], block):
        seg = audio[i:i + block]
        m = (seg[:, 0] + seg[:, 1]) / 2
        sd = (seg[:, 0] - seg[:, 1]) / 2 * k
        seg[:, 0], seg[:, 1] = m + sd, m - sd
    return audio, k


def limit(x, sr, ceiling_db=CEILING_DB, lookahead_ms=6.0, pre_gain=1.0):
    """Look-ahead peak limiter with smooth gain reduction. Returns a new float32 array."""
    ceiling = np.float32(10 ** (ceiling_db / 20))
    n = max(int(sr * lookahead_ms / 1000), 1)
    gain = np.empty(x.shape[0], dtype="float32")
    for i in range(0, x.shape[0], 1 << 20):
        gain[i:i + (1 << 20)] = np.abs(x[i:i + (1 << 20)]).max(axis=1)
    gain *= np.float32(pre_gain)
    np.maximum(gain, 1e-9, out=gain)
    np.divide(ceiling, gain, out=gain)
    np.minimum(gain, 1.0, out=gain)
    gain = minimum_filter1d(gain, 2 * n + 1, mode="nearest")
    gain = uniform_filter1d(gain, n + 1, mode="nearest")  # attack ramp
    y = x * (gain * np.float32(pre_gain))[:, None]
    np.clip(y, -ceiling, ceiling, out=y)
    return y


def _match_loudness(audio, sr, target_lufs):
    """Gain + limit iteratively until the limited output hits the target LUFS."""
    y = audio
    gain_db = target_lufs - _lufs(audio, sr)
    for _ in range(4):
        y = limit(audio, sr, pre_gain=10 ** (gain_db / 20))
        err = target_lufs - _lufs(y, sr)
        if abs(err) < 0.3:
            break
        gain_db += err
    return y


def master(target_path, reference_path, output_path, max_seconds=None):
    audio, sr = load_audio(target_path, max_seconds=max_seconds)
    ref, _ = load_audio(reference_path, sr=sr)
    in_lufs, in_peak = _lufs(audio, sr), _peak_db(audio)
    ref_lufs = _lufs(ref, sr)

    freqs, eq, a_s, r_s = _eq_curve(audio, ref, sr)
    y = _apply_eq(audio, freqs, eq, sr)
    del audio
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
