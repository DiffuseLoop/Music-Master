import numpy as np
import soundfile as sf

from app import mastering
from app.spotify import SpotifyError, parse_track_id
import pytest

SR = 44100


def _noise(seconds, tilt, amp, seed):
    """Stereo noise with a spectral tilt (higher tilt = darker)."""
    rng = np.random.default_rng(seed)
    n = SR * seconds
    out = []
    for _ in range(2):
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        spec /= np.maximum(f, 20) ** tilt
        x = np.fft.irfft(spec, n)
        out.append(x / np.abs(x).max() * amp)
    return np.stack(out, axis=1).astype("float32")


def test_master_matches_reference(tmp_path):
    sf.write(tmp_path / "t.wav", _noise(8, 0.9, 0.2, 1), SR)   # dark & quiet
    sf.write(tmp_path / "r.wav", _noise(8, 0.3, 0.4, 2), SR)   # bright & louder
    rep = mastering.master(tmp_path / "t.wav", tmp_path / "r.wav", tmp_path / "o.wav")
    out, _ = sf.read(tmp_path / "o.wav")
    assert abs(rep.output_lufs - rep.reference_lufs) < 1.0
    assert np.abs(out).max() <= 10 ** (-1.0 / 20) + 1e-4
    # a dark track matched to a bright one must get treble boosted
    assert max(rep.eq_gain_db[-20:]) > 3


def test_spotify_link_parsing():
    assert parse_track_id("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC?si=x") == "4uLU6hMCjMI75M1A2tKUQC"
    with pytest.raises(SpotifyError):
        parse_track_id("https://example.com")
