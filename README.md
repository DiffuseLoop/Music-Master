# Music Master

Reference-based automatic mastering. Upload a track, give it a reference (a Spotify link or an uploaded file), and get a mastered WAV that matches the reference's:

- **Tonal balance** – smoothed long-term spectrum matching via a linear-phase EQ (capped at ±9 dB)
- **Stereo width** – side/mid ratio matching
- **Loudness** – integrated LUFS (ITU-R BS.1770) matching, with a look-ahead peak limiter at -1 dBFS

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Open http://localhost:8000. Tests: `pytest`.

## Spotify notes

Spotify doesn't allow downloading full tracks, so a Spotify reference uses the 30 s preview from the public embed player (best effort; it may be unavailable for some tracks). Uploading the full reference file gives the most accurate match.

To enable in-app search, set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` (developer.spotify.com). Pasting a link works without credentials.

Supported inputs: WAV, FLAC, MP3, OGG. Output: 24-bit WAV.
