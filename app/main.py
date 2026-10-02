import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import mastering, spotify

MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = float(os.environ.get("MAX_TRACK_MINUTES", "4")) * 60  # memory bound on small hosts
JOBS = Path(tempfile.gettempdir()) / "music-master-jobs"
JOBS.mkdir(exist_ok=True)
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Music Master")


async def _save(upload: UploadFile, dest: Path):
    size = 0
    with dest.open("wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                raise HTTPException(413, "File too large (200 MB max).")
            f.write(chunk)


@app.get("/api/config")
def config():
    return {"spotify_search": spotify.search_enabled()}


@app.get("/api/spotify/search")
def spotify_search(q: str):
    try:
        return spotify.search(q)
    except spotify.SpotifyError as e:
        raise HTTPException(400, str(e))


@app.post("/api/master")
async def master(track: UploadFile = File(...), reference: UploadFile | None = File(None),
                 spotify_link: str = Form("")):
    if not reference and not spotify_link.strip():
        raise HTTPException(400, "Provide a reference file or a Spotify link.")
    job = uuid.uuid4().hex
    d = JOBS / job
    d.mkdir()
    try:
        await _save(track, d / "track")
        ref_name = reference.filename if reference else ""
        if reference:
            await _save(reference, d / "reference")
        else:
            try:
                ref_name, audio = spotify.resolve_preview(spotify_link.strip())
            except spotify.SpotifyError as e:
                raise HTTPException(400, str(e))
            (d / "reference").write_bytes(audio)
        try:
            report = mastering.master(d / "track", d / "reference", d / "mastered.wav", MAX_SECONDS)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(422, str(e) if "limit is" in str(e) else f"Couldn't read audio: {e}. Use WAV, FLAC, MP3 or OGG.")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    stem = Path(track.filename or "track").stem
    return JSONResponse({"job": job, "reference_name": ref_name, "filename": f"{stem}-mastered.wav",
                         "report": report.to_dict()})


@app.get("/api/download/{job}")
def download(job: str, name: str = "mastered.wav"):
    if not job.isalnum():
        raise HTTPException(404)
    p = JOBS / job / "mastered.wav"
    if not p.exists():
        raise HTTPException(404, "Not found (results are temporary).")
    return FileResponse(p, media_type="audio/wav", filename=Path(name).name)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
