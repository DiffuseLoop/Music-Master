"""Resolve a Spotify track to a reference audio clip.

Spotify does not offer full-track audio downloads, and the official `preview_url`
field was withdrawn for new API apps. We therefore read the 30-second preview that
Spotify's public embed player uses. This is best-effort: if it is unavailable the
caller should ask the user to upload the reference file instead.
"""
import base64
import json
import os
import re
import time

import httpx

TRACK_RE = re.compile(r"(?:open\.spotify\.com/(?:intl-[a-z]+/)?track/|spotify:track:)([A-Za-z0-9]{22})")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) MusicMaster/1.0"}


class SpotifyError(Exception):
    pass


def parse_track_id(link):
    m = TRACK_RE.search(link or "")
    if not m:
        raise SpotifyError("That doesn't look like a Spotify track link.")
    return m.group(1)


def _find_preview(obj):
    if isinstance(obj, dict):
        ap = obj.get("audioPreview")
        if isinstance(ap, dict) and ap.get("url"):
            return ap["url"]
        if isinstance(obj.get("preview_url"), str):
            return obj["preview_url"]
        for v in obj.values():
            r = _find_preview(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_preview(v)
            if r:
                return r
    return None


def resolve_preview(link, client=None):
    """Return (title, preview_audio_bytes) for a Spotify track link."""
    tid = parse_track_id(link)
    c = client or httpx.Client(timeout=20, follow_redirects=True, headers=UA)
    try:
        page = c.get(f"https://open.spotify.com/embed/track/{tid}")
        page.raise_for_status()
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page.text, re.S)
        if not m:
            raise SpotifyError("Couldn't read Spotify's track page.")
        data = json.loads(m.group(1))
        url = _find_preview(data)
        if not url:
            raise SpotifyError("Spotify has no audio preview for this track. Upload the reference file instead.")
        title = tid
        try:
            title = data["props"]["pageProps"]["state"]["data"]["entity"]["name"]
        except (KeyError, TypeError):
            pass
        audio = c.get(url)
        audio.raise_for_status()
        return title, audio.content
    except httpx.HTTPError as e:
        raise SpotifyError(f"Couldn't reach Spotify ({e.__class__.__name__}). Upload the reference file instead.")


_token = {"value": None, "exp": 0}


def search_enabled():
    return bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"))


def search(query, limit=8):
    """Search tracks via the official API (client-credentials; needs env vars)."""
    if not search_enabled():
        raise SpotifyError("Search needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET; paste a link instead.")
    with httpx.Client(timeout=20) as c:
        if time.time() > _token["exp"]:
            cred = base64.b64encode(
                f"{os.environ['SPOTIFY_CLIENT_ID']}:{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()).decode()
            r = c.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"},
                       headers={"Authorization": f"Basic {cred}"})
            if r.status_code != 200:
                raise SpotifyError("Spotify credentials were rejected.")
            _token.update(value=r.json()["access_token"], exp=time.time() + r.json()["expires_in"] - 60)
        r = c.get("https://api.spotify.com/v1/search", params={"q": query, "type": "track", "limit": limit},
                  headers={"Authorization": f"Bearer {_token['value']}"})
        if r.status_code != 200:
            raise SpotifyError("Spotify search failed.")
        return [{"id": t["id"], "name": t["name"], "artists": ", ".join(a["name"] for a in t["artists"]),
                 "url": t["external_urls"]["spotify"]} for t in r.json()["tracks"]["items"]]
