"""
Spotify Web API client: OAuth2 Authorization Code flow with automatic
access-token refresh, plus simple playback controls (play/pause/resume/skip)
for Jarvis's Spotify tool.

Setup (one time): create an app at https://developer.spotify.com/dashboard,
set its Redirect URI to exactly REDIRECT_URI below, and set the
SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET environment variables to the
app's Client ID/Secret. Then visit /spotify/login in the Jarvis web UI once
to authorize.

The refresh token (the long-lived credential) is stored in the Windows
Credential Manager via `keyring`, never in a plaintext file. The access
token is short-lived and kept in memory only, re-derived from the refresh
token as needed -- nothing sensitive touches disk.
"""

import base64
import os
import time
from urllib.parse import urlencode

import httpx
import keyring

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = "http://127.0.0.1:5000/spotify/callback"
SCOPES = "user-modify-playback-state user-read-playback-state user-read-currently-playing"

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

_KEYRING_SERVICE = "jarvis-spotify"
_KEYRING_USER = "refresh_token"

_access_token = None
_access_token_expires_at = 0.0


def is_configured():
    """True once SPOTIFY_CLIENT_ID/SECRET are set (the developer.spotify.com app exists)."""
    return bool(CLIENT_ID and CLIENT_SECRET)


def is_connected():
    """True once the user has completed the /spotify/login authorization flow."""
    return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER) is not None


def get_authorize_url(state):
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _basic_auth_header():
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def exchange_code_for_tokens(code):
    """Trades an authorization code for an access + refresh token pair.
    Stores the refresh token in the OS credential store. Raises on failure."""
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Authorization": _basic_auth_header()},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, data["refresh_token"])

    global _access_token, _access_token_expires_at
    _access_token = data["access_token"]
    _access_token_expires_at = time.time() + data.get("expires_in", 3600) - 30


def disconnect():
    """Forgets the stored refresh token (does not revoke it on Spotify's side)."""
    global _access_token, _access_token_expires_at
    try:
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
    except keyring.errors.PasswordDeleteError:
        pass
    _access_token = None
    _access_token_expires_at = 0.0


def _refresh_access_token():
    refresh_token = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
    if not refresh_token:
        raise RuntimeError("Spotify is not connected yet")

    resp = httpx.post(
        TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={"Authorization": _basic_auth_header()},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    global _access_token, _access_token_expires_at
    _access_token = data["access_token"]
    _access_token_expires_at = time.time() + data.get("expires_in", 3600) - 30
    if "refresh_token" in data:  # Spotify occasionally rotates it
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, data["refresh_token"])


def _get_access_token():
    if _access_token is None or time.time() >= _access_token_expires_at:
        _refresh_access_token()
    return _access_token


def _auth_headers():
    return {"Authorization": f"Bearer {_get_access_token()}"}


def _friendly_error(resp):
    reason = ""
    try:
        reason = resp.json().get("error", {}).get("reason", "")
    except Exception:
        pass
    if resp.status_code == 403 and reason == "PREMIUM_REQUIRED":
        return (
            "Spotify says playback control requires Premium -- this account "
            "appears to be on the Free tier."
        )
    if resp.status_code == 404:
        return "No active Spotify device found. Open Spotify on a device first."
    return f"Spotify request failed (HTTP {resp.status_code})."


def _active_device():
    """Returns (device_id, is_already_active) for the best device to target,
    or (None, False) if no devices are available at all."""
    resp = httpx.get(f"{API_BASE}/me/player/devices", headers=_auth_headers(), timeout=10)
    resp.raise_for_status()
    devices = resp.json().get("devices", [])
    for device in devices:
        if device.get("is_active"):
            return device["id"], True
    if devices:
        return devices[0]["id"], False
    return None, False


def _not_connected_message():
    msg = "Spotify isn't connected. Ask the user to authorize it from the Jarvis web UI first."
    print(f"[Spotify] {msg}")
    return msg


def play(query):
    """Searches Spotify for `query` (song and/or artist name) and starts
    playback on the active (or first available) device. Returns a short
    human-readable result string -- never raises. Every failure is also
    printed to the console so it's clear whether a voice command failed at
    the Spotify API step vs. transcription/tool-calling."""
    if not is_connected():
        return _not_connected_message()
    try:
        print(f"[Spotify] searching for '{query}'...")
        search = httpx.get(
            f"{API_BASE}/search",
            params={"q": query, "type": "track,artist", "limit": 1},
            headers=_auth_headers(),
            timeout=10,
        )
        search.raise_for_status()
        data = search.json()
        track_items = data.get("tracks", {}).get("items", [])
        artist_items = data.get("artists", {}).get("items", [])

        device_id, is_active = _active_device()
        if device_id is None:
            msg = "No active Spotify device found. Open Spotify on a device first."
            print(f"[Spotify] {msg}")
            return msg

        params = {} if is_active else {"device_id": device_id}

        if track_items:
            track = track_items[0]
            body = {"uris": [track["uri"]]}
            label = f"{track['name']} by {track['artists'][0]['name']}"
        elif artist_items:
            artist = artist_items[0]
            body = {"context_uri": artist["uri"]}
            label = artist["name"]
        else:
            msg = f"Couldn't find anything on Spotify matching '{query}'."
            print(f"[Spotify] {msg}")
            return msg

        resp = httpx.put(
            f"{API_BASE}/me/player/play", params=params, json=body,
            headers=_auth_headers(), timeout=10,
        )
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] play request failed: {error} (HTTP {resp.status_code}: {resp.text[:200]})")
            return error
        print(f"[Spotify] now playing: {label}")
        return f"Now playing {label} on Spotify."
    except httpx.HTTPError as exc:
        print(f"[Spotify] play() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] play() error: {exc}")
        return str(exc)


def pause():
    if not is_connected():
        return _not_connected_message()
    try:
        resp = httpx.put(f"{API_BASE}/me/player/pause", headers=_auth_headers(), timeout=10)
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] pause failed: {error} (HTTP {resp.status_code})")
            return error
        return "Paused Spotify."
    except httpx.HTTPError as exc:
        print(f"[Spotify] pause() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] pause() error: {exc}")
        return str(exc)


def resume():
    if not is_connected():
        return _not_connected_message()
    try:
        resp = httpx.put(f"{API_BASE}/me/player/play", headers=_auth_headers(), timeout=10)
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] resume failed: {error} (HTTP {resp.status_code})")
            return error
        return "Resumed Spotify."
    except httpx.HTTPError as exc:
        print(f"[Spotify] resume() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] resume() error: {exc}")
        return str(exc)


def skip():
    if not is_connected():
        return _not_connected_message()
    try:
        resp = httpx.post(f"{API_BASE}/me/player/next", headers=_auth_headers(), timeout=10)
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] skip failed: {error} (HTTP {resp.status_code})")
            return error
        return "Skipped to the next track."
    except httpx.HTTPError as exc:
        print(f"[Spotify] skip() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] skip() error: {exc}")
        return str(exc)


def previous():
    if not is_connected():
        return _not_connected_message()
    try:
        resp = httpx.post(f"{API_BASE}/me/player/previous", headers=_auth_headers(), timeout=10)
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] previous failed: {error} (HTTP {resp.status_code})")
            return error
        return "Skipped to the previous track."
    except httpx.HTTPError as exc:
        print(f"[Spotify] previous() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] previous() error: {exc}")
        return str(exc)


def set_volume(percent):
    if not is_connected():
        return _not_connected_message()
    try:
        percent = max(0, min(100, int(percent)))
        resp = httpx.put(
            f"{API_BASE}/me/player/volume", params={"volume_percent": percent},
            headers=_auth_headers(), timeout=10,
        )
        if resp.status_code >= 400:
            error = _friendly_error(resp)
            print(f"[Spotify] set_volume({percent}) failed: {error} (HTTP {resp.status_code})")
            return error
        return f"Volume set to {percent} percent."
    except (TypeError, ValueError):
        return "That doesn't look like a valid volume -- give me a number from 0 to 100."
    except httpx.HTTPError as exc:
        print(f"[Spotify] set_volume() request error: {exc}")
        return f"Spotify request failed: {exc}"
    except RuntimeError as exc:
        print(f"[Spotify] set_volume() error: {exc}")
        return str(exc)


def get_now_playing():
    """Returns a dict describing current playback (track/artist/art/
    progress/volume), or None if nothing is loaded or Spotify isn't
    connected. Never raises -- errors are logged and treated as "nothing
    playing" so the UI panel just stays minimal rather than breaking."""
    if not is_connected():
        return None
    try:
        resp = httpx.get(f"{API_BASE}/me/player", headers=_auth_headers(), timeout=10)
        if resp.status_code == 204 or not resp.content:
            return None
        if resp.status_code >= 400:
            print(f"[Spotify] now-playing fetch failed: HTTP {resp.status_code}")
            return None
        data = resp.json()
        item = data.get("item")
        if not item:
            return None

        images = ((item.get("album") or {}).get("images")) or []
        art_url = None
        if images:
            art_url = images[1]["url"] if len(images) > 1 else images[0]["url"]

        device = data.get("device") or {}
        return {
            "is_playing": bool(data.get("is_playing")),
            "track": item.get("name"),
            "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
            "album_art": art_url,
            "progress_ms": data.get("progress_ms") or 0,
            "duration_ms": item.get("duration_ms") or 0,
            "volume_percent": device.get("volume_percent"),
        }
    except httpx.HTTPError as exc:
        print(f"[Spotify] get_now_playing() request error: {exc}")
        return None
    except RuntimeError as exc:
        print(f"[Spotify] get_now_playing() error: {exc}")
        return None
