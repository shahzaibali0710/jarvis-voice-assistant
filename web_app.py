"""
Jarvis - local web UI (Flask).

Serves the HUD-style single page from templates/index.html and exposes a
tiny JSON API the frontend polls/calls:
  GET  /api/state           -> current status, conversation history, cpu/mic
                                info, Spotify connection status
  POST /api/talk             -> starts a mic-based interaction (the "Talk to
                                 Jarvis" button), skipping the wake word
  POST /api/message          -> {"text": "..."} starts a typed interaction,
                                 skipping audio entirely (cancels whatever
                                 is in flight rather than returning "busy")
  POST /api/settings         -> {"voice_enabled": bool, "speak_typed_replies":
                                 bool} -- persisted to settings.json
  POST /api/reset            -> clears conversation history ("New conversation")
  POST /api/cancel           -> interrupts whatever the current interaction is
                                 doing (listening/thinking/speaking)
  GET  /spotify/login        -> redirects to Spotify's authorization page
  GET  /spotify/callback     -> OAuth redirect target; exchanges the code for
                                 tokens, then redirects back to "/"
  GET  /health/login         -> redirects to Google's consent screen for the
                                 Google Health API (google_health_client.py)
  GET  /health/callback      -> OAuth redirect target; exchanges the code,
                                 stores the refresh token, redirects to "/"
  GET  /api/spotify/now_playing -> current track/artist/art/progress/volume,
                                    or {} if nothing is loaded
  POST /api/spotify/play        -> resume playback (Now Playing panel button)
  POST /api/spotify/pause       -> pause playback (Now Playing panel button)
  POST /api/spotify/next        -> skip to next track
  POST /api/spotify/previous    -> skip to previous track
  POST /api/spotify/volume      -> {"volume": 0-100} sets playback volume
  GET  /api/applications        -> logged job applications (job_tracker.py),
                                    most recently applied first
  POST /api/applications/<id>/status -> {"status": "..."} updates one
                                    application's status (UI panel dropdown)
  GET  /api/job_search_results  -> parsed results from the most recent
                                    search_jobs / job hunting mode search
  GET  /api/transit             -> next departures for the saved default
                                    stop (entur_client.py), for the TRANSIT
                                    panel; {"configured": false} if none set

Runs alongside the wake-word background loop, which keeps working via the
terminal exactly as before -- this just adds a browser window on top.
"""

import html
import secrets
import threading
import time
import webbrowser

from flask import Flask, jsonify, redirect, render_template, request

import spotify_client
import jarvis
import job_tracker
import hotkey
import entur_client
import google_health_client

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__)

_state = None
_pause_event = None
_pending_oauth_state = None
_pending_health_state = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    snap = _state.snapshot()
    snap["cpu_percent"] = psutil.cpu_percent(interval=None) if psutil else None
    snap["mic_active"] = snap["status"] == "Listening"
    snap["spotify_configured"] = spotify_client.is_configured()
    snap["spotify_connected"] = spotify_client.is_connected()
    snap["health_status"] = google_health_client.status()  # connected | not_connected | not_configured
    return jsonify(snap)


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    jarvis.cancel_event.set()
    return jsonify(ok=True)


@app.route("/spotify/login")
def spotify_login():
    global _pending_oauth_state
    if not spotify_client.is_configured():
        return (
            "Spotify isn't configured yet -- set SPOTIFY_CLIENT_ID and "
            "SPOTIFY_CLIENT_SECRET (from your app at "
            "developer.spotify.com/dashboard) and restart Jarvis.",
            400,
        )
    _pending_oauth_state = secrets.token_urlsafe(16)
    return redirect(spotify_client.get_authorize_url(_pending_oauth_state))


@app.route("/spotify/callback")
def spotify_callback():
    global _pending_oauth_state
    error = request.args.get("error")
    if error:
        return f"Spotify authorization failed: {html.escape(error)}", 400

    state = request.args.get("state")
    if not state or state != _pending_oauth_state:
        return "Spotify authorization failed: state mismatch. Please try again.", 400
    _pending_oauth_state = None

    code = request.args.get("code")
    if not code:
        return "Spotify authorization failed: no code returned.", 400

    try:
        spotify_client.exchange_code_for_tokens(code)
    except Exception as exc:
        return f"Spotify authorization failed: {html.escape(str(exc))}", 500

    return redirect("/")


def _health_message_page(title, body_html, status=200):
    page = (
        "<!doctype html><meta charset=utf-8><title>J.A.R.V.I.S. - Google Health</title>"
        "<body style=\"font:15px system-ui;background:#07070b;color:#ececf1;max-width:560px;"
        "margin:12vh auto;padding:0 20px;line-height:1.6\">"
        f"<h2 style=\"margin:0 0 12px\">{title}</h2>{body_html}"
        "<p style=\"margin-top:28px\"><a style=\"color:#a99fff\" href=\"/\">Back to Jarvis</a></p>"
    )
    return page, status


@app.route("/health/login")
def health_login():
    global _pending_health_state
    if not google_health_client.is_configured():
        return _health_message_page(
            "Google Health isn't configured yet",
            "<p>Set the <code>GOOGLE_HEALTH_CLIENT_ID</code> and <code>GOOGLE_HEALTH_CLIENT_SECRET</code> "
            "environment variables (from your OAuth client in Google Cloud Console) and restart Jarvis.</p>",
            400,
        )
    _pending_health_state = secrets.token_urlsafe(16)
    return redirect(google_health_client.get_authorize_url(_pending_health_state))


@app.route("/health/callback")
def health_callback():
    global _pending_health_state
    error = request.args.get("error")
    if error:
        reason = ("You declined access on Google's consent screen." if error == "access_denied"
                  else f"Google returned an error: {html.escape(error)}.")
        return _health_message_page("Google Health authorization failed", f"<p>{reason}</p>", 400)

    state = request.args.get("state")
    if not state or state != _pending_health_state:
        return _health_message_page("Google Health authorization failed",
                                    "<p>State mismatch. Please try connecting again from Jarvis.</p>", 400)
    _pending_health_state = None

    code = request.args.get("code")
    if not code:
        return _health_message_page("Google Health authorization failed", "<p>No authorization code returned.</p>", 400)

    try:
        missing = google_health_client.exchange_code_for_tokens(code)
    except google_health_client.HealthError as exc:
        return _health_message_page("Google Health authorization failed", f"<p>{html.escape(str(exc))}</p>", 502)
    except Exception as exc:
        return _health_message_page("Google Health authorization failed", f"<p>{html.escape(str(exc))}</p>", 500)

    if missing:
        names = html.escape(", ".join(s.rsplit("googlehealth.", 1)[-1] for s in missing))
        return _health_message_page(
            "Connected, with a missing permission",
            f"<p>Jarvis is connected, but you didn't grant: <b>{names}</b>. Anything that needs it "
            "will say so. To fix it, click Connect again and leave every box ticked.</p>",
        )
    return redirect("/")


@app.route("/api/spotify/now_playing")
def api_spotify_now_playing():
    return jsonify(spotify_client.get_now_playing() or {})


@app.route("/api/spotify/play", methods=["POST"])
def api_spotify_play():
    return jsonify(message=spotify_client.resume())


@app.route("/api/spotify/pause", methods=["POST"])
def api_spotify_pause():
    return jsonify(message=spotify_client.pause())


@app.route("/api/spotify/next", methods=["POST"])
def api_spotify_next():
    return jsonify(message=spotify_client.skip())


@app.route("/api/spotify/previous", methods=["POST"])
def api_spotify_previous():
    return jsonify(message=spotify_client.previous())


@app.route("/api/spotify/volume", methods=["POST"])
def api_spotify_volume():
    data = request.get_json(silent=True) or {}
    volume = data.get("volume")
    if volume is None:
        return jsonify(ok=False, reason="missing volume"), 400
    return jsonify(message=spotify_client.set_volume(volume))


@app.route("/api/talk", methods=["POST"])
def api_talk():
    if jarvis.interaction_lock.locked():
        return jsonify(ok=False, reason="busy"), 409

    def worker():
        _pause_event.set()
        time.sleep(0.15)  # give the wake-word loop a moment to release the mic
        try:
            jarvis.run_interaction(_state)
        finally:
            _pause_event.clear()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(ok=True)


@app.route("/api/message", methods=["POST"])
def api_message():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(ok=False, reason="empty"), 400

    if jarvis.interaction_lock.locked():
        # Keyboard-first: typing while Jarvis is still thinking/speaking means
        # "never mind that, here's the next thing" -- same as pressing Escape
        # -- rather than bouncing the message with a "busy" error. Give the
        # in-flight interaction a moment to unwind and release the lock.
        jarvis.cancel_event.set()
        deadline = time.time() + 3.0
        while jarvis.interaction_lock.locked() and time.time() < deadline:
            time.sleep(0.02)
        if jarvis.interaction_lock.locked():
            return jsonify(ok=False, reason="busy"), 409

    threading.Thread(
        target=jarvis.run_interaction, args=(_state,), kwargs={"typed_text": text},
        daemon=True,
    ).start()
    return jsonify(ok=True)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """{"voice_enabled": bool, "speak_typed_replies": bool} -- either or both."""
    data = request.get_json(silent=True) or {}
    voice = data.get("voice_enabled")
    speak = data.get("speak_typed_replies")
    if not all(v is None or isinstance(v, bool) for v in (voice, speak)):
        return jsonify(ok=False, reason="values must be booleans"), 400
    _state.set_preferences(voice_enabled=voice, speak_typed_replies=speak)
    return jsonify(ok=True)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _state.clear_history()
    return jsonify(ok=True)


@app.route("/api/applications")
def api_applications():
    return jsonify(job_tracker.list_applications())


@app.route("/api/applications/<entry_id>/status", methods=["POST"])
def api_update_application_status(entry_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status", "")
    ok, message = job_tracker.update_status_by_id(entry_id, status)
    return jsonify(ok=ok, message=message), (200 if ok else 400)


@app.route("/api/transit")
def api_transit():
    return jsonify(entur_client.panel_data())


@app.route("/api/job_search_results")
def api_job_search_results():
    return jsonify(_state.snapshot_job_search_results())


def run(state, pause_event, stop_event, host="127.0.0.1", port=5000):
    global _state, _pause_event
    _state = state
    _pause_event = pause_event

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}/")

    threading.Thread(target=open_browser, daemon=True).start()
    hotkey.start(state, stop_event)

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    finally:
        stop_event.set()
