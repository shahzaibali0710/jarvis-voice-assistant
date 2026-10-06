"""
Google Health API client (the successor to the legacy Fitbit Web API, which is
shut down on 2026-10-30): OAuth 2.0 authorization-code flow with automatic
access-token refresh, plus sleep / workout / weekly-activity summaries for
Jarvis's health tools.

Everything here follows Google's current documentation and its machine-readable
API description (https://health.googleapis.com/$discovery/rest?version=v4):

  * Base URL      https://health.googleapis.com/v4/users/me/...
  * Auth          standard Google OAuth 2.0 (accounts.google.com / oauth2.googleapis.com)
  * Read scopes   https://www.googleapis.com/auth/googlehealth.sleep.readonly
                  https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly
  * Sleep         GET  users/me/dataTypes/sleep/dataPoints            (filter on civil_end_time)
  * Workouts      GET  users/me/dataTypes/exercise/dataPoints         (filter on civil_start_time)
  * Weekly rollup POST users/me/dataTypes/{steps,distance,total-calories,
                       active-zone-minutes}/dataPoints:dailyRollUp

None of the old Fitbit endpoints (api.fitbit.com, /1.2/user/-/sleep/...) are used.

Setup (one time) is in the README section of the hand-off message; in short:
create an OAuth "Web application" client in Google Cloud, register
REDIRECT_URI below exactly, and put GOOGLE_HEALTH_CLIENT_ID /
GOOGLE_HEALTH_CLIENT_SECRET in your *environment variables* (never in chat or
in a file). Then click CONNECT in the Jarvis UI once.

Like spotify_client.py, the refresh token (the long-lived credential) lives in
the Windows Credential Manager via `keyring`; the access token is short-lived
and held in memory only.

Until credentials exist, the summary functions return clearly-labelled SAMPLE
data (run through the very same parsing code as real responses), so the tools
and UI can be exercised end to end. Run `python google_health_client.py check`
after connecting to verify live responses against these assumptions.
"""

import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import keyring

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://health.googleapis.com/v4"

_SCOPE_PREFIX = "https://www.googleapis.com/auth/googlehealth."
SCOPE_SLEEP = _SCOPE_PREFIX + "sleep.readonly"
SCOPE_ACTIVITY = _SCOPE_PREFIX + "activity_and_fitness.readonly"
# Least privilege: read-only, and only the two categories the tools need
# (heart-rate-zone, active-zone-minutes, calories and distance all live under
# activity_and_fitness; nothing here writes or touches other health data).
SCOPES = [SCOPE_SLEEP, SCOPE_ACTIVITY]

# Must match, character for character, the URI registered on the OAuth client.
# Google allows plain-http loopback redirects for "Web application" clients.
REDIRECT_URI = os.environ.get("GOOGLE_HEALTH_REDIRECT_URI", "http://127.0.0.1:5000/health/callback")

_KEYRING_SERVICE = "jarvis-google-health"
_KEYRING_USER = "refresh_token"

_access_token = None
_access_token_expires_at = 0.0

SAMPLE_PREFIX = (
    "[SAMPLE DATA -- Google Health is not connected yet, so these figures are "
    "placeholders, not the user's real numbers. Say so plainly.] "
)

# Every health tool answer starts with exactly one of these two labels, so the
# source (real account data vs made-up placeholders) is part of the text Claude
# relays and can't be silently dropped.
LIVE_PREFIX = "[LIVE DATA from the user's Google Health account, fetched {when}] "

# "Today" and "last night" are the user's own calendar days, not the PC's or
# UTC's. Google's date filters (civil_end_time etc.) are in the local time at
# the place the data was recorded, so the windows below are built from dates in
# this zone.
TIMEZONE_NAME = os.environ.get("GOOGLE_HEALTH_TIMEZONE", "Europe/Oslo")
_tz_warned = False


def _tz():
    global _tz_warned
    try:
        return ZoneInfo(TIMEZONE_NAME)
    except (ZoneInfoNotFoundError, ValueError):
        if not _tz_warned:
            _tz_warned = True
            print(f"[Google Health] time zone '{TIMEZONE_NAME}' unavailable (pip install tzdata); "
                  f"falling back to this PC's local time zone")
        return None


def _now():
    return datetime.now(_tz()) if _tz() else datetime.now().astimezone()


def _today():
    return _now().date()


class HealthError(Exception):
    """A problem with a message that is safe to speak aloud."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


# ── Credentials / connection state ───────────────────────────────────────────

def _client_id():
    return os.environ.get("GOOGLE_HEALTH_CLIENT_ID")


def _client_secret():
    return os.environ.get("GOOGLE_HEALTH_CLIENT_SECRET")


def is_configured():
    """True once GOOGLE_HEALTH_CLIENT_ID/SECRET are set (the Cloud OAuth client exists)."""
    return bool(_client_id() and _client_secret())


def is_connected():
    """True once the user has completed the /health/login authorization flow."""
    try:
        return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER) is not None
    except Exception as exc:  # a broken credential store shouldn't crash the UI
        print(f"[Google Health] credential store unavailable: {exc}")
        return False


def can_use_live():
    """Live data needs BOTH the saved authorization and the OAuth client
    credentials in this process's environment (they're needed to refresh the
    access token). A token without credentials -- e.g. Jarvis started from a
    shell that lacks the env vars -- can't be used."""
    return is_configured() and is_connected()


def status():
    """'connected' | 'not_connected' | 'not_configured' -- for the UI row."""
    if not is_configured():
        return "not_configured"
    return "connected" if is_connected() else "not_connected"


# ── OAuth 2.0 ────────────────────────────────────────────────────────────────

def get_authorize_url(state):
    params = {
        "client_id": _client_id(),
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        # offline => Google returns a refresh token; consent => it does so every
        # time the user (re)connects, not just the very first authorization.
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _token_error(resp):
    try:
        body = resp.json()
        return f"{body.get('error', 'error')}: {body.get('error_description', '')}".strip(": ")
    except Exception:
        return f"HTTP {resp.status_code}"


def exchange_code_for_tokens(code):
    """Trades an authorization code for tokens and stores the refresh token in
    the OS credential store. Returns the list of required scopes the user did
    NOT grant (partial consent is possible). Raises HealthError on failure."""
    resp = httpx.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if resp.status_code >= 400:
        raise HealthError(f"Google rejected the authorization code ({_token_error(resp)}).", resp.status_code)
    data = resp.json()
    if "refresh_token" not in data:
        raise HealthError(
            "Google didn't return a refresh token. Remove Jarvis from "
            "https://myaccount.google.com/permissions and connect again."
        )
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, data["refresh_token"])

    global _access_token, _access_token_expires_at
    _access_token = data["access_token"]
    _access_token_expires_at = time.time() + data.get("expires_in", 3600) - 30

    granted = data.get("scope", "").split()
    return [s for s in SCOPES if s not in granted]


def disconnect():
    """Forgets the stored refresh token (does not revoke it on Google's side)."""
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
        raise HealthError("Google Health isn't connected yet.")

    resp = httpx.post(
        TOKEN_URL,
        data={
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    if resp.status_code >= 400:
        detail = _token_error(resp)
        print(f"[Google Health] token refresh failed: {detail}")
        if "invalid_grant" in detail:
            # Expired or revoked. While the Cloud project is in "Testing" mode this
            # happens every 7 days. Forget the dead token so the UI offers CONNECT.
            disconnect()
            raise HealthError(
                "Google Health authorization expired, so I've disconnected it. "
                "Click Connect in the Jarvis UI to authorize again."
            )
        raise HealthError("I couldn't refresh the Google Health authorization.", resp.status_code)

    data = resp.json()
    global _access_token, _access_token_expires_at
    _access_token = data["access_token"]
    _access_token_expires_at = time.time() + data.get("expires_in", 3600) - 30
    if "refresh_token" in data:  # Google normally doesn't rotate it, but handle it
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, data["refresh_token"])


def _get_access_token():
    if _access_token is None or time.time() >= _access_token_expires_at:
        _refresh_access_token()
    return _access_token


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _raise_for_response(resp):
    if resp.status_code < 400:
        return
    text = resp.text[:600]
    print(f"[Google Health] HTTP {resp.status_code}: {text}")
    if "ACCOUNT_NOT_LINKED" in text:
        raise HealthError(
            "Your Google account isn't linked to Google Health yet. Open the Google "
            "Health app on your phone, sign in, then try again.",
            resp.status_code,
        )
    if resp.status_code == 403:
        raise HealthError(
            "Google Health refused access, probably because a permission wasn't "
            "granted. Reconnect Health in the Jarvis UI and tick every box.",
            403,
        )
    if resp.status_code == 429:
        raise HealthError("Google Health is rate-limiting me. Try again in a minute.", 429)
    raise HealthError(f"Google Health request failed (HTTP {resp.status_code}).", resp.status_code)


def _api(method, path, params=None, body=None):
    resp = None
    for attempt in (1, 2):
        resp = httpx.request(
            method,
            f"{API_BASE}/{path}",
            params=params,
            json=body,
            headers={"Authorization": f"Bearer {_get_access_token()}"},
            timeout=20,
        )
        if resp.status_code == 401 and attempt == 1:
            global _access_token
            _access_token = None  # stale/revoked access token: refresh once and retry
            continue
        break
    _raise_for_response(resp)
    return resp.json()


def _list_points(data_type, filter_expr, page_size=25, max_pages=3, reconcile=False):
    """All data points of `data_type` matching an AIP-160 filter (paged).

    reconcile=True uses the `dataPoints:reconcile` method, which Google documents
    as merging data points from multiple sources into a single stream (so the
    same night recorded by a watch and a phone isn't counted twice). Its items
    have the same `sleep` / `exercise` payload as `list`, wrapped with a
    `dataPointName` instead of `name`."""
    points, token = [], None
    suffix = "dataPoints:reconcile" if reconcile else "dataPoints"
    for _ in range(max_pages):
        params = {"pageSize": page_size, "filter": filter_expr}
        if token:
            params["pageToken"] = token
        data = _api("GET", f"users/me/dataTypes/{data_type}/{suffix}", params=params)
        points.extend(data.get("dataPoints", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return points


# The reference page shows civil times in two shapes (flat, and nested
# {date, time}); the API description says nested. Use nested, and if the API
# answers 400 to it, remember and use the flat form instead.
_rollup_flat = False


def _civil(day, flat):
    if flat:
        return {"year": day.year, "month": day.month, "day": day.day, "hours": 0, "minutes": 0, "seconds": 0}
    return {
        "date": {"year": day.year, "month": day.month, "day": day.day},
        "time": {"hours": 0, "minutes": 0, "seconds": 0},
    }


def _daily_rollup(data_type, start_day, end_day_exclusive):
    """One rollup bucket per civil day in [start_day, end_day_exclusive)."""
    global _rollup_flat

    def body(flat):
        return {
            "range": {"start": _civil(start_day, flat), "end": _civil(end_day_exclusive, flat)},
            "windowSizeDays": 1,
        }

    path = f"users/me/dataTypes/{data_type}/dataPoints:dailyRollUp"
    try:
        data = _api("POST", path, body=body(_rollup_flat))
    except HealthError as exc:
        if exc.status != 400:
            raise
        data = _api("POST", path, body=body(not _rollup_flat))
        _rollup_flat = not _rollup_flat
    return data.get("rollupDataPoints", [])


# ── Small parsing helpers (API quirks: int64 fields are strings; durations are "123.5s") ──

_FRACTION = re.compile(r"(\.\d{6})\d+")


def _secs(duration):
    if not duration:
        return 0.0
    return float(str(duration).rstrip("s"))


def _num(value, default=0):
    if value is None or value == "":
        return default
    return float(value)


def _parse_ts(ts):
    return datetime.fromisoformat(_FRACTION.sub(r"\1", ts).replace("Z", "+00:00"))


def _local(ts, utc_offset):
    """Wall-clock datetime (naive) at the place/time the user was, from a UTC
    timestamp plus the record's own UTC offset."""
    return (_parse_ts(ts) + timedelta(seconds=_secs(utc_offset))).replace(tzinfo=None)


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _hm(minutes):
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    parts = []
    if hours:
        parts.append(_plural(hours, "hour"))
    if mins or not parts:
        parts.append(_plural(mins, "minute"))
    return " ".join(parts)


def _clock(dt):
    return dt.strftime("%H:%M")


def _parse_day(text):
    if not text:
        return None
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        raise HealthError(f"I couldn't read '{text}' as a date. Use YYYY-MM-DD.")


# ── Sleep ────────────────────────────────────────────────────────────────────

def parse_sleep(point):
    s = point["sleep"]
    iv = s["interval"]
    summary = s.get("summary", {})
    meta = s.get("metadata", {})
    stages = {x["type"]: _num(x.get("minutes")) for x in summary.get("stagesSummary", [])}
    start = _local(iv["startTime"], iv.get("startUtcOffset"))
    end = _local(iv["endTime"], iv.get("endUtcOffset"))
    return {
        "start": start,                      # local wall-clock (naive), from the record's own UTC offset
        "end": end,
        "start_utc": _parse_ts(iv["startTime"]),
        "end_utc": _parse_ts(iv["endTime"]),
        "wake_day": end.date(),              # the civil day the session ENDS on
        "asleep_min": _num(summary.get("minutesAsleep")),
        "awake_min": _num(summary.get("minutesAwake")),
        "in_bed_min": _num(summary.get("minutesInSleepPeriod")),
        "to_sleep_min": _num(summary.get("minutesToFallAsleep"), None),
        "stages": stages,
        "main": bool(meta.get("mainSleep")),
        "nap": bool(meta.get("nap")),
        "has_stages": s.get("type") == "STAGES" and any(k in stages for k in ("DEEP", "REM", "LIGHT")),
        "processed": meta.get("processed", True),
    }


# A night often comes back from the API as SEVERAL sessions (e.g. you wake up,
# get up for a bit, go back to sleep) -- and Google may label one of them a
# "nap" even when it's part of the same night. Picking just one session (the
# bug this replaced) under-counts: live data showed 130 min + 351 min = 8 h 1 min,
# which is what the Google Health app reports, while only the 351 min session
# had been used. So sessions are chained into "sleep blocks": a session that
# starts within SLEEP_BLOCK_GAP_MIN of the previous one's end joins its block.
# A block belongs to the civil day it ENDS on, so sleep that crosses midnight is
# neither cut off nor split between two days.
SLEEP_BLOCK_GAP_MIN = 120


def group_sleep_blocks(sessions):
    """Chains sessions (from parse_sleep) into blocks. Each block sums the
    sessions' minutes; `asleep_min` is time actually asleep, `in_bed_min` is the
    sum of the sessions' sleep periods (bed time, awake time included, the gaps
    BETWEEN sessions excluded)."""
    blocks = []
    for s in sorted(sessions, key=lambda x: x["start_utc"]):
        gap = None
        if blocks:
            gap = (s["start_utc"] - blocks[-1]["end_utc"]).total_seconds() / 60
        if blocks and gap <= SLEEP_BLOCK_GAP_MIN:
            b = blocks[-1]
            if gap < -10:
                print(f"[Google Health] warning: sleep sessions overlap by {-gap:.0f} min; "
                      f"totals may double-count (reconcile should have merged them)")
            b["gap_min"] += max(0.0, gap)
            b["sessions"].append(s)
            if s["end_utc"] > b["end_utc"]:
                b["end_utc"], b["end"] = s["end_utc"], s["end"]
        else:
            blocks.append({"sessions": [s], "gap_min": 0.0, "start": s["start"], "start_utc": s["start_utc"],
                           "end": s["end"], "end_utc": s["end_utc"]})
    for b in blocks:
        ss = b["sessions"]
        b["wake_day"] = b["end"].date()
        b["asleep_min"] = sum(s["asleep_min"] for s in ss)
        b["awake_min"] = sum(s["awake_min"] for s in ss)
        b["in_bed_min"] = sum(s["in_bed_min"] for s in ss)
        b["to_sleep_min"] = ss[0]["to_sleep_min"]
        b["has_stages"] = any(s["has_stages"] for s in ss)
        stages = {}
        for s in ss:
            for k, v in s["stages"].items():
                stages[k] = stages.get(k, 0) + v
        b["stages"] = stages
    return blocks


def pick_sleep_block(blocks, wake_day=None):
    """(main block, other blocks that day). With no day given: the most recent
    day that has any sleep. The main block is the one with the most time asleep;
    any others are separate sleeps (naps)."""
    pool = [b for b in blocks if wake_day is None or b["wake_day"] == wake_day]
    if not pool:
        return None, []
    if wake_day is None:
        latest = max(b["wake_day"] for b in pool)
        pool = [b for b in pool if b["wake_day"] == latest]
    main = max(pool, key=lambda b: b["asleep_min"])
    return main, sorted((b for b in pool if b is not main), key=lambda b: b["start_utc"])


def _day_phrase(day):
    today = _today()
    if day == today:
        return "today"
    if day == today - timedelta(days=1):
        return "yesterday"
    return day.strftime("%A %d %B")


def format_sleep(main, others=()):
    if main is None:
        return "I couldn't find any sleep data for that night."
    n = len(main["sessions"])
    text = (f"Total sleep ending {_day_phrase(main['wake_day'])} at {_clock(main['end'])}: "
            f"{_hm(main['asleep_min'])} actually asleep")
    if main["in_bed_min"]:
        text += f" ({_hm(main['in_bed_min'])} in bed, {_hm(main['awake_min'])} awake in bed)"
    text += "."
    if n == 1:
        text += f" Bedtime {_clock(main['start'])}, woke {_clock(main['end'])}."
    else:
        parts = []
        for s in main["sessions"]:
            label = f"{_clock(s['start'])}-{_clock(s['end'])} ({_hm(s['asleep_min'])} asleep"
            if s["nap"] and not s["main"]:
                label += "; Google tags this one a nap"
            parts.append(label + ")")
        text += (f" This was {n} sessions only {_hm(main['gap_min'])} apart, counted as one sleep and added "
                 f"together: " + "; ".join(parts) + f". Overall from {_clock(main['start'])} to {_clock(main['end'])}.")
    if main["in_bed_min"]:
        text += f" Efficiency {round(100 * main['asleep_min'] / main['in_bed_min'])}% (time asleep divided by time in bed)."
    if main["has_stages"]:
        names = [("DEEP", "deep"), ("REM", "REM"), ("LIGHT", "light")]
        bits = [f"{label} {_hm(main['stages'][key])}" for key, label in names if key in main["stages"]]
        if main["awake_min"]:
            bits.append(f"awake {_hm(main['awake_min'])}")
        text += " Stages (summed over all sessions): " + ", ".join(bits) + "."
    else:
        text += " (No sleep-stage breakdown for this sleep.)"
    if main["to_sleep_min"]:
        text += f" It took about {_hm(main['to_sleep_min'])} to fall asleep."
    for b in others:
        text += (f" Separately that day there was another sleep at {_clock(b['start'])}-{_clock(b['end'])}: "
                 f"{_hm(b['asleep_min'])} asleep (not included in the total above).")
    # The API has no sleep score field at all (checked against its schema).
    text += " Google Health doesn't provide a sleep score, so there isn't one to report."
    return text


# ── Workouts ─────────────────────────────────────────────────────────────────

_ZONE_ORDER = [("lightTime", "light"), ("moderateTime", "moderate"), ("vigorousTime", "vigorous"), ("peakTime", "peak")]


def parse_exercise(point):
    ex = point["exercise"]
    iv = ex["interval"]
    m = ex.get("metricsSummary", {})
    start = _local(iv["startTime"], iv.get("startUtcOffset"))
    end = _local(iv["endTime"], iv.get("endUtcOffset"))
    active = _secs(ex.get("activeDuration")) or (end - start).total_seconds()
    zones_raw = m.get("heartRateZoneDurations") or {}
    name = ex.get("displayName") or ex.get("exerciseType", "workout").replace("_", " ").lower()
    return {
        "name": name,
        "start": start,
        "minutes": active / 60,
        "kcal": _num(m.get("caloriesKcal"), None),
        "km": _num(m.get("distanceMillimeters"), None) / 1_000_000 if m.get("distanceMillimeters") else None,
        "avg_hr": int(_num(m.get("averageHeartRateBeatsPerMinute"))) or None,
        "azm": int(_num(m.get("activeZoneMinutes"))) or None,
        "zones": {label: _secs(zones_raw[key]) / 60 for key, label in _ZONE_ORDER if key in zones_raw},
    }


def format_workout(w):
    day = "Today" if w["start"].date() == _today() else w["start"].strftime("%A %d %B")
    text = f"{day} {_clock(w['start'])}: {w['name'].lower()}, {_hm(w['minutes'])}"
    extras = []
    if w["km"]:
        extras.append(f"{w['km']:.1f} km")
    if w["kcal"]:
        extras.append(f"{round(w['kcal'])} kcal")
    if w["avg_hr"]:
        extras.append(f"average heart rate {w['avg_hr']}")
    if extras:
        text += ", " + ", ".join(extras)
    text += "."
    zone_bits = [f"{label} {round(mins)} min" for label, mins in w["zones"].items() if round(mins) > 0]
    if zone_bits:
        text += " Time in heart rate zones: " + ", ".join(zone_bits) + "."
    if w["azm"]:
        text += f" Active Zone Minutes: {w['azm']}."
    return text


# ── Weekly rollup ────────────────────────────────────────────────────────────

def _bucket_day(p):
    """The civil date of a dailyRollUp bucket (None for sample buckets, which carry no date)."""
    d = (p.get("civilStartTime") or {}).get("date")
    if not d:
        return None
    return date(int(d["year"]), int(d["month"]), int(d["day"]))


def _sum_rollup(points, getter):
    total = 0.0
    for p in points:
        total += getter(p)
    return total


def build_weekly(steps, distance, calories, azm, exercises, sleep_blocks, days):
    """Combines the raw per-type results into one 7-day summary dict.
    `sleep_blocks` are the merged sleeps from group_sleep_blocks."""
    week = {"days": days}
    today = _today()

    def is_complete_day(p):
        d = _bucket_day(p)
        return d is not None and d != today          # today is still in progress

    if steps:
        week["steps"] = int(_sum_rollup(steps, lambda p: _num(p.get("steps", {}).get("countSum"))))
        full = [p for p in steps if is_complete_day(p)]
        if full:
            week["steps_per_full_day"] = sum(_num(p.get("steps", {}).get("countSum")) for p in full) / len(full)
    if distance:
        week["km"] = _sum_rollup(distance, lambda p: _num(p.get("distance", {}).get("millimetersSum"))) / 1_000_000
    # Calories and Active Zone Minutes come from heart-rate data, which may only
    # exist for some of the days (new watch, watch not worn). Only average them
    # when every day has data; otherwise say how many days are covered.
    if calories:
        buckets = [p.get("totalCalories", {}).get("kcalSum") for p in calories]
        buckets = [_num(b) for b in buckets if b is not None]
        if buckets:
            week["kcal_days"] = len(buckets)
            week["kcal_per_day"] = sum(buckets) / len(buckets)
    if azm:
        def zone_total(p):
            v = p.get("activeZoneMinutes", {})
            return sum(_num(v.get(k)) for k in ("sumInFatBurnHeartZone", "sumInCardioHeartZone", "sumInPeakHeartZone"))
        week["azm"] = int(_sum_rollup(azm, zone_total))
        week["azm_days"] = len(azm)
    if exercises:
        week["workouts"] = len(exercises)
        week["workout_minutes"] = sum(w["minutes"] for w in exercises)
    # One night per civil day: that day's biggest merged sleep (naps excluded),
    # with all of that night's sessions already summed inside the block.
    nights = {}
    for b in sleep_blocks:
        best = nights.get(b["wake_day"])
        if best is None or b["asleep_min"] > best["asleep_min"]:
            nights[b["wake_day"]] = b
    nights = [n for n in nights.values() if n["asleep_min"] > 0]
    if nights:
        week["sleep_nights"] = len(nights)
        week["sleep_avg_min"] = sum(n["asleep_min"] for n in nights) / len(nights)
    return week


def format_weekly(w):
    n = w["days"]
    parts = []
    if "steps" in w:
        step_text = f"{w['steps']:,} steps"
        if "steps_per_full_day" in w:
            step_text += f" (about {round(w['steps_per_full_day']):,} on a full day; today isn't over yet)"
        parts.append(step_text)
    if "km" in w:
        parts.append(f"{w['km']:.1f} km covered")
    partial = []   # heart-rate-based figures that don't cover the whole week
    if "kcal_per_day" in w:
        if w["kcal_days"] >= n:
            parts.append(f"about {round(w['kcal_per_day']):,} kcal burned a day")
        else:
            partial.append(f"calories ({round(w['kcal_per_day']):,} kcal a day on those {w['kcal_days']} days, today partial)")
    if "azm" in w:
        if w["azm_days"] >= n:
            parts.append(f"{w['azm']} Active Zone Minutes")
        else:
            partial.append(f"Active Zone Minutes ({w['azm']} in total)")
    text = f"Last {n} days: " + ", ".join(parts) + "." if parts else f"No activity data found for the last {n} days."
    if partial:
        days_cov = min(w.get("kcal_days", n), w.get("azm_days", n))
        text += (f" Heart-rate-based figures only cover about {_plural(days_cov, 'day')} of the {n}, so there's no "
                 f"full-week average for them: " + "; ".join(partial) + ".")
    if "workouts" in w:
        text += f" {_plural(w['workouts'], 'workout')}, {_hm(w['workout_minutes'])} in total."
    else:
        text += " No workouts logged."
    if "sleep_avg_min" in w:
        if w["sleep_nights"] >= n - 1:
            text += f" Sleep averaged {_hm(w['sleep_avg_min'])} over {_plural(w['sleep_nights'], 'night')}."
        else:
            text += (f" Sleep data only covers {_plural(w['sleep_nights'], 'night')} of the {n} days "
                     f"({_hm(w['sleep_avg_min'])} asleep on average), so it isn't a full-week average.")
    return text


# ── Placeholder data (same shapes as real API responses) ─────────────────────

def _sample_sleep_points():
    today = _today()
    points = []
    for back, asleep, deep, rem in [(0, 432, 78, 101), (1, 405, 64, 92), (2, 468, 88, 110), (3, 390, 55, 85),
                                    (4, 441, 80, 99), (5, 420, 70, 96), (6, 455, 84, 105)]:
        end_day = today - timedelta(days=back)
        end_utc = datetime(end_day.year, end_day.month, end_day.day, 4, 42)      # 06:42 at +02:00
        in_bed = asleep + 38
        start_utc = end_utc - timedelta(minutes=in_bed)
        light = asleep - deep - rem
        points.append({
            "name": f"users/me/dataTypes/sleep/dataPoints/sample{back}",
            "sleep": {
                "interval": {
                    "startTime": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "startUtcOffset": "7200s",
                    "endTime": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "endUtcOffset": "7200s",
                },
                "type": "STAGES",
                "summary": {
                    "minutesAsleep": str(asleep), "minutesAwake": "38", "minutesInSleepPeriod": str(in_bed),
                    "minutesToFallAsleep": "12",
                    "stagesSummary": [
                        {"type": "DEEP", "minutes": str(deep), "count": "4"},
                        {"type": "REM", "minutes": str(rem), "count": "5"},
                        {"type": "LIGHT", "minutes": str(light), "count": "27"},
                        {"type": "AWAKE", "minutes": "38", "count": "19"},
                    ],
                },
                "metadata": {"mainSleep": True, "processed": True, "stagesStatus": "SUCCEEDED"},
            },
        })
        if back == 0:
            # Today's sample night is split in two (an earlier session Google tags as
            # a nap, then the main one 16 min later), like real data, so the sample
            # path exercises the same merge-and-sum logic.
            early_end = start_utc - timedelta(minutes=16)
            early_start = early_end - timedelta(minutes=105)
            points.append({
                "name": "users/me/dataTypes/sleep/dataPoints/sample0b",
                "sleep": {
                    "interval": {
                        "startTime": early_start.strftime("%Y-%m-%dT%H:%M:%SZ"), "startUtcOffset": "7200s",
                        "endTime": early_end.strftime("%Y-%m-%dT%H:%M:%SZ"), "endUtcOffset": "7200s",
                    },
                    "type": "STAGES",
                    "summary": {
                        "minutesAsleep": "95", "minutesAwake": "10", "minutesInSleepPeriod": "105",
                        "minutesToFallAsleep": "0",
                        "stagesSummary": [
                            {"type": "DEEP", "minutes": "15", "count": "1"}, {"type": "REM", "minutes": "20", "count": "1"},
                            {"type": "LIGHT", "minutes": "60", "count": "3"}, {"type": "AWAKE", "minutes": "10", "count": "2"},
                        ],
                    },
                    "metadata": {"nap": True, "processed": True, "stagesStatus": "SUCCEEDED"},
                },
            })
    return points


def _sample_exercise_points():
    today = _today()
    specs = [(1, "Run", "RUNNING", 17, 42, 520.0, 8_100_000.0, 152, 31, {"lightTime": "300s", "moderateTime": "1260s", "vigorousTime": "840s", "peakTime": "120s"}),
             (3, "Cycling", "OUTDOOR_BIKE", 18, 65, 610.0, 24_500_000.0, 138, 24, {"lightTime": "900s", "moderateTime": "2400s", "vigorousTime": "600s"}),
             (5, "Strength training", "STRENGTH_TRAINING", 7, 50, 280.0, None, 118, 9, {"lightTime": "1500s", "moderateTime": "900s"})]
    points = []
    for back, label, etype, hour, minutes, kcal, dist, hr, azm, zones in specs:
        day = today - timedelta(days=back)
        start_utc = datetime(day.year, day.month, day.day, hour - 2, 0)
        end_utc = start_utc + timedelta(minutes=minutes)
        summary = {"caloriesKcal": kcal, "averageHeartRateBeatsPerMinute": str(hr), "activeZoneMinutes": str(azm),
                   "heartRateZoneDurations": zones}
        if dist:
            summary["distanceMillimeters"] = dist
        points.append({
            "name": f"users/me/dataTypes/exercise/dataPoints/sample{back}",
            "exercise": {
                "displayName": label, "exerciseType": etype, "activeDuration": f"{minutes * 60}s",
                "interval": {"startTime": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "startUtcOffset": "7200s",
                             "endTime": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "endUtcOffset": "7200s"},
                "metricsSummary": summary,
            },
        })
    return points


def _sample_rollup(data_type, days):
    out = []
    for i in range(days):
        if data_type == "steps":
            val = {"steps": {"countSum": str([8200, 10450, 6900, 12100, 9300, 7600, 8800][i % 7])}}
        elif data_type == "distance":
            val = {"distance": {"millimetersSum": str([6_100_000, 8_300_000, 5_000_000, 9_700_000, 7_000_000, 5_600_000, 6_500_000][i % 7])}}
        elif data_type == "total-calories":
            val = {"totalCalories": {"kcalSum": [2380.0, 2610.0, 2250.0, 2720.0, 2440.0, 2300.0, 2400.0][i % 7]}}
        else:
            val = {"activeZoneMinutes": {"sumInFatBurnHeartZone": str([14, 22, 8, 30, 18, 10, 12][i % 7]),
                                         "sumInCardioHeartZone": str([6, 10, 0, 16, 8, 4, 6][i % 7]),
                                         "sumInPeakHeartZone": str([0, 4, 0, 6, 2, 0, 0][i % 7])}}
        out.append(val)
    return out


# ── Public entry points used by Jarvis's tools (never raise) ─────────────────

def _sample_note():
    if is_configured():
        return SAMPLE_PREFIX + "(Credentials are set; the user just needs to click Connect in the Jarvis UI.) "
    if is_connected():
        return SAMPLE_PREFIX + ("(A saved authorization exists, but GOOGLE_HEALTH_CLIENT_ID/SECRET aren't set in "
                                "this Jarvis process -- restart Jarvis from a shell that has them.) ")
    return SAMPLE_PREFIX + "(The Google Cloud OAuth client hasn't been set up yet.) "


def _safely(label, producer):
    sample = not can_use_live()
    try:
        text = producer(sample)
    except HealthError as exc:
        return str(exc)
    except httpx.HTTPError as exc:
        print(f"[Google Health] network error during {label}: {exc}")
        return "I couldn't reach Google Health right now."
    except Exception as exc:
        print(f"[Google Health] unexpected error during {label}: {type(exc).__name__}: {exc}")
        return "Sorry, the Google Health lookup failed unexpectedly."
    if sample:
        return _sample_note() + text
    # Always labelled, so "is this real?" is answered by the text itself.
    return LIVE_PREFIX.format(when=_now().strftime("%a %H:%M %Z").strip()) + text


def _sleep_window(day):
    """The civil-date range [first, end) of sessions to fetch, filtered on the
    day each session ENDS (sleep.interval.civil_end_time, which Google evaluates
    in the local time recorded with the session, i.e. Oslo time).

    * a specific night (`day` = the day you woke up): [day-1, day+2). The extra
      day either side pulls in the pre-midnight part of a night that was recorded
      as several sessions, so group_sleep_blocks can chain it to the rest;
      only blocks that END on `day` are then reported.
    * default: [today-3, today+1) in Europe/Oslo dates; the latest day with sleep wins."""
    if day:
        return day - timedelta(days=1), day + timedelta(days=2)
    today = _today()
    return today - timedelta(days=3), today + timedelta(days=1)


def _sleep_points(sample, first_day, end_exclusive):
    if sample:
        return _sample_sleep_points()
    flt = (f'sleep.interval.civil_end_time >= "{first_day.isoformat()}" AND '
           f'sleep.interval.civil_end_time < "{end_exclusive.isoformat()}"')
    print(f"[Google Health] sleep window ({TIMEZONE_NAME}, today={_today()}): {flt}")
    return _list_points("sleep", flt, reconcile=True)


def _exercise_points(sample, first_day, end_exclusive):
    if sample:
        return _sample_exercise_points()
    flt = (f'exercise.interval.civil_start_time >= "{first_day.isoformat()}" AND '
           f'exercise.interval.civil_start_time < "{end_exclusive.isoformat()}"')
    return _list_points("exercise", flt)


def sleep_summary_text(day_text=None):
    """Sleep for the night ending on `day_text` (YYYY-MM-DD); default: the most recent night."""
    def run(sample):
        day = _parse_day(day_text)
        first, end = _sleep_window(day)
        sessions = [parse_sleep(p) for p in _sleep_points(sample, first, end)]
        blocks = group_sleep_blocks(sessions)
        main, others = pick_sleep_block(blocks, wake_day=day)
        return format_sleep(main, others)
    return _safely("sleep summary", run)


def workout_summary_text(day_text=None, count=1):
    """Workouts on `day_text`, or the most recent `count` in the last 14 days."""
    def run(sample):
        day = _parse_day(day_text)
        today = _today()
        first, end = (day, day + timedelta(days=1)) if day else (today - timedelta(days=14), today + timedelta(days=1))
        workouts = sorted((parse_exercise(p) for p in _exercise_points(sample, first, end)),
                          key=lambda w: w["start"], reverse=True)
        if sample:
            workouts = [w for w in workouts if first <= w["start"].date() < end]
        workouts = workouts[: max(1, min(int(count or 1), 5))]
        if not workouts:
            return "I couldn't find any workouts for that period."
        return " ".join(format_workout(w) for w in workouts)
    return _safely("workout summary", run)


def weekly_activity_text(days=7):
    """Rollup of the last `days` days (today included)."""
    def run(sample):
        today = _today()
        first, end = today - timedelta(days=days - 1), today + timedelta(days=1)

        def rollup(data_type):
            return _sample_rollup(data_type, days) if sample else _daily_rollup(data_type, first, end)

        exercises = [parse_exercise(p) for p in _exercise_points(sample, first, end)]
        if sample:
            exercises = [w for w in exercises if first <= w["start"].date() < end]
        sleep_blocks = group_sleep_blocks([parse_sleep(p) for p in _sleep_points(sample, first, end)])
        sleep_blocks = [b for b in sleep_blocks if first <= b["wake_day"] < end]
        week = build_weekly(rollup("steps"), rollup("distance"), rollup("total-calories"),
                            rollup("active-zone-minutes"), exercises, sleep_blocks, days)
        return format_weekly(week)
    return _safely("weekly summary", run)


# ── `python google_health_client.py check` ───────────────────────────────────

def _check():
    """Verifies the live API against this module's assumptions. Run once after connecting."""
    import json
    print(f"configured: {is_configured()}   connected: {is_connected()}   redirect URI: {REDIRECT_URI}")
    if not can_use_live():
        print("Can't run live checks: need GOOGLE_HEALTH_CLIENT_ID/SECRET in this shell's environment AND a "
              "saved authorization (click Connect on the Health row in the Jarvis UI once).")
        return 1
    ok = True
    today = _today()
    print(f"time zone: {TIMEZONE_NAME}   now: {_now().isoformat()}   today (user's date): {today}")
    first, end = _sleep_window(None)
    print(f"sleep window (civil_end_time): [{first}, {end})")
    try:
        sessions = [parse_sleep(p) for p in _sleep_points(False, first, end)]
        print(f"\nsleep sessions returned: {len(sessions)}")
        for i, s in enumerate(sorted(sessions, key=lambda x: x["start_utc"]), 1):
            print(f"  #{i} {s['start']:%a %Y-%m-%d %H:%M} -> {s['end']:%H:%M} local | in bed {s['in_bed_min']:.0f} min, "
                  f"asleep {s['asleep_min']:.0f}, awake {s['awake_min']:.0f} | "
                  f"{'main' if s['main'] else ''}{'nap' if s['nap'] else ''} | stages {({k: int(v) for k, v in s['stages'].items()})}")
        blocks = group_sleep_blocks(sessions)
        for b in blocks:
            print(f"  block ending {b['wake_day']}: {len(b['sessions'])} session(s), asleep {b['asleep_min']:.0f} min "
                  f"= {_hm(b['asleep_min'])}, in bed {_hm(b['in_bed_min'])}")
    except Exception as exc:
        ok = False
        print(f"FAIL sleep sessions: {exc}")
    print()
    probes = [
        ("sleep reconcile", lambda: _list_points("sleep", f'sleep.interval.civil_end_time >= "{(today - timedelta(days=3)).isoformat()}"', page_size=1, max_pages=1, reconcile=True)),
        ("exercise list", lambda: _list_points("exercise", f'exercise.interval.civil_start_time >= "{(today - timedelta(days=14)).isoformat()}"', page_size=1, max_pages=1)),
        ("steps dailyRollUp", lambda: _daily_rollup("steps", today - timedelta(days=2), today + timedelta(days=1))),
    ]
    for label, call in probes:
        try:
            result = call()
            print(f"OK   {label}: {len(result)} item(s)")
            if result:
                print("     first item keys:", sorted(result[0].keys()))
        except Exception as exc:
            ok = False
            print(f"FAIL {label}: {exc}")
    print(f"dailyRollUp civil-time format in use: {'flat' if _rollup_flat else 'nested {date,time}'}")
    for label, fn in [("sleep", sleep_summary_text), ("workout", workout_summary_text), ("week", weekly_activity_text)]:
        print(f"\n{label}: {fn()}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(_check() if sys.argv[1:] == ["check"] else print(__doc__))
