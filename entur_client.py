"""Entur (Norwegian public transit) client -- departures and journey planning.

Entur's APIs are free and need no key, but their usage guidelines ask every
client to identify itself with an `ET-Client-Name` header ("company-app"
format), which is sent here together with a matching User-Agent:
https://developer.entur.org/pages-intro-authentication

  - Geocoder (stop/address lookup):  https://api.entur.io/geocoder/v1/autocomplete
  - Journey Planner v3 (GraphQL):    https://api.entur.io/journey-planner/v3/graphql

The same structured data feeds both the voice tools in jarvis.py (which turn
it into short, speakable text) and the TRANSIT panel in the web UI.
"""
import json
import os
import threading
import time
from datetime import datetime

import httpx

# Entur asks every client to identify itself as "company-application" so they can
# contact you if it misbehaves. Replace with your own before heavy use.
CLIENT_NAME = "personal-jarvis"
HEADERS = {"ET-Client-Name": CLIENT_NAME, "User-Agent": CLIENT_NAME}

GEOCODER_URL = "https://api.entur.io/geocoder/v1/autocomplete"
GRAPHQL_URL = "https://api.entur.io/journey-planner/v3/graphql"

# Geocoder results are biased toward Oslo so an ambiguous name ("Majorstuen",
# "Sentrum") resolves to the Oslo one first. A strong bias, not a filter --
# stops elsewhere in Norway are still found.
FOCUS_LAT, FOCUS_LON = 59.9139, 10.7522

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transit_settings.json")
_settings_lock = threading.Lock()

TIMEOUT = httpx.Timeout(8.0, connect=5.0)

# Entur transport modes -> short spoken/displayed words.
MODE_WORDS = {
    "bus": "bus", "coach": "coach", "tram": "tram", "rail": "train",
    "metro": "metro", "water": "ferry", "air": "flight", "foot": "walk",
}
# What a user might say -> the Entur mode to filter on.
MODE_ALIASES = {
    "bus": "bus", "buses": "bus", "train": "rail", "trains": "rail", "rail": "rail",
    "tram": "tram", "trams": "tram", "metro": "metro", "subway": "metro", "tbane": "metro",
    "t-bane": "metro", "ferry": "water", "boat": "water",
}


class EnturError(Exception):
    """A lookup/network problem with a message that is safe to speak."""


def _get(url, params):
    try:
        r = httpx.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise EnturError("I couldn't reach Entur right now.") from exc


def _graphql(query):
    try:
        r = httpx.post(GRAPHQL_URL, json={"query": query},
                       headers={**HEADERS, "Content-Type": "application/json"}, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as exc:
        raise EnturError("I couldn't reach Entur right now.") from exc
    if payload.get("errors"):
        print(f"[Entur GraphQL errors]: {payload['errors']}")
        raise EnturError("Entur returned an error for that request.")
    return payload["data"]


def _geocode(text, layers=None, size=5):
    params = {
        "text": text, "size": size, "lang": "no",
        "focus.point.lat": FOCUS_LAT, "focus.point.lon": FOCUS_LON,
    }
    if layers:
        params["layers"] = layers
    return _get(GEOCODER_URL, params).get("features", [])


def resolve_stop(text):
    """Finds the best-matching transit stop (an NSR:StopPlace) for free text.
    Returns {"id", "name", "locality"} or None."""
    for feature in _geocode(text, layers="venue", size=10):
        props = feature["properties"]
        if props.get("id", "").startswith("NSR:StopPlace:"):
            return {"id": props["id"], "name": props["name"], "locality": props.get("locality") or ""}
    return None


def _resolve_place(text):
    """Resolves a stop OR address to an Entur trip-planner `Location` dict
    (plus a display name). Stops go by id; addresses/other places by
    coordinates."""
    features = _geocode(text, size=5)
    if not features:
        return None
    # Prefer a real stop when one matches well; otherwise the top hit.
    top = features[0]
    props = top["properties"]
    name = props.get("name") or text
    if props.get("id", "").startswith("NSR:StopPlace:"):
        return {"location": {"place": props["id"]}, "name": name}
    lon, lat = top["geometry"]["coordinates"]
    return {"location": {"coordinates": {"latitude": lat, "longitude": lon}, "name": name}, "name": name}


# --- Departures -------------------------------------------------------------

def _parse_time(iso):
    return datetime.fromisoformat(iso)


def get_departures(stop_id, count=5, mode=None):
    """Upcoming departures for a stop place id, as structured data:
    {"stop_name": str, "departures": [{line, mode, destination, minutes,
    clock, expected_iso, delay_min, realtime, cancelled, quay}]}.
    `mode` ("bus", "rail", "tram", "metro", "water") filters client-side."""
    query = """{ stopPlace(id: "%s") { name
      estimatedCalls(numberOfDepartures: 40, timeRange: 14400) {
        realtime cancellation aimedDepartureTime expectedDepartureTime
        destinationDisplay { frontText } quay { publicCode }
        serviceJourney { journeyPattern { line { publicCode transportMode } } } } } }""" % stop_id
    stop = _graphql(query).get("stopPlace")
    if not stop:
        raise EnturError("I couldn't find that stop in Entur.")

    now = datetime.now().astimezone()
    departures = []
    for call in stop["estimatedCalls"]:
        line = call["serviceJourney"]["journeyPattern"]["line"]
        if mode and line["transportMode"] != mode:
            continue
        expected = _parse_time(call["expectedDepartureTime"])
        aimed = _parse_time(call["aimedDepartureTime"])
        minutes = max(0, int((expected - now).total_seconds() // 60))
        delay = int(round((expected - aimed).total_seconds() / 60))
        departures.append({
            "line": line["publicCode"] or "",
            "mode": line["transportMode"],
            "destination": ((call["destinationDisplay"] or {}).get("frontText") or "").strip(),
            "minutes": minutes,
            "clock": expected.strftime("%H:%M"),
            "expected_iso": call["expectedDepartureTime"],
            "delay_min": delay,
            "realtime": bool(call["realtime"]),
            "cancelled": bool(call["cancellation"]),
            "quay": (call["quay"] or {}).get("publicCode") or "",
        })
        if len(departures) >= count:
            break
    return {"stop_name": stop["name"], "departures": departures}


def _when(dep):
    """'now', 'in 6 min', or 'at 14:32' (far-off departures read better as a clock time)."""
    if dep["minutes"] < 1:
        return "now"
    if dep["minutes"] < 60:
        return f"in {dep['minutes']} min"
    return f"at {dep['clock']}"


def format_departures(data, mode=None):
    """Short, speakable text for the voice tool -- not a raw data dump."""
    name = data["stop_name"]
    deps = data["departures"]
    if not deps:
        what = MODE_WORDS.get(mode, mode) if mode else "departures"
        return f"No {what} departures found from {name} in the next few hours."
    parts = []
    for d in deps:
        word = MODE_WORDS.get(d["mode"], d["mode"])
        label = f"{word} {d['line']}".strip() if d["line"] else word
        text = f"{label} to {d['destination']} {_when(d)}" if d["destination"] else f"{label} {_when(d)}"
        if d["cancelled"]:
            text += " (cancelled)"
        elif d["delay_min"] >= 2:
            text += f" ({d['delay_min']} min late)"
        parts.append(text)
    return f"From {name}: " + "; ".join(parts) + "."


def next_departures_text(stop_text=None, count=4, mode=None):
    """Voice-tool entry point: resolve the stop (or use the saved default),
    fetch departures, return speakable text."""
    mode = MODE_ALIASES.get((mode or "").strip().lower()) if mode else None
    if stop_text and stop_text.strip():
        stop = resolve_stop(stop_text.strip())
        if not stop:
            return f"I couldn't find a stop called '{stop_text}'."
    else:
        stop = get_default_stop()
        if not stop:
            return ("No stop given and no default stop saved yet. Ask the user which stop, "
                    "or have them say 'set my default stop to <name>'.")
    count = max(1, min(int(count or 4), 8))
    return format_departures(get_departures(stop["id"], count=count, mode=mode), mode=mode)


# --- Journey planning -------------------------------------------------------

def plan_journey_text(origin, destination):
    """Voice-tool entry point: best public-transit options between two
    places (stops or addresses), leaving now."""
    a = _resolve_place(origin)
    b = _resolve_place(destination)
    if not a:
        return f"I couldn't find '{origin}'."
    if not b:
        return f"I couldn't find '{destination}'."

    def loc(place):
        # Build the GraphQL input literal for a Location.
        l = place["location"]
        if "place" in l:
            return '{place: "%s"}' % l["place"]
        c = l["coordinates"]
        return '{coordinates: {latitude: %s, longitude: %s}, name: %s}' % (
            c["latitude"], c["longitude"], json.dumps(l["name"]))

    query = """{ trip(from: %s, to: %s, numTripPatterns: 3) { tripPatterns {
      expectedStartTime expectedEndTime duration
      legs { mode duration expectedStartTime line { publicCode }
             fromPlace { name } toPlace { name }
             fromEstimatedCall { destinationDisplay { frontText } } } } } }""" % (loc(a), loc(b))
    patterns = _graphql(query)["trip"]["tripPatterns"]
    if not patterns:
        return f"No public transit route found from {a['name']} to {b['name']}."

    def describe(tp, detailed):
        start = _parse_time(tp["expectedStartTime"]).strftime("%H:%M")
        end = _parse_time(tp["expectedEndTime"]).strftime("%H:%M")
        mins = round(tp["duration"] / 60)
        head = f"leave {start}, arrive {end} ({mins} min)"
        if not detailed:
            return head
        steps = []
        for leg in tp["legs"]:
            if leg["mode"] == "foot":
                if leg["duration"] >= 120:  # skip trivial walking hops
                    steps.append(f"walk {round(leg['duration'] / 60)} min")
                continue
            word = MODE_WORDS.get(leg["mode"], leg["mode"])
            line = (leg["line"] or {}).get("publicCode", "")
            dest = ((leg["fromEstimatedCall"] or {}).get("destinationDisplay") or {}).get("frontText", "")
            clock = _parse_time(leg["expectedStartTime"]).strftime("%H:%M")
            steps.append(f"{word} {line}{' towards ' + dest if dest else ''} from "
                         f"{leg['fromPlace']['name']} at {clock}, off at {leg['toPlace']['name']}")
        return head + ": " + "; then ".join(steps)

    out = f"{a['name']} to {b['name']}. Best: {describe(patterns[0], True)}."
    if len(patterns) > 1:
        out += " Alternatives: " + "; ".join(describe(p, False) for p in patterns[1:]) + "."
    return out


# --- Default stop (for the TRANSIT panel and "next bus" with no stop given) --

def get_default_stop():
    with _settings_lock:
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                stop = json.load(f).get("default_stop")
            if isinstance(stop, dict) and stop.get("id") and stop.get("name"):
                return stop
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Transit settings read error]: {exc}")
    return None


def set_default_stop(stop_text):
    """Resolves `stop_text` and saves it as the default. Returns a
    confirmation string naming exactly which stop was chosen."""
    stop = resolve_stop((stop_text or "").strip())
    if not stop:
        return f"I couldn't find a stop called '{stop_text}'. Nothing was saved."
    with _settings_lock:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"default_stop": stop}, f, indent=2)
    where = f" in {stop['locality']}" if stop["locality"] else ""
    _invalidate_panel_cache()
    return f"Default stop saved: {stop['name']}{where}."


# --- TRANSIT panel data (cached so several tabs/polls don't hammer Entur) ----

_PANEL_TTL = 20.0
_panel_cache = {"at": 0.0, "data": None}
_panel_lock = threading.Lock()


def _invalidate_panel_cache():
    with _panel_lock:
        _panel_cache["at"] = 0.0
        _panel_cache["data"] = None


def panel_data(count=5):
    """What the UI's TRANSIT panel shows: the default stop's next departures.
    Never raises -- problems are returned as an `error` string for display."""
    stop = get_default_stop()
    if not stop:
        return {"configured": False}
    with _panel_lock:
        if _panel_cache["data"] is not None and time.time() - _panel_cache["at"] < _PANEL_TTL:
            return _panel_cache["data"]
    try:
        data = get_departures(stop["id"], count=count)
        result = {"configured": True, "stop_name": data["stop_name"],
                  "departures": data["departures"]}
    except EnturError as exc:
        result = {"configured": True, "stop_name": stop["name"], "departures": [], "error": str(exc)}
    with _panel_lock:
        _panel_cache["at"] = time.time()
        _panel_cache["data"] = result
    return result
