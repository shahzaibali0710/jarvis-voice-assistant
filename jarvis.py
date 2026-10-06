"""
Jarvis: Wake Word -> Record -> Transcribe -> LLM -> Speak
Flow: listen for "hey jarvis" -> record until the user stops talking ->
transcribe with faster-whisper (auto-detecting language) -> send to Claude
(with a Jarvis persona system prompt and the web_search tool) -> speak the
reply with edge-tts -> go back to listening.

Runs a local Flask web UI (see web_app.py) alongside the wake-word loop
by default, opened automatically in the browser. The UI's "Talk to
Jarvis" button and text box run the exact same pipeline as the wake
word, just skipping the "wait for wake word" / "record audio" steps
respectively. Use --no-ui for the original console-only mode.
"""

import argparse
import asyncio
import concurrent.futures
import glob
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import wave

import edge_tts
import numpy as np
import pyaudio
import pygame
import pygame.sndarray
from faster_whisper import WhisperModel
from anthropic import Anthropic, APIError, APIConnectionError
from openwakeword.model import Model

import spotify_client
import job_tracker
import entur_client
import google_health_client

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

# Python block-buffers stdout when it's not attached to an interactive
# terminal (e.g. redirected to a log file, as happens when this is launched
# in the background) -- print() calls can sit unflushed for a long time,
# which defeats the point of the diagnostic logging below (transcription
# results, Spotify tool calls, API errors). Force line buffering so every
# print shows up immediately in whatever is capturing this process's output.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _ensure_ffmpeg_on_path():
    """Whisper shells out to ffmpeg. Make sure it's importable even if the
    current process's PATH env var predates the winget install (Windows
    only refreshes PATH for new sessions, not already-running ones)."""
    if shutil.which("ffmpeg"):
        return
    candidates = glob.glob(
        os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin"
        )
    )
    for bin_dir in candidates:
        if os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
            return
    print(
        "WARNING: ffmpeg not found on PATH and not found in the usual winget "
        "install location. Whisper transcription will fail. Install it with:\n"
        "  winget install --id=Gyan.FFmpeg -e"
    )


_ensure_ffmpeg_on_path()

# ---- Config ----
CHUNK = 1280            # 80ms at 16kHz, required by openWakeWord
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
COMMAND_WAV = "command.wav"

# ---- Wake word tuning ----
# openWakeWord's confidence score for "hey jarvis" must exceed this to
# trigger. Raise it if background noise/chatter is causing false triggers;
# lower it if genuine "hey jarvis" utterances are being missed. 0.5 was the
# original default and turned out too permissive; 0.65 still let room
# noise/TV/conversation through often enough to be annoying, so it was
# raised to 0.72 -- but that turned out to reject real "hey jarvis" attempts
# too, not just noise. Backed off to a middle ground. Noise false-positive
# protection no longer rests on this threshold alone now that the VAD layer
# (see the recording pipeline) also helps filter non-speech noise during
# actual recording, so a somewhat lower wake threshold is safer than it
# would have been before that was added. Re-tune using the confidence
# scores logged below (WAKE_LOG_THRESHOLD) from real "hey jarvis" attempts.
WAKE_THRESHOLD = 0.6

# Any wake-word confidence score above this gets printed, even if it didn't
# cross WAKE_THRESHOLD -- lets you see exactly what score your real voice is
# landing at (and what background noise peaks at) to tune WAKE_THRESHOLD
# from real data instead of guessing. Set above 1.0 to disable this logging
# entirely.
WAKE_LOG_THRESHOLD = 0.3

# Below this RMS energy, a mic chunk is treated as silence/background noise
# and skipped entirely -- the (comparatively expensive) wake word model
# never even runs on it. Keeps the wake-word loop from being tempted by
# quiet room tone at all, on top of the confidence threshold above. Set to
# 0 to disable this gate. Tune alongside MIN_SILENCE_THRESHOLD below, which
# uses the same RMS scale for a similar purpose during recording.
WAKE_MIN_ENERGY = 200

# ---- Silence-based recording ----
# Rather than always recording a fixed window, keep recording while the
# mic is picking up speech-level energy and stop once it's been quiet for
# a bit. All energy is measured as RMS of raw int16 samples.
MIN_RECORD_SECONDS = 0.6      # always capture at least this much before silence can end it
SILENCE_DURATION = 1.2        # seconds of continuous silence that ends recording
MAX_RECORD_SECONDS = 15       # hard safety cap in case silence is never detected
AMBIENT_SAMPLE_SECONDS = 0.24 # leading audio used to estimate the room's noise floor
SILENCE_MULTIPLIER = 2.5      # audio below noise_floor * this is "silence"
MIN_SILENCE_THRESHOLD = 150   # absolute floor so a near-silent room doesn't yield ~0

# How much a single "speech-like" chunk decays the silence run-length,
# rather than resetting it to 0 outright. A single stray noise spike (a
# creak, a cough from another room, a brief TV blip) used to fully reset the
# countdown to silence, meaning recording could stay open indefinitely
# across a string of intermittent noises even after the user had finished
# talking. Decaying instead of resetting still requires several consecutive
# speech-like chunks to meaningfully delay the cutoff (real continued
# speech), while one-off spikes only cost a couple hundred ms of progress.
SILENCE_RUN_DECAY = 2

# webrtcvad's aggressiveness mode, 0 (least aggressive about filtering out
# non-speech) to 3 (most aggressive). Used alongside the RMS energy check
# below -- RMS alone can't tell loud non-speech noise (TV, music, someone
# else talking) apart from the user's own voice; VAD adds a speech-pattern
# check on top. It's not perfect either (it still can't distinguish the
# user's voice from someone else's speech), but it meaningfully helps with
# non-speech noise.
VAD_AGGRESSIVENESS = 2

# Extra chunks of padding kept on each side of the detected speech span when
# trimming the recording before it's handed to Whisper. Keeps natural
# word onsets/offsets intact while dropping the silence/noise padding that
# accumulates while waiting to confirm the user has actually stopped.
TRIM_PADDING_CHUNKS = 3

_vad = None
if webrtcvad is not None:
    try:
        _vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    except Exception as exc:
        print(f"[VAD unavailable, falling back to energy-only detection]: {exc}")
        _vad = None
else:
    print("[webrtcvad not installed, falling back to energy-only detection]")

# ---- Conversation memory ----
CONVERSATION_TIMEOUT = 300    # seconds of inactivity after which history auto-clears
MAX_HISTORY_MESSAGES = 20     # trim oldest turns beyond this so context doesn't grow unbounded

# ---- Follow-up mode ----
# After speaking, Jarvis listens again for a bit without needing the wake
# word, so a back-and-forth doesn't require "hey jarvis" every time.
FOLLOWUP_WINDOW_SECONDS = 7   # how long to wait for the user to start a follow-up

# Voice picked per Whisper-detected language ("ur"/"no"/"en" are Whisper's
# own ISO codes). Falls back to DEFAULT_VOICE for anything else.
# NOTE: Whisper's base/small models routinely mis-tag spoken Urdu as "hi"
# (Hindi) since the two are phonetically almost identical -- they mainly
# differ in script, which spoken audio doesn't have. Since Hindi isn't one
# of this assistant's supported languages, "hi" is treated as Urdu here.
VOICE_BY_LANGUAGE = {
    "ur": "ur-PK-AsadNeural",
    "hi": "ur-PK-AsadNeural",
    "no": "nb-NO-FinnNeural",
    "en": "en-GB-RyanNeural",
}
DEFAULT_VOICE = "en-GB-RyanNeural"  # composed, British-butler-ish -- closest to Iron Man's Jarvis

# Haiku 4.5: fast, cheap, plenty capable for casual assistant queries -- Jarvis
# doesn't need heavy reasoning to answer "what's the weather" or chat.
CLAUDE_MODEL = "claude-haiku-4-5"

# Server-side tool: Claude decides on its own when a query needs current
# information (weather, news, prices, "what's the latest on X") and searches
# automatically -- no client-side execution loop needed. Haiku 4.5 isn't in
# the model set that supports the newer dynamic-filtering "_20260209" tool
# variant, so this uses the basic "_20250305" one.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

# Client-side tools: Claude decides to call these, we execute them locally
# against the Spotify Web API (spotify_client.py) and hand back a short
# result string. Unlike web_search, these need a manual tool-use loop --
# see ask_claude().
SPOTIFY_TOOLS = [
    {
        "name": "play_music",
        "description": (
            "Play a song or artist on Spotify by name. Searches Spotify and "
            "starts playback on the user's active device. Use this whenever "
            "the user asks to play, put on, or listen to music."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Song title, artist name, or both -- e.g. "
                    "'Bohemian Rhapsody' or 'Taylor Swift'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "pause_music",
        "description": "Pause Spotify playback.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "resume_music",
        "description": "Resume/unpause Spotify playback.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "skip_music",
        "description": "Skip to the next track on Spotify.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "previous_music",
        "description": "Go back to the previous track on Spotify.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_volume",
        "description": "Set Spotify playback volume to a specific percentage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "volume": {
                    "type": "integer",
                    "description": "Volume level from 0 to 100.",
                    "minimum": 0,
                    "maximum": 100,
                }
            },
            "required": ["volume"],
        },
    },
]

VALID_STATUSES = job_tracker.VALID_STATUSES

# Included in every job search so results are scored for relevance without
# restating your background each time. It lives in candidate_profile.txt next
# to this file -- a local, git-ignored file, so personal details never end up
# in the repository. See candidate_profile.example.txt for a template. The file
# is re-read on every search, so edits take effect without restarting Jarvis.
_CANDIDATE_PROFILE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_profile.txt")
_NO_PROFILE = (
    "No candidate profile has been set up. Judge relevance from the search "
    "keywords and location alone, and do NOT claim or invent any personal "
    "background in the 'reason' for a result."
)
_profile_warned = False


def _candidate_profile():
    global _profile_warned
    try:
        with open(_CANDIDATE_PROFILE_FILE, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[Job search] couldn't read candidate_profile.txt: {exc}")
    if not _profile_warned:
        _profile_warned = True
        print("[Job search] no candidate_profile.txt found -- searching without a personal "
              "profile (copy candidate_profile.example.txt to candidate_profile.txt and fill it in).")
    return _NO_PROFILE

JOB_TOOLS = [
    {
        "name": "search_jobs",
        "description": (
            "Search the web for current, real job openings, automatically scored "
            "for relevance against the candidate's own background (degree, thesis, "
            "skills, preferences) -- so it finds genuinely relevant postings even "
            "with no criteria given. Checks sources like Finn.no, LinkedIn, NAV "
            "(nav.no), and relevant company career pages. Returns title/company/"
            "location/link/why-it-fits for each -- not full job descriptions. Full "
            "results are shown in the UI's JOB SEARCH RESULTS panel automatically. "
            "Use this whenever the user asks to find, search for, or look up job "
            "openings/postings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Role type and/or keywords to search for, e.g. "
                    "'FPGA engineer' or 'embedded systems robotics'. Omit this "
                    "entirely if the user didn't specify -- the search will then "
                    "draw broadly on the candidate's full background instead of "
                    "guessing a narrower default.",
                },
                "location": {
                    "type": "string",
                    "description": "Location to search in, e.g. 'Oslo, Norway'. "
                    "Defaults to Oslo, Norway if not given.",
                },
            },
        },
    },
    {
        "name": "activate_job_hunting_mode",
        "description": (
            "Turns on Job Hunting Mode: shows a persistent 'JOB HUNTING' indicator "
            "in the UI and immediately runs a broad job search across the "
            "candidate's full range of interests (not just one category), "
            "returning as many genuinely relevant current listings as can "
            "reasonably be found rather than just a few. Full results go to the "
            "UI's JOB SEARCH RESULTS panel. Use when the user says something like "
            "'Jarvis, job hunting mode' or 'start job hunting mode'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "deactivate_job_hunting_mode",
        "description": (
            "Turns off Job Hunting Mode and clears its UI indicator. Use when the "
            "user says something like 'exit job hunting mode' or 'turn off job "
            "hunting mode'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "log_application",
        "description": (
            "Records a new job application (or, if one's already logged for the "
            "same company and role, updates its status instead of duplicating). "
            "Use whenever the user mentions applying to a job, saving one to "
            "consider later, or a status change -- e.g. 'I applied to the FPGA "
            "role at Kongsberg' or 'save the Nammo listing, I haven't applied yet'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name."},
                "role": {"type": "string", "description": "Job title/role."},
                "status": {
                    "type": "string",
                    "description": "One of: not applied, applied, interview, "
                    "rejected, offer. Defaults to 'applied' for a new entry.",
                    "enum": VALID_STATUSES,
                },
            },
            "required": ["company", "role"],
        },
    },
    {
        "name": "update_application_status",
        "description": (
            "Updates the status of an already-logged application, found by company "
            "name (and role, if given, to disambiguate multiple roles at the same "
            "company). Use for voice commands like 'mark Kongsberg as interview' "
            "or 'I got rejected by Nammo' where the user names the company without "
            "necessarily repeating the exact role. If the tool reports multiple "
            "matches, ask the user which role they meant and call it again with "
            "role set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name (partial match)."},
                "role": {"type": "string", "description": "Role, to disambiguate if needed."},
                "status": {
                    "type": "string",
                    "description": "One of: not applied, applied, interview, rejected, offer.",
                    "enum": VALID_STATUSES,
                },
            },
            "required": ["company", "status"],
        },
    },
    {
        "name": "get_application_status",
        "description": (
            "Looks up logged job applications -- either all of them, or filtered by "
            "company and/or status. Use whenever the user asks what they've applied "
            "to, or the status of a specific application."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Filter by company name (partial match)."},
                "status": {
                    "type": "string",
                    "description": "Filter by status: not applied, applied, interview, rejected, or offer.",
                    "enum": VALID_STATUSES,
                },
            },
        },
    },
]
TRANSIT_TOOLS = [
    {
        "name": "get_next_departures",
        "description": (
            "Real-time upcoming bus/train/tram/metro/ferry departures from a stop in "
            "Norway (Entur). Use whenever the user asks when the next bus/train/tram/"
            "metro leaves, or what's departing from somewhere. Omit `stop` to use "
            "their saved default stop. Never answer departure times from memory -- "
            "always call this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stop": {"type": "string", "description": "Stop name, e.g. 'Jernbanetorget' or 'Majorstuen'. Omit to use the saved default stop."},
                "mode": {"type": "string", "description": "Optional filter: bus, train, tram, metro, or ferry."},
                "count": {"type": "integer", "description": "How many departures (default 4, max 8).", "minimum": 1, "maximum": 8},
            },
        },
    },
    {
        "name": "plan_journey",
        "description": (
            "Public-transit directions between two places in Norway (stops or "
            "addresses), leaving now, via Entur. Use when the user asks how to get "
            "somewhere by bus/train/tram/metro. Never invent routes -- always call this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Where from -- a stop name or address."},
                "destination": {"type": "string", "description": "Where to -- a stop name or address."},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "set_default_stop",
        "description": (
            "Saves the user's usual transit stop. It's used when they ask for "
            "departures without naming a stop, and shown in the UI's TRANSIT panel. "
            "Use when they say things like 'my usual stop is Majorstuen' or 'set my "
            "default stop to ...'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"stop": {"type": "string", "description": "Stop name."}},
            "required": ["stop"],
        },
    },
]
HEALTH_TOOLS = [
    {
        "name": "get_sleep_summary",
        "description": (
            "The user's sleep from Google Health (Fitbit/Pixel Watch data): time asleep, "
            "bed and wake times, efficiency and sleep stages. Defaults to the most recent "
            "night. Use whenever they ask how they slept. Never state sleep numbers from "
            "memory -- always call this. Google Health has no sleep score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Optional: the date they woke up, as YYYY-MM-DD (for 'how did I sleep on Tuesday'). Omit for last night."},
            },
        },
    },
    {
        "name": "get_workout_summary",
        "description": (
            "Recent workouts from Google Health: type, duration, calories, distance, average "
            "heart rate, time in heart-rate zones and Active Zone Minutes. Defaults to the "
            "most recent workout. Use whenever they ask about a workout, run, ride or "
            "training session. Never invent workout numbers -- always call this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Optional: workouts on this day, YYYY-MM-DD."},
                "count": {"type": "integer", "description": "How many recent workouts (default 1, max 5).", "minimum": 1, "maximum": 5},
            },
        },
    },
    {
        "name": "get_weekly_activity_summary",
        "description": (
            "A 7-day rollup from Google Health: steps, distance, calories, Active Zone "
            "Minutes, number of workouts and average sleep. Use for 'how was my week' or "
            "'how active have I been'. Never state these from memory -- always call this."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]
_CLIENT_TOOL_NAMES = {tool["name"] for tool in SPOTIFY_TOOLS + JOB_TOOLS + TRANSIT_TOOLS + HEALTH_TOOLS}


def _parse_job_listings(text):
    """Parses the pipe-delimited "<Title> | <Company> | <Location> | <URL> |
    <Reason>" lines _search_jobs asks the nested search call to return, into
    structured dicts for the UI's JOB SEARCH RESULTS panel. Lines that don't
    match (e.g. the "No matching listings found." sentinel, or stray prose)
    are silently skipped rather than raising -- worst case the panel just
    shows fewer/no rows, which is safer than crashing the tool call."""
    listings = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4 or not parts[0] or not parts[3].startswith("http"):
            continue
        listings.append({
            "title": parts[0],
            "company": parts[1],
            "location": parts[2],
            "url": parts[3],
            "reason": parts[4] if len(parts) > 4 else "",
        })
    return listings


def _search_jobs(keywords=None, location=None, broad=False, state=None):
    """Runs a dedicated, focused Claude + web_search round trip to find real
    current job postings scored against candidate_profile.txt, rather than
    letting the general web_search tool loose on a vague query. Kept
    separate from the main conversation so the search prompt (which sources
    to check, exact output format, relevance scoring) can be tailored
    specifically for job listings without cluttering the Jarvis persona
    prompt.

    Parsed results are written to state.job_search_results (if a state is
    given) for the UI's JOB SEARCH RESULTS panel -- independent of whatever
    summary Claude ultimately says in the conversation, so the panel always
    reflects the real search output even if Claude's spoken summary is
    abridged."""
    keywords = (keywords or "").strip()
    location = (location or "Oslo, Norway").strip()
    result_cap = 20 if broad else 8

    if keywords:
        focus_line = f'Focus the search specifically on: "{keywords}". '
    else:
        focus_line = (
            "The user gave no specific keywords -- search broadly across the "
            "candidate's full range of interests described above, not just one "
            "category. "
        )

    prompt = (
        f"{_candidate_profile()}\n\n"
        f"Search the web for current, real, currently-open job postings located "
        f"in or near {location}. {focus_line}"
        "Check sources like Finn.no, LinkedIn, NAV (nav.no), and relevant company "
        "career pages (defense/aerospace, robotics, embedded/FPGA employers in "
        "that area). Only include postings you actually find via search -- never "
        "invent or guess at listings. Skip roles that clearly match the "
        "candidate's stated 'poor fit' list (PLC/automation, offshore) unless the "
        "user explicitly asked for those. For each posting, give the job title, "
        "company, location, a direct URL, and a short one-sentence reason it's a "
        "good fit for THIS candidate's specific background (thesis, skills, "
        "target area) -- not a generic reason. Reply with ONLY a list, one "
        "posting per line, in this exact format:\n"
        "<Job Title> | <Company> | <Location> | <URL> | <Reason it fits>\n"
        f"List up to {result_cap} of the most relevant and recent postings"
        + (
            ", casting a genuinely wide net across FPGA, embedded, robotics, and "
            "AI/ML-adjacent roles rather than just one category"
            if broad else ""
        )
        + ". If nothing relevant turns up, reply with exactly: No matching listings found."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = None
        for _ in range(4):  # allows a couple of pause_turn continuations for the search
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                tools=[WEB_SEARCH_TOOL],
                messages=messages,
            )
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            break
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        text = text or "No matching listings found."
        if state is not None:
            state.update(job_search_results=_parse_job_listings(text))
        return text
    except (APIError, APIConnectionError) as exc:
        print(f"[Job search error]: {exc}")
        return "Sorry, the job search failed due to an API error."
    except Exception as exc:
        print(f"[Job search error]: {exc}")
        return "Sorry, the job search failed unexpectedly."


def _activate_job_hunting_mode(state):
    state.update(job_hunting_mode=True)
    results_text = _search_jobs(broad=True, state=state)
    return (
        "Job Hunting Mode activated. Broad search results:\n" + results_text
    )


# Testing turned up Claude (Haiku) reliably calling activate_job_hunting_mode
# for "job hunting mode", but NOT reliably calling deactivate_job_hunting_mode
# for "exit job hunting mode" -- it would just reply "Job Hunting Mode is
# off" conversationally without ever invoking the tool, leaving the UI
# indicator stuck on. Turning it off is a simple, unambiguous command with
# no need for LLM judgment, so it's handled deterministically here instead
# of relying on tool-call reliability for it.
_JOB_HUNTING_DEACTIVATE_PATTERN = re.compile(
    r"\b(exit|stop|turn off|end|deactivate|leave|cancel)\b[\s\w]{0,25}\bjob hunting mode\b"
    r"|\bjob hunting mode\b[\s\w]{0,25}\b(off|exit|end|stop)\b",
    re.IGNORECASE,
)


def _is_job_hunting_deactivate_command(text):
    return bool(_JOB_HUNTING_DEACTIVATE_PATTERN.search(text or ""))


def _execute_client_tool(name, tool_input, state=None):
    # Logged unconditionally so a failed voice-triggered tool call can be
    # diagnosed at a glance: this line confirms the tool was actually
    # invoked (transcription + tool-calling worked), and the result line
    # right after it shows whether the failure was inside the underlying
    # call (Spotify API, job search, application log) itself.
    print(f"[Tool call] {name}({tool_input})")
    if name == "play_music":
        result = spotify_client.play(tool_input.get("query", ""))
    elif name == "pause_music":
        result = spotify_client.pause()
    elif name == "resume_music":
        result = spotify_client.resume()
    elif name == "skip_music":
        result = spotify_client.skip()
    elif name == "previous_music":
        result = spotify_client.previous()
    elif name == "set_volume":
        result = spotify_client.set_volume(tool_input.get("volume", 50))
    elif name == "search_jobs":
        result = _search_jobs(tool_input.get("keywords"), tool_input.get("location"), state=state)
    elif name == "activate_job_hunting_mode":
        result = _activate_job_hunting_mode(state) if state is not None else "Unavailable right now."
    elif name == "deactivate_job_hunting_mode":
        if state is not None:
            state.update(job_hunting_mode=False)
        result = "Job Hunting Mode deactivated."
    elif name == "log_application":
        result = job_tracker.log_application(
            tool_input.get("company", ""),
            tool_input.get("role", ""),
            status=tool_input.get("status", "applied"),
        )
    elif name == "update_application_status":
        result = job_tracker.update_status(
            tool_input.get("company", ""),
            tool_input.get("status", ""),
            role=tool_input.get("role"),
        )
    elif name == "get_application_status":
        entries = job_tracker.list_applications(tool_input.get("company"), tool_input.get("status"))
        result = job_tracker.format_applications_for_speech(entries)
    elif name in ("get_next_departures", "plan_journey", "set_default_stop"):
        try:
            if name == "get_next_departures":
                result = entur_client.next_departures_text(
                    tool_input.get("stop"), count=tool_input.get("count") or 4,
                    mode=tool_input.get("mode"))
            elif name == "plan_journey":
                result = entur_client.plan_journey_text(
                    tool_input.get("origin", ""), tool_input.get("destination", ""))
            else:
                result = entur_client.set_default_stop(tool_input.get("stop", ""))
        except entur_client.EnturError as exc:
            result = str(exc)  # already phrased to be spoken
        except Exception as exc:
            print(f"[Transit tool error]: {exc}")
            result = "Sorry, the transit lookup failed unexpectedly."
    elif name == "get_sleep_summary":
        result = google_health_client.sleep_summary_text(tool_input.get("date"))
    elif name == "get_workout_summary":
        result = google_health_client.workout_summary_text(tool_input.get("date"), tool_input.get("count") or 1)
    elif name == "get_weekly_activity_summary":
        result = google_health_client.weekly_activity_text()
    else:
        result = f"Unknown tool: {name}"
    print(f"[Tool result] {name} -> {result}")
    return result


_HEALTH_TOOL_NAMES = {tool["name"] for tool in HEALTH_TOOLS}


def _health_source(tool_name, result_text):
    """'sample' or 'live' for a health tool result (the client labels every
    answer), None for anything else -- including error messages, which claim
    no data at all."""
    if tool_name not in _HEALTH_TOOL_NAMES:
        return None
    if result_text.startswith("[SAMPLE DATA"):
        return "sample"
    if result_text.startswith("[LIVE DATA"):
        return "live"
    return None


def _ensure_health_source(text, sources):
    """Guarantees a health answer says where its numbers came from. The tool
    text is labelled, but Claude could still drop that when summarizing -- and
    placeholder figures presented as real are exactly the failure to avoid. So
    if the reply doesn't mention the source itself, a short tag is prepended."""
    if not sources or not text:
        return text
    lower = text.lower()
    norwegian = detect_language(text) == "no"
    if "sample" in sources:
        if any(w in lower for w in ("sample", "eksempel", "placeholder", "made-up", "made up")):
            return text
        return ("Eksempeldata, ikke dine egne tall: " if norwegian else "Sample data, not your real numbers: ") + text
    if any(w in lower for w in ("live", "google health", "fitbit", "direkte")):
        return text
    return ("Live fra Google Health: " if norwegian else "Live from Google Health: ") + text


# Set True if you want Jarvis to address you as "sir" occasionally.
ADDRESS_AS_SIR = False

JARVIS_SYSTEM_PROMPT = (
    "You are Jarvis, a calm, witty, highly capable AI assistant in the spirit of "
    "Tony Stark's AI from Iron Man. You are composed and unflappable, with a "
    "subtly dry, understated sense of humor -- never goofy, never over the top. "
    + (
        "Address the user respectfully as \"sir\" every so often, but don't overdo it. "
        if ADDRESS_AS_SIR
        else ""
    )
    + "Be concise and efficient rather than chatty: your replies are spoken out loud, "
    "not read as text, so keep them short and conversational -- generally a few "
    "sentences unless the user clearly wants detail. Be genuinely helpful and "
    "direct; skip the flattery and unnecessary apologies. A touch of wit is "
    "welcome, sycophancy is not. Reply in the same language the user wrote in. "
    "You do retain memory of everything said earlier in this current "
    "conversation -- use it naturally. That memory resets when the user starts "
    "a new conversation or after a few minutes of silence, but never claim you "
    "have no memory at all; if something genuinely wasn't mentioned yet, just "
    "say you don't have that information. You have a web_search tool -- use it "
    "for anything time-sensitive or past your training data (weather, news, "
    "scores, prices, current events, \"what's the latest on X\"), and answer "
    "from your own knowledge otherwise. When you do search, summarize what you "
    "found in a sentence or two in your own words -- never read back raw "
    "search result text, long excerpts, or lists of links. If a search turns "
    "up several things (e.g. multiple news stories), lead with the single most "
    "relevant one rather than listing them all -- the user can ask for more. "
    "You also have Spotify tools (play_music, pause_music, resume_music, "
    "skip_music, previous_music, set_volume) -- use them whenever the user "
    "asks to play, pause, resume, skip, go back, or change the volume of "
    "music. If a tool reports Spotify isn't connected, or that Premium is "
    "required, or that there's no active device, just relay that plainly "
    "and briefly -- don't apologize at length or guess at fixes. "
    "You also have job search tools. search_jobs looks up real, current "
    "openings and scores them against the candidate's own background -- "
    "call it with no keywords at all if the user didn't specify a role type "
    "(e.g. just 'find me some jobs'), and it'll search broadly using their "
    "real profile rather than a guessed default; pass keywords/location only "
    "when the user actually gives them, to narrow the search. Every result "
    "already shows in full in the UI's JOB SEARCH RESULTS panel (title, "
    "company, location, link, and why it's a fit for them specifically) -- "
    "so your reply should just be a brief, natural spoken summary (roughly "
    "how many you found and one or two standout highlights), NOT a repeat "
    "of the full list -- the user can see the details on screen. "
    "activate_job_hunting_mode turns on Job Hunting Mode (a persistent UI "
    "indicator) and runs one broad search across the candidate's whole "
    "range of interests, again shown in full in the panel -- use it when "
    "the user says something like 'job hunting mode'. "
    "deactivate_job_hunting_mode turns it back off -- use it when they say "
    "something like 'exit job hunting mode'. "
    "log_application records a job application, or a job the user is "
    "considering but hasn't applied to yet (status 'not applied'). "
    "update_application_status changes the status of one already logged -- "
    "use it for things like 'mark Kongsberg as interview' or 'I got "
    "rejected by Nammo', where they name the company without necessarily "
    "repeating the exact role; if it reports multiple matches, ask which "
    "role they meant and call it again with role set. "
    "IMPORTANT: activate_job_hunting_mode, deactivate_job_hunting_mode, "
    "log_application, and update_application_status are the ONLY things "
    "that actually change anything -- nothing is on, saved, or updated "
    "until you call the matching tool THIS turn, even for what feels like "
    "a trivial toggle or a small status update. Never reply as if Job "
    "Hunting Mode changed, or an application was logged or updated, unless "
    "you actually called that exact tool in this same turn and are "
    "relaying its real result -- saying so without calling it leaves the "
    "user thinking something happened when nothing did. "
    "get_application_status looks up or lists what they've applied to -- "
    "use it whenever they ask, rather than answering from memory of "
    "earlier in the conversation. "
    "You also have Norwegian public transit tools (Entur). "
    "get_next_departures gives real-time departures from a stop (omit the "
    "stop to use their saved default), and plan_journey gives directions "
    "between two places -- call them for any question about when something "
    "leaves or how to get somewhere by bus/train/tram/metro, and never "
    "state departure times or routes from memory. Keep the reply short and "
    "spoken-style, e.g. 'Next bus from Jernbanetorget is the 54 to Kjelsås, "
    "in 6 minutes, then another in 9' -- the first one or two departures "
    "plus anything notable (a delay or cancellation), not the whole list. "
    "set_default_stop saves their usual stop (also shown in the on-screen "
    "TRANSIT panel) -- like the job tools, it only takes effect if you call "
    "it this turn, so never say a default stop was saved unless you did, and "
    "name the exact stop the tool reports so they can catch a wrong match. "
    "You also have Google Health tools for the user's Fitbit/Pixel Watch data: "
    "get_sleep_summary, get_workout_summary and get_weekly_activity_summary. "
    "Call them for any question about sleep, workouts or activity and never "
    "state health numbers from memory. Answer conversationally and briefly -- "
    "lead with the headline (e.g. 'You slept about seven hours, with a good "
    "amount of deep sleep') plus one or two notable details, rather than "
    "reading every figure back; the user can ask for more. Google Health has "
    "no sleep score, so if asked, say it isn't available instead of guessing. "
    "Every health tool result starts with a label: '[LIVE DATA ...]' for the "
    "user's real Google Health numbers, or '[SAMPLE DATA ...]' for placeholders. "
    "Always say which in your reply, briefly ('Live from Google Health: ...' or "
    "'Sample data: ...'). For sleep, the headline is the TIME ASLEEP (not time in "
    "bed); when the result says a night was several sessions added together, "
    "give the total and mention that briefly. "
    "Call the matching health tool for EVERY new health question, even if an "
    "earlier answer in this conversation was sample data. "
    "IMPORTANT: if a health tool result begins with '[SAMPLE DATA', Google "
    "Health isn't connected yet and the figures are made-up placeholders. "
    "Still answer the question briefly using those placeholder figures, so "
    "the user can see how it will work, but open by saying plainly that it's "
    "sample data (e.g. 'Sample data, since Health isn't connected yet: you "
    "slept about seven hours...') and never present the numbers as the "
    "user's real results."
)

# Strips emoji / pictograph characters before they're handed to TTS (they're
# still shown when replies are printed to the terminal or the UI).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # emoticons, symbols, transport, supplemental pictographs
    "\U00002600-\U000027BF"   # misc symbols & dingbats
    "\U00002300-\U000023FF"   # misc technical (watch, hourglass, etc.)
    "\U00002B00-\U00002BFF"   # stars, arrows
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0000FE0F"              # variation selector-16 (emoji presentation)
    "\U0000200D"              # zero-width joiner (combined emoji)
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text):
    return _EMOJI_PATTERN.sub("", text).strip()


# Strips URLs before text is handed to TTS -- same idea as strip_emojis:
# a raw link (e.g. from search_jobs results) reads out as unintelligible
# noise when spoken aloud, but is genuinely useful in the on-screen
# conversation log, so it stays in the text that's displayed/stored and is
# only removed from what's actually spoken.
_URL_PATTERN = re.compile(r"https?://\S+")


def strip_urls_for_speech(text):
    without_urls = _URL_PATTERN.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", without_urls).strip()


# ---- Which voice speaks a reply? Decided from the TEXT, not the input method.
# Typed messages used to be hard-coded as "en", so a Norwegian reply to a typed
# message was read by the English voice (measured: 70% word errors when the
# result was transcribed back -- essentially unintelligible). Whisper's tag only
# exists for spoken input, and only describes what the user said, not the
# language of Claude's reply. So both the reply and the user's message are
# classified directly. Only the three supported languages matter here.
# Words chosen to be distinctive: ones that are also common in the *other*
# language (Norwegian "to"=two, "i", "at", "for", "over", "bare"...) are left out.
_NORWEGIAN_WORDS = frozenset(
    "og jeg ikke det er på til som med har av å vi du den kan så fra vil skal var deg "
    "meg hva hvor når også være blir dette noe ved etter litt ja nei takk hei god "
    "neste går om minutter timer dag været har fant ble hvordan hvilken vært".split()
)
_ENGLISH_WORDS = frozenset(
    "the and is are you your of that this it with have was will can not be what when "
    "where which my from there they would could should just here been were has had "
    "hello thanks yes no please".split()
)


def detect_language(text):
    """'ur', 'no' or 'en' for `text`, or None when there isn't enough signal
    (e.g. a one-word reply) -- callers then fall back to something else."""
    if not text:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    arabic_script = sum(
        "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" or "ﭐ" <= c <= "﷿"
        or "ﹰ" <= c <= "﻿" for c in letters
    )
    if arabic_script / len(letters) > 0.3:
        return "ur"
    lowered = text.lower()
    words = re.findall(r"[a-zæøåé']+", lowered)
    no = sum(w in _NORWEGIAN_WORDS for w in words)
    en = sum(w in _ENGLISH_WORDS for w in words)
    if any(c in lowered for c in "æøå"):
        no += 2  # æ/ø/å is a strong Norwegian signal (English text never has them)
    if no > en and no >= 2:
        return "no"
    if en > no and en >= 1:
        return "en"
    if en == 0 and no == 0 and not any(c in lowered for c in "æøå"):
        # No stopwords either way: plain ASCII text of a few words is far more
        # likely English than anything else here, but a single word is a guess.
        return "en" if len(words) >= 4 else None
    return None


# ---- Setup ----
print("Loading wake word model...")
owwModel = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print("Loading Whisper model...")
# faster-whisper (CTranslate2) is a CPU-optimized reimplementation of the same
# "small" model -- same accuracy, much faster inference than openai-whisper.
# int8 quantization trades a hair of precision for a large CPU speedup.
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# Reads ANTHROPIC_API_KEY from your environment variable automatically
client = Anthropic()

audio = pyaudio.PyAudio()
pygame.mixer.init()


def _make_confirmation_chime():
    """Builds a short, subtle two-tone "command captured" chime entirely in
    memory (no asset file, no network call) -- a quick rising blip so it
    reads as an acknowledgment rather than an alert. Played on its own
    pygame Channel/Sound (not pygame.mixer.music, which speak() uses for
    TTS) so the two never conflict."""
    sample_rate = pygame.mixer.get_init()[0] if pygame.mixer.get_init() else 44100
    tones = [(880, 0.06), (1320, 0.08)]  # (Hz, seconds) -- short rising blip
    gap_seconds = 0.015
    pieces = []
    for freq, duration in tones:
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        tone = np.sin(2 * np.pi * freq * t)
        # Fade in/out to avoid audible clicks at the start/end of each tone.
        fade = min(int(sample_rate * 0.01), n // 2)
        if fade > 0:
            envelope = np.ones(n)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            tone *= envelope
        pieces.append(tone)
        pieces.append(np.zeros(int(sample_rate * gap_seconds)))
    waveform = np.concatenate(pieces) * 0.25 * 32767  # quiet -- a cue, not an alert
    stereo = np.column_stack([waveform, waveform]).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


try:
    _confirmation_chime = _make_confirmation_chime()
except Exception as exc:
    print(f"[Could not build confirmation chime, skipping]: {exc}")
    _confirmation_chime = None


def play_confirmation_chime():
    """Fire-and-forget playback of the "command captured" cue. Never
    raises or blocks the pipeline."""
    if _confirmation_chime is None:
        return
    try:
        _confirmation_chime.play()
    except Exception as exc:
        print(f"[Chime playback error, continuing]: {exc}")

# Serializes full interactions (wake word / button / typed) so only one
# record-transcribe-ask-speak sequence runs at a time.
interaction_lock = threading.Lock()

# Set by the UI's Cancel button / Escape key to interrupt whatever the
# current interaction is doing (listening, thinking, or speaking) and
# return to idle. Cleared at the start of every new interaction.
cancel_event = threading.Event()

# The Claude SDK call and edge-tts synthesis are both blocking with no
# built-in cancellation, so they run on this small pool -- ask_claude() and
# speak() poll their future instead of blocking directly, so either can
# bail out as soon as cancel_event is set instead of waiting for the call
# to finish. An abandoned call keeps running here in the background; its
# result is simply discarded.
_background_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="jarvis-bg"
)


SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# Keyboard is the primary input; voice is optional. Defaults keep the
# previous behavior (wake word listening on, replies spoken) until toggled
# in the UI -- toggles are persisted to SETTINGS_FILE so they survive restarts.
DEFAULT_SETTINGS = {
    "voice_enabled": True,         # wake-word listening on/off (mic released when off)
    "speak_typed_replies": True,   # read replies aloud for typed messages
}


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        import json
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)
        for key in DEFAULT_SETTINGS:
            if isinstance(stored.get(key), bool):
                settings[key] = stored[key]
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[Settings read error, using defaults]: {exc}")
    return settings


def save_settings(settings):
    try:
        import json
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({k: settings[k] for k in DEFAULT_SETTINGS}, f, indent=2)
    except OSError as exc:
        print(f"[Settings write error]: {exc}")


class JarvisState:
    """Thread-safe status -- including conversation history -- shared
    between the wake-word loop, manual UI triggers, and the web UI that
    displays them."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "Idle"
        self.last_transcript = ""
        self.last_reply = ""
        self.pending_user_text = ""    # message being answered right now, shown in the log immediately
        self.history = []          # [{"role": "user"/"assistant", "content": str}, ...]
        self.last_activity = time.time()
        self.follow_up_active = False
        self.job_hunting_mode = False
        self.job_search_results = []   # [{"title", "company", "location", "url", "reason"}, ...]
        settings = load_settings()
        self.voice_enabled = settings["voice_enabled"]
        self.speak_typed_replies = settings["speak_typed_replies"]
        self.focus_request = 0         # bumped by the global hotkey; the UI focuses its input when it changes

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self):
        with self._lock:
            return {
                "status": self.status,
                "last_transcript": self.last_transcript,
                "last_reply": self.last_reply,
                "pending_user_text": self.pending_user_text,
                "history": list(self.history),
                "follow_up_active": self.follow_up_active,
                "job_hunting_mode": self.job_hunting_mode,
                "voice_enabled": self.voice_enabled,
                "speak_typed_replies": self.speak_typed_replies,
                "focus_request": self.focus_request,
            }

    def set_preferences(self, voice_enabled=None, speak_typed_replies=None):
        """Updates and persists the user-toggleable settings."""
        with self._lock:
            if voice_enabled is not None:
                self.voice_enabled = bool(voice_enabled)
            if speak_typed_replies is not None:
                self.speak_typed_replies = bool(speak_typed_replies)
            settings = {
                "voice_enabled": self.voice_enabled,
                "speak_typed_replies": self.speak_typed_replies,
            }
        save_settings(settings)

    def voice_is_enabled(self):
        with self._lock:
            return self.voice_enabled

    def should_speak_typed(self):
        with self._lock:
            return self.speak_typed_replies

    def request_focus(self):
        with self._lock:
            self.focus_request += 1

    def snapshot_job_search_results(self):
        with self._lock:
            return list(self.job_search_results)

    def touch_and_maybe_reset(self):
        """Call once at the start of a new top-level interaction: clears
        the conversation if it's been idle longer than CONVERSATION_TIMEOUT,
        then marks this moment as the latest activity."""
        with self._lock:
            now = time.time()
            if self.history and (now - self.last_activity) > CONVERSATION_TIMEOUT:
                self.history = []
            self.last_activity = now

    def append_exchange(self, user_text, reply_text):
        with self._lock:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply_text})
            if len(self.history) > MAX_HISTORY_MESSAGES:
                self.history = self.history[-MAX_HISTORY_MESSAGES:]
            self.pending_user_text = ""
            self.last_activity = time.time()

    def clear_history(self):
        with self._lock:
            self.history = []
            self.last_transcript = ""
            self.last_reply = ""
            self.job_hunting_mode = False


def speak(text, voice=DEFAULT_VOICE, cancel_evt=None):
    """Synthesizes `text` with edge-tts and plays it back. Never raises --
    a TTS/network failure just gets logged and skipped. If `cancel_evt` is
    set at any point -- including during synthesis, before any audio has
    played -- stops/abandons promptly and returns.

    Synthesis (a network call with no built-in cancellation) runs on
    _background_executor so this can poll for cancellation instead of
    blocking on it directly, same pattern as ask_claude()."""
    clean_text = strip_emojis(text)
    if not clean_text:
        return
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="jarvis_tts_")
        os.close(fd)

        def _synthesize():
            asyncio.run(edge_tts.Communicate(clean_text, voice=voice).save(path))

        future = _background_executor.submit(_synthesize)
        synthesized = False
        while cancel_evt is None or not cancel_evt.is_set():
            try:
                future.result(timeout=0.05)
                synthesized = True
                break
            except concurrent.futures.TimeoutError:
                continue
        if not synthesized:
            return  # cancelled while synthesizing, before any audio played

        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if cancel_evt is not None and cancel_evt.is_set():
                pygame.mixer.music.stop()
                break
            pygame.time.wait(25)  # tight poll so we don't linger after playback ends
        pygame.mixer.music.unload()
    except Exception as exc:
        print(f"[TTS error, continuing without speech]: {exc}")
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def open_listen_stream():
    return audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                       input=True, frames_per_buffer=CHUNK)


def _rms(frame_bytes):
    """Root-mean-square energy of a chunk of int16 PCM audio."""
    samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _normalize_volume(raw_bytes, target_peak_ratio=0.7, max_gain=10.0):
    """Boosts quiet int16 PCM audio toward `target_peak_ratio` of full
    scale so Whisper gets a stronger signal. Never reduces volume, and
    caps the gain so near-silent noise doesn't get amplified absurdly."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    if samples.size == 0:
        return raw_bytes
    peak = int(np.abs(samples).max())
    if peak == 0:
        return raw_bytes
    gain = min((target_peak_ratio * 32767) / peak, max_gain)
    if gain <= 1.0:
        return raw_bytes
    boosted = np.clip(samples.astype(np.float64) * gain, -32768, 32767).astype(np.int16)
    return boosted.tobytes()


_VAD_FRAME_MS = 20
_VAD_FRAME_BYTES = int(RATE * (_VAD_FRAME_MS / 1000.0)) * 2  # 16-bit samples


def _is_speech_chunk(data, rms, silence_threshold):
    """Classifies one CHUNK (80ms) of audio as speech-like or not.

    RMS energy is the first gate -- below the (room-adaptive) silence
    threshold is always treated as silence, VAD or not. Above it, if VAD is
    available, the chunk is split into the 20ms sub-frames webrtcvad
    requires and majority-voted, which helps tell the user's voice apart
    from loud non-speech noise (TV, music, a fan) that would otherwise pass
    the energy check alone. VAD can't distinguish the user's own voice from
    someone else's speech in the room -- that's a hard limit of any
    single-mic setup, not something tunable away.
    """
    if rms < silence_threshold:
        return False
    if _vad is None:
        return True
    votes = 0
    total = 0
    for offset in range(0, len(data) - _VAD_FRAME_BYTES + 1, _VAD_FRAME_BYTES):
        sub = data[offset:offset + _VAD_FRAME_BYTES]
        total += 1
        try:
            if _vad.is_speech(sub, RATE):
                votes += 1
        except Exception:
            return True  # VAD failed on this frame -- fall back to energy-only
    if total == 0:
        return True
    return votes / total >= 0.5


def _capture_utterance(next_chunk, max_wait_seconds=None, cancel_evt=None):
    """Reads chunks from `next_chunk()` (a callable returning CHUNK-sized
    raw bytes) and returns one spoken utterance as a list of raw chunk
    bytes, or None if `max_wait_seconds` is given and no speech ever
    started, or if `cancel_evt` gets set mid-capture.

    - max_wait_seconds=None (normal "record a command"): starts capturing
      immediately and keeps going until roughly SILENCE_DURATION seconds
      of continuous non-speech audio follow at least MIN_RECORD_SECONDS of
      capture, or MAX_RECORD_SECONDS is hit as a safety cap.
    - max_wait_seconds=N (follow-up listening): first waits up to N
      seconds for speech to appear at all; if it never does, returns None.
      Once speech begins, the same min-record/silence-stop logic takes
      over.

    The "silence" threshold is estimated from the first
    AMBIENT_SAMPLE_SECONDS of this very recording, so it adapts to the
    room/mic instead of relying on one fixed magic number. Each chunk is
    then classified speech/non-speech by _is_speech_chunk (energy + VAD).
    A run of non-speech chunks needed to stop recording decays rather than
    resets on an occasional speech-like blip (SILENCE_RUN_DECAY), so a
    single noise spike during the trailing pause doesn't restart the whole
    countdown -- this is what makes the cutoff reliable across different
    speaking volumes/pause lengths and in a noisier room.

    The returned frames are trimmed to the detected speech span (plus
    TRIM_PADDING_CHUNKS of padding on each side) rather than including the
    full trailing silence window, which both tightens up latency and gives
    Whisper less silence/noise to potentially mishear.
    """
    chunk_duration = CHUNK / RATE
    min_chunks = max(1, int(MIN_RECORD_SECONDS / chunk_duration))
    silence_chunks_needed = max(1, int(SILENCE_DURATION / chunk_duration))
    max_chunks = int(MAX_RECORD_SECONDS / chunk_duration)
    ambient_chunks = max(1, int(AMBIENT_SAMPLE_SECONDS / chunk_duration))
    max_wait_chunks = int(max_wait_seconds / chunk_duration) if max_wait_seconds else None

    frames = []
    ambient_levels = []
    silence_threshold = None
    speech_started = max_wait_seconds is None
    silent_run = 0
    first_speech_idx = None
    last_speech_idx = None

    for i in range(max_chunks):
        if cancel_evt is not None and cancel_evt.is_set():
            return None

        if max_wait_chunks is not None and not speech_started and i >= max_wait_chunks:
            return None

        data = next_chunk()
        frames.append(data)
        rms = _rms(data)

        if i < ambient_chunks:
            ambient_levels.append(rms)
            continue

        if silence_threshold is None:
            noise_floor = max(ambient_levels) if ambient_levels else 0.0
            silence_threshold = max(noise_floor * SILENCE_MULTIPLIER, MIN_SILENCE_THRESHOLD)

        is_speech = _is_speech_chunk(data, rms, silence_threshold)
        if is_speech:
            if first_speech_idx is None:
                first_speech_idx = i
            last_speech_idx = i

        if not speech_started:
            if is_speech:
                speech_started = True
            continue

        if is_speech:
            silent_run = max(0, silent_run - SILENCE_RUN_DECAY)
        else:
            silent_run += 1

        if len(frames) >= min_chunks and silent_run >= silence_chunks_needed:
            break

    if max_wait_seconds is not None and not speech_started:
        return None

    if first_speech_idx is None:
        return frames  # nothing classified as speech -- return as-is, don't guess

    start = max(0, first_speech_idx - TRIM_PADDING_CHUNKS)
    end = min(len(frames), last_speech_idx + TRIM_PADDING_CHUNKS + 1)
    return frames[start:end]


def record_command(max_wait_seconds=None, cancel_evt=None):
    """Records one spoken utterance, stopping automatically once the user
    goes quiet (see _capture_utterance). If `max_wait_seconds` is given,
    first waits that long for speech to begin at all (used for follow-up
    listening). Returns True on success, False if the mic failed, or None
    if max_wait_seconds expired with nothing said or `cancel_evt` fired."""
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                             input=True, frames_per_buffer=CHUNK)
    except OSError as exc:
        print(f"[Mic error, could not open input stream]: {exc}")
        return False

    print("Listening for your command...")
    try:
        frames = _capture_utterance(
            lambda: stream.read(CHUNK, exception_on_overflow=False),
            max_wait_seconds=max_wait_seconds,
            cancel_evt=cancel_evt,
        )
    except OSError as exc:
        print(f"[Mic error while recording]: {exc}")
        return False
    finally:
        stream.stop_stream()
        stream.close()

    if frames is None:
        return None

    raw = _normalize_volume(b"".join(frames))

    try:
        wf = wave.open(COMMAND_WAV, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(raw)
        wf.close()
    except OSError as exc:
        print(f"[Error writing {COMMAND_WAV}]: {exc}")
        return False

    return True


def transcribe_command():
    """Runs Whisper on the recorded command, auto-detecting the spoken
    language. Returns (text, language_code), or (None, None) on failure."""
    try:
        segments, info = whisper_model.transcribe(COMMAND_WAV)
        text = "".join(segment.text for segment in segments).strip()
        language = info.language or "en"
        return text, language
    except FileNotFoundError as exc:
        print(f"[Transcription error, ffmpeg not found]: {exc}")
        return None, None
    except Exception as exc:
        print(f"[Transcription error]: {exc}")
        return None, None


def _call_claude(user_text, history, state=None):
    """The actual (blocking) Claude round trip, including the manual loop
    for client-side tool calls (Spotify, job search/tracking) and
    pause_turn continuations for the server-side web_search tool. Runs on
    _background_executor -- see ask_claude() for why. `state` is threaded
    through to tool calls that need to update shared UI state directly
    (job_search_results, job_hunting_mode) independent of Claude's own
    reply text."""
    messages = list(history or []) + [{"role": "user", "content": user_text}]
    all_tools = [WEB_SEARCH_TOOL] + SPOTIFY_TOOLS + JOB_TOOLS + TRANSIT_TOOLS + HEALTH_TOOLS
    response = None
    health_sources = set()   # "live" / "sample" for every health tool result this turn

    for _ in range(6):  # caps total round trips (search pauses + tool calls)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=JARVIS_SYSTEM_PROMPT,
            tools=all_tools,
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name in _CLIENT_TOOL_NAMES:
                    content = _execute_client_tool(block.name, block.input, state=state)
                    source = _health_source(block.name, content)
                    if source:
                        health_sources.add(source)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            if not tool_results:
                break  # tool_use for something we don't handle -- bail out
            messages.append({"role": "user", "content": tool_results})
            continue

        break  # end_turn (or anything else) -- final answer is in `response`

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return _ensure_health_source(text, health_sources)


def ask_claude(user_text, history=None, cancel_evt=None, state=None):
    """Sends the transcribed text to Claude -- with prior conversation
    history, the web_search tool, and the Spotify/job tools available --
    and returns the reply text, or None on failure/cancellation.

    The Claude SDK call is blocking with no built-in cancellation, so the
    actual work runs on _background_executor and this function just polls the
    future, checking `cancel_evt` between polls -- letting it return almost
    immediately on cancel instead of waiting for the HTTP call to finish.
    An abandoned call keeps running in the background; its result is simply
    never used.
    """
    future = _background_executor.submit(_call_claude, user_text, history, state)
    while cancel_evt is None or not cancel_evt.is_set():
        try:
            return future.result(timeout=0.05)
        except concurrent.futures.TimeoutError:
            continue
        except APIConnectionError as exc:
            print(f"[Claude API connection error]: {exc}")
            return None
        except APIError as exc:
            print(f"[Claude API error]: {exc}")
            return None
        except Exception as exc:
            print(f"[Unexpected error calling Claude]: {exc}")
            return None

    print("[Claude call cancelled by user]")
    return None


def _get_utterance(state, typed_text, cancel_evt=None):
    """Gets the first utterance of an interaction: the given typed text,
    or a fresh mic recording + transcription. Returns (text, language),
    or (None, None) on mic failure / no speech recognized / cancellation."""
    if typed_text is not None:
        state.update(status="Thinking")
        return typed_text, detect_language(typed_text) or "en"

    state.update(status="Listening")
    ok = record_command(cancel_evt=cancel_evt)
    if ok is False:
        speak("Sorry, I had a microphone problem.")
        return None, None
    if ok is None:
        return None, None  # cancelled

    play_confirmation_chime()
    state.update(status="Thinking")
    text, language = transcribe_command()
    if not text:
        print("[No speech recognized]")
        return None, None
    return text, language


def run_interaction(state, typed_text=None):
    """Runs a Jarvis interaction and updates `state` at each stage:
    - typed_text is None: record + transcribe (auto language) the mic, as
      used by the wake word and the "Talk to Jarvis" button.
    - typed_text given: skip audio entirely, as used by the UI text box.

    Each turn is asked with the running conversation history for context,
    then appended to it. After speaking, listens for a follow-up for
    FOLLOWUP_WINDOW_SECONDS without needing the wake word again; if the
    user keeps talking this repeats, otherwise control returns to normal
    wake-word-only listening. Only one interaction runs at a time; a
    second caller while one is in flight is simply ignored.

    Checks `cancel_event` (module-level) after every stage -- listening,
    thinking, speaking -- so the UI's Cancel button / Escape key can
    interrupt at any point and land cleanly back on Idle.
    """
    if not interaction_lock.acquire(blocking=False):
        print("[Busy with another interaction, ignoring]")
        return
    typed = typed_text is not None
    try:
        cancel_event.clear()
        if typed:
            # Show the message in the log and flip to Thinking right away --
            # nothing below is allowed to delay that for typed input.
            state.update(pending_user_text=typed_text, status="Thinking")
        state.touch_and_maybe_reset()
        text, language = _get_utterance(state, typed_text, cancel_evt=cancel_event)
        if not text:
            return

        while True:
            print(f"You said: {text}")
            state.update(last_transcript=text, pending_user_text=text)

            if _is_job_hunting_deactivate_command(text):
                state.update(job_hunting_mode=False)
                reply = "Job Hunting Mode deactivated."
            else:
                history = state.snapshot()["history"]
                reply = ask_claude(text, history=history, cancel_evt=cancel_event, state=state)
                if cancel_event.is_set():
                    print("[Cancelled while thinking]")
                    return
                if not reply:
                    # Show the failure in the log too -- with typed input and
                    # speech off there's otherwise no sign anything went wrong.
                    reply = "Sorry, I couldn't reach Claude right now."

            print(f"Jarvis: {reply}")
            state.append_exchange(text, reply)
            state.update(last_reply=reply)

            # Voice-initiated turns are always answered aloud; for typed
            # messages it's the "speak replies" toggle. The reply text is
            # already in the log by this point, so speech never delays it.
            if not typed or state.should_speak_typed():
                state.update(status="Speaking")
                # Voice follows the language of the reply itself; the user's
                # own language (typed-text detection / Whisper tag) is only
                # the fallback for replies too short to classify ("Ti.").
                spoken_language = detect_language(reply) or language
                voice = VOICE_BY_LANGUAGE.get(spoken_language, DEFAULT_VOICE)
                print(f"[Voice] {voice} (reply language: {spoken_language})")
                speak(strip_urls_for_speech(reply), voice, cancel_evt=cancel_event)
                if cancel_event.is_set():
                    print("[Cancelled while speaking]")
                    return

            if reply == "Sorry, I couldn't reach Claude right now.":
                return

            # Typed input ends here: no mic, no follow-up window. (Follow-up
            # listening exists so a spoken back-and-forth doesn't need the
            # wake word every time; it makes no sense after typing, and it
            # kept the interaction lock held -- blocking the next message.)
            if typed:
                return

            # Follow-up window: listen again without requiring the wake word.
            state.update(status="Listening", follow_up_active=True)
            follow_up_ok = record_command(
                max_wait_seconds=FOLLOWUP_WINDOW_SECONDS, cancel_evt=cancel_event
            )
            state.update(follow_up_active=False)

            if not follow_up_ok:  # False (mic error), None (nothing said / cancelled)
                return

            play_confirmation_chime()
            state.update(status="Thinking")
            text, language = transcribe_command()
            if not text:
                return
    finally:
        state.update(status="Idle", follow_up_active=False, pending_user_text="")
        interaction_lock.release()


def wake_word_loop(state, stop_event, pause_event):
    """Continuously listens for "hey jarvis" and runs an interaction on
    detection. Pauses (releasing the mic stream) whenever `pause_event` is
    set, so a UI-triggered manual interaction can use the mic exclusively.
    """
    listen_stream = None
    print("Listening for wake word 'Hey Jarvis'..." if state.voice_is_enabled()
          else "Voice (wake word) is off -- keyboard only. Toggle it in the UI.")

    try:
        while not stop_event.is_set():
            # Paused for a manual interaction, or voice turned off in the UI:
            # release the mic entirely rather than listening and ignoring it.
            if pause_event.is_set() or not state.voice_is_enabled():
                if listen_stream is not None:
                    try:
                        listen_stream.stop_stream()
                        listen_stream.close()
                    except Exception:
                        pass
                    listen_stream = None
                    if not state.voice_is_enabled():
                        print("[Voice off: wake-word listening stopped, microphone released]")
                time.sleep(0.1)
                continue

            if listen_stream is None:
                try:
                    listen_stream = open_listen_stream()
                except OSError as exc:
                    print(f"[Mic error opening listen stream, retrying]: {exc}")
                    time.sleep(0.5)
                    continue
                # Drop any audio the model buffered before the mic was last
                # released, so a stale half-heard phrase can't trigger it.
                try:
                    owwModel.reset()
                except Exception:
                    pass
                print("[Microphone open: listening for wake word 'Hey Jarvis']")

            try:
                chunk = listen_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as exc:
                print(f"[Mic error while listening, reopening stream]: {exc}")
                try:
                    listen_stream.stop_stream()
                    listen_stream.close()
                except Exception:
                    pass
                listen_stream = None
                continue

            if _rms(chunk) < WAKE_MIN_ENERGY:
                # Near-silent chunk -- skip the (comparatively expensive)
                # wake word model entirely rather than running it on nothing.
                continue

            audio_data = np.frombuffer(chunk, dtype=np.int16)

            try:
                prediction = owwModel.predict(audio_data)
            except Exception as exc:
                print(f"[Wake word model error, skipping frame]: {exc}")
                continue

            best_score = max(prediction.values()) if prediction else 0.0
            if best_score > WAKE_LOG_THRESHOLD:
                print(f"[Wake word confidence] {best_score:.3f} "
                      f"(threshold {WAKE_THRESHOLD})")

            if any(score > WAKE_THRESHOLD for score in prediction.values()):
                print("Wake word detected!")
                listen_stream.stop_stream()
                listen_stream.close()
                listen_stream = None

                try:
                    run_interaction(state)
                except Exception as exc:
                    # Catch-all so one bad interaction never kills the loop.
                    print(f"[Unexpected error handling interaction]: {exc}")

                print("\nListening for wake word 'Hey Jarvis'...")

    finally:
        if listen_stream is not None:
            try:
                listen_stream.stop_stream()
                listen_stream.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Jarvis voice assistant")
    parser.add_argument("--no-ui", action="store_true",
                         help="run in console-only mode, no desktop UI")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    state = JarvisState()
    stop_event = threading.Event()
    pause_event = threading.Event()

    wake_thread = threading.Thread(
        target=wake_word_loop, args=(state, stop_event, pause_event), daemon=True
    )
    wake_thread.start()

    try:
        if args.no_ui:
            print("Running console-only (Ctrl+C to stop)...")
            while wake_thread.is_alive():
                wake_thread.join(timeout=0.5)
        else:
            import web_app
            web_app.run(state, pause_event, stop_event)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        wake_thread.join(timeout=2)
        audio.terminate()


if __name__ == "__main__":
    # web_app.py does `import jarvis` to reach run_interaction/interaction_lock/
    # etc. Since this file is running as __main__ here, that import wouldn't
    # find it in sys.modules under the name "jarvis" and would re-execute the
    # whole module from scratch -- a second owwModel, whisper_model, audio
    # (PyAudio) instance, and critically a second, *separate* interaction_lock,
    # so wake-word and web-triggered interactions could run concurrently and
    # collide over the mic and command.wav. Aliasing this already-loaded
    # module under "jarvis" makes web_app's import reuse it instead.
    sys.modules.setdefault("jarvis", sys.modules[__name__])
    main()
