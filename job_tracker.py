"""Local job-application tracker -- a simple JSON file on disk, no external
database. Companion to the log_application/update_application_status/
get_application_status tools in jarvis.py, and to the "JOB APPLICATIONS"
panel in the web UI (which can also update status directly).
"""
import json
import os
import threading
import uuid
from datetime import date

APPLICATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "applications.json")
VALID_STATUSES = ["not applied", "applied", "interview", "rejected", "offer"]

_lock = threading.Lock()


def _load():
    if not os.path.exists(APPLICATIONS_FILE):
        return []
    try:
        with open(APPLICATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[Application log read error, treating as empty]: {exc}")
        return []


def _save(entries):
    with open(APPLICATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def log_application(company, role, status="applied", date_applied=None):
    """Records a new application, or -- if one already exists for the same
    company + role (case-insensitive) -- updates its status instead of
    creating a duplicate. This lets "I got an interview with X for the Y
    role" naturally update the existing "applied" entry rather than
    piling up a second row for the same job."""
    company = (company or "").strip()
    role = (role or "").strip()
    if not company or not role:
        return "Need both a company and a role to log an application."

    status = (status or "applied").strip().lower()
    if status not in VALID_STATUSES:
        status = "applied"

    with _lock:
        entries = _load()
        existing = next(
            (e for e in entries
             if e["company"].lower() == company.lower() and e["role"].lower() == role.lower()),
            None,
        )
        if existing:
            existing["status"] = status
            if date_applied:
                existing["date_applied"] = date_applied
            _save(entries)
            return f"Updated: {role} at {company} is now '{status}'."

        entry = {
            "id": uuid.uuid4().hex[:8],
            "company": company,
            "role": role,
            "date_applied": date_applied or date.today().isoformat(),
            "status": status,
        }
        entries.append(entry)
        _save(entries)
        return f"Logged: {role} at {company} ({status}, {entry['date_applied']})."


def update_status(company, status, role=None):
    """Updates the status of an existing application, matched by company
    (and role, if given) via case-insensitive substring match. Used for
    voice commands like "mark Kongsberg as interview" that name the
    company without necessarily repeating the exact role. If more than one
    application matches, nothing is changed and the caller (Claude) is told
    to ask the user to narrow it down by role."""
    status = (status or "").strip().lower()
    if status not in VALID_STATUSES:
        return f"'{status}' isn't a recognized status ({', '.join(VALID_STATUSES)})."
    company = (company or "").strip()
    if not company:
        return "Need a company name to update an application status."

    with _lock:
        entries = _load()
        matches = [e for e in entries if company.lower() in e["company"].lower()]
        if role:
            narrowed = [e for e in matches if role.lower() in e["role"].lower()]
            if narrowed:
                matches = narrowed
        if not matches:
            return f"No logged application found for {company}."
        if len(matches) > 1:
            options = "; ".join(f"{e['role']} at {e['company']}" for e in matches)
            return f"Multiple applications match '{company}': {options}. Which role did you mean?"
        matches[0]["status"] = status
        _save(entries)
        return f"Updated: {matches[0]['role']} at {matches[0]['company']} is now '{status}'."


def update_status_by_id(entry_id, status):
    """Same as update_status, but for the UI panel where the exact entry is
    already known by id (no ambiguity to resolve)."""
    status = (status or "").strip().lower()
    if status not in VALID_STATUSES:
        return False, f"'{status}' isn't a recognized status."
    with _lock:
        entries = _load()
        entry = next((e for e in entries if e["id"] == entry_id), None)
        if entry is None:
            return False, "Application not found."
        entry["status"] = status
        _save(entries)
    return True, f"Updated to '{status}'."


def list_applications(company=None, status=None):
    """Returns matching entries, most recently applied first. No filters
    returns everything."""
    entries = _load()
    if company:
        company = company.lower()
        entries = [e for e in entries if company in e["company"].lower()]
    if status:
        status = status.strip().lower()
        entries = [e for e in entries if e["status"] == status]
    return sorted(entries, key=lambda e: e["date_applied"], reverse=True)


def format_applications_for_speech(entries, max_items=10):
    """A concise, voice-friendly summary of a list of entries."""
    if not entries:
        return "No matching applications found."
    shown = entries[:max_items]
    lines = [f"{e['role']} at {e['company']}: {e['status']}, applied {e['date_applied']}" for e in shown]
    summary = "; ".join(lines)
    if len(entries) > max_items:
        summary += f"; and {len(entries) - max_items} more."
    return summary
