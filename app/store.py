"""
Durable state.

The previous version kept drafts and paused conversations in plain module-level
dicts. That is fine on a laptop you never restart, but on Railway the container
is recycled on every deploy, crash or idle-sleep — the agent would forget a
half-finished conversation mid-sentence and the user would have to start over.

Everything here is written through to JSON on DATA_DIR with atomic replace, so
state survives restarts as long as DATA_DIR points at a mounted volume.

Writes are cheap (a few KB) and infrequent (once per Telegram turn), so plain
JSON is the right level of machinery here — no database needed.
"""

import json
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from config import (
    APPLICATIONS_PATH,
    CONVERSATIONS_PATH,
    DATA_DIR,
    NOTES_PATH,
    PROFILE_PATH,
    SEED_DIR,
    logger,
)

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Low-level IO
# ---------------------------------------------------------------------------


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not read {path}: {e} — falling back to default.")
        return default


def _write_json(path, data) -> bool:
    """Atomic write: temp file in the same dir, then os.replace."""
    with _LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception as e:
            logger.error(f"Could not write {path}: {e}")
            return False


def bootstrap() -> None:
    """
    On first boot against an empty volume, copy the seed profile/CV shipped in
    the repo into DATA_DIR so the agent has something to work with immediately.
    Never overwrites an existing file.
    """
    if DATA_DIR.resolve() == SEED_DIR.resolve():
        return
    for name in ("profile.json", "cv.pdf"):
        src = SEED_DIR / name
        dst = DATA_DIR / name
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
                logger.info(f"Bootstrapped {name} into {DATA_DIR}")
            except Exception as e:
                logger.warning(f"Could not bootstrap {name}: {e}")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def load_profile() -> Dict[str, Any]:
    profile = _read_json(PROFILE_PATH, {})
    if not profile:
        logger.warning(f"No profile found at {PROFILE_PATH}.")
    return profile


def save_profile(profile: Dict[str, Any]) -> bool:
    return _write_json(PROFILE_PATH, profile)


def update_profile_field(field: str, value: Any) -> Dict[str, Any]:
    """Set one top-level profile field and persist. Returns the new profile."""
    with _LOCK:
        profile = load_profile()
        profile[field] = value
        save_profile(profile)
        return profile


def profile_summary_text(profile: Optional[Dict[str, Any]] = None) -> str:
    p = profile if profile is not None else load_profile()
    if not p:
        return "No profile is set yet."
    skills = p.get("skills") or []
    lines = [
        f"Name: {p.get('name', '—')}",
        f"Headline: {p.get('headline', '—')}",
        f"Location: {p.get('location', '—')}",
        f"Email: {p.get('email', '—')}",
        f"Target roles: {', '.join(p.get('target_roles') or []) or '—'}",
        f"Experience entries: {len(p.get('experience') or [])}",
        f"Projects: {len(p.get('projects') or [])}",
        f"Skills on file: {len(skills)}",
        f"Certifications: {len(p.get('certifications') or [])}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Applications tracker
# ---------------------------------------------------------------------------


def load_applications() -> List[Dict[str, Any]]:
    data = _read_json(APPLICATIONS_PATH, [])
    return data if isinstance(data, list) else []


def log_application(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append (or update by job_url) an application record."""
    with _LOCK:
        apps = load_applications()
        entry.setdefault("created_at", time.time())
        entry["updated_at"] = time.time()
        url = (entry.get("job_url") or "").strip()

        if url:
            for i, existing in enumerate(apps):
                if (existing.get("job_url") or "").strip() == url:
                    existing.update(entry)
                    _write_json(APPLICATIONS_PATH, apps)
                    return existing

        entry.setdefault("id", f"app_{int(time.time() * 1000)}")
        apps.append(entry)
        _write_json(APPLICATIONS_PATH, apps)
        return entry


def set_application_status(app_id_or_url: str, status: str) -> bool:
    with _LOCK:
        apps = load_applications()
        for a in apps:
            if a.get("id") == app_id_or_url or a.get("job_url") == app_id_or_url:
                a["status"] = status
                a["updated_at"] = time.time()
                return _write_json(APPLICATIONS_PATH, apps)
    return False


def application_stats() -> Dict[str, Any]:
    apps = load_applications()
    by_status: Dict[str, int] = {}
    for a in apps:
        by_status[a.get("status", "unknown")] = by_status.get(a.get("status", "unknown"), 0) + 1
    week_ago = time.time() - 7 * 86400
    return {
        "total": len(apps),
        "by_status": by_status,
        "last_7_days": sum(1 for a in apps if (a.get("created_at") or 0) >= week_ago),
    }


# ---------------------------------------------------------------------------
# Conversations (one thread per Telegram user)
# ---------------------------------------------------------------------------


def _load_all_conversations() -> Dict[str, Any]:
    data = _read_json(CONVERSATIONS_PATH, {})
    return data if isinstance(data, dict) else {}


def load_conversation(user_id: int) -> List[Dict[str, Any]]:
    convo = _load_all_conversations().get(str(user_id)) or {}
    msgs = convo.get("messages") or []
    return msgs if isinstance(msgs, list) else []


def save_conversation(user_id: int, messages: List[Dict[str, Any]]) -> bool:
    """
    Merge into the user's slot — never replace it. Replacing was silently
    dropping the staged draft (CV path, email body) that lives in the same slot,
    so by the time the user tapped Approve there was nothing left to send.
    """
    with _LOCK:
        all_convos = _load_all_conversations()
        slot = all_convos.setdefault(str(user_id), {})
        slot["messages"] = messages
        slot["updated_at"] = time.time()
        return _write_json(CONVERSATIONS_PATH, all_convos)


def clear_conversation(user_id: int) -> bool:
    with _LOCK:
        all_convos = _load_all_conversations()
        all_convos.pop(str(user_id), None)
        return _write_json(CONVERSATIONS_PATH, all_convos)


# ---------------------------------------------------------------------------
# Drafts (pending approval) — stored alongside the conversation
# ---------------------------------------------------------------------------


def save_draft(user_id: int, draft: Dict[str, Any]) -> bool:
    with _LOCK:
        all_convos = _load_all_conversations()
        slot = all_convos.setdefault(str(user_id), {})
        slot["draft"] = draft
        slot["updated_at"] = time.time()
        return _write_json(CONVERSATIONS_PATH, all_convos)


def load_draft(user_id: int) -> Optional[Dict[str, Any]]:
    return (_load_all_conversations().get(str(user_id)) or {}).get("draft")


def clear_draft(user_id: int) -> bool:
    with _LOCK:
        all_convos = _load_all_conversations()
        slot = all_convos.get(str(user_id))
        if not slot:
            return True
        slot.pop("draft", None)
        return _write_json(CONVERSATIONS_PATH, all_convos)


# ---------------------------------------------------------------------------
# Free-form notes the agent is told to remember
# ---------------------------------------------------------------------------


def load_notes() -> List[Dict[str, Any]]:
    data = _read_json(NOTES_PATH, [])
    return data if isinstance(data, list) else []


def add_note(text: str) -> Dict[str, Any]:
    with _LOCK:
        notes = load_notes()
        note = {"text": text, "created_at": time.time()}
        notes.append(note)
        notes = notes[-100:]  # keep it bounded
        _write_json(NOTES_PATH, notes)
        return note
