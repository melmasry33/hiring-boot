"""
Central configuration.

Everything is environment-driven so the exact same image runs locally and on
Railway. The only difference between the two is which env vars are set.

Railway notes:
- The filesystem is EPHEMERAL. Anything written outside a mounted volume is
  lost on every redeploy/restart. `DATA_DIR` therefore defaults to the repo's
  ./data locally, but you should point it at a Railway Volume mount path
  (e.g. /data) in production so applications history and profile edits survive.
- `PORT` is injected by Railway. We bind a tiny health server to it so the
  service passes healthchecks and doesn't get flagged as unresponsive.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_raw_data_dir = _env("DATA_DIR", str(BASE_DIR / "data"))
DATA_DIR = Path(_raw_data_dir).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Seed data shipped with the repo. If DATA_DIR is a fresh Railway volume, the
# bootstrap step in store.py copies these in on first boot.
SEED_DIR = BASE_DIR / "data"

PROFILE_PATH = DATA_DIR / "profile.json"
APPLICATIONS_PATH = DATA_DIR / "applications.json"
CONVERSATIONS_PATH = DATA_DIR / "conversations.json"
NOTES_PATH = DATA_DIR / "notes.json"

# Generated PDFs are disposable — they get regenerated on demand and are sent
# straight to Telegram, so keeping them in /tmp avoids bloating the volume.
GENERATED_CVS_DIR = Path(_env("GENERATED_CVS_DIR", "/tmp/generated_cvs"))
try:
    GENERATED_CVS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    GENERATED_CVS_DIR = DATA_DIR / "generated_cvs"
    GENERATED_CVS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM provider (OpenAI-compatible: NVIDIA NIM, Groq, OpenRouter, OpenAI, ...)
# ---------------------------------------------------------------------------
# LLM_* is the canonical name. NVIDIA_* is kept as a fallback so an existing
# .env from the previous version keeps working with no edits.

LLM_API_KEY = _env("LLM_API_KEY") or _env("NVIDIA_API_KEY")
_RAW_LLM_BASE_URL = (
    _env("LLM_BASE_URL")
    or _env("NVIDIA_BASE_URL")
    or "https://integrate.api.nvidia.com/v1"
)
# A base URL with no /v1 (or with /chat/completions pasted on the end) produces
# a 404 whose response body is EMPTY -- which the openai SDK renders as a bare
# "Error code: 404" with nothing after it. Normalise before anyone can trip on it.
from llm_preflight import normalize_base_url as _normalize_base_url  # noqa: E402

LLM_BASE_URL, LLM_BASE_URL_NOTES = _normalize_base_url(_RAW_LLM_BASE_URL)
LLM_MODEL = _env("LLM_MODEL") or _env("NVIDIA_MODEL") or "meta/muse-glimmer-30b"

# Ordered degradation chain. Tried in order when LLM_MODEL is not served by the
# endpoint (404) — a newly-released NIM that your key isn't entitled to yet is
# the common case, and it should not take the whole bot down.
LLM_FALLBACK_MODELS = [
    m.strip() for m in (_env("LLM_FALLBACK_MODELS") or "meta/muse-glimmer-30b").split(",") if m.strip()
]
LLM_MODEL_CHAIN = [LLM_MODEL] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL]
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 90)
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.3") or 0.3)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2048)

# Hard ceiling on tool-calling steps inside one agent turn.
MAX_TOOL_STEPS = _env_int("MAX_TOOL_STEPS", 12)
# How many messages of conversation history to keep per user before trimming.
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 40)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
_raw_allowed = _env("TELEGRAM_ALLOWED_USER_ID") or _env("TELEGRAM_ALLOWED_USER_IDS")
TELEGRAM_ALLOWED_USER_IDS = [
    int(uid.strip()) for uid in _raw_allowed.split(",") if uid.strip().lstrip("-").isdigit()
]

# ---------------------------------------------------------------------------
# Email (SMTP)
# ---------------------------------------------------------------------------

SMTP_HOST = _env("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USER = _env("SMTP_USER")
SMTP_PASS = _env("SMTP_PASS")
SENDER_EMAIL = _env("SENDER_EMAIL") or SMTP_USER
SENDER_NAME = _env("SENDER_NAME")

# ---------------------------------------------------------------------------
# Job sources
# ---------------------------------------------------------------------------

LINKEDIN_BASE_URL = _env("LINKEDIN_BASE_URL", "https://www.linkedin.com").rstrip("/")
HTTP_TIMEOUT = _env_int("HTTP_TIMEOUT", 20)
HTTP_USER_AGENT = _env(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)

# Playwright is OPTIONAL and off by default. It cannot work on Railway the way
# it did locally (no persistent logged-in LinkedIn browser profile, datacenter
# IPs get the authwall), so the cloud path uses LinkedIn's public guest
# endpoints over plain HTTP instead. Set ENABLE_PLAYWRIGHT=true only when
# running locally with a logged-in ./browser_profile.
ENABLE_PLAYWRIGHT = _env_bool("ENABLE_PLAYWRIGHT", False)
_raw_browser_profile = _env("BROWSER_PROFILE_PATH", str(BASE_DIR / "browser_profile"))
BROWSER_PROFILE_PATH = Path(_raw_browser_profile).resolve()
HEADLESS = _env_bool("HEADLESS", True)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

PORT = _env_int("PORT", 8080)
ENV = _env("ENV", "production")
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
# httpx logs every request at INFO, which is noise in Railway logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Application").setLevel(logging.WARNING)

logger = logging.getLogger("career_agent")


def startup_report() -> list:
    """Return human-readable config problems, worst first. Empty list = good."""
    problems = []
    if not TELEGRAM_BOT_TOKEN:
        problems.append("FATAL: TELEGRAM_BOT_TOKEN is not set — the bot cannot start.")
    if not LLM_API_KEY:
        problems.append("FATAL: LLM_API_KEY (or NVIDIA_API_KEY) is not set — the agent cannot think.")
    for note in LLM_BASE_URL_NOTES:
        problems.append(f"WARN: {note}")
    if not PROFILE_PATH.exists():
        problems.append(f"WARN: no profile at {PROFILE_PATH} — run /profile in the bot to set one up.")
    if not (SMTP_USER and SMTP_PASS):
        problems.append("WARN: SMTP_USER/SMTP_PASS not set — drafting works, sending will fail.")
    if not TELEGRAM_ALLOWED_USER_IDS:
        problems.append(
            "WARN: TELEGRAM_ALLOWED_USER_ID is empty — ANYONE who finds your bot can use "
            "your profile, your CV and your email quota. Set it to your numeric Telegram ID."
        )
    return problems
