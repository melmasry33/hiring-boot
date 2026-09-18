"""
llm_preflight.py — fail loudly at boot instead of 404-ing on the first message.

WHY THIS EXISTS
---------------
`get_client()` only checks that LLM_API_KEY is non-empty. Constructing a
ChatOpenAI makes no network call, so a wrong LLM_BASE_URL or an LLM_MODEL the
provider doesn't serve stays invisible until a real user sends a real message
— at which point it surfaces as a bare `openai.NotFoundError: Error code: 404`
buried in a 60-line LangGraph traceback. /diag reports "LLM key: set" and looks
green the whole time.

This module makes the endpoint prove itself at startup with one cheap
`GET {base_url}/models` call, and normalises the two base-URL mistakes that
cause a path-level 404 (no `/v1` suffix, or `/chat/completions` pasted in).
"""

from __future__ import annotations

import difflib
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx

# Path segments that mean "this URL already points at an API version root".
_VERSION_SEGMENTS = {"v1", "v1beta", "openai", "api"}

# NVIDIA hosted endpoint lifecycle: this model reached end-of-life on
# 2026-08-26 and returns HTTP 410 even though older model catalogues may still
# list it. Keep it out of automatic model selection.
RETIRED_MODELS = {
    "meta/llama-3.3-70b-instruct",
}


def normalize_base_url(raw: str) -> Tuple[str, List[str]]:
    """
    Return (cleaned_url, notes). Fixes the two mistakes that produce a 404 with
    an EMPTY response body — i.e. the HTTP path doesn't exist, as opposed to a
    404 whose JSON body says the model doesn't exist.

      https://integrate.api.nvidia.com            -> .../v1        (+note)
      https://api.groq.com/openai/v1/chat/completions -> .../openai/v1 (+note)
    """
    notes: List[str] = []
    url = (raw or "").strip().rstrip("/")
    if not url:
        return url, notes

    for suffix in ("/chat/completions", "/completions", "/responses"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            notes.append(
                f"LLM_BASE_URL ended in '{suffix}' — the SDK appends that itself. "
                f"Trimmed to {url}"
            )
            break

    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if not segments or segments[-1] not in _VERSION_SEGMENTS:
        url = f"{url}/v1"
        notes.append(
            f"LLM_BASE_URL had no version segment — OpenAI-compatible endpoints "
            f"live under /v1. Using {url}"
        )

    return url, notes


def list_models(base_url: str, api_key: str, timeout: float = 15.0) -> Tuple[Optional[List[str]], str]:
    """
    GET {base_url}/models. Returns (model_ids, detail).
    model_ids is None when the catalogue could not be read; detail then
    explains why in a form you can act on.
    """
    url = f"{base_url}/models"
    try:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as e:
        return None, f"could not reach {url} ({type(e).__name__}: {e})"

    if resp.status_code in (401, 403):
        return None, (
            f"{url} returned {resp.status_code} — the API key is rejected. "
            "Check LLM_API_KEY belongs to this provider and is still active."
        )
    if resp.status_code == 404:
        return None, (
            f"{url} returned 404 — this host serves no OpenAI-compatible catalogue "
            "at that path. LLM_BASE_URL is pointing at the wrong place."
        )
    if resp.status_code >= 400:
        return None, f"{url} returned {resp.status_code}: {resp.text[:200]}"

    try:
        payload = resp.json()
    except ValueError:
        return None, f"{url} returned {resp.status_code} but the body was not JSON — likely a proxy or landing page, not an API."

    ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]
    if not ids:
        return None, f"{url} answered but listed no models."
    return ids, f"{len(ids)} models available at {base_url}"


def probe_chat_model(base_url: str, model: str, api_key: str, timeout: float = 8.0) -> Tuple[bool, str]:
    """Verify the model can actually answer a Chat Completions request."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}: {e}"
    if resp.status_code < 300:
        return True, "chat/completions probe succeeded"
    if resp.status_code in (401, 403):
        return False, f"authentication rejected ({resp.status_code})"
    if resp.status_code == 404:
        return False, "model is not available to this key/endpoint (HTTP 404)"
    if resp.status_code == 429:
        return False, "provider rate-limited the probe (HTTP 429)"
    return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def resolve_model(
    base_url: str, chain: List[str], api_key: str, timeout: float = 8.0
) -> Tuple[Optional[str], List[str]]:
    """Choose the first model that passes a real chat/completions probe."""
    notes: List[str] = []
    if not chain:
        return None, ["FATAL: no model configured."]

    ids, detail = list_models(base_url, api_key, min(timeout, 8.0))
    if ids is None:
        return None, [f"FATAL: LLM endpoint unusable — {detail}"]

    available = {i.lower(): i for i in ids}
    for candidate in chain:
        if candidate.lower() in RETIRED_MODELS:
            notes.append(f"WARN: '{candidate}' is retired at the hosted endpoint — skipping.")
            continue

        ok, probe_detail = probe_chat_model(base_url, candidate, api_key, timeout)
        if ok:
            if notes:
                notes.append(f"Falling back to '{candidate}'.")
            return candidate, notes

        if "authentication rejected" in probe_detail:
            return None, [f"FATAL: provider rejected the API key while probing '{candidate}'."]
        if "HTTP 503" in probe_detail or "HTTP 429" in probe_detail:
            # Capacity/rate-limit errors are transient. Do not hammer every
            # fallback model during boot; select this candidate provisionally
            # and let the runtime retry with backoff.
            notes.append(f"WARN: '{candidate}' is temporarily unavailable — {probe_detail}. Selecting provisionally.")
            return candidate, notes
        notes.append(f"WARN: '{candidate}' failed real chat probe — {probe_detail}.")

    suggestions = difflib.get_close_matches(chain[0].lower(), list(available), n=3, cutoff=0.4)
    notes.append(
        f"FATAL: none of {chain} passed a real chat/completions probe at {base_url}."
        + (" Closest catalogue ids: " + ", ".join(available[s] for s in suggestions) if suggestions else "")
    )
    return None, notes


def preflight(base_url: str, model: str, api_key: str, timeout: float = 15.0) -> List[str]:
    """
    Return a list of human-readable problems, worst first. Empty list = the
    endpoint answered and serves `model`. Never raises — a preflight that
    itself blows up must not take the bot down.
    """
    problems: List[str] = []
    if not api_key:
        return ["FATAL: LLM_API_KEY is not set — skipping endpoint preflight."]

    try:
        ids, detail = list_models(base_url, api_key, timeout)
    except Exception as e:  # pragma: no cover — defensive
        return [f"WARN: LLM preflight crashed ({type(e).__name__}: {e}) — continuing anyway."]

    if ids is None:
        return [f"FATAL: LLM endpoint unusable — {detail}"]

    if model in RETIRED_MODELS:
        return [f"WARN: LLM_MODEL '{model}' is retired at the hosted endpoint; use an active model such as meta/muse-glimmer-30b."]

    if model in ids:
        return []

    # Model not served here. Near-matches are the useful part: the same weights
    # are published under a different slug by almost every provider
    # (meta/muse-glimmer-30b vs meta-models/Muse-Glimmer-30B vs
    # positron-ai/Muse-Glimmer-30B), and picking the wrong one is a 404.
    lowered = {i.lower(): i for i in ids}
    suggestions = difflib.get_close_matches(model.lower(), list(lowered), n=3, cutoff=0.5)
    if not suggestions:
        tail = model.split("/")[-1].lower()
        suggestions = [k for k in lowered if tail and tail in k][:3]

    msg = f"FATAL: LLM_MODEL '{model}' is not served at {base_url} ({detail})."
    if suggestions:
        msg += " Closest available: " + ", ".join(lowered[s] for s in suggestions)
    else:
        msg += f" Example ids there: {', '.join(ids[:3])}"
    problems.append(msg)
    return problems
