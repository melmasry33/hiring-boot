"""
Job sourcing over plain HTTP.

Why this replaces the Playwright path for deployment:
the old linkedin.py drove a persistent Chromium profile that you had logged
into by hand. That works on your laptop and cannot work on Railway — there is
no browser profile to inherit, no way to solve a CAPTCHA, and LinkedIn serves
datacenter IPs an authwall almost immediately. Shipping it as the primary path
would mean a bot that looks deployed and fails on the first real link.

LinkedIn's *guest* endpoints (the ones that render job pages for logged-out
visitors) need no session and answer plain GETs:
  /jobs-guest/jobs/api/jobPosting/{job_id}          -> one job's description
  /jobs-guest/jobs/api/seeMoreJobPostings/search    -> a page of search results

They are rate-limited and occasionally return 429, so every function here
degrades gracefully and returns a structured error the agent can read and
explain to the user, instead of raising.

Playwright is still importable for local use (ENABLE_PLAYWRIGHT=true) but is
never required and is not installed in the Docker image.
"""

import html as html_lib
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from config import HTTP_TIMEOUT, HTTP_USER_AGENT, LINKEDIN_BASE_URL, logger

_HEADERS = {
    "User-Agent": HTTP_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def strip_html(raw: str, limit: int = 8000) -> str:
    """Crude but effective tag stripper — good enough for job postings."""
    if not raw:
        return ""
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|li|div|h[1-6]|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>", "• ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()[:limit]


def extract_emails(text: str) -> List[str]:
    """Pull plausible contact emails out of a posting, newest-first order kept."""
    found = []
    for match in EMAIL_RE.findall(text or ""):
        low = match.lower()
        # Filter the obvious non-contacts that show up in page boilerplate.
        if any(bad in low for bad in ("sentry.io", "example.com", ".png", ".jpg", "noreply", "no-reply")):
            continue
        if low not in found:
            found.append(low)
    return found[:5]


def linkedin_job_id(url: str) -> Optional[str]:
    """Extract the numeric job id from any LinkedIn job URL shape."""
    if not url:
        return None
    patterns = [
        r"/jobs/view/(?:[^/]*-)?(\d{6,})",
        r"currentJobId=(\d{6,})",
        r"/jobs-guest/jobs/api/jobPosting/(\d{6,})",
        r"(?:^|[^\d])(\d{10})(?:[^\d]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


async def fetch_page(url: str, limit: int = 8000) -> Dict[str, Any]:
    """Fetch any URL and return cleaned text. Never raises."""
    if not url or not url.startswith(("http://", "https://")):
        return {"ok": False, "url": url, "error": "Not a valid http(s) URL."}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=HTTP_TIMEOUT, headers=_HEADERS
        ) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                return {
                    "ok": False,
                    "url": url,
                    "status": resp.status_code,
                    "error": f"Server returned HTTP {resp.status_code}.",
                }
            text = strip_html(resp.text, limit=limit)
            return {
                "ok": True,
                "url": str(resp.url),
                "status": resp.status_code,
                "content": text,
                "emails_found": extract_emails(text),
            }
    except Exception as e:
        logger.warning(f"fetch_page failed for {url}: {e}")
        return {"ok": False, "url": url, "error": f"Could not fetch the page: {e}"}


async def get_linkedin_job(url: str) -> Dict[str, Any]:
    """
    Read a LinkedIn job via the logged-out guest endpoint.

    Falls back to fetching the public job URL directly if the guest endpoint
    refuses, and finally to Playwright if it is explicitly enabled locally.
    """
    job_id = linkedin_job_id(url)
    if not job_id:
        # Not a recognisable LinkedIn job URL — treat it as a normal page.
        page = await fetch_page(url)
        if page.get("ok"):
            return {
                "ok": True,
                "source": "generic",
                "url": url,
                "title": "",
                "company": "",
                "location": "",
                "description": page["content"],
                "emails_found": page.get("emails_found", []),
            }
        return {"ok": False, "url": url, "error": page.get("error", "Unknown fetch error.")}

    guest_url = f"{LINKEDIN_BASE_URL}/jobs-guest/jobs/api/jobPosting/{job_id}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=HTTP_TIMEOUT, headers=_HEADERS
        ) as client:
            resp = await client.get(guest_url)
            if resp.status_code == 200 and resp.text.strip():
                raw = resp.text
                title = _first_group(raw, [r'<h2[^>]*top-card-layout__title[^>]*>([\s\S]*?)</h2>',
                                           r'<h1[^>]*>([\s\S]*?)</h1>'])
                company = _first_group(raw, [r'topcard__org-name-link[^>]*>([\s\S]*?)</a>',
                                             r'topcard__flavor[^>]*>([\s\S]*?)</span>'])
                location = _first_group(raw, [r'topcard__flavor--bullet[^>]*>([\s\S]*?)</span>'])
                description = strip_html(raw)
                return {
                    "ok": True,
                    "source": "linkedin_guest",
                    "url": f"{LINKEDIN_BASE_URL}/jobs/view/{job_id}",
                    "job_id": job_id,
                    "title": strip_html(title, 200),
                    "company": strip_html(company, 200),
                    "location": strip_html(location, 200),
                    "description": description,
                    "emails_found": extract_emails(description),
                }
            logger.info(f"LinkedIn guest endpoint returned {resp.status_code} for job {job_id}.")
    except Exception as e:
        logger.warning(f"LinkedIn guest fetch failed for {job_id}: {e}")

    # Fallback 1: the public job page itself.
    public_url = f"{LINKEDIN_BASE_URL}/jobs/view/{job_id}"
    page = await fetch_page(public_url)
    if page.get("ok") and len(page.get("content", "")) > 400:
        return {
            "ok": True,
            "source": "linkedin_public",
            "url": public_url,
            "job_id": job_id,
            "title": "",
            "company": "",
            "location": "",
            "description": page["content"],
            "emails_found": page.get("emails_found", []),
        }

    # Fallback 2: local Playwright, only if explicitly enabled.
    pw = await _try_playwright(public_url)
    if pw:
        return pw

    return {
        "ok": False,
        "url": public_url,
        "job_id": job_id,
        "error": (
            "LinkedIn did not return the posting to this server (it rate-limits or "
            "authwalls cloud IPs). Ask the user to paste the job description text "
            "directly — everything else works exactly the same on pasted text."
        ),
    }


def _first_group(raw: str, patterns: List[str]) -> str:
    for pat in patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


async def search_linkedin_jobs(
    keywords: str,
    location: str = "Egypt",
    limit: int = 10,
    remote_only: bool = False,
    posted_within_days: int = 0,
) -> Dict[str, Any]:
    """
    Search LinkedIn's guest job feed. Returns a list of {title, company,
    location, url} — no description (fetch the job for that).
    """
    params = {
        "keywords": keywords,
        "location": location,
        "start": "0",
        "sortBy": "DD",
    }
    if remote_only:
        params["f_WT"] = "2"  # LinkedIn's "Remote" workplace-type filter
    if posted_within_days and posted_within_days > 0:
        params["f_TPR"] = f"r{int(posted_within_days) * 86400}"

    search_url = (
        f"{LINKEDIN_BASE_URL}/jobs-guest/jobs/api/seeMoreJobPostings/search?"
        + urllib.parse.urlencode(params)
    )

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=HTTP_TIMEOUT, headers=_HEADERS
        ) as client:
            resp = await client.get(search_url)
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "error": (
                        f"LinkedIn search returned HTTP {resp.status_code}. It throttles "
                        f"cloud IPs; suggest the user paste a job link or description instead."
                    ),
                    "results": [],
                }
            results = _parse_search_cards(resp.text, limit=limit)
            return {
                "ok": True,
                "query": {"keywords": keywords, "location": location},
                "count": len(results),
                "results": results,
            }
    except Exception as e:
        logger.warning(f"LinkedIn search failed: {e}")
        return {"ok": False, "error": f"Search failed: {e}", "results": []}


def _parse_search_cards(raw_html: str, limit: int = 10) -> List[Dict[str, str]]:
    cards = re.split(r"<li[\s>]", raw_html)
    out: List[Dict[str, str]] = []
    for card in cards[1:]:
        link_m = re.search(r'href="(https://[^"]*?/jobs/view/[^"?]+)', card)
        if not link_m:
            continue
        url = html_lib.unescape(link_m.group(1))
        title = strip_html(
            _first_group(card, [r'class="[^"]*base-search-card__title[^"]*"[^>]*>([\s\S]*?)</h3>']), 200
        )
        company = strip_html(
            _first_group(card, [r'class="[^"]*base-search-card__subtitle[^"]*"[^>]*>([\s\S]*?)</h4>']), 200
        )
        location = strip_html(
            _first_group(card, [r'class="[^"]*job-search-card__location[^"]*"[^>]*>([\s\S]*?)</span>']), 200
        )
        posted = strip_html(
            _first_group(card, [r'datetime="([^"]+)"']), 50
        )
        out.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "posted": posted,
                "url": url.split("?")[0],
            }
        )
        if len(out) >= limit:
            break
    return out


async def _try_playwright(url: str) -> Optional[Dict[str, Any]]:
    """Local-only escape hatch. Returns None unless ENABLE_PLAYWRIGHT is on."""
    from config import ENABLE_PLAYWRIGHT

    if not ENABLE_PLAYWRIGHT:
        return None
    try:
        import linkedin_browser  # noqa: WPS433 — optional local dependency

        data = await linkedin_browser.get_job(url)
        data["ok"] = True
        data["source"] = "playwright"
        return data
    except Exception as e:
        logger.warning(f"Playwright fallback unavailable/failed: {e}")
        return None
