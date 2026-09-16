"""
OPTIONAL local-only LinkedIn scraper (Playwright).

This is the original browser-driven scraper, kept for running on your own
machine with a logged-in ./browser_profile. It is NOT used in the deployed
container: Railway has no persistent browser session, cannot show you a CAPTCHA,
and LinkedIn authwalls datacenter IPs. jobs.py handles the cloud path over HTTP.

Enable locally with ENABLE_PLAYWRIGHT=true and `playwright install chromium`.
"""

import asyncio
import urllib.parse
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError

from config import (
    BROWSER_PROFILE_PATH,
    HEADLESS,
    LINKEDIN_BASE_URL,
    logger,
)



async def get_browser_context(playwright_instance) -> BrowserContext:
    """Launch or attach to a persistent browser context preserving LinkedIn login session."""
    context = await playwright_instance.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_PROFILE_PATH),
        headless=HEADLESS,
        viewport={"width": 1280, "height": 850},
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )
    return context


async def check_for_challenges(page: Page) -> Optional[str]:
    """Check if LinkedIn presents a CAPTCHA, MFA, or login verification challenge."""
    current_url = page.url.lower()
    
    if "checkpoint/challenge" in current_url or "checkpoint" in current_url:
        msg = "LinkedIn security challenge/checkpoint detected. Please complete the verification manually in the open browser window."
        logger.warning(msg)
        return msg
    
    if "authwall" in current_url:
        msg = "LinkedIn Authwall detected. Please log into LinkedIn in the browser window."
        logger.warning(msg)
        return msg

    # Check for visible CAPTCHA or verification text on page
    try:
        captcha_elem = await page.query_selector("iframe[src*='captcha'], #captcha-internal, div[data-test-captcha]")
        if captcha_elem:
            msg = "CAPTCHA detected on page. Please solve it manually in the open browser window."
            logger.warning(msg)
            return msg
    except Exception:
        pass

    return None


async def search_jobs(
    keywords: str,
    location: str = "Egypt",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for jobs on LinkedIn using keywords and location.
    Saves discovered jobs to data/jobs.json and returns structured job list.
    """
    logger.info(f"Searching jobs... Keywords: '{keywords}', Location: '{location}', Max: {max_results}")
    
    encoded_keywords = urllib.parse.quote_plus(keywords)
    encoded_location = urllib.parse.quote_plus(location)
    
    # LinkedIn job search URL with sorting
    search_url = f"{LINKEDIN_BASE_URL}/jobs/search/?keywords={encoded_keywords}&location={encoded_location}&sortBy=DD"
    
    results: List[Dict[str, Any]] = []

    async with async_playwright() as p:
        context = await get_browser_context(p)
        page = await context.new_page()
        try:
            logger.info(f"Navigating to {search_url}")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2500)

            challenge = await check_for_challenges(page)
            if challenge:
                return [{"error": challenge}]

            # Wait for job container or job cards
            card_selectors = [
                "div.job-card-container",
                "li.jobs-search-results__list-item",
                "ul.scaffold-layout__list-container > li",
                "li.ember-view.jobs-search-results__list-item",
                "div.base-card",
                "div.job-search-card",
            ]

            found_selector = None
            for sel in card_selectors:
                try:
                    await page.wait_for_selector(sel, timeout=4000)
                    found_selector = sel
                    break
                except PlaywrightTimeoutError:
                    continue

            if not found_selector:
                logger.warning("No job cards selector found immediately. Attempting to scroll.")
                await page.evaluate("window.scrollBy(0, 400)")
                await page.wait_for_timeout(1500)

            # Scroll multiple times to load dynamic results
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, 600)")
                await page.wait_for_timeout(1000)
                cards = await page.query_selector_all(found_selector or "div.job-card-container, div.base-card, li.jobs-search-results__list-item")
                if len(cards) >= max_results:
                    break

            cards = await page.query_selector_all(found_selector or "div.job-card-container, div.base-card, li.jobs-search-results__list-item")
            logger.info(f"Found {len(cards)} raw job cards.")

            for card in cards[:max_results]:
                try:
                    # Extract Title & Link
                    title_elem = await card.query_selector(
                        "a.job-card-list__title, a.job-card-container__link, a.base-card__full-link, h3, a[data-control-name='job_card_title']"
                    )
                    title = (await title_elem.inner_text()).strip() if title_elem else "Unknown Title"
                    # Clean up any newlines or trailing badges in title
                    title = title.split("\n")[0].strip()

                    url = ""
                    if title_elem:
                        href = await title_elem.get_attribute("href") or ""
                        if href:
                            if href.startswith("/"):
                                url = f"{LINKEDIN_BASE_URL}{href.split('?')[0]}"
                            else:
                                url = href.split("?")[0]

                    # Extract Company
                    company_elem = await card.query_selector(
                        ".job-card-container__primary-description, .job-card-container__company-name, .base-search-card__subtitle, .artdeco-entity-lockup__subtitle"
                    )
                    company = (await company_elem.inner_text()).strip() if company_elem else "Unknown Company"
                    company = company.split("\n")[0].strip()

                    # Extract Location
                    location_elem = await card.query_selector(
                        ".job-card-container__metadata-item, .job-search-card__location, .base-search-card__metadata"
                    )
                    loc = (await location_elem.inner_text()).strip() if location_elem else location
                    loc = loc.split("\n")[0].strip()

                    # Snippet or short text if available
                    snippet_elem = await card.query_selector(".job-card-list__snippet, .base-card__metadata")
                    snippet = (await snippet_elem.inner_text()).strip() if snippet_elem else ""

                    if url:
                        job_item = {
                            "title": title,
                            "company": company,
                            "location": loc,
                            "url": url,
                            "description": snippet,
                        }
                        results.append(job_item)
                except Exception as card_err:
                    logger.debug(f"Failed extracting one job card: {card_err}")
                    continue

            logger.info(f"Successfully extracted {len(results)} jobs.")
            return results

        except Exception as e:
            logger.error(f"Error during search_jobs: {e}")
            return [{"error": f"Job search failed: {str(e)}"}]
        finally:
            await context.close()


async def get_job(url: str) -> Dict[str, Any]:
    """
    Fetch comprehensive job details from a specific LinkedIn job URL.
    Returns title, company, location, description, and job_url.
    """
    logger.info(f"Opening job details: {url}")
    
    async with async_playwright() as p:
        context = await get_browser_context(p)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)

            challenge = await check_for_challenges(page)
            if challenge:
                return {"error": challenge, "job_url": url}

            # Try to click "See more" or "Show more" button for full description
            try:
                see_more_btn = await page.query_selector(
                    "button.show-more-less-html__button, button[aria-label*='Show more'], button[aria-label*='see more'], button.artdeco-card__action"
                )
                if see_more_btn:
                    await see_more_btn.click(timeout=2000)
                    await page.wait_for_timeout(500)
            except Exception:
                pass

            # Extract Title
            title = ""
            for t_sel in [
                "h1.job-details-jobs-unified-top-card__job-title",
                "h1.top-card-layout__title",
                "h1.jobs-title",
                "h1",
            ]:
                t_elem = await page.query_selector(t_sel)
                if t_elem:
                    title = (await t_elem.inner_text()).strip()
                    if title:
                        break

            # Extract Company
            company = ""
            for c_sel in [
                "div.job-details-jobs-unified-top-card__company-name a",
                "a.topcard__org-name-link",
                "span.jobs-unified-top-card__company-name",
                "div.job-details-jobs-unified-top-card__company-name",
                "a.jobs-unified-top-card__company-name",
            ]:
                c_elem = await page.query_selector(c_sel)
                if c_elem:
                    company = (await c_elem.inner_text()).strip()
                    if company:
                        break

            # Extract Location
            location = ""
            for l_sel in [
                "span.job-details-jobs-unified-top-card__bullet",
                "span.topcard__flavor--bullet",
                "span.jobs-unified-top-card__bullet",
                "div.job-details-jobs-unified-top-card__primary-description span",
            ]:
                l_elem = await page.query_selector(l_sel)
                if l_elem:
                    location = (await l_elem.inner_text()).strip()
                    if location:
                        break

            # Extract Description
            description = ""
            for d_sel in [
                "#job-details",
                "div.jobs-description-content__text",
                "div.jobs-description__content",
                "div.show-more-less-html__markup",
                "div.jobs-box__html-content",
                "article.jobs-description",
            ]:
                d_elem = await page.query_selector(d_sel)
                if d_elem:
                    description = (await d_elem.inner_text()).strip()
                    if description:
                        break

            job_data = {
                "title": title or "Unknown Title",
                "company": company or "Unknown Company",
                "location": location or "Unknown Location",
                "description": description or "No description could be extracted.",
                "job_url": url,
            }

            # Update cache
            logger.info(f"Extracted job: '{job_data['title']}' at '{job_data['company']}'")
            return job_data

        except Exception as e:
            logger.error(f"Error fetching job {url}: {e}")
            return {
                "title": "Error",
                "company": "Error",
                "location": "Error",
                "description": f"Failed to extract job details: {str(e)}",
                "job_url": url,
            }
        finally:
            await context.close()
