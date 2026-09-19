"""
Resend HTTP delivery.

Uses Resend's HTTPS API instead of SMTP so delivery does not depend on outbound
mail ports. The public interface remains unchanged for the rest of the agent.
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from config import RESEND_API_KEY, SENDER_EMAIL, SENDER_NAME, logger

RESEND_URL = "https://api.resend.com/emails"


def _send_sync(
    recipient_email: str,
    subject: str,
    body: str,
    cv_file_path: Optional[str],
    reply_to: Optional[str],
) -> Dict[str, Any]:
    sender = SENDER_EMAIL.strip()
    if not sender:
        raise RuntimeError("SENDER_EMAIL is not configured.")

    from_value = f"{SENDER_NAME} <{sender}>" if SENDER_NAME else sender

    payload: Dict[str, Any] = {
        "from": from_value,
        "to": [recipient_email],
        "subject": subject,
        "text": body,
    }

    if reply_to:
        payload["reply_to"] = [reply_to]

    if cv_file_path and os.path.exists(cv_file_path):
        file_path = Path(cv_file_path)
        payload["attachments"] = [
            {
                "filename": file_path.name,
                "content": base64.b64encode(file_path.read_bytes()).decode("ascii"),
            }
        ]
        logger.info("Attached CV: %s", file_path.name)

    logger.info("Sending email to %s through Resend HTTPS API ...", recipient_email)

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            RESEND_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.is_success:
        data = response.json()
        email_id = data.get("id")
        return {
            "success": True,
            "status": "SENT",
            "message": f"Application email sent to {recipient_email}.",
            "recipient": recipient_email,
            "subject": subject,
            "id": email_id,
        }

    try:
        error_payload = response.json()
        detail = error_payload.get("message") or error_payload.get("name") or str(error_payload)
    except ValueError:
        detail = response.text.strip() or response.reason_phrase

    raise RuntimeError(f"Resend API returned HTTP {response.status_code}: {detail}")


async def send_application_email(
    recipient_email: str,
    subject: str,
    body: str,
    cv_file_path: Optional[str] = None,
    approved: bool = False,
    reply_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an application email. Refuses unless approved is exactly True."""
    if approved is not True:
        msg = "Refused: explicit user approval is required before any email is sent."
        logger.warning(msg)
        return {"success": False, "status": "REFUSED_APPROVAL_REQUIRED", "message": msg}

    recipient_email = (recipient_email or "").strip()
    if not recipient_email or "@" not in recipient_email or " " in recipient_email:
        return {
            "success": False,
            "status": "INVALID_RECIPIENT",
            "message": f"Invalid recipient address: '{recipient_email}'",
        }

    if not RESEND_API_KEY:
        return {
            "success": False,
            "status": "RESEND_NOT_CONFIGURED",
            "message": "RESEND_API_KEY is not set in Railway.",
        }

    if not SENDER_EMAIL:
        return {
            "success": False,
            "status": "SENDER_NOT_CONFIGURED",
            "message": "SENDER_EMAIL is not set. Configure a verified Resend sender address.",
        }

    try:
        return await asyncio.to_thread(
            _send_sync, recipient_email, subject, body, cv_file_path, reply_to
        )
    except httpx.HTTPError as e:
        logger.error("Resend HTTP request failed: %s", e)
        return {
            "success": False,
            "status": "HTTP_ERROR",
            "message": f"Resend HTTP request failed: {e}",
        }
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return {"success": False, "status": "SEND_ERROR", "message": f"Failed to send: {e}"}
