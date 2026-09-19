"""
Gmail API delivery over HTTPS.

Uses Gmail's REST API instead of SMTP/Resend. The authenticated Gmail account
is the sender. The public send_application_email(...) interface stays unchanged.
"""

import asyncio
import base64
import os
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import (
    GMAIL_CLIENT_ID,
    GMAIL_CLIENT_SECRET,
    GMAIL_REFRESH_TOKEN,
    GMAIL_TOKEN_URI,
    SENDER_EMAIL,
    SENDER_NAME,
    logger,
)

GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def _build_gmail_service():
    if not (GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN):
        raise RuntimeError(
            "Gmail OAuth is not configured. Set GMAIL_CLIENT_ID, "
            "GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN."
        )

    credentials = Credentials(
        token=None,
        refresh_token=GMAIL_REFRESH_TOKEN,
        token_uri=GMAIL_TOKEN_URI,
        client_id=GMAIL_CLIENT_ID,
        client_secret=GMAIL_CLIENT_SECRET,
        scopes=[GMAIL_SEND_SCOPE],
    )
    credentials.refresh(Request())
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _send_sync(
    recipient_email: str,
    subject: str,
    body: str,
    cv_file_path: Optional[str],
    reply_to: Optional[str],
) -> Dict[str, Any]:
    service = _build_gmail_service()

    # The gmail.send scope is intentionally the only Gmail permission we request.
    # Do not call users.getProfile() here: that endpoint requires additional
    # mailbox/profile scopes and causes 403 insufficientPermissions with gmail.send.
    sender = (SENDER_EMAIL or "").strip()
    if not sender:
        raise RuntimeError("SENDER_EMAIL is required for Gmail API sending.")

    msg = MIMEMultipart()
    from_addr = formataddr((SENDER_NAME, sender)) if SENDER_NAME else sender
    msg["From"] = from_addr
    msg["To"] = recipient_email
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to

    msg.attach(MIMEText(body, "plain", "utf-8"))

    if cv_file_path and os.path.exists(cv_file_path):
        file_path = Path(cv_file_path)
        with file_path.open("rb") as fh:
            part = MIMEBase("application", "pdf")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{file_path.name}"',
        )
        msg.attach(part)
        logger.info("Attached CV: %s", file_path.name)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    result = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
    )

    return {
        "success": True,
        "status": "SENT",
        "message": f"Application email sent to {recipient_email}.",
        "recipient": recipient_email,
        "subject": subject,
        "id": result.get("id"),
        "thread_id": result.get("threadId"),
        "sender": sender,
    }


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

    if not (GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN):
        return {
            "success": False,
            "status": "GMAIL_NOT_CONFIGURED",
            "message": (
                "Gmail API is not configured. Set GMAIL_CLIENT_ID, "
                "GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN in Railway."
            ),
        }

    try:
        return await asyncio.to_thread(
            _send_sync, recipient_email, subject, body, cv_file_path, reply_to
        )
    except HttpError as e:
        detail = str(e)
        logger.error("Gmail API send failed: %s", detail)
        return {
            "success": False,
            "status": "GMAIL_API_ERROR",
            "message": f"Gmail API rejected the send: {detail}",
        }
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return {"success": False, "status": "SEND_ERROR", "message": f"Failed to send: {e}"}
