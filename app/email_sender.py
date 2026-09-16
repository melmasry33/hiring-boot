"""
SMTP delivery.

Two changes from the original that matter in production:

1. `smtplib` is blocking. Called directly from a coroutine it freezes the whole
   bot for the duration of the handshake — under Gmail that is routinely 3-8
   seconds during which the bot answers nobody. Everything is now wrapped in
   `asyncio.to_thread`.
2. The approval gate is kept exactly as strict as before: nothing leaves this
   process without approved=True, which only the Telegram button sets.
"""

import asyncio
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    SENDER_EMAIL,
    SENDER_NAME,
    SMTP_HOST,
    SMTP_PASS,
    SMTP_PORT,
    SMTP_USER,
    logger,
)


def _send_sync(
    recipient_email: str,
    subject: str,
    body: str,
    cv_file_path: Optional[str],
    reply_to: Optional[str],
) -> Dict[str, Any]:
    msg = MIMEMultipart()
    from_addr = SENDER_EMAIL or SMTP_USER
    msg["From"] = formataddr((SENDER_NAME, from_addr)) if SENDER_NAME else from_addr
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    if reply_to:
        msg["Reply-To"] = reply_to

    msg.attach(MIMEText(body, "plain", "utf-8"))

    if cv_file_path and os.path.exists(cv_file_path):
        file_name = Path(cv_file_path).name
        with open(cv_file_path, "rb") as f:
            part = MIMEBase("application", "pdf")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{file_name}"')
        msg.attach(part)
        logger.info(f"Attached CV: {file_name}")

    logger.info(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} ...")
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

    return {
        "success": True,
        "status": "SENT",
        "message": f"Application email sent to {recipient_email}.",
        "recipient": recipient_email,
        "subject": subject,
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

    if not (SMTP_USER and SMTP_PASS):
        return {
            "success": False,
            "status": "SMTP_NOT_CONFIGURED",
            "message": (
                "SMTP_USER / SMTP_PASS are not set. For Gmail, create an App Password "
                "(myaccount.google.com → Security → App passwords) and set it in Railway."
            ),
        }

    try:
        return await asyncio.to_thread(
            _send_sync, recipient_email, subject, body, cv_file_path, reply_to
        )
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "status": "AUTH_FAILED",
            "message": (
                "SMTP login was rejected. With Gmail you must use a 16-character App "
                "Password, not your normal account password."
            ),
        }
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return {"success": False, "status": "SEND_ERROR", "message": f"Failed to send: {e}"}
