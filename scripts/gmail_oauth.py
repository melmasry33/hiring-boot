#!/usr/bin/env python3
"""Authorize Gmail API locally and print/save a refresh token.

Run this on the same computer where your browser is available:
    python scripts/gmail_oauth.py path/to/credentials.json

The generated refresh token is sensitive. Do not commit it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/gmail_oauth.py path/to/credentials.json")
        return 2

    credentials_path = Path(sys.argv[1]).expanduser().resolve()
    if not credentials_path.is_file():
        print(f"credentials file not found: {credentials_path}")
        return 2

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        SCOPES,
    )
    creds = flow.run_local_server(
        host="localhost",
        port=0,
        access_type="offline",
        prompt="consent",
    )

    token_path = Path("gmail_token.json").resolve()
    token_path.write_text(
        json.dumps(
            {
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "scopes": creds.scopes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved Gmail OAuth token to: {token_path}")
    print("Upload gmail_token.json here when it exists; do not paste the token into chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
