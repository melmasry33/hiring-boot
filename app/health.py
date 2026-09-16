"""
Health endpoint.

A Telegram long-polling bot has no inbound HTTP, but Railway expects a process
to bind $PORT and will keep restarting/flagging a service that never does. This
is a ~40-line stdlib server on a daemon thread — no FastAPI, no uvicorn, no
extra dependency — so `GET /` and `GET /health` answer 200 while the bot polls.

It also gives you a URL you can hit (or point an uptime pinger at) to confirm
the container is alive without opening Telegram.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import ENV, LLM_MODEL, PORT, logger

_STARTED_AT = time.time()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — stdlib naming
        body = json.dumps(
            {
                "status": "ok",
                "service": "telegram-career-agent",
                "env": ENV,
                "model": LLM_MODEL,
                "uptime_seconds": int(time.time() - _STARTED_AT),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request noise in Railway logs
        return


def start_health_server() -> None:
    def _serve():
        try:
            server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
            logger.info(f"Health server listening on 0.0.0.0:{PORT}")
            server.serve_forever()
        except Exception as e:
            # A missing health endpoint must never take the bot down.
            logger.warning(f"Health server could not start on port {PORT}: {e}")

    threading.Thread(target=_serve, daemon=True, name="health").start()
