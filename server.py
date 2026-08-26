#!/usr/bin/env python3
"""Local server for the AI GPU Dashboard.

Serves index.html and the static data files, and proxies /api/chat so the LLM
key can stay server-side rather than being shipped to the browser.

It used to serve web_dashboard.html at both / and /index.html -- so running it
locally showed the superseded dashboard even when you asked for the current one
-- and carried seventeen /api/* data routes that answered from the gpu_data.py
seed constants. index.html hardcodes IS_STATIC = true and reads data.json, so
those routes were unreachable; anything that had reached them would have been
served hand-seeded prices from before live scraping existed.
"""

import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

sys.path.insert(0, os.path.dirname(__file__))

from config import WEB_PORT


def _load_env():
    """Load variables from .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

CHAT_API_URL = os.environ.get("CHAT_API_URL", "https://api.hyperfusion.io/v1/chat/completions")
CHAT_API_KEY = os.environ.get("CHAT_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen/qwen3-32b")

# Rate limiter
_rate_map = {}
_RATE_LIMIT = 20
_RATE_WINDOW = 60

def _is_rate_limited(ip):
    import time
    now = time.time()
    entry = _rate_map.get(ip)
    if not entry or now - entry["start"] > _RATE_WINDOW:
        _rate_map[ip] = {"start": now, "count": 1}
        return False
    entry["count"] += 1
    return entry["count"] > _RATE_LIMIT

# CORS allowlist
_ALLOWED_ORIGINS = {"http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:5500"}

def _cors_origin(headers):
    origin = headers.get("Origin", "")
    if not origin:
        return None
    if origin in _ALLOWED_ORIGINS:
        return origin
    return None


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

    def _set_cors(self):
        origin = _cors_origin(self.headers)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self.handle_chat_proxy()
        else:
            self.send_error(404, "Not Found")

    def handle_chat_proxy(self):
        # Rate limiting
        client_ip = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()
        if _is_rate_limited(client_ip):
            self.send_json({"error": "Too many requests. Please wait a moment."}, status=429)
            return

        if not CHAT_API_KEY:
            self.send_json({"error": "Chat service unavailable."}, status=503)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 100_000:
                self.send_json({"error": "Request too large."}, status=413)
                return
            body = self.rfile.read(content_length)
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            self.send_json({"error": "Invalid request body."}, status=400)
            return

        # Input validation
        messages = payload.get("messages", [])
        if not messages or not isinstance(messages, list):
            self.send_json({"error": "No messages provided."}, status=400)
            return
        if len(messages) > 50:
            self.send_json({"error": "Too many messages."}, status=400)
            return
        for msg in messages:
            if not isinstance(msg, dict) or not isinstance(msg.get("role"), str) or not isinstance(msg.get("content"), str):
                self.send_json({"error": "Invalid message format."}, status=400)
                return
            if msg["role"] not in ("system", "user", "assistant"):
                self.send_json({"error": "Invalid message role."}, status=400)
                return
            limit = 60000 if msg["role"] == "system" else 10000
            if len(msg["content"]) > limit:
                self.send_json({"error": "Message too long."}, status=400)
                return

        proxy_payload = json.dumps({
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": min(payload.get("max_tokens", 1024), 2048),
            "temperature": 0.7,
        }).encode("utf-8")

        req = Request(
            CHAT_API_URL,
            data=proxy_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CHAT_API_KEY}",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            self.send_json(resp_data)
        except HTTPError:
            self.send_json({"error": "AI service returned an error. Please try again."}, status=502)
        except (URLError, TimeoutError):
            self.send_json({"error": "AI service temporarily unavailable."}, status=502)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_file("index.html", "text/html")
        # Retired: /api/summary, /matrix, /gpu, /regional, /indicators,
        # /workloads, /historical, /specs, /providers, /tco, /inference, /spot,
        # /news, /forecasts, /competitive, /sustainability, /supplychain.
        # index.html sets IS_STATIC = true and reads data.json directly, so none
        # of them were reachable -- and each answered from the gpu_data.py seed
        # constants rather than the pipeline's output, meaning any caller got
        # hand-seeded prices from before live scraping existed.
        else:
            super().do_GET()

    def send_json(self, data, status=200):
        content = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def send_file(self, filename, content_type):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        pass  # Suppress default logging


def run_server(port=None):
    port = port or WEB_PORT
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  AI GPU Dashboard Server running at:")
    print(f"  -> http://localhost:{port}")
    print(f"  -> Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else WEB_PORT
    run_server(port)
