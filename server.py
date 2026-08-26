#!/usr/bin/env python3
"""Local server for the AI GPU Dashboard.

Serves index.html and the static data files. That is all it does now, and all
it needs to do: the dashboard reads data.json and ai_analysis.json directly,
and every language-model output is written by the pipeline rather than fetched
at request time, so nothing here handles a credential.

It previously served web_dashboard.html at both / and /index.html -- running it
locally showed the superseded dashboard even when you asked for the current one
-- alongside seventeen /api/* data routes answering from the gpu_data.py seed
constants (unreachable, since index.html hardcodes IS_STATIC = true, and they
would have served hand-seeded prices from before live scraping existed) and an
/api/chat proxy for the since-removed chat panel.
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# config.py is gitignored, so a fresh clone does not have it. Fall back rather
# than failing to start over a port number.
try:
    sys.path.insert(0, PROJECT_DIR)
    from config import WEB_PORT
except ImportError:
    WEB_PORT = int(os.environ.get("WEB_PORT", "8050"))


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_DIR, **kwargs)

    def end_headers(self):
        # The pipeline rewrites data.json daily and the page carries an hourly
        # cache key; a stale local copy on top of that only confuses testing.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # quiet by default


def run_server(port=None):
    port = port or WEB_PORT
    httpd = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == "__main__":
    run_server(int(sys.argv[1]) if len(sys.argv) > 1 else None)
