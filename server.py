#!/usr/bin/env python3
"""
Web server for the AI GPU Dashboard.
Serves the HTML dashboard and provides API endpoints for live data + AI analysis.
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

from gpu_data import (
    GPU_SPECS, CLOUD_PRICING, HISTORICAL_PRICING, MARKET_INDICATORS,
    REGIONAL_DATA, WORKLOAD_RECOMMENDATIONS,
    TCO_COMPONENTS, INFERENCE_BENCHMARKS, SPOT_MARKET, NEWS_FEED,
    get_cheapest_by_gpu, get_price_comparison_matrix,
    generate_market_summary, get_regional_summary, get_workload_recommendations,
    get_utilization_summary, get_reservation_analysis, get_price_forecasts,
    get_competitive_landscape, get_sustainability_summary, get_supply_chain_summary,
    get_model_hardware_fit,
)
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
            if len(msg["content"]) > 10000:
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
            self.send_file("web_dashboard.html", "text/html")
        elif path == "/api/summary":
            self.send_json(generate_market_summary())
        elif path == "/api/matrix":
            self.send_json(get_price_comparison_matrix())
        elif path == "/api/gpu":
            params = parse_qs(parsed.query)
            gpu_id = params.get("id", ["H100-SXM"])[0]
            providers = get_cheapest_by_gpu(gpu_id)
            spec = GPU_SPECS.get(gpu_id, {})
            trends = HISTORICAL_PRICING.get(gpu_id, {})
            self.send_json({"spec": spec, "providers": providers, "trends": trends})
        elif path == "/api/regional":
            self.send_json(get_regional_summary())
        elif path == "/api/indicators":
            self.send_json(MARKET_INDICATORS)
        elif path == "/api/workloads":
            self.send_json(get_workload_recommendations())
        elif path == "/api/historical":
            self.send_json(HISTORICAL_PRICING)
        elif path == "/api/specs":
            self.send_json(GPU_SPECS)
        elif path == "/api/providers":
            self.send_json(CLOUD_PRICING)
        elif path.startswith("/api/ai/"):
            params = parse_qs(parsed.query)
            use_cache = "nocache" not in params
            ai_route = path[8:]  # strip "/api/ai/"
            ai_types = {
                "summary": "quick_summary", "trends": "market_trends",
                "regional": "regional_analysis", "investment": "investment_outlook",
                "notes": "market_notes", "efficiency": "efficiency_optimization",
                "forecast": "price_forecasts", "sustainability": "sustainability_risk",
            }
            if ai_route == "all":
                self.send_ai_all()
            elif ai_route == "gpu":
                gpu_id = params.get("id", ["H100-SXM"])[0]
                self.send_ai_gpu(gpu_id)
            elif ai_route in ai_types:
                self.send_ai_analysis(ai_types[ai_route], use_cache=use_cache)
            else:
                super().do_GET()
        elif path == "/api/tco":
            self.send_json(TCO_COMPONENTS)
        elif path == "/api/inference":
            self.send_json(INFERENCE_BENCHMARKS)
        elif path == "/api/spot":
            self.send_json(SPOT_MARKET)
        elif path == "/api/news":
            try:
                from ai_analyzer import generate_daily_news
                self.send_json(generate_daily_news())
            except Exception:
                self.send_json(NEWS_FEED)
        elif path == "/api/utilization":
            self.send_json(get_utilization_summary())
        elif path == "/api/reservations":
            self.send_json(get_reservation_analysis())
        elif path == "/api/forecasts":
            self.send_json(get_price_forecasts())
        elif path == "/api/competitive":
            self.send_json(get_competitive_landscape())
        elif path == "/api/sustainability":
            self.send_json(get_sustainability_summary())
        elif path == "/api/supplychain":
            self.send_json(get_supply_chain_summary())
        elif path == "/api/modelfit":
            self.send_json(get_model_hardware_fit())
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

    def send_ai_analysis(self, analysis_type, use_cache=True):
        try:
            from ai_analyzer import (
                get_quick_summary, analyze_market_trends,
                analyze_regional_opportunities, analyze_investment_outlook,
                generate_market_notes, analyze_efficiency_optimization,
                analyze_price_forecasts, analyze_sustainability_risk,
            )
            funcs = {
                "quick_summary": get_quick_summary,
                "market_trends": analyze_market_trends,
                "regional_analysis": analyze_regional_opportunities,
                "investment_outlook": analyze_investment_outlook,
                "market_notes": generate_market_notes,
                "efficiency_optimization": analyze_efficiency_optimization,
                "price_forecasts": analyze_price_forecasts,
                "sustainability_risk": analyze_sustainability_risk,
            }
            result = funcs[analysis_type](use_cache=use_cache)
            self.send_json({"analysis": result, "type": analysis_type, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            self.send_json({"error": str(e)})

    def send_ai_all(self):
        try:
            from ai_analyzer import get_all_analyses
            result = get_all_analyses(use_cache=True)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)})

    def send_ai_gpu(self, gpu_id):
        try:
            from ai_analyzer import analyze_specific_gpu
            result = analyze_specific_gpu(gpu_id)
            self.send_json({"analysis": result, "gpu_id": gpu_id, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            self.send_json({"error": str(e)})

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
