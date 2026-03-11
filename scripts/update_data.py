#!/usr/bin/env python3
"""
GPU Market Data Updater
=======================
Fetches live GPU market data from free APIs and updates data.json and ai_analysis.json.

Usage:
    python scripts/update_data.py

Dependencies:
    pip install requests feedparser
"""

import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from xml.etree import ElementTree

from forecast_engine import generate_forecasts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_JSON = os.path.join(PROJECT_DIR, "data.json")
EMBEDDED_DATA_JSON = os.path.join(PROJECT_DIR, "embedded_data.json")
AI_ANALYSIS_JSON = os.path.join(PROJECT_DIR, "ai_analysis.json")
EMBEDDED_AI_JSON = os.path.join(PROJECT_DIR, "embedded_ai.json")
CONFIG_PY = os.path.join(PROJECT_DIR, "config.py")

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_results = {"success": [], "failed": []}


def log_info(msg):
    print(f"  [INFO]  {msg}")


def log_ok(source, detail=""):
    msg = f"{source}: {detail}" if detail else source
    print(f"  [ OK ]  {msg}")
    _results["success"].append(source)


def log_fail(source, detail=""):
    msg = f"{source}: {detail}" if detail else source
    print(f"  [FAIL]  {msg}")
    _results["failed"].append(source)


# ---------------------------------------------------------------------------
# Load existing data (template)
# ---------------------------------------------------------------------------
def load_existing(path, fallback_path=None):
    """Load existing JSON file, falling back to embedded copy."""
    for p in [path, fallback_path]:
        if p and os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                log_info(f"Warning: could not parse {p}: {exc}")
    return {}


def save_json(path, data):
    """Atomically write JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)
    log_info(f"Wrote {path} ({os.path.getsize(path):,} bytes)")


# ---------------------------------------------------------------------------
# HTTP helper (uses requests if available, falls back to urllib)
# ---------------------------------------------------------------------------
try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def http_get(url, headers=None, timeout=30, params=None):
    """GET request returning (status_code, response_body_str)."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    if _HAS_REQUESTS:
        resp = _req.get(url, headers=headers or {}, timeout=timeout)
        return resp.status_code, resp.text
    else:
        import urllib.request as _ureq
        import ssl
        ctx = ssl.create_default_context()
        req = _ureq.Request(url, headers=headers or {})
        with _ureq.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body


def http_post(url, json_body, headers=None, timeout=30):
    """POST request returning (status_code, response_body_str)."""
    if _HAS_REQUESTS:
        resp = _req.post(url, json=json_body, headers=headers or {}, timeout=timeout)
        return resp.status_code, resp.text
    else:
        import urllib.request as _ureq
        import ssl
        ctx = ssl.create_default_context()
        data = json.dumps(json_body).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        req = _ureq.Request(url, data=data, headers=hdrs, method="POST")
        with _ureq.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body


# ===================================================================
# 1. GPU CLOUD PRICING FETCHERS
# ===================================================================

def fetch_vastai_pricing():
    """Fetch on-demand GPU pricing from Vast.ai API."""
    log_info("Fetching Vast.ai pricing...")
    # Fetch a large batch sorted by price ascending to get consumer + datacenter GPUs
    query = json.dumps({
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "num_gpus": {"gte": 1},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 3000,
    })
    url = "https://cloud.vast.ai/api/v0/bundles/"
    status, body = http_get(url, params={"q": query}, timeout=60)
    if status != 200:
        raise RuntimeError(f"Vast.ai returned HTTP {status}")

    data = json.loads(body)
    offers = data if isinstance(data, list) else data.get("offers", [])

    # Aggregate: cheapest price per GPU model
    gpu_prices = {}
    for offer in offers:
        gpu_name = offer.get("gpu_name", "")
        price_hr = offer.get("dph_total")  # dollars per hour total
        num_gpus = offer.get("num_gpus", 1)
        if not gpu_name or not price_hr or num_gpus < 1:
            continue
        per_gpu = round(price_hr / num_gpus, 4)
        if gpu_name not in gpu_prices or per_gpu < gpu_prices[gpu_name]["min_price"]:
            gpu_prices[gpu_name] = {
                "min_price": per_gpu,
                "num_offers": gpu_prices.get(gpu_name, {}).get("num_offers", 0) + 1,
            }
        else:
            gpu_prices[gpu_name]["num_offers"] += 1

    log_ok("Vast.ai", f"{len(gpu_prices)} GPU models, {len(offers)} total offers")
    return gpu_prices


def fetch_runpod_pricing():
    """Fetch GPU pricing from RunPod GraphQL API."""
    log_info("Fetching RunPod pricing...")
    url = "https://api.runpod.io/graphql"
    query = {
        "query": """
        {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
    }
    headers = {"Content-Type": "application/json"}
    status, body = http_post(url, query, headers=headers, timeout=30)
    if status != 200:
        raise RuntimeError(f"RunPod returned HTTP {status}")

    data = json.loads(body)
    gpu_types = data.get("data", {}).get("gpuTypes", [])
    if not gpu_types:
        raise RuntimeError("RunPod returned empty gpuTypes")

    result = {}
    for gpu in gpu_types:
        name = gpu.get("displayName", gpu.get("id", "unknown"))
        lowest = gpu.get("lowestPrice", {}) or {}
        on_demand = lowest.get("uninterruptablePrice")
        spot = lowest.get("minimumBidPrice")
        result[name] = {
            "on_demand_price": on_demand,
            "spot_price": spot,
            "memory_gb": gpu.get("memoryInGb"),
            "secure_cloud": gpu.get("secureCloud"),
            "community_cloud": gpu.get("communityCloud"),
        }

    log_ok("RunPod", f"{len(result)} GPU types")
    return result


# GPU name normalization map: maps external names to internal IDs
_GPU_NAME_MAP = {
    "RTX 4090": "RTX-4090",
    "RTX 5090": "RTX-5090",
    "GeForce RTX 4090": "RTX-4090",
    "GeForce RTX 5090": "RTX-5090",
    "A100 80GB": "A100-80GB",
    "A100 40GB": "A100-40GB",
    "A100-80GB SXM": "A100-80GB",
    "A100 PCIE 80GB": "A100-80GB",
    "A100 PCIE 40GB": "A100-40GB",
    "A100 SXM 80GB": "A100-80GB",
    "A100_80GB": "A100-80GB",
    "A100_PCIE_80GB": "A100-80GB",
    "H100 SXM": "H100-SXM",
    "H100 PCIe": "H100-PCIe",
    "H100 80GB SXM": "H100-SXM",
    "H100_SXM": "H100-SXM",
    "H100_PCIE": "H100-PCIe",
    "H100 NVL": "H100-SXM",
    "H200": "H200",
    "H200 SXM": "H200",
    "B200": "B200",
    "L40S": "L40S",
    "L40": "L40S",
    "L4": "L4",
    "T4": "T4",
    "MI300X": "MI300X",
    "MI250X": "MI250X",
    "Tesla T4": "T4",
    "A100 PCIE": "A100-80GB",
    "A100 SXM": "A100-80GB",
    "A100-SXM4-80GB": "A100-80GB",
    "A100-SXM4-40GB": "A100-40GB",
    "A100-PCIE-80GB": "A100-80GB",
    "A100-PCIE-40GB": "A100-40GB",
    "H100 80GB HBM3": "H100-SXM",
    "H100 NVL 94GB": "H100-SXM",
    "H100-SXM5-80GB": "H100-SXM",
    "H100-PCIE-80GB": "H100-PCIe",
    "MI300X": "MI300X",
    "MI325X": "MI325X",
    "B200": "B200",
    "B300": "B300",
    "GB200": "GB200",
}


def normalize_gpu_name(name):
    """Try to map an external GPU name to our internal ID."""
    if not name:
        return None
    # Direct lookup
    if name in _GPU_NAME_MAP:
        return _GPU_NAME_MAP[name]
    # Case-insensitive
    name_upper = name.upper().strip()
    for ext, internal in _GPU_NAME_MAP.items():
        if ext.upper() == name_upper:
            return internal
    # Substring matching for common patterns
    for ext, internal in _GPU_NAME_MAP.items():
        if ext.upper() in name_upper:
            return internal
    return None


def get_hardcoded_fallback_prices():
    """Return hardcoded prices for providers without free APIs."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "GCP": {
            "last_verified": "2026-02-23T00:00:00+00:00",
            "source": "hardcoded_fallback",
            "gpus": {
                "H100-SXM": {"price_per_gpu_hr": 3.40},
                "H100-PCIe": {"price_per_gpu_hr": 2.85},
                "A100-80GB": {"price_per_gpu_hr": 2.21},
                "A100-40GB": {"price_per_gpu_hr": 1.38},
                "L4": {"price_per_gpu_hr": 0.70},
                "T4": {"price_per_gpu_hr": 0.35},
            },
        },
        "Azure": {
            "last_verified": "2026-02-23T00:00:00+00:00",
            "source": "hardcoded_fallback",
            "gpus": {
                "H100-SXM": {"price_per_gpu_hr": 3.67},
                "A100-80GB": {"price_per_gpu_hr": 2.52},
                "A100-40GB": {"price_per_gpu_hr": 1.97},
                "T4": {"price_per_gpu_hr": 0.53},
            },
        },
        "Lambda": {
            "last_verified": "2026-02-23T00:00:00+00:00",
            "source": "hardcoded_fallback",
            "gpus": {
                "H200": {"price_per_gpu_hr": 3.29},
                "H100-SXM": {"price_per_gpu_hr": 2.49},
                "A100-80GB": {"price_per_gpu_hr": 1.29},
                "A100-40GB": {"price_per_gpu_hr": 1.10},
            },
        },
        "CoreWeave": {
            "last_verified": "2026-02-23T00:00:00+00:00",
            "source": "hardcoded_fallback",
            "gpus": {
                "B200": {"price_per_gpu_hr": 3.75},
                "H200": {"price_per_gpu_hr": 3.49},
                "H100-SXM": {"price_per_gpu_hr": 2.23},
                "H100-PCIe": {"price_per_gpu_hr": 2.06},
                "A100-80GB": {"price_per_gpu_hr": 2.06},
                "A100-40GB": {"price_per_gpu_hr": 1.62},
                "L40S": {"price_per_gpu_hr": 1.14},
            },
        },
        "AWS": {
            "last_verified": "2026-02-23T00:00:00+00:00",
            "source": "hardcoded_fallback",
            "gpus": {
                "B200": {"price_per_gpu_hr": 5.35},
                "H100-SXM": {"price_per_gpu_hr": 4.28},
                "A100-80GB": {"price_per_gpu_hr": 2.21},
                "A100-40GB": {"price_per_gpu_hr": 1.38},
                "L4": {"price_per_gpu_hr": 0.80},
                "T4": {"price_per_gpu_hr": 0.53},
            },
        },
    }


def merge_live_pricing_into_data(data, vastai_prices, runpod_prices):
    """Merge live pricing data into the providers section of data.json."""
    now = datetime.now(timezone.utc).isoformat()
    providers = data.get("providers", {})

    # -- Vast.ai --
    if vastai_prices:
        vast_prov = None
        for pk, pv in providers.items():
            if "vast" in pk.lower() or "vast" in pv.get("provider_name", "").lower():
                vast_prov = pv
                break
        if vast_prov is None:
            providers["Vast.ai"] = {"provider_name": "Vast.ai", "type": "marketplace", "gpus": {}}
            vast_prov = providers["Vast.ai"]
        vast_gpus = vast_prov.get("gpus", {})
        for ext_name, info in vastai_prices.items():
            internal = normalize_gpu_name(ext_name)
            if not internal or internal not in vast_gpus:
                continue
            gpu_entry = vast_gpus[internal]
            if isinstance(gpu_entry, dict):
                gpu_entry["price_per_gpu_hr"] = info["min_price"]
                gpu_entry["source"] = "vastai_api"
                gpu_entry["last_updated"] = now
        vast_prov["last_updated"] = now

    # -- RunPod --
    if runpod_prices:
        rp_prov = None
        for pk, pv in providers.items():
            if "runpod" in pk.lower() or "runpod" in pv.get("provider_name", "").lower():
                rp_prov = pv
                break
        if rp_prov is None:
            providers["RunPod"] = {"provider_name": "RunPod", "type": "marketplace", "gpus": {}}
            rp_prov = providers["RunPod"]
        rp_gpus = rp_prov.get("gpus", {})
        for ext_name, info in runpod_prices.items():
            internal = normalize_gpu_name(ext_name)
            if not internal or internal not in rp_gpus:
                continue
            gpu_entry = rp_gpus[internal]
            if isinstance(gpu_entry, dict) and info.get("on_demand_price") is not None:
                gpu_entry["price_per_gpu_hr"] = info["on_demand_price"]
                gpu_entry["source"] = "runpod_api"
                gpu_entry["last_updated"] = now
        rp_prov["last_updated"] = now

    # -- Hardcoded fallbacks for providers without free APIs --
    fallbacks = get_hardcoded_fallback_prices()
    for provider_key, fb_data in fallbacks.items():
        if provider_key not in providers:
            providers[provider_key] = {
                "provider_name": provider_key,
                "type": "cloud",
                "gpus": {},
            }
        prov = providers[provider_key]
        existing_gpus = prov.setdefault("gpus", {})
        for gpu_id, gpu_info in fb_data["gpus"].items():
            if gpu_id not in existing_gpus:
                existing_gpus[gpu_id] = {}
            existing_entry = existing_gpus[gpu_id]
            # Only overwrite price if there is no live data
            if "source" not in existing_entry or existing_entry.get("source") == "hardcoded_fallback":
                existing_entry["price_per_gpu_hr"] = gpu_info["price_per_gpu_hr"]
                existing_entry["source"] = "hardcoded_fallback"
                existing_entry["last_verified"] = fb_data["last_verified"]

    data["providers"] = providers
    return data


# ===================================================================
# 1b. RECALCULATE MATRIX, HISTORICAL, SPOT from provider prices
# ===================================================================

def recalculate_matrix(data):
    """Rebuild the matrix array from current providers + specs data."""
    providers = data.get("providers", {})
    specs = data.get("specs", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Collect all prices per GPU across providers
    gpu_prices = {}  # gpu_id -> [{price, provider, type}]
    for prov_key, prov in providers.items():
        prov_type = prov.get("type", "cloud")
        prov_name = prov.get("provider_name", prov_key)
        for gpu_id, gpu_info in (prov.get("gpus") or {}).items():
            price = gpu_info.get("price_per_gpu_hr")
            if price is not None and price > 0:
                gpu_prices.setdefault(gpu_id, []).append({
                    "price": price, "provider": prov_name, "type": prov_type
                })

    matrix = []
    for gpu_id, price_list in gpu_prices.items():
        spec = specs.get(gpu_id, {})
        prices_sorted = sorted(price_list, key=lambda x: x["price"])
        cheapest = prices_sorted[0]
        most_expensive = prices_sorted[-1]
        avg_price = round(sum(p["price"] for p in prices_sorted) / len(prices_sorted), 2)
        spread = round((most_expensive["price"] - cheapest["price"]) / cheapest["price"] * 100, 1) if cheapest["price"] > 0 else 0

        # Compute FLOPS/$ using FP16 TFLOPS
        fp16 = spec.get("fp16_tflops", 0)
        flops_per_dollar = round(fp16 / cheapest["price"], 1) if cheapest["price"] > 0 and fp16 else 0

        # MoM change from historical
        hist = data.get("historical", {}).get(gpu_id, {})
        periods = sorted(hist.keys())
        monthly_change = 0
        if len(periods) >= 2:
            prev_avg = hist[periods[-2]].get("avg")
            cur_avg = hist[periods[-1]].get("avg")
            if prev_avg and cur_avg:
                monthly_change = round((cur_avg - prev_avg) / prev_avg * 100, 1)

        matrix.append({
            "gpu_id": gpu_id,
            "name": spec.get("name", gpu_id),
            "vendor": spec.get("vendor", "Unknown"),
            "vram_gb": spec.get("vram_gb"),
            "arch": spec.get("arch", ""),
            "tier": spec.get("tier", ""),
            "cheapest_price": cheapest["price"],
            "cheapest_provider": cheapest["provider"],
            "cheapest_provider_type": cheapest["type"],
            "most_expensive": most_expensive["price"],
            "avg_price": avg_price,
            "num_providers": len(prices_sorted),
            "price_spread_pct": spread,
            "monthly_change_pct": monthly_change,
            "flops_per_dollar": flops_per_dollar,
            "vram_per_dollar": round(spec.get("vram_gb", 0) / cheapest["price"], 1) if cheapest["price"] > 0 and spec.get("vram_gb") else 0,
        })

    # Sort by flops_per_dollar descending
    matrix.sort(key=lambda x: x["flops_per_dollar"], reverse=True)
    data["matrix"] = matrix
    log_ok("Matrix", f"{len(matrix)} GPUs recalculated from {len(providers)} providers")
    return data


def update_historical(data):
    """Append a new monthly data point from current provider prices."""
    providers = data.get("providers", {})
    historical = data.get("historical", {})
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")

    # Collect per-GPU prices from providers
    gpu_prices = {}
    for prov_key, prov in providers.items():
        for gpu_id, gpu_info in (prov.get("gpus") or {}).items():
            price = gpu_info.get("price_per_gpu_hr")
            if price is not None and price > 0:
                gpu_prices.setdefault(gpu_id, []).append(price)

    updated_count = 0
    for gpu_id, prices in gpu_prices.items():
        if gpu_id not in historical:
            historical[gpu_id] = {}
        # Only update if we have data and don't already have this month with live source
        entry = historical[gpu_id].get(current_month, {})
        avg_price = round(sum(prices) / len(prices), 2)
        min_price = round(min(prices), 2)
        max_price = round(max(prices), 2)

        historical[gpu_id][current_month] = {
            "avg": avg_price,
            "min": min_price,
            "max": max_price,
            "availability": entry.get("availability", "available"),
        }
        updated_count += 1

    data["historical"] = historical
    log_ok("Historical", f"{updated_count} GPUs updated for {current_month}")
    return data


def update_spot(data):
    """Refresh spot data from current provider prices and discounts."""
    providers = data.get("providers", {})
    spot = data.get("spot", {})

    # Collect per-GPU on-demand prices and compute spot/reserved estimates
    gpu_prices = {}
    for prov_key, prov in providers.items():
        spot_disc = prov.get("spot_discount", 0)
        res1_disc = prov.get("reserved_1yr_discount", 0)
        res3_disc = prov.get("reserved_3yr_discount", 0)
        for gpu_id, gpu_info in (prov.get("gpus") or {}).items():
            price = gpu_info.get("price_per_gpu_hr")
            if price is not None and price > 0:
                gpu_prices.setdefault(gpu_id, []).append({
                    "price": price,
                    "spot_disc": spot_disc,
                    "res1_disc": res1_disc,
                    "res3_disc": res3_disc,
                })

    for gpu_id, entries in gpu_prices.items():
        on_demand = [e["price"] for e in entries]
        reserved_1yr = [e["price"] * (1 - e["res1_disc"]) for e in entries if e["res1_disc"] > 0]
        reserved_3yr = [e["price"] * (1 - e["res3_disc"]) for e in entries if e["res3_disc"] > 0]

        existing = spot.get(gpu_id, {})
        # Preserve quarterly_trend, just append/shift
        trend = existing.get("quarterly_trend", [])
        avg_od = round(sum(on_demand) / len(on_demand), 2)

        spot[gpu_id] = {
            "on_demand_low": round(min(on_demand), 2),
            "on_demand_avg": avg_od,
            "on_demand_high": round(max(on_demand), 2),
            "reserved_1yr_low": round(min(reserved_1yr), 2) if reserved_1yr else existing.get("reserved_1yr_low"),
            "reserved_1yr_avg": round(sum(reserved_1yr) / len(reserved_1yr), 2) if reserved_1yr else existing.get("reserved_1yr_avg"),
            "reserved_1yr_high": round(max(reserved_1yr), 2) if reserved_1yr else existing.get("reserved_1yr_high"),
            "reserved_3yr_low": round(min(reserved_3yr), 2) if reserved_3yr else existing.get("reserved_3yr_low"),
            "reserved_3yr_avg": round(sum(reserved_3yr) / len(reserved_3yr), 2) if reserved_3yr else existing.get("reserved_3yr_avg"),
            "reserved_3yr_high": round(max(reserved_3yr), 2) if reserved_3yr else existing.get("reserved_3yr_high"),
            "res1_savings_pct": existing.get("res1_savings_pct", 30),
            "res3_savings_pct": existing.get("res3_savings_pct", 50),
            "num_providers": len(entries),
            "quarterly_trend": (trend + [avg_od])[-4:],  # Keep last 4 quarters
            "bid": existing.get("bid", avg_od),
            "ask": existing.get("ask", avg_od),
            "last_trade": avg_od,
        }

    data["spot"] = spot
    log_ok("Spot", f"{len(gpu_prices)} GPUs refreshed")
    return data


# ===================================================================
# 2. STOCK PRICE FETCHERS
# ===================================================================

def fetch_stock_price(ticker):
    """Fetch stock data from Yahoo Finance v8 chart API."""
    log_info(f"Fetching stock price for {ticker}...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1y", "interval": "1d"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    status, body = http_get(url, headers=headers, params=params, timeout=20)
    if status != 200:
        raise RuntimeError(f"Yahoo Finance returned HTTP {status} for {ticker}")

    data = json.loads(body)
    chart = data.get("chart", {})
    result = chart.get("result", [])
    if not result:
        raise RuntimeError(f"No chart data for {ticker}")

    r = result[0]
    meta = r.get("meta", {})
    current_price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose")

    # Extract 52-week range from indicators
    indicators = r.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    highs = [h for h in (quotes.get("high") or []) if h is not None]
    lows = [l for l in (quotes.get("low") or []) if l is not None]

    week52_high = max(highs) if highs else None
    week52_low = min(lows) if lows else None

    # Calculate YTD change (approximate from first trading day data)
    timestamps = r.get("timestamp", [])
    closes = quotes.get("close") or []

    # Find first close of current year
    current_year = datetime.now().year
    ytd_start_price = None
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts)
        if dt.year == current_year and i < len(closes) and closes[i] is not None:
            ytd_start_price = closes[i]
            break

    ytd_change = None
    if ytd_start_price and current_price:
        ytd_change = round(((current_price - ytd_start_price) / ytd_start_price) * 100, 2)

    stock_data = {
        "ticker": ticker,
        "current_price": current_price,
        "previous_close": prev_close,
        "52_week_high": round(week52_high, 2) if week52_high else None,
        "52_week_low": round(week52_low, 2) if week52_low else None,
        "ytd_change_pct": ytd_change,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    log_ok(f"Stock/{ticker}", f"${current_price:.2f}" if current_price else "price unavailable")
    return stock_data


def merge_stocks_into_data(data, stocks):
    """Merge stock data into the indicators section, matching existing structure."""
    if not stocks:
        return data
    data["stocks"] = {s["ticker"]: s for s in stocks}
    indicators = data.get("indicators", {})
    for s in stocks:
        tk = s["ticker"]
        if tk == "NVDA" and "nvidia_stock" in indicators:
            indicators["nvidia_stock"]["current"] = s["current_price"]
            if s.get("ytd_change_pct") is not None:
                indicators["nvidia_stock"]["ytd_pct"] = s["ytd_change_pct"]
            if s.get("week52_high") is not None:
                indicators["nvidia_stock"]["52w_range"] = f"${s['week52_low']:.2f}-${s['week52_high']:.2f}"
        elif tk == "AMD" and "amd_stock" in indicators:
            indicators["amd_stock"]["current"] = s["current_price"]
            if s.get("ytd_change_pct") is not None:
                indicators["amd_stock"]["ytd_pct"] = s["ytd_change_pct"]
            if s.get("week52_high") is not None:
                indicators["amd_stock"]["52w_range"] = f"${s['week52_low']:.2f}-${s['week52_high']:.2f}"
    data["indicators"] = indicators
    return data


# ===================================================================
# 3. NEWS FETCHER (Google News RSS)
# ===================================================================

def fetch_news_rss():
    """Fetch GPU/AI news headlines from Google News RSS."""
    log_info("Fetching news from Google News RSS...")
    query = "nvidia gpu ai datacenter"
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    full_url = url + "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    status, body = http_get(full_url, headers=headers, timeout=20)
    if status != 200:
        raise RuntimeError(f"Google News RSS returned HTTP {status}")

    # Try feedparser first, fall back to stdlib XML
    articles = []
    try:
        import feedparser
        feed = feedparser.parse(body)
        for entry in feed.entries[:20]:
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)
            source = ""
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title
            articles.append({
                "headline": entry.get("title", ""),
                "source": source or _extract_source_from_title(entry.get("title", "")),
                "url": entry.get("link", ""),
                "date": pub_date or datetime.now().strftime("%Y-%m-%d"),
                "category": "news",
                "sentiment": "neutral",
                "impact": "medium",
            })
    except ImportError:
        # Fallback: stdlib XML parsing
        root = ElementTree.fromstring(body)
        for item in root.findall(".//item")[:20]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            source_el = item.find("source")
            headline = title_el.text if title_el is not None else ""
            articles.append({
                "headline": headline,
                "source": (source_el.text if source_el is not None
                           else _extract_source_from_title(headline)),
                "url": link_el.text if link_el is not None else "",
                "date": _parse_rss_date(pub_el.text) if pub_el is not None else datetime.now().strftime("%Y-%m-%d"),
                "category": "news",
                "sentiment": _guess_sentiment(headline),
                "impact": "medium",
            })

    log_ok("Google News RSS", f"{len(articles)} articles")
    return articles


def _extract_source_from_title(title):
    """Google News often appends ' - Source Name' to titles."""
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "Unknown"


def _parse_rss_date(date_str):
    """Parse RSS date string to YYYY-MM-DD."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        # RFC 822 format: "Mon, 10 Mar 2026 12:00:00 GMT"
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _guess_sentiment(headline):
    """Simple keyword-based sentiment guess."""
    hl = headline.lower()
    bullish = ["surge", "soar", "record", "beat", "boost", "growth", "rally", "strong", "gain"]
    bearish = ["crash", "drop", "fall", "decline", "loss", "weak", "risk", "warning", "shortage"]
    if any(w in hl for w in bullish):
        return "bullish"
    if any(w in hl for w in bearish):
        return "bearish"
    return "neutral"


def merge_news_into_data(data, articles):
    """Replace news in data with fresh articles."""
    if articles:
        data["news"] = articles
    return data


# ===================================================================
# 3b. LIVE SENTIMENT — Reddit + HuggingFace
# ===================================================================

# GPU search terms for Reddit/HF queries
_GPU_SEARCH_TERMS = {
    "H100-SXM": ["H100", "H100 SXM"],
    "B300": ["B300", "Blackwell Ultra"],
    "B200": ["B200", "Blackwell B200"],
    "H200": ["H200"],
    "A100-80GB": ["A100", "A100 80GB"],
    "MI300X": ["MI300X", "MI300"],
    "L40S": ["L40S", "L40"],
    "RTX-4090": ["RTX 4090", "4090"],
}

_REDDIT_SUBREDDITS = ["MachineLearning", "LocalLLaMA", "deeplearning", "nvidia", "mlops"]


def fetch_reddit_sentiment():
    """Fetch GPU mention counts and sentiment from Reddit public JSON API."""
    log_info("Fetching Reddit sentiment...")
    results = {}
    headers = {"User-Agent": "GPUDashboard/1.0 (market research)"}

    for gpu_id, terms in _GPU_SEARCH_TERMS.items():
        total_mentions = 0
        total_score = 0
        total_upvote_ratio = 0.0
        post_count = 0
        hot_topics = []

        for term in terms:
            for sub in _REDDIT_SUBREDDITS:
                try:
                    url = f"https://www.reddit.com/r/{sub}/search.json"
                    params = {
                        "q": term,
                        "restrict_sr": "on",
                        "sort": "new",
                        "t": "month",
                        "limit": "25",
                    }
                    status, body = http_get(url, headers=headers, params=params, timeout=15)
                    if status != 200:
                        continue
                    data = json.loads(body)
                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        pd = post.get("data", {})
                        total_mentions += 1
                        total_score += pd.get("score", 0)
                        total_upvote_ratio += pd.get("upvote_ratio", 0.5)
                        post_count += 1
                        # Capture top posts as hot topics
                        if pd.get("score", 0) > 10 and len(hot_topics) < 5:
                            hot_topics.append({
                                "title": pd.get("title", "")[:100],
                                "score": pd.get("score", 0),
                                "subreddit": sub,
                            })
                    time.sleep(0.5)  # Rate limit: Reddit allows ~30 req/min unauthenticated
                except Exception:
                    continue

        avg_upvote = total_upvote_ratio / post_count if post_count > 0 else 0.5
        # Sort hot topics by score
        hot_topics.sort(key=lambda x: x["score"], reverse=True)

        results[gpu_id] = {
            "mentions_30d": total_mentions,
            "total_score": total_score,
            "avg_upvote_ratio": round(avg_upvote, 3),
            "reddit_sentiment": round(avg_upvote, 2),  # upvote ratio as sentiment proxy
            "hot_topics": hot_topics[:3],
        }

    log_ok("Reddit Sentiment", f"{len(results)} GPUs, {sum(r['mentions_30d'] for r in results.values())} total mentions")
    return results


def fetch_huggingface_models():
    """Fetch model counts per GPU from HuggingFace API."""
    log_info("Fetching HuggingFace model counts...")
    results = {}
    # HF API: search models with GPU keywords in tags/description
    hf_gpu_tags = {
        "H100-SXM": "h100",
        "B300": "b300",
        "B200": "b200",
        "H200": "h200",
        "A100-80GB": "a100",
        "MI300X": "mi300",
        "L40S": "l40s",
        "RTX-4090": "4090",
    }

    for gpu_id, tag in hf_gpu_tags.items():
        try:
            url = "https://huggingface.co/api/models"
            params = {"search": tag, "limit": "1", "sort": "downloads"}
            headers = {"User-Agent": "GPUDashboard/1.0"}
            status, body = http_get(url, headers=headers, params=params, timeout=15)
            if status != 200:
                results[gpu_id] = {"model_count": 0}
                continue

            # The API doesn't return total count directly with search,
            # so we fetch with a larger limit to estimate
            params["limit"] = "200"
            status2, body2 = http_get(url, headers=headers, params=params, timeout=20)
            if status2 == 200:
                models = json.loads(body2)
                count = len(models) if isinstance(models, list) else 0
                # If we got exactly 200, there are likely more
                results[gpu_id] = {
                    "model_count": count,
                    "estimated": count >= 200,
                }
            else:
                results[gpu_id] = {"model_count": 0}
            time.sleep(0.3)
        except Exception:
            results[gpu_id] = {"model_count": 0}

    log_ok("HuggingFace", f"{len(results)} GPUs queried")
    return results


def fetch_github_compat():
    """Fetch GitHub repository/issue counts mentioning each GPU."""
    log_info("Fetching GitHub compatibility scores...")
    results = {}
    headers = {
        "User-Agent": "GPUDashboard/1.0",
        "Accept": "application/vnd.github.v3+json",
    }

    for gpu_id, terms in _GPU_SEARCH_TERMS.items():
        total_repos = 0
        for term in terms[:1]:  # Use first term only to stay within rate limits
            try:
                url = "https://api.github.com/search/repositories"
                params = {
                    "q": f"{term} GPU language:python",
                    "sort": "updated",
                    "per_page": "1",
                }
                status, body = http_get(url, headers=headers, params=params, timeout=15)
                if status == 200:
                    data = json.loads(body)
                    total_repos = data.get("total_count", 0)
                time.sleep(2)  # GitHub rate limit: 10 req/min unauthenticated
            except Exception:
                continue

        # Normalize to a 0-100 score (H100 ~95, newer GPUs lower)
        # Scale: 1000+ repos = 95+, 100 = ~70, 10 = ~40, 0 = 20
        if total_repos >= 1000:
            score = min(98, 85 + int((total_repos - 1000) / 500))
        elif total_repos >= 100:
            score = 60 + int((total_repos - 100) / 30)
        elif total_repos >= 10:
            score = 30 + int((total_repos - 10) / 3)
        else:
            score = max(15, total_repos * 3)

        results[gpu_id] = {
            "repo_count": total_repos,
            "github_compat_score": min(score, 99),
        }

    log_ok("GitHub Compat", f"{len(results)} GPUs scored")
    return results


def merge_sentiment_into_data(data, reddit, hf, github):
    """Merge live sentiment data into the existing sentiment section."""
    sentiment = data.get("sentiment", {})
    now = datetime.now(timezone.utc).isoformat()

    for gpu_id in _GPU_SEARCH_TERMS:
        if gpu_id not in sentiment:
            sentiment[gpu_id] = {
                "score": 50, "ecosystem": "early", "adoption": "stable",
                "community_pick": False, "pros": [], "cons": [],
                "top_use_case": "",
            }
        s = sentiment[gpu_id]

        # Store previous score for trend tracking
        prev_score = s.get("score", 50)
        history = s.get("score_history", [])

        # Reddit data
        rd = reddit.get(gpu_id, {})
        if rd.get("mentions_30d", 0) > 0:
            s["reddit_sentiment"] = rd["reddit_sentiment"]
            s["mentions_30d"] = rd["mentions_30d"]
            s["hot_topics"] = rd.get("hot_topics", [])

        # HuggingFace data
        hfd = hf.get(gpu_id, {})
        if hfd.get("model_count", 0) > 0:
            s["hf_models_trained"] = hfd["model_count"]
            if hfd.get("estimated"):
                s["hf_models_estimated"] = True

        # GitHub data
        ghd = github.get(gpu_id, {})
        if ghd.get("repo_count", 0) > 0:
            s["github_compat_score"] = ghd["github_compat_score"]
            s["github_repos"] = ghd["repo_count"]

        # Recalculate composite score:
        # 35% reddit sentiment + 30% github compat + 20% HF activity + 15% mentions volume
        reddit_score = s.get("reddit_sentiment", 0.5) * 100
        github_score = s.get("github_compat_score", 50)
        mentions = s.get("mentions_30d", 0)
        mention_score = min(100, mentions / 300 * 100)  # 300+ mentions = 100
        hf_count = s.get("hf_models_trained", 0)
        hf_score = min(100, hf_count / 400 * 100)  # 400+ models = 100

        new_score = int(reddit_score * 0.35 + github_score * 0.30 + hf_score * 0.20 + mention_score * 0.15)
        s["score"] = max(10, min(99, new_score))

        # Update adoption trend based on score change
        score_delta = s["score"] - prev_score
        if score_delta > 3:
            s["adoption"] = "rising"
        elif score_delta < -3:
            s["adoption"] = "declining"
        else:
            s["adoption"] = "stable"

        # Append to score history (keep last 8 weeks)
        history.append({"date": now[:10], "score": s["score"]})
        s["score_history"] = history[-8:]

        s["last_updated"] = now

        # Update community_pick: top 2 scores get the badge
        sentiment[gpu_id] = s

    # Set community_pick for top 2
    sorted_gpus = sorted(sentiment.items(), key=lambda x: x[1].get("score", 0), reverse=True)
    for i, (gid, gs) in enumerate(sorted_gpus):
        gs["community_pick"] = i < 2

    data["sentiment"] = sentiment
    log_ok("Sentiment Merge", f"{len(sentiment)} GPUs updated")
    return data


# ===================================================================
# 4. AI ANALYSIS UPDATER
# ===================================================================

def load_config():
    """Try to load LLM config from config.py."""
    try:
        # Use direct file reading to avoid import issues
        config = {}
        config_path = CONFIG_PY
        if not os.path.isfile(config_path):
            return None
        with open(config_path) as f:
            exec(f.read(), config)
        api_base = config.get("LLM_API_BASE")
        api_key = config.get("LLM_API_KEY")
        model = config.get("LLM_MODEL")
        if api_base and api_key and model:
            return {"base": api_base, "key": api_key, "model": model}
    except Exception as exc:
        log_info(f"Could not load config.py: {exc}")
    return None


def regenerate_ai_analysis(data, config):
    """Call the LLM API to regenerate the summary analysis."""
    log_info("Regenerating AI analysis via LLM API...")
    url = f"{config['base']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['key']}",
    }

    # Build a compact market snapshot for the LLM
    providers = data.get("providers", {})
    price_summary = {}
    for prov_name, prov in providers.items():
        gpus = prov.get("gpus", {})
        for gpu_id, gpu_info in gpus.items():
            price = gpu_info.get("price_per_gpu_hr")
            if price is not None:
                if gpu_id not in price_summary:
                    price_summary[gpu_id] = []
                price_summary[gpu_id].append({"provider": prov_name, "price": price})

    # Sort by cheapest
    for gpu_id in price_summary:
        price_summary[gpu_id].sort(key=lambda x: x["price"])

    stocks = data.get("stocks", {})
    news_headlines = [a.get("headline", "") for a in data.get("news", [])[:10]]

    snapshot = json.dumps({
        "gpu_pricing": {k: v[:5] for k, v in price_summary.items()},
        "stocks": stocks,
        "recent_headlines": news_headlines,
        "date": datetime.now().strftime("%Y-%m-%d"),
    }, indent=2, default=str)

    system_prompt = (
        "You are a GPU compute market analyst. Provide a concise market brief "
        "(300-500 words) covering: current GPU pricing trends, key stock movements, "
        "notable news, and actionable insights for organizations planning GPU procurement. "
        "Use markdown formatting. Be specific with numbers and prices."
    )

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate a market brief based on this data:\n\n{snapshot}"},
        ],
        "max_tokens": 3000,
        "temperature": 0.3,
    }

    status, body = http_post(url, payload, headers=headers, timeout=120)
    if status != 200:
        raise RuntimeError(f"LLM API returned HTTP {status}: {body[:200]}")

    resp = json.loads(body)
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("LLM returned empty content")

    log_ok("AI Analysis (LLM)", f"{len(content)} chars generated")
    return content


def update_ai_analysis(data, existing_ai):
    """Update ai_analysis.json, optionally regenerating via LLM."""
    now = datetime.now(timezone.utc).isoformat()
    config = load_config()

    if config:
        try:
            new_summary = regenerate_ai_analysis(data, config)
            existing_ai["summary"] = {
                "analysis": new_summary,
                "type": "quick_summary",
                "timestamp": now,
            }
        except Exception as exc:
            log_fail("AI Analysis (LLM)", str(exc))
    else:
        log_info("No LLM config found -- keeping existing ai_analysis.json unchanged")

    return existing_ai


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 60)
    print("  GPU Market Data Updater")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Load existing data as template
    data = load_existing(DATA_JSON, EMBEDDED_DATA_JSON)
    if not data:
        log_info("No existing data.json or embedded_data.json found. Starting fresh.")
        data = {}

    existing_ai = load_existing(AI_ANALYSIS_JSON, EMBEDDED_AI_JSON)
    if not existing_ai:
        existing_ai = {}

    # Track update timestamp
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # ---- 1. GPU Cloud Pricing ----
    print("[1/10] GPU Cloud Pricing")
    print("-" * 40)

    vastai_prices = None
    try:
        vastai_prices = fetch_vastai_pricing()
    except Exception as exc:
        log_fail("Vast.ai", str(exc))

    runpod_prices = None
    try:
        runpod_prices = fetch_runpod_pricing()
    except Exception as exc:
        log_fail("RunPod", str(exc))

    data = merge_live_pricing_into_data(data, vastai_prices, runpod_prices)
    print()

    # ---- 2. Recalculate derived data (matrix, historical, spot) ----
    print("[2/10] Recalculate Historical")
    print("-" * 40)
    try:
        data = update_historical(data)
    except Exception as exc:
        log_fail("Historical", str(exc))
    print()

    print("[3/10] Recalculate Spot")
    print("-" * 40)
    try:
        data = update_spot(data)
    except Exception as exc:
        log_fail("Spot", str(exc))
    print()

    print("[4/10] Recalculate Matrix")
    print("-" * 40)
    try:
        data = recalculate_matrix(data)
    except Exception as exc:
        log_fail("Matrix", str(exc))
    print()

    # ---- 5. Stock Prices ----
    print("[5/10] Stock Prices")
    print("-" * 40)

    stocks = []
    for ticker in ["NVDA", "AMD"]:
        try:
            stock = fetch_stock_price(ticker)
            stocks.append(stock)
        except Exception as exc:
            log_fail(f"Stock/{ticker}", str(exc))

    data = merge_stocks_into_data(data, stocks)
    print()

    # ---- 6. News ----
    print("[6/10] News Headlines")
    print("-" * 40)

    try:
        articles = fetch_news_rss()
        data = merge_news_into_data(data, articles)
    except Exception as exc:
        log_fail("Google News RSS", str(exc))
    print()

    # ---- 7. Community Sentiment (Reddit + HuggingFace + GitHub) ----
    print("[7/10] Reddit Sentiment")
    print("-" * 40)

    reddit_data = {}
    try:
        reddit_data = fetch_reddit_sentiment()
    except Exception as exc:
        log_fail("Reddit Sentiment", str(exc))
    print()

    print("[8/10] HuggingFace + GitHub")
    print("-" * 40)

    hf_data = {}
    github_data = {}
    try:
        hf_data = fetch_huggingface_models()
    except Exception as exc:
        log_fail("HuggingFace", str(exc))

    try:
        github_data = fetch_github_compat()
    except Exception as exc:
        log_fail("GitHub Compat", str(exc))

    if reddit_data or hf_data or github_data:
        data = merge_sentiment_into_data(data, reddit_data, hf_data, github_data)
    print()

    # ---- 9. Price Forecasts ----
    print("[9/10] Price Forecasts")
    print("-" * 40)

    try:
        forecasts = generate_forecasts(data)
        if forecasts:
            data["forecasts"] = forecasts
            log_ok("Forecasts", f"{len(forecasts)} GPUs forecasted")
        else:
            log_info("No historical data available for forecasting")
    except Exception as exc:
        log_fail("Forecasts", str(exc))
    print()

    # ---- 10. AI Analysis ----
    print("[10/10] AI Analysis")
    print("-" * 40)

    existing_ai = update_ai_analysis(data, existing_ai)
    print()

    # ---- Save ----
    print("Saving files...")
    print("-" * 40)
    save_json(DATA_JSON, data)
    if existing_ai:
        save_json(AI_ANALYSIS_JSON, existing_ai)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if _results["success"]:
        print(f"  Updated ({len(_results['success'])}):")
        for s in _results["success"]:
            print(f"    + {s}")
    if _results["failed"]:
        print(f"  Failed ({len(_results['failed'])}):")
        for f in _results["failed"]:
            print(f"    - {f}")
    if not _results["failed"]:
        print("  All data sources fetched successfully.")
    print()
    print(f"  data.json:        {DATA_JSON}")
    print(f"  ai_analysis.json: {AI_ANALYSIS_JSON}")
    print("=" * 60)

    return 0 if not _results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
