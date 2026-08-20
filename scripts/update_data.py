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
from inference_market import refresh_inference_market
from prediction_markets import (
    build_prediction_markets,
    fetch_kalshi_gpu_markets,
    fetch_polymarket_gpu_markets,
)

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
    """Fetch on-demand GPU pricing from Vast.ai API.

    The endpoint caps a response at ~64 offers no matter what `limit` says, so a
    single price-ascending query returns nothing but the cheapest consumer cards
    -- every datacenter GPU fell off the end and the section quietly served
    carried-forward prices. Query the cheap tail and the datacenter tier
    separately (the VRAM filter keeps the latter in its own 64-offer budget) and
    merge on the cheapest offer per model.
    """
    log_info("Fetching Vast.ai pricing...")
    url = "https://cloud.vast.ai/api/v0/bundles/"
    base = {
        "verified": {"eq": True},
        "external": {"eq": False},
        "rentable": {"eq": True},
        "num_gpus": {"gte": 1},
        "type": "on-demand",
    }
    # (extra filters, ordering) -- one pass per price tier
    passes = [
        ({}, [["dph_total", "asc"]]),                             # consumer / cheap
        ({"gpu_ram": {"gte": 20000}}, [["dph_total", "asc"]]),    # datacenter, cheapest first
        ({"gpu_ram": {"gte": 60000}}, [["dph_total", "asc"]]),    # big-VRAM (B200/B300/H200)
    ]

    gpu_prices = {}
    total_offers = 0
    for extra, order in passes:
        query = json.dumps({**base, **extra, "order": order, "limit": 3000})
        status, body = http_get(url, params={"q": query}, timeout=60)
        if status != 200:
            raise RuntimeError(f"Vast.ai returned HTTP {status}")
        data = json.loads(body)
        offers = data if isinstance(data, list) else data.get("offers", [])
        total_offers += len(offers)

        for offer in offers:
            gpu_name = offer.get("gpu_name", "")
            price_hr = offer.get("dph_total")  # dollars per hour total
            num_gpus = offer.get("num_gpus", 1)
            if not gpu_name or not price_hr or num_gpus < 1:
                continue
            per_gpu = round(price_hr / num_gpus, 4)
            entry = gpu_prices.setdefault(gpu_name, {"min_price": per_gpu, "num_offers": 0})
            entry["min_price"] = min(entry["min_price"], per_gpu)
            entry["num_offers"] += 1

    log_ok("Vast.ai", f"{len(gpu_prices)} GPU models, {total_offers} offers across {len(passes)} passes")
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
                securePrice
                communityPrice
                secureSpotPrice
                communitySpotPrice
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
        secure = gpu.get("securePrice")
        community = gpu.get("communityPrice")
        # RunPod sells two tiers: Secure Cloud (RunPod-operated datacenters, the
        # published on-demand rate) and Community Cloud (third-party hosts).
        # `lowestPrice.uninterruptablePrice` tracks whatever is cheapest and
        # available right now, which is usually a Community host -- so reading
        # it as "the on-demand price" quietly mixed tiers across GPUs (a Secure
        # H100 PCIe rate next to a Community B200 rate) and made RunPod
        # incomparable to the dedicated clouds beside it in the price grid.
        # Publish the Secure rate as on-demand and carry Community separately.
        tier = None
        on_demand = None
        if secure and secure > 0:
            on_demand, tier = secure, "secure"
        elif community and community > 0:
            on_demand, tier = community, "community"
        secure_spot = gpu.get("secureSpotPrice")
        community_spot = gpu.get("communitySpotPrice")
        spot = secure_spot if (tier == "secure" and secure_spot) else None
        if not spot:
            spot = lowest.get("minimumBidPrice") or community_spot
        result[name] = {
            "on_demand_price": on_demand,
            "tier": tier,
            "community_price": community if community and community > 0 else None,
            "spot_price": spot,
            "memory_gb": gpu.get("memoryInGb"),
            "secure_cloud": gpu.get("secureCloud"),
            "community_cloud": gpu.get("communityCloud"),
        }

    log_ok("RunPod", f"{len(result)} GPU types")
    return result


# ===================================================================
# 1b. ADDITIONAL LIVE PRICING FETCHERS (Azure API + HTML scrapers)
# ===================================================================

import re as _re
import html as _html


def _extract_prices_from_html(html, patterns):
    """Apply a list of (gpu_id, regex) pairs to html, return {gpu_id: price_float}.

    Each regex must capture the price as group 1 (dollar amount, float).
    """
    found = {}
    for gpu_id, pattern in patterns:
        m = _re.search(pattern, html, _re.IGNORECASE | _re.DOTALL)
        if not m:
            continue
        try:
            price = float(m.group(1))
            if 0.05 <= price <= 500:  # sanity bounds per GPU-hr
                found[gpu_id] = price
        except (ValueError, IndexError):
            continue
    return found


def fetch_azure_pricing():
    """Azure Retail Prices API — free, unauthenticated.

    Docs: https://learn.microsoft.com/rest/api/cost-management/retail-prices/azure-retail-prices
    Returns {gpu_id: per_gpu_hr_price}.

    Each query targets a specific SKU prefix with a known GPU count, avoiding
    the "single-GPU SKU divided by 8" trap.
    """
    log_info("Fetching Azure Retail Prices API...")
    # (skuName substring, required substring 2 for 8-GPU, internal GPU id, GPUs per VM)
    # ND96 family = 8 GPU NVLink trainers; NC* = inference (1-4 GPU)
    sku_queries = [
        ("ND96", "H100", "H100-SXM", 8),
        ("ND96", "H200", "H200", 8),
        ("ND96", "B200", "B200", 8),
        ("ND", "B300", "B300", 8),
        ("ND", "GB200", "GB200", 4),
        ("ND96", "A100", "A100-80GB", 8),
        ("NC24", "A100", "A100-80GB", 1),  # may overwrite with cheaper per-GPU
        ("NC6s_v3", None, "V100", 1),
        ("NV36", "A10", "A10", 1),
    ]
    base = "https://prices.azure.com/api/retail/prices"
    result = {}
    for prefix, must_contain, gpu_id, gpus_per_vm in sku_queries:
        # No priceType filter: 'Consumption' would exclude the Reservation rows
        # we need for the committed rates. They are separated below by
        # reservationTerm.
        filt = (
            f"serviceName eq 'Virtual Machines' and armRegionName eq 'eastus' "
            f"and startswith(skuName, '{prefix}')"
        )
        try:
            # The retail API throttles bursts; a single 429 used to silently
            # drop that GPU for the run (a refresh once published Azure with
            # one SKU instead of three). Retry before giving up on a query.
            body = None
            for attempt in range(3):
                status, body = http_get(
                    base, params={"$filter": filt, "currencyCode": "USD"}, timeout=25
                )
                if status == 200:
                    break
                body = None
                time.sleep(1.5 * (attempt + 1))
            if body is None:
                continue
            items = json.loads(body).get("Items", [])
            # Azure prices each SKU three ways in one response: pay-as-you-go
            # (no reservationTerm), Spot / Low Priority (flagged in meterName),
            # and reservations (reservationTerm "1 Year" / "3 Years", where the
            # value is the *whole-term* total despite unitOfMeasure saying
            # "1 Hour"). Split them out instead of taking one median across the
            # lot, which is how a reserved total could land in an on-demand row.
            on_demand, spot, res = [], [], {1: [], 3: []}
            on_demand_sku = None
            for it in items:
                sku = it.get("skuName", "")
                meter = (it.get("meterName") or "").lower()
                if must_contain and must_contain not in sku:
                    continue
                up = it.get("retailPrice") or it.get("unitPrice")
                if not up or up <= 0:
                    continue
                up = float(up)
                price_type = (it.get("priceType") or "").strip()
                term = (it.get("reservationTerm") or "").strip()
                if term:
                    years = {"1 Year": 1, "3 Years": 3}.get(term)
                    if years:
                        res[years].append(up / (years * 8760) / gpus_per_vm)
                    continue
                # Everything below is a pay-as-you-go rate. DevTest and other
                # price types are not comparable to published on-demand.
                if price_type and price_type != "Consumption":
                    continue
                if "low priority" in meter:
                    continue
                if "spot" in meter:
                    spot.append(up / gpus_per_vm)
                    continue
                if (it.get("unitOfMeasure") or "").lower() not in ("1 hour", "1hour"):
                    continue
                on_demand.append(up / gpus_per_vm)
                on_demand_sku = on_demand_sku or sku

            if not on_demand:
                continue
            on_demand.sort()
            rates = {"on_demand": round(on_demand[len(on_demand) // 2], 4)}
            if on_demand_sku:
                rates["instance"] = on_demand_sku
            # A "spot" meter that is not below on-demand is a mislabelled row,
            # not a bargain. Drop it rather than publish a 0% discount.
            cheap_spot = [s for s in spot if s < rates["on_demand"]]
            if cheap_spot:
                rates["spot"] = round(min(cheap_spot), 4)
            for years, key in ((1, "reserved_1yr"), (3, "reserved_3yr")):
                if res[years]:
                    rates[key] = round(min(res[years]), 4)
            prev = result.get(gpu_id)
            if not prev or rates["on_demand"] < prev["on_demand"]:
                result[gpu_id] = rates
        except Exception:
            continue
    if not result:
        raise RuntimeError("Azure API returned no parseable pricing")
    log_ok("Azure", f"{len(result)} GPU types")
    return result


def fetch_lambda_pricing():
    """Scrape Lambda Labs on-demand GPU prices from the pricing page."""
    log_info("Fetching Lambda Labs pricing...")
    url = "https://lambda.ai/service/gpu-cloud"
    status, body = http_get(url, headers={"User-Agent": "Mozilla/5.0 (pricing-bot)"}, timeout=20)
    if status != 200:
        raise RuntimeError(f"Lambda returned HTTP {status}")
    # Each entry matches the GPU-name row followed by a "$X.XX / GPU / hour" or "$X.XX" token.
    # Pattern captures the first dollar price occurring within 400 chars after the GPU name.
    patterns = [
        ("B200", r"B200[^$]{0,400}?\$(\d+\.\d{2})"),
        ("GH200", r"GH200[^$]{0,400}?\$(\d+\.\d{2})"),
        ("H100-SXM", r"H100\s*SXM[^$]{0,400}?\$(\d+\.\d{2})"),
        ("H100-PCIe", r"H100\s*PCIe[^$]{0,400}?\$(\d+\.\d{2})"),
        ("A100-80GB", r"A100\s*(?:SXM\s*)?\(?80\s*GB\)?[^$]{0,400}?\$(\d+\.\d{2})"),
        ("A100-40GB", r"A100\s*(?:SXM|PCIe)?\s*\(?40\s*GB\)?[^$]{0,400}?\$(\d+\.\d{2})"),
    ]
    result = _extract_prices_from_html(body, patterns)
    if not result:
        raise RuntimeError("Lambda: no prices parsed from HTML")
    log_ok("Lambda", f"{len(result)} GPU types")
    return result


def fetch_coreweave_pricing():
    """Scrape CoreWeave on-demand GPU prices.

    Page shows per-instance totals; we divide by GPU-count.
    """
    log_info("Fetching CoreWeave pricing...")
    url = "https://www.coreweave.com/pricing"
    status, body = http_get(url, headers={"User-Agent": "Mozilla/5.0 (pricing-bot)"}, timeout=20)
    if status != 200:
        raise RuntimeError(f"CoreWeave returned HTTP {status}")
    # (html_anchor, internal_gpu_id, gpus_per_instance)
    rows = [
        ("GB200 NVL72", "GB200", 4),
        ("HGX B300", "B300", 8),
        ("HGX B200", "B200", 8),
        ("HGX H200", "H200", 8),
        ("HGX H100", "H100-SXM", 8),
        ("A100", "A100-80GB", 8),
        ("L40S", "L40S", 8),
        ("GH200", "GH200", 1),
    ]
    result = {}
    for anchor, gpu_id, gpus in rows:
        m = _re.search(_re.escape(anchor), body)
        if not m:
            continue
        chunk = body[m.start():m.start() + 2500]
        price_m = _re.search(r"On-Demand Price:\s*\$(\d+\.\d{2})", chunk)
        if not price_m:
            # Fall back to any dollar amount within 500 chars
            price_m = _re.search(r"\$(\d+\.\d{2})", chunk[:500])
        if not price_m:
            continue
        try:
            per_gpu = float(price_m.group(1)) / gpus
            if 0.2 <= per_gpu <= 100:
                # keep lowest observed
                if gpu_id not in result or per_gpu < result[gpu_id]:
                    result[gpu_id] = per_gpu
        except ValueError:
            continue
    if not result:
        raise RuntimeError("CoreWeave: no prices parsed from HTML")
    log_ok("CoreWeave", f"{len(result)} GPU types")
    return result


def fetch_together_pricing():
    """Scrape Together.ai GPU-cluster on-demand prices.

    The pricing page carries three tables that all mention `HGX <gpu>`:

      1. Dedicated Inference -- managed single-tenant endpoints (H100 $5.49,
         B200 $8.99). A different product; not a raw GPU rental.
      2. GPU Clusters, "Hardware | Hourly" (H100 $3.99, H200 $5.99, B200 $8.19).
      3. GPU Clusters, "Hardware | ON-Demand | Reserved | 7-30 days | ..."
         -- same on-demand column plus commitment tiers.

    A plain `HGX <gpu> ... $x.xx` regex matches table 1 first, which is how the
    published H100 rate drifted to $5.49 -- 38% above Together's actual
    on-demand cluster price. Parse tables instead and keep only the GPU-cluster
    ones, identified by their header row.
    """
    log_info("Fetching Together.ai HGX cluster pricing...")
    url = "https://www.together.ai/pricing"
    status, body = http_get(url, headers={"User-Agent": "Mozilla/5.0 (pricing-bot)"}, timeout=20)
    if status != 200:
        raise RuntimeError(f"Together returned HTTP {status}")

    def cells(row_html):
        raw = _re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, _re.IGNORECASE | _re.DOTALL)
        return [_re.sub(r"\s+", " ", _html.unescape(_re.sub(r"<[^>]+>", " ", c))).strip() for c in raw]

    def preceding_text(upto, window=2000):
        pre = body[max(0, upto - window):upto]
        pre = _re.sub(r"<script.*?</script>", " ", pre, flags=_re.IGNORECASE | _re.DOTALL)
        pre = _re.sub(r"<style.*?</style>", " ", pre, flags=_re.IGNORECASE | _re.DOTALL)
        return _re.sub(r"\s+", " ", _html.unescape(_re.sub(r"<[^>]+>", " ", pre))).lower()

    anchors = [("B200", "HGX B200"), ("H200", "HGX H200"), ("H100-SXM", "HGX H100")]
    result = {}
    for match in _re.finditer(r"<table[^>]*>.*?</table>", body, _re.IGNORECASE | _re.DOTALL):
        table = match.group(0)
        if "HGX" not in table:
            continue
        # Keep only tables sitting under the "GPU Clusters" heading. The tables
        # carry no usable header row (some have none; the tiered one splits its
        # header across two rows), so the enclosing section heading is the only
        # reliable discriminator.
        pre = preceding_text(match.start())
        cluster_at = pre.rfind("gpu cluster")
        dedicated_at = pre.rfind("dedicated inference")
        if cluster_at < 0 or cluster_at < dedicated_at:
            continue
        rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", table, _re.IGNORECASE | _re.DOTALL)
        for row in rows:
            cs = cells(row)
            if not cs:
                continue
            label = " ".join(cs).replace("\xa0", " ")
            label = _re.sub(r"\s+", " ", label)
            for gpu_id, anchor in anchors:
                if anchor.lower() not in label.lower():
                    continue
                # First dollar amount in the row is the on-demand rate; later
                # columns are commitment tiers.
                pm = _re.search(r"\$(\d+\.\d{2})", label)
                if not pm:
                    continue
                price = float(pm.group(1))
                if not (0.05 <= price <= 500):
                    continue
                # Same rate appears in both cluster tables; lowest wins if they
                # ever disagree.
                if gpu_id not in result or price < result[gpu_id]:
                    result[gpu_id] = price
                break

    if not result:
        raise RuntimeError("Together: no GPU-cluster prices parsed from HTML")
    log_ok("Together", f"{len(result)} GPU types")
    return result


def fetch_aws_pricing():
    """AWS pricing: EC2 on-demand prices via the public pricing JSON endpoint.

    The full offers file is huge, so we use a per-region bulk endpoint for US East.
    Falls back to hardcoded if the JSON layout changes.
    """
    log_info("Fetching AWS pricing (us-east-1 bulk)...")
    url = (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/"
        "current/us-east-1/index.json"
    )
    try:
        status, body = http_get(url, timeout=45)
    except Exception as exc:
        raise RuntimeError(f"AWS pricing fetch failed: {exc}")
    if status != 200:
        raise RuntimeError(f"AWS returned HTTP {status}")

    try:
        doc = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"AWS JSON parse failed: {exc}")

    products = doc.get("products", {})
    terms = doc.get("terms", {}).get("OnDemand", {})
    reserved_terms = doc.get("terms", {}).get("Reserved", {})

    # instance_type -> (internal GPU id, GPUs per VM)
    instance_map = {
        "p5.48xlarge": ("H100-SXM", 8),
        "p5e.48xlarge": ("H200", 8),
        "p5en.48xlarge": ("H200", 8),
        "p6-b200.48xlarge": ("B200", 8),
        "p6-b300.48xlarge": ("B300", 8),
        "p4d.24xlarge": ("A100-40GB", 8),
        "p4de.24xlarge": ("A100-80GB", 8),
        "g5.48xlarge": ("A10G", 8),
        "g6.48xlarge": ("L4", 8),
        "g6e.48xlarge": ("L40S", 8),
    }

    # Map sku -> instance_type for products with Linux/Shared tenancy/No pre-install
    sku_to_instance = {}
    for sku, p in products.items():
        attrs = p.get("attributes", {})
        if (
            attrs.get("tenancy") == "Shared"
            and attrs.get("operatingSystem") == "Linux"
            and attrs.get("preInstalledSw") == "NA"
            and attrs.get("capacitystatus") == "Used"
        ):
            it = attrs.get("instanceType")
            if it in instance_map:
                sku_to_instance[sku] = it

    def reserved_rate(sku, years):
        """Effective $/hr for a Standard RI over `years`, No Upfront preferred.

        Standard (not Convertible) No Upfront is the closest thing AWS has to a
        like-for-like committed rate: no capital outlay, so it compares
        directly against on-demand. Fall back to the other purchase options,
        amortising any upfront over the term.
        """
        want = f"{years}yr"
        best = None
        for _oid, term in (reserved_terms.get(sku) or {}).items():
            ta = term.get("termAttributes", {})
            if ta.get("LeaseContractLength") != want:
                continue
            if ta.get("OfferingClass") != "standard":
                continue
            upfront = hourly = 0.0
            for _pid, pd in (term.get("priceDimensions") or {}).items():
                try:
                    val = float(pd.get("pricePerUnit", {}).get("USD", 0))
                except (TypeError, ValueError):
                    continue
                if pd.get("unit") == "Quantity":
                    upfront = val
                elif pd.get("unit") == "Hrs":
                    hourly = val
            effective = hourly + upfront / (years * 8760)
            if effective <= 0:
                continue
            prefer = ta.get("PurchaseOption") == "No Upfront"
            if best is None or (prefer and not best[1]) or (prefer == best[1] and effective < best[0]):
                best = (effective, prefer)
        return best[0] if best else None

    result = {}
    for sku, inst_type in sku_to_instance.items():
        offer = terms.get(sku, {})
        for _oid, odata in offer.items():
            pds = odata.get("priceDimensions", {})
            for _pid, pd in pds.items():
                price = pd.get("pricePerUnit", {}).get("USD")
                if not price:
                    continue
                try:
                    p = float(price)
                except ValueError:
                    continue
                if p <= 0:
                    continue
                gpu_id, gpus = instance_map[inst_type]
                rates = {"on_demand": round(p / gpus, 4), "instance": inst_type}
                for years, key in ((1, "reserved_1yr"), (3, "reserved_3yr")):
                    rr = reserved_rate(sku, years)
                    # Newer instance families (p6-*) have no RI terms yet. No
                    # rate is the honest answer; a discount guess is not.
                    if rr:
                        rates[key] = round(rr / gpus, 4)
                prev = result.get(gpu_id)
                if not prev or rates["on_demand"] < prev["on_demand"]:
                    result[gpu_id] = rates

    if not result:
        raise RuntimeError("AWS: no prices parsed from index.json")
    log_ok("AWS", f"{len(result)} GPU types")
    return result


def fetch_gcp_pricing():
    """GCP accelerator-optimized rates, parsed by column.

    The table carries several rates per machine type, and its columns are:

        Machine type | GPU | Components | Price | DWS Flex-start | DWS Calendar
        | Current Spot | Compute Resource CUDs - 1 Year | ... - 3 Year

    The old parser took the first dollar amount after the machine type. That
    silently read the wrong column whenever "Price" was N/A: a4-highgpu-8g has
    no published on-demand rate, so B200 was published at $8.055/GPU-hr, which
    is really the DWS Flex-start price. Match the header row, then read cells
    by index -- and leave a GPU out entirely when its on-demand cell is N/A
    rather than substituting the next number along.

    Returns {gpu_id: {"on_demand": x, "spot": y, "reserved_1yr": z,
    "reserved_3yr": w}} with missing rates omitted.
    """
    log_info("Fetching GCP accelerator-optimized pricing...")
    url = "https://cloud.google.com/products/compute/pricing/accelerator-optimized"
    status, body = http_get(url, headers={"User-Agent": "Mozilla/5.0 (pricing-bot)"}, timeout=30)
    if status != 200:
        raise RuntimeError(f"GCP returned HTTP {status}")

    # machine_type -> (gpu_id, gpus_per_vm)
    wanted = {
        "a3-highgpu-8g": ("H100-SXM", 8),
        "a3-ultragpu-8g": ("H200", 8),
        "a4-highgpu-8g": ("B200", 8),
        "a2-ultragpu-8g": ("A100-80GB", 8),
        "a2-highgpu-8g": ("A100-40GB", 8),
        "g2-standard-96": ("L4", 8),
    }
    # our key -> substring of the column header
    COLUMNS = {
        "on_demand": "price (usd)",
        "spot": "current spot",
        "reserved_1yr": "cuds - 1 year",
        "reserved_3yr": "cuds - 3 year",
    }

    def cells(row_html):
        raw = _re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, _re.IGNORECASE | _re.DOTALL)
        return [_re.sub(r"\s+", " ", _html.unescape(_re.sub(r"<[^>]+>", " ", c))).strip() for c in raw]

    def money(cell):
        if not cell or "n/a" in cell.lower():
            return None
        m = _re.search(r"\$\s*([\d,]+\.?\d*)", cell)
        return float(m.group(1).replace(",", "")) if m else None

    result = {}
    for table in _re.findall(r"<table[^>]*>.*?</table>", body, _re.IGNORECASE | _re.DOTALL):
        rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", table, _re.IGNORECASE | _re.DOTALL)
        if not rows:
            continue
        header = [c.lower() for c in cells(rows[0])]
        idx = {}
        for key, needle in COLUMNS.items():
            for i, col in enumerate(header):
                if needle in col:
                    idx[key] = i
                    break
        if "on_demand" not in idx:
            continue
        for row in rows[1:]:
            cs = cells(row)
            if not cs:
                continue
            mt = cs[0].strip()
            if mt not in wanted:
                continue
            gpu_id, gpus = wanted[mt]
            rates = {"instance": mt}
            for key, i in idx.items():
                val = money(cs[i]) if i < len(cs) else None
                if val is None:
                    continue
                per_gpu = val / gpus
                if 0.01 <= per_gpu <= 200:
                    rates[key] = round(per_gpu, 4)
            # No published on-demand rate means no listing. The other columns
            # are different products (Flex-start, Calendar), not a substitute.
            if "on_demand" not in rates:
                log_info(f"GCP: {mt} has no on-demand rate published, skipping")
                continue
            prev = result.get(gpu_id)
            if not prev or rates["on_demand"] < prev["on_demand"]:
                result[gpu_id] = rates

    if not result:
        raise RuntimeError("GCP: no prices parsed from HTML")
    log_ok("GCP", f"{len(result)} GPU types")
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
    "A100 SXM 40GB": "A100-40GB",
    "A100_80GB": "A100-80GB",
    "A100_PCIE_80GB": "A100-80GB",
    "H100 SXM": "H100-SXM",
    "H100 PCIe": "H100-PCIe",
    "H100 80GB SXM": "H100-SXM",
    "H100_SXM": "H100-SXM",
    "H100_PCIE": "H100-PCIe",
    # H100 NVL (94GB PCIe) and H200 NVL (143GB PCIe) are distinct, cheaper parts
    # than the SXM modules. Folding them into H100-SXM / H200 let an NVL quote
    # be published as an SXM price -- on RunPod that put H100 SXM at $2.59 when
    # the SXM rate was $3.29. They have no spec entry, so they are dropped
    # rather than mislabelled.
    "H200": "H200",
    "H200 SXM": "H200",
    "B200": "B200",
    "L40S": "L40S",
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
    "H100-SXM5-80GB": "H100-SXM",
    "H100-PCIE-80GB": "H100-PCIe",
    "MI300X": "MI300X",
    "MI325X": "MI325X",
    "B200": "B200",
    "B300": "B300",
    "GB200": "GB200",
}


# Parts we deliberately do not track, because the substring fallback below would
# otherwise resolve them to a neighbour and publish the wrong card's price:
#   "H200 NVL" / "H100 NVL" -> cheaper PCIe parts, not the SXM modules
#   "L40"                   -> would fall through to "L4"
_GPU_NAME_UNTRACKED = ("NVL", "L40")


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
    if any(_re.search(rf"\b{tok}\b", name_upper) for tok in _GPU_NAME_UNTRACKED):
        return None
    # Substring matching, longest key first -- "A100 SXM 40GB" must not resolve
    # via the shorter "A100 SXM" key, and "RTX 4090 Ti" must not beat "RTX 4090".
    for ext, internal in sorted(_GPU_NAME_MAP.items(), key=lambda kv: -len(kv[0])):
        if ext.upper() in name_upper:
            return internal
    return None


def get_hardcoded_fallback_prices():
    """Last-known-good prices for provider/GPU pairs with no usable live feed.

    Every figure here was verified against the provider's own published pricing
    on VERIFIED_ON. This table is a *floor*, not a source of truth: it is only
    written when an entry has no live-sourced price at all (see the guard in
    merge_live_pricing_into_data), and it must never contain a price nobody
    publishes. The previous table carried invented entries -- FluidStack B200/
    B300, Azure H200/B200/B300/MI300X, CoreWeave B300 -- for hardware those
    providers do not sell, and its cloud prices ran 24-69% below list, which is
    how a single failed scrape could understate a whole provider.
    """
    # Bump this whenever a price below is re-checked against the provider.
    VERIFIED_ON = "2026-07-28"
    return {
        # AWS EC2 on-demand, us-east-1 (per-GPU = instance price / GPU count).
        # Cross-checked against the AWS bulk pricing API and instances.vantage.sh.
        "AWS": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                "B300": {"price_per_gpu_hr": 17.802},      # p6-b300.48xlarge $142.416/8
                "B200": {"price_per_gpu_hr": 14.242},      # p6-b200.48xlarge $113.934/8
                "H200": {"price_per_gpu_hr": 7.912},       # p5en.48xlarge
                "H100-SXM": {"price_per_gpu_hr": 6.880},   # p5.48xlarge $55.04/8
                "A100-80GB": {"price_per_gpu_hr": 3.431},  # p4de.24xlarge
                "A100-40GB": {"price_per_gpu_hr": 2.745},  # p4d.24xlarge
                "L40S": {"price_per_gpu_hr": 3.766},       # g6e
                "L4": {"price_per_gpu_hr": 1.669},         # g6
                "T4": {"price_per_gpu_hr": 0.526},         # g4dn.xlarge
            },
        },
        # Google Cloud on-demand, us-central1.
        "GCP": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                # No B200: Google publishes no on-demand rate for a4-highgpu-8g.
                # The 8.055 that used to sit here was $64.44/8 -- the DWS
                # Flex-start price, a different product, picked up by a parser
                # that took the first dollar amount in the row. It survived the
                # parser fix because this table put it straight back.
                "H200": {"price_per_gpu_hr": 10.601},      # a3-ultragpu-8g
                "H100-SXM": {"price_per_gpu_hr": 11.061},  # a3-highgpu-8g
                "A100-80GB": {"price_per_gpu_hr": 5.069},  # a2-ultragpu-1g
                "A100-40GB": {"price_per_gpu_hr": 3.673},  # a2-highgpu-1g
                "L4": {"price_per_gpu_hr": 1.000},         # g2-standard
                "T4": {"price_per_gpu_hr": 0.350},
            },
        },
        # Azure Retail Prices API, eastus. Azure does not publish retail pricing
        # for H200 / B200 / B300 / MI300X / MI325X / L40S in this region, so
        # they are deliberately absent rather than guessed.
        "Azure": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                "GB200": {"price_per_gpu_hr": 28.512},     # ND128isrNDRGB200v6 $114.048/4
                "H100-SXM": {"price_per_gpu_hr": 11.613},  # ND96is*H100v5 median /8
                "A100-80GB": {"price_per_gpu_hr": 3.952},  # ND96amsA100v4
            },
        },
        # Lambda Labs public on-demand rates.
        "Lambda": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                "B200": {"price_per_gpu_hr": 6.690},
                "H100-SXM": {"price_per_gpu_hr": 3.990},
                "H100-PCIe": {"price_per_gpu_hr": 3.290},
                "GH200": {"price_per_gpu_hr": 2.290},
            },
        },
        # CoreWeave public on-demand rates.
        "CoreWeave": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                "GB200": {"price_per_gpu_hr": 10.500},
                "B200": {"price_per_gpu_hr": 8.600},
                "GH200": {"price_per_gpu_hr": 6.500},
                "H200": {"price_per_gpu_hr": 6.305},
                "H100-SXM": {"price_per_gpu_hr": 6.155},
                "A100-80GB": {"price_per_gpu_hr": 2.700},
                "L40S": {"price_per_gpu_hr": 2.250},
            },
        },
        # Together.ai GPU-cluster on-demand rates, read from its published price
        # list (the HTML scraper is unreliable). It does not publish A100 or
        # GB200/GB300 rates -- those are quote-only, so they are absent here.
        "Together": {
            "last_verified": VERIFIED_ON,
            "source": "hardcoded_fallback",
            "gpus": {
                "B200": {"price_per_gpu_hr": 8.19},
                "H200": {"price_per_gpu_hr": 5.99},
                "H100-SXM": {"price_per_gpu_hr": 3.99},
            },
        },
        # FluidStack was here. It no longer publishes a self-serve price list,
        # so there is nothing to fall back to -- see the retired-provider note
        # in merge_live_pricing_into_data.
    }


# Committed / interruptible rates a fetcher may return alongside on-demand.
# Absent means the provider does not publish one, which is a fact worth
# showing; it is never filled in with a modelled discount.
_RATE_KEYS = (("spot", "spot_hr"), ("reserved_1yr", "reserved_1yr_hr"),
              ("reserved_3yr", "reserved_3yr_hr"))


def _apply_scraped_prices(providers, provider_key, prices, source_tag, tracked_specs, now):
    """Merge scraped prices into providers[provider_key].gpus.

    `prices` maps gpu_id to either a bare on-demand float (older fetchers) or a
    {"on_demand": x, "spot": y, "reserved_1yr": z, "reserved_3yr": w} dict.
    """
    if not prices:
        return
    if provider_key not in providers:
        providers[provider_key] = {
            "provider_name": provider_key,
            "type": "cloud",
            "gpus": {},
        }
    prov = providers[provider_key]
    prov_gpus = prov.setdefault("gpus", {})

    # A successful fetch that omits a GPU is evidence the provider no longer
    # lists it -- not a fetch failure. Without this the entry sat there forever
    # marked `stale: true` and kept being published: GCP's B200 stayed on the
    # grid at $8.055 after we established that Google publishes no on-demand
    # price for a4-highgpu-8g and that $8.055 was the DWS Flex-start rate read
    # from the wrong column. Only prune what this same source put there;
    # fallbacks and other sources are not ours to withdraw.
    withdrawn = [
        gpu_id for gpu_id, entry in prov_gpus.items()
        if entry.get("source") == source_tag and gpu_id not in prices
    ]
    for gpu_id in withdrawn:
        prov_gpus.pop(gpu_id, None)
    if withdrawn:
        log_info(f"{provider_key}: no longer listed, removed {', '.join(sorted(withdrawn))}")

    for gpu_id, value in prices.items():
        if gpu_id not in tracked_specs:
            continue
        rates = value if isinstance(value, dict) else {"on_demand": value}
        if rates.get("on_demand") is None:
            continue
        if gpu_id not in prov_gpus:
            prov_gpus[gpu_id] = {}
        entry = prov_gpus[gpu_id]
        entry["price_per_gpu_hr"] = round(float(rates["on_demand"]), 3)
        if rates.get("instance"):
            entry["instance"] = rates["instance"]
        for src_key, out_key in _RATE_KEYS:
            val = rates.get(src_key)
            if val is not None:
                entry[out_key] = round(float(val), 3)
            else:
                # Stop publishing last run's rate if the provider withdrew it.
                entry.pop(out_key, None)
        entry["source"] = source_tag
        entry["last_updated"] = now
        entry["last_verified"] = now
        entry.pop("stale", None)
    prov["last_updated"] = now


def merge_live_pricing_into_data(
    data,
    vastai_prices,
    runpod_prices,
    azure_prices=None,
    lambda_prices=None,
    coreweave_prices=None,
    together_prices=None,
    aws_prices=None,
    gcp_prices=None,
):
    """Merge live pricing data into the providers section of data.json."""
    now = datetime.now(timezone.utc).isoformat()
    providers = data.get("providers", {})

    # Drop providers we no longer track.
    #
    # FluidStack retired 2026-08-20: it has taken down its public price list
    # (fluidstack.io/pricing 404s) and now sells contracted capacity rather
    # than self-serve instances. Its last figures were constants copied from
    # third-party aggregators, and because they were the lowest numbers in the
    # set they were being published as the cheapest listed H100 ($2.10) and
    # H200 ($2.30) on the dashboard -- a price nobody could actually book, in
    # the one column readers trust most. No public rate, no listing.
    for retired in ("Oracle", "FluidStack"):
        providers.pop(retired, None)

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
        vast_gpus = vast_prov.setdefault("gpus", {})
        tracked = set(data.get("specs", {}).keys())
        for ext_name, info in vastai_prices.items():
            internal = normalize_gpu_name(ext_name)
            if not internal or internal not in tracked:
                continue
            if internal not in vast_gpus:
                vast_gpus[internal] = {}
            gpu_entry = vast_gpus[internal]
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
        rp_gpus = rp_prov.setdefault("gpus", {})
        tracked = set(data.get("specs", {}).keys())
        for ext_name, info in runpod_prices.items():
            internal = normalize_gpu_name(ext_name)
            if not internal or internal not in tracked:
                continue
            if info.get("on_demand_price") is None:
                continue
            if internal not in rp_gpus:
                rp_gpus[internal] = {}
            gpu_entry = rp_gpus[internal]
            # Several RunPod SKUs collapse to one internal id (A100 PCIe and
            # A100 SXM both land on A100-80GB). Dict order decided the winner
            # before; take the cheapest published rate so the result is stable.
            already = gpu_entry.get("price_per_gpu_hr")
            if (
                already is not None
                and gpu_entry.get("last_updated") == now
                and already <= info["on_demand_price"]
            ):
                continue
            gpu_entry["price_per_gpu_hr"] = info["on_demand_price"]
            gpu_entry["source"] = "runpod_api"
            gpu_entry["last_updated"] = now
            if info.get("tier"):
                gpu_entry["tier"] = info["tier"]
            if info.get("community_price") is not None:
                gpu_entry["community_price"] = info["community_price"]
            if info.get("spot_price") is not None:
                gpu_entry["spot_price"] = info["spot_price"]
            if info.get("memory_gb"):
                gpu_entry["memory_gb"] = info["memory_gb"]
        rp_prov["last_updated"] = now

    # -- Scraped/API-fetched providers --
    tracked = set(data.get("specs", {}).keys())
    _apply_scraped_prices(providers, "Azure", azure_prices, "azure_retail_api", tracked, now)
    _apply_scraped_prices(providers, "Lambda", lambda_prices, "lambda_scrape", tracked, now)
    _apply_scraped_prices(providers, "CoreWeave", coreweave_prices, "coreweave_scrape", tracked, now)
    _apply_scraped_prices(providers, "Together", together_prices, "together_scrape", tracked, now)
    _apply_scraped_prices(providers, "AWS", aws_prices, "aws_pricing_api", tracked, now)
    _apply_scraped_prices(providers, "GCP", gcp_prices, "gcp_scrape", tracked, now)

    # -- Hardcoded fallbacks for providers without free APIs --
    # Only fill in GPUs that no live source has already populated this run.
    live_sources = {
        "vastai_api", "runpod_api", "azure_retail_api", "lambda_scrape",
        "coreweave_scrape", "together_scrape", "aws_pricing_api", "gcp_scrape",
    }
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
            # Never downgrade a real, provider-sourced price to a constant. The
            # old guard only protected a scrape from *this* run, so any transient
            # scraper failure replaced a verified API price with a hardcoded one
            # -- and those constants run 24-69% below real list prices, so a
            # single failed fetch silently understated a whole provider. Keep the
            # last real price and mark it stale instead.
            if existing_entry.get("source") in live_sources and existing_entry.get("price_per_gpu_hr"):
                if existing_entry.get("last_updated") != now:
                    existing_entry["stale"] = True
                continue
            existing_entry["price_per_gpu_hr"] = gpu_info["price_per_gpu_hr"]
            existing_entry["source"] = "hardcoded_fallback"
            existing_entry["last_verified"] = fb_data["last_verified"]
            existing_entry.pop("stale", None)

        # Mirror of the prune in _apply_scraped_prices. Dropping a bad constant
        # from the table above is not enough on its own: the entry it wrote on
        # an earlier run is still sitting in data.json under the
        # hardcoded_fallback source, which also shields it from the live-fetch
        # prune. GCP's B200 survived two separate fixes that way. If we no
        # longer stand behind a constant, withdraw the price it produced.
        orphaned = [
            gpu_id for gpu_id, entry in existing_gpus.items()
            if entry.get("source") == "hardcoded_fallback" and gpu_id not in fb_data["gpus"]
        ]
        for gpu_id in orphaned:
            existing_gpus.pop(gpu_id, None)
        if orphaned:
            log_info(f"{provider_key}: withdrawn from fallback table, removed {', '.join(sorted(orphaned))}")

    # Every provider carries a last_updated so the Data tab can report its
    # freshness. Fallback-only providers (e.g. FluidStack) had none at all.
    for prov in providers.values():
        prov.setdefault("last_updated", now)
        # Retire the discount constants. reserved_1yr_discount /
        # reserved_3yr_discount / spot_discount were seeded once and never
        # fetched -- grep showed them being read in four places and written in
        # none -- yet they priced every reserved and spot figure on the site,
        # including the TCO break-even a buyer commits real money against.
        # Committed and interruptible rates now come from the provider's own
        # price list, per GPU, and are simply absent where none is published.
        for stale_key in ("reserved_1yr_discount", "reserved_3yr_discount", "spot_discount"):
            prov.pop(stale_key, None)

    # Record which providers actually publish term pricing, so the UI can say
    # "3 of 8" rather than implying the whole market offers a commitment.
    publishes = {"reserved_1yr_hr": [], "reserved_3yr_hr": [], "spot_hr": []}
    for prov_key, prov in providers.items():
        for key, names in publishes.items():
            if any(g.get(key) for g in (prov.get("gpus") or {}).values()):
                names.append(prov_key)
    data["rate_coverage"] = {
        "providers_total": len(providers),
        "reserved_1yr": sorted(publishes["reserved_1yr_hr"]),
        "reserved_3yr": sorted(publishes["reserved_3yr_hr"]),
        "spot": sorted(publishes["spot_hr"]),
        "note": (
            "Committed and spot rates are read from each provider's published "
            "price list. Providers absent from a list do not publish that rate "
            "publicly -- it is not shown as a discount off on-demand."
        ),
    }

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

    # Collect per-GPU rates. Committed and spot rates come from the provider's
    # own price list where it publishes one -- see _RATE_KEYS. They used to be
    # derived by applying a per-provider discount constant to on-demand, but
    # those constants were carried forward from the original seed file and
    # nothing ever verified them: they were only ever read, never fetched. The
    # real spread is wider and less uniform than any constant (Azure spot runs
    # ~80% below on-demand, not the 40% that was assumed), and newer parts have
    # no committed rate at all.
    gpu_prices = {}
    for prov_key, prov in providers.items():
        for gpu_id, gpu_info in (prov.get("gpus") or {}).items():
            price = gpu_info.get("price_per_gpu_hr")
            if price is not None and price > 0:
                gpu_prices.setdefault(gpu_id, []).append({
                    "price": price,
                    "spot": gpu_info.get("spot_hr"),
                    "res1": gpu_info.get("reserved_1yr_hr"),
                    "res3": gpu_info.get("reserved_3yr_hr"),
                })

    for gpu_id, entries in gpu_prices.items():
        on_demand = [e["price"] for e in entries]
        # Only providers that publish a committed rate contribute one. A GPU
        # nobody publishes a term price for reports none, rather than a number
        # implying a discount that is not on offer anywhere.
        reserved_1yr = [e["res1"] for e in entries if e["res1"]]
        reserved_3yr = [e["res3"] for e in entries if e["res3"]]
        spot_rates = [e["spot"] for e in entries if e["spot"]]

        existing = spot.get(gpu_id, {})
        avg_od = round(sum(on_demand) / len(on_demand), 2)

        # quarterly_trend used to append on *every* run, so the "last 4
        # quarters" were really the last 4 runs -- a few updates in one day
        # flattened it and the header ticker's change-since read 0%. Track the
        # quarter each point belongs to and overwrite within the same quarter.
        trend = list(existing.get("quarterly_trend") or [])
        now = datetime.now(timezone.utc)
        quarter = f"{now.year}Q{(now.month - 1) // 3 + 1}"
        if existing.get("trend_quarter") == quarter and trend:
            trend[-1] = avg_od
        else:
            trend.append(avg_od)
        trend = trend[-4:]

        # Retired: bid / ask / spread_pct. Nothing trades against a provider
        # list price, so those three were a fixed percentage band painted
        # around the average and dressed up as an order book. Real quotes now
        # live in the prediction_markets section, which is an actual venue.
        # on_demand_low/avg/high already say what the listings do.

        def band(key):
            """Low / avg / high for a rate, plus the saving against the SAME
            providers' on-demand prices.

            Comparing a committed average taken over three providers against an
            on-demand average taken over eight measures basket composition, not
            the discount. Match them.
            """
            pairs = [(e[key], e["price"]) for e in entries if e[key]]
            if not pairs:
                return None
            rates = [r for r, _ in pairs]
            base = sum(od for _, od in pairs)
            return {
                "low": round(min(rates), 2),
                "avg": round(sum(rates) / len(rates), 2),
                "high": round(max(rates), 2),
                "savings_pct": round((1 - sum(rates) / base) * 100) if base else None,
                # On-demand over the SAME providers. Without it the committed
                # average looks dearer than on-demand whenever the cheap
                # marketplaces (which publish no term rate) drag the all-provider
                # on-demand average down -- H100-SXM reads $7.38 committed
                # against a $6.16 blended on-demand, which is a basket artefact,
                # not a price. Anything comparing levels must use this.
                "vs_ondemand": round(base / len(pairs), 2),
                "providers": len(pairs),
            }

        res1, res3, sp = band("res1"), band("res3"), band("spot")
        record = {
            "on_demand_low": round(min(on_demand), 2),
            "on_demand_avg": avg_od,
            "on_demand_high": round(max(on_demand), 2),
            "num_providers": len(entries),
            "quarterly_trend": trend,
            "trend_quarter": quarter,
        }
        # Absent means nobody publishes that rate for this GPU. Previously these
        # fell back to `existing`, which quietly resurrected the old modelled
        # numbers run after run.
        for prefix, vals in (("reserved_1yr", res1), ("reserved_3yr", res3), ("spot", sp)):
            if not vals:
                continue
            record[f"{prefix}_low"] = vals["low"]
            record[f"{prefix}_avg"] = vals["avg"]
            record[f"{prefix}_high"] = vals["high"]
            record[f"{prefix}_savings_pct"] = vals["savings_pct"]
            record[f"{prefix}_vs_ondemand"] = vals["vs_ondemand"]
            record[f"{prefix}_providers"] = vals["providers"]
        spot[gpu_id] = record

    # Drop GPUs no tracked provider prices any more, rather than leaving the
    # last known quote sitting there looking current.
    dropped = [g for g in list(spot) if g not in gpu_prices]
    for gpu_id in dropped:
        spot.pop(gpu_id, None)
        (data.get("forecasts") or {}).pop(gpu_id, None)
    if dropped:
        log_info(f"Spot: dropped {', '.join(dropped)} (no provider lists them)")

    data["spot"] = spot
    log_ok("Spot", f"{len(gpu_prices)} GPUs refreshed")
    return data


def _cheapest_provider_for(providers, gpu_id):
    """(price, provider_name) for the cheapest live listing of a GPU, or None."""
    best = None
    for name, prov in (providers or {}).items():
        info = (prov.get("gpus") or {}).get(gpu_id) or {}
        price = info.get("price_per_gpu_hr")
        if price and price > 0 and (best is None or price < best[0]):
            best = (round(price, 4), name)
    return best


def refresh_workload_recs(data):
    """Re-price the workload recommendations from live provider data.

    current_prices used to be a frozen snapshot, so it drifted away from the
    Pricing tab -- e.g. GB200 was listed as "cheapest $27.04" when the actual
    cheapest live listing was $10.50 (27.04 was the *most expensive*).
    """
    recs = data.get("workload_recs")
    if not recs:
        return data

    providers = data.get("providers") or {}
    repriced = 0
    for _workload, rec in recs.items():
        prices = rec.get("current_prices")
        if not prices:
            continue
        for gpu_id in list(prices.keys()):
            best = _cheapest_provider_for(providers, gpu_id)
            if not best:
                continue
            price, provider = best
            prices[gpu_id] = {
                "cheapest": price,
                "provider": provider,
                "monthly_1gpu": round(price * 730, 2),
            }
            repriced += 1

    log_ok("Workload Recs", f"{repriced} GPU prices re-derived from live listings")
    return data


def _avg_spot_rate(providers, gpu_id):
    """Average published spot rate for a GPU, or None if nobody publishes one.

    This used to apply a per-provider spot_discount constant to the on-demand
    price, which produced a spot rate for every GPU at every provider whether
    or not one existed. Only Azure and GCP publish interruptible rates through
    an API; AWS spot needs a signed EC2 call and is not in the public bulk
    pricing file, so AWS contributes nothing here.
    """
    rates = [
        info["spot_hr"]
        for prov in (providers or {}).values()
        for gid, info in (prov.get("gpus") or {}).items()
        if gid == gpu_id and info.get("spot_hr")
    ]
    return round(sum(rates) / len(rates), 2) if rates else None


def refresh_tco(data):
    """Point the TCO and reservation models at live cloud prices.

    Both carried their own frozen cloud rates, so the same GPU was quoted at
    three different on-demand prices depending on the tab -- H100-SXM was
    $2.49/hr in TCO and $2.18/hr in reservations while Pricing showed a $5.73
    average. TCO's cloud-vs-self-hosted breakeven is computed from those rates,
    so a stale number there changes the conclusion, not just the display.
    """
    spot = data.get("spot") or {}
    providers = data.get("providers") or {}
    if not spot:
        return data

    synced = 0
    for gpu_id, s in spot.items():
        od = s.get("on_demand_avg")
        res1 = s.get("reserved_1yr_avg")
        res3 = s.get("reserved_3yr_avg")
        spot_rate = _avg_spot_rate(providers, gpu_id)
        if od is None:
            continue

        tco = (data.get("tco") or {}).get(gpu_id)
        if tco:
            tco["cloud_on_demand_hr"] = od
            if res1 is not None:
                tco["cloud_reserved_1yr_hr"] = res1
            if res3 is not None:
                tco["cloud_reserved_3yr_hr"] = res3
            if spot_rate is not None:
                tco["cloud_spot_hr"] = spot_rate
            synced += 1

        # Break-even used to live in a parallel `reservations` dataset that
        # duplicated this maths. It is now a field on the TCO profile.
        if tco and res1 is not None and res3 is not None:
            tco["commitment"] = {
                "spot_rate": spot_rate,
                "reserved_1yr_rate": res1,
                "reserved_3yr_rate": res3,
                # A commitment is billed for every hour of the term, so at
                # utilisation u its effective per-used-hour cost is rate / u.
                # Break-even is where that equals the on-demand rate.
                "breakeven_utilization_1yr_pct": round(res1 / od * 100),
                "breakeven_utilization_3yr_pct": round(res3 / od * 100),
                "savings_at_utilization": {
                    f"{int(u*100)}_pct": {
                        "spot": round((1 - spot_rate / od) * 100) if spot_rate else None,
                        "reserved_1yr": round((1 - (res1 / u) / od) * 100),
                        "reserved_3yr": round((1 - (res3 / u) / od) * 100),
                    } for u in (0.4, 0.6, 0.8, 1.0)
                },
            }
        synced += 1

    data.pop("reservations", None)
    log_ok("TCO", f"{synced} profiles synced to live cloud rates")
    return data


def refresh_lead_times(data):
    """Make the three places that report lead times agree with each other.

    indicators.gpu_lead_times is the maintained per-GPU source. The flagship
    lead-time series and the per-vendor supply-chain figures were separate
    hand-written numbers, so the dashboard simultaneously claimed the flagship
    lead time was 36 weeks (series), 8 weeks (per-GPU) and 1 week (vendor).
    Both derived views are now computed from the per-GPU table.
    """
    indicators = data.get("indicators") or {}
    gpu_lead = indicators.get("gpu_lead_times") or {}
    specs = data.get("specs") or {}
    if not gpu_lead or not specs:
        return data

    def newest_flagship(vendor=None):
        candidates = [
            gid for gid, spec in specs.items()
            if spec.get("tier") == "flagship"
            and gid in gpu_lead
            and (vendor is None or spec.get("vendor") == vendor)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda g: specs[g].get("release_year") or 0)

    flagship = newest_flagship()
    if flagship:
        weeks = gpu_lead[flagship].get("weeks")
        if weeks is not None:
            series = indicators.setdefault("flagship_lead_time_weeks", {})
            series[datetime.now(timezone.utc).strftime("%Y-%m")] = weeks
            log_ok("Lead Times", f"flagship = {flagship} @ {weeks} wk")

    for vendor_key, vendor in (data.get("supplychain", {}).get("vendors") or {}).items():
        gid = newest_flagship(vendor_key)
        if not gid:
            continue
        weeks = gpu_lead[gid].get("weeks")
        if weeks is None:
            continue
        vendor["lead_time_weeks"] = weeks
        trend = vendor.get("lead_time_trend")
        if isinstance(trend, list) and trend:
            trend[-1] = weeks

    return data


def refresh_summary(data):
    """Rebuild the summary block from the live data it is supposed to summarize.

    Everything here used to be a hand-written snapshot that was never
    recalculated, so the Overview tab reported a March timestamp, GPU/provider
    counts that no longer matched the dataset, and stock prices months out of
    date.
    """
    summary = data.setdefault("summary", {})
    specs = data.get("specs") or {}
    providers = data.get("providers") or {}
    matrix = data.get("matrix") or []
    indicators = data.get("indicators") or {}

    summary["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary["total_gpus_tracked"] = len(specs)
    summary["total_providers_tracked"] = len(providers)

    if matrix:
        summary["comparison_matrix"] = matrix

        best_flops = max(
            (r for r in matrix if r.get("flops_per_dollar")),
            key=lambda r: r["flops_per_dollar"],
            default=None,
        )
        if best_flops:
            summary["best_flops_per_dollar"] = {
                "gpu": best_flops.get("name"),
                "value": round(best_flops["flops_per_dollar"], 1),
                "at_price": best_flops.get("cheapest_price"),
                "provider": best_flops.get("cheapest_provider"),
            }

        best_vram = max(
            (r for r in matrix if r.get("vram_per_dollar")),
            key=lambda r: r["vram_per_dollar"],
            default=None,
        )
        if best_vram:
            summary["best_vram_per_dollar"] = {
                "gpu": best_vram.get("name"),
                "value": round(best_vram["vram_per_dollar"], 1),
                "at_price": best_vram.get("cheapest_price"),
                "provider": best_vram.get("cheapest_provider"),
            }

        biggest_drop = min(
            (r for r in matrix if r.get("monthly_change_pct") is not None),
            key=lambda r: r["monthly_change_pct"],
            default=None,
        )
        if biggest_drop:
            summary["biggest_price_drop"] = {
                "gpu": biggest_drop.get("name"),
                "change_pct": round(biggest_drop["monthly_change_pct"], 1),
            }

        competitive = max(
            (r for r in matrix if r.get("num_providers")),
            key=lambda r: (r["num_providers"], r.get("price_spread_pct") or 0),
            default=None,
        )
        if competitive:
            summary["most_competitive_market"] = {
                "gpu": competitive.get("name"),
                "num_providers": competitive["num_providers"],
                "price_spread_pct": round(competitive.get("price_spread_pct") or 0, 1),
            }

    # Keep the summary's copy of the market indicators in step with the live
    # ones rather than letting it sit at whatever it was seeded with.
    mi = summary.setdefault("market_indicators", {})
    for key, value in indicators.items():
        if isinstance(value, dict):
            mi[key] = value

    if data.get("sentiment"):
        summary["market_sentiment"] = data["sentiment"]

    log_ok("Summary", f"{summary['total_gpus_tracked']} GPUs / {summary['total_providers_tracked']} providers")
    return data


# ===================================================================
# 1c. REAL REGIONAL PRICING, VOLATILITY, MODEL FIT, CHANGE FEED
# ===================================================================

# Azure publishes the same VM SKU in every region through one API, which makes
# it the only provider here that can give a like-for-like regional comparison
# cheaply. The old Regional tab applied invented multipliers (eu = us x 1.10)
# to invented base prices; this measures the real spread instead.
AZURE_REGIONS = [
    ("eastus", "US East", "North America"),
    ("westus2", "US West", "North America"),
    ("northeurope", "Ireland", "Europe"),
    ("westeurope", "Netherlands", "Europe"),
    ("uksouth", "UK South", "Europe"),
    ("japaneast", "Japan East", "Asia Pacific"),
    ("southeastasia", "Singapore", "Asia Pacific"),
    ("australiaeast", "Australia East", "Asia Pacific"),
]

# (skuName prefix, required substring, internal GPU id, GPUs per VM)
_AZURE_REGION_SKUS = [
    ("ND96", "H100", "H100-SXM", 8),
    ("ND96", "A100", "A100-80GB", 8),
    ("ND", "GB200", "GB200", 4),
]


def fetch_regional_pricing():
    """Same SKU priced across regions, straight from the Azure Retail API."""
    log_info("Fetching regional pricing (Azure, %d regions)..." % len(AZURE_REGIONS))
    base = "https://prices.azure.com/api/retail/prices"
    regions = {}

    for arm, label, continent in AZURE_REGIONS:
        prices = {}
        for prefix, must, gpu_id, per_vm in _AZURE_REGION_SKUS:
            filt = (
                f"serviceName eq 'Virtual Machines' and armRegionName eq '{arm}' "
                f"and priceType eq 'Consumption' and startswith(skuName, '{prefix}')"
            )
            try:
                status, body = http_get(
                    base, params={"$filter": filt, "currencyCode": "USD"}, timeout=25
                )
                if status != 200:
                    continue
                found = []
                for it in json.loads(body).get("Items", []):
                    sku = it.get("skuName", "")
                    meter = (it.get("meterName") or "").lower()
                    if "spot" in meter or "low priority" in meter:
                        continue
                    if must and must not in sku:
                        continue
                    if (it.get("unitOfMeasure") or "").lower() not in ("1 hour", "1hour"):
                        continue
                    up = it.get("unitPrice") or it.get("retailPrice")
                    if up and up > 0:
                        found.append(float(up) / per_vm)
                if found:
                    found.sort()
                    prices[gpu_id] = round(found[len(found) // 2], 3)
            except Exception:
                continue
        if prices:
            regions[label] = {
                "arm_region": arm,
                "continent": continent,
                "gpu_pricing": prices,
            }

    if not regions:
        raise RuntimeError("no regional pricing returned")

    # Premium vs the cheapest region, per GPU and overall.
    for gpu_id in {g for r in regions.values() for g in r["gpu_pricing"]}:
        quotes = [(r["gpu_pricing"][gpu_id], name) for name, r in regions.items()
                  if gpu_id in r["gpu_pricing"]]
        cheapest = min(quotes)[0]
        for name, r in regions.items():
            if gpu_id in r["gpu_pricing"]:
                r.setdefault("premium_pct", {})[gpu_id] = round(
                    (r["gpu_pricing"][gpu_id] / cheapest - 1) * 100, 1
                )

    for r in regions.values():
        prem = list((r.get("premium_pct") or {}).values())
        r["avg_premium_pct"] = round(sum(prem) / len(prem), 1) if prem else 0.0

    log_ok("Regional Pricing", f"{len(regions)} regions, {len(_AZURE_REGION_SKUS)} SKUs")
    return regions


def build_regional(data):
    """Replace the modelled regional block with measured cross-region pricing."""
    try:
        regions = fetch_regional_pricing()
    except Exception as exc:
        log_fail("Regional Pricing", str(exc))
        return data

    existing = data.get("regional") or {}
    out = {}
    for name, r in regions.items():
        # Carry across the two curated fields worth keeping: the published
        # industrial electricity rate (feeds TCO) and the hub list.
        prior = next((v for k, v in existing.items() if k.startswith(r["continent"][:6])), {})
        out[name] = {
            **r,
            "energy_cost_kwh": prior.get("energy_cost_kwh"),
            "key_hubs": prior.get("key_hubs", []),
            "source": "azure_retail_api",
            "last_verified": datetime.now(timezone.utc).isoformat(),
        }
    data["regional"] = out
    data.pop("regional_summary", None)
    return data


def compute_volatility(data):
    """Describe what the price history actually shows, instead of forecasting it.

    A 6-month point forecast off ~12 monthly observations is not defensible.
    Realized volatility, drawdown and a trend classification are.
    """
    historical = data.get("historical") or {}
    spot = data.get("spot") or {}
    out = {}

    for gpu_id, series in historical.items():
        if gpu_id not in spot:
            continue
        # Each month is {"avg", "min", "max", "availability"}; track the average.
        def _avg(v):
            if isinstance(v, dict):
                return v.get("avg")
            return v if isinstance(v, (int, float)) else None

        months = sorted(series)
        pts = [(m, _avg(series[m])) for m in months]
        pts = [(m, v) for m, v in pts if isinstance(v, (int, float)) and v > 0]
        if len(pts) < 4:
            continue

        values = [v for _, v in pts]

        # Only take returns between genuinely consecutive months -- the early
        # history is quarterly, and treating a 3-month gap as one step would
        # overstate monthly volatility.
        def _months(label):
            y, m = label.split("-")
            return int(y) * 12 + int(m)

        rets = []
        for i in range(1, len(pts)):
            if _months(pts[i][0]) - _months(pts[i - 1][0]) == 1:
                rets.append(values[i] / values[i - 1] - 1)

        peak = max(values)
        current = values[-1]

        def change_over(n):
            if len(values) <= n:
                return None
            return round((current / values[-1 - n] - 1) * 100, 1)

        monthly_vol = None
        if len(rets) >= 3:
            mean = sum(rets) / len(rets)
            monthly_vol = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5

        # Classify on the compounded 3-month move, not the mean of returns: on a
        # choppy series +67% then -35% then -28% averages to ~0 while the actual
        # 3-month change is -23%.
        trend = change_over(3)
        if trend is None:
            trend = change_over(1) or 0.0
        if trend <= -5:
            regime, note = "falling", "Prices trending down — favour on-demand and short commitments."
        elif trend >= 5:
            regime, note = "tightening", "Prices trending up — supply tightening; consider locking in capacity."
        else:
            regime, note = "stable", "Prices flat within noise."

        out[gpu_id] = {
            "current": round(current, 3),
            "observations": len(values),
            "first_month": pts[0][0],
            "monthly_volatility_pct": round(monthly_vol * 100, 1) if monthly_vol is not None else None,
            "annualized_volatility_pct": round(monthly_vol * (12 ** 0.5) * 100, 1) if monthly_vol is not None else None,
            "monthly_observations": len(rets) + 1,
            "peak": round(peak, 3),
            "drawdown_from_peak_pct": round((current / peak - 1) * 100, 1),
            "change_3mo_pct": change_over(3),
            "change_6mo_pct": change_over(6),
            "change_12mo_pct": change_over(12),
            "regime": regime,
            "note": note,
        }

    data["volatility"] = out
    data.pop("forecasts", None)
    log_ok("Volatility", f"{len(out)} GPUs described from price history")
    return data


# Published transformer architectures. Every number here is from the model's own
# config, which is what makes the VRAM figures arithmetic rather than estimates.
MODEL_ARCHITECTURES = {
    "Llama-3.1-8B":    {"params_b": 8.03,  "layers": 32,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Llama-3.1-70B":   {"params_b": 70.6,  "layers": 80,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Llama-3.1-405B":  {"params_b": 405.9, "layers": 126, "kv_heads": 8, "head_dim": 128, "open": True},
    "Qwen2.5-72B":     {"params_b": 72.7,  "layers": 80,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Qwen2.5-32B":     {"params_b": 32.8,  "layers": 64,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Qwen2.5-7B":      {"params_b": 7.6,   "layers": 28,  "kv_heads": 4, "head_dim": 128, "open": True},
    "Mistral-7B":      {"params_b": 7.25,  "layers": 32,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Mixtral-8x7B":    {"params_b": 46.7,  "layers": 32,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Mixtral-8x22B":   {"params_b": 141.0, "layers": 56,  "kv_heads": 8, "head_dim": 128, "open": True},
    "Gemma-2-27B":     {"params_b": 27.2,  "layers": 46,  "kv_heads": 16, "head_dim": 128, "open": True},
}

PRECISIONS = {"fp16": 2, "fp8": 1, "int4": 0.5}
GIB = 1024 ** 3


def build_modelfit(data):
    """Which tracked GPUs can actually hold a given model, and what that costs.

    Replaces the old table of invented throughput/batch numbers. Everything here
    is computed:
        weights   = params x bytes_per_param
        kv_cache  = 2 x layers x kv_heads x head_dim x ctx x batch x bytes
        overhead  = 20% of weights (activations, fragmentation, CUDA context)
    """
    specs = data.get("specs") or {}
    providers = data.get("providers") or {}
    contexts = [8192, 32768, 131072]
    out = {}

    for model, arch in MODEL_ARCHITECTURES.items():
        entry = {
            "params_b": arch["params_b"],
            "layers": arch["layers"],
            "kv_heads": arch["kv_heads"],
            "head_dim": arch["head_dim"],
            "open_source": arch["open"],
            "precisions": {},
        }
        for prec, nbytes in PRECISIONS.items():
            weights_gib = arch["params_b"] * 1e9 * nbytes / GIB
            overhead_gib = weights_gib * 0.20
            per_ctx = {}
            for ctx in contexts:
                # KV cache is held at fp16 even when weights are quantised.
                kv_gib = (2 * arch["layers"] * arch["kv_heads"] * arch["head_dim"]
                          * ctx * 2) / GIB
                total = weights_gib + overhead_gib + kv_gib
                fits = []
                for gpu_id, spec in specs.items():
                    vram = spec.get("vram_gb")
                    if not vram:
                        continue
                    n_gpus = max(1, -(-int(total) // int(vram)) if total > vram else 1)
                    if total > vram * 8:
                        continue
                    n_gpus = 1 if total <= vram else int(-(-total // vram))
                    cheapest = _cheapest_provider_for(providers, gpu_id)
                    if not cheapest:
                        continue
                    price, prov = cheapest
                    fits.append({
                        "gpu": gpu_id,
                        "gpus_needed": n_gpus,
                        "vram_total_gb": round(vram * n_gpus, 1),
                        "headroom_pct": round((vram * n_gpus - total) / (vram * n_gpus) * 100, 1),
                        "cheapest_provider": prov,
                        "usd_per_hr": round(price * n_gpus, 3),
                    })
                fits.sort(key=lambda f: (f["usd_per_hr"], f["gpus_needed"]))
                per_ctx[str(ctx)] = {
                    "kv_cache_gib": round(kv_gib, 2),
                    "total_vram_gib": round(total, 2),
                    "options": fits[:6],
                    "cheapest_usd_per_hr": fits[0]["usd_per_hr"] if fits else None,
                }
            entry["precisions"][prec] = {
                "weights_gib": round(weights_gib, 2),
                "overhead_gib": round(overhead_gib, 2),
                "contexts": per_ctx,
            }
        out[model] = entry

    data["modelfit"] = out
    log_ok("Model Fit", f"{len(out)} models x {len(PRECISIONS)} precisions computed")
    return data


def build_changelog(data, previous):
    """Record what moved since the last run so the console has a change feed."""
    if not previous:
        return data

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = []

    old_p, new_p = previous.get("providers") or {}, data.get("providers") or {}
    for prov, pdata in new_p.items():
        old_gpus = (old_p.get(prov) or {}).get("gpus") or {}
        new_gpus = pdata.get("gpus") or {}
        for gpu_id, info in new_gpus.items():
            new_price = info.get("price_per_gpu_hr")
            old_price = (old_gpus.get(gpu_id) or {}).get("price_per_gpu_hr")
            if old_price and new_price:
                delta = (new_price / old_price - 1) * 100
                if abs(delta) >= 3:
                    ev = {
                        "date": stamp, "type": "price", "provider": prov, "gpu": gpu_id,
                        "from": old_price, "to": new_price, "change_pct": round(delta, 1),
                    }
                    # A >100% week-on-week move in list pricing is almost always a
                    # scraper artefact, not the market. Flag rather than silently
                    # publish it as a price signal.
                    if abs(delta) > 100:
                        ev["needs_review"] = True
                    events.append(ev)
            elif not old_price:
                events.append({"date": stamp, "type": "listed", "provider": prov,
                               "gpu": gpu_id, "to": new_price})
        for gpu_id in old_gpus:
            if gpu_id not in new_gpus:
                events.append({"date": stamp, "type": "delisted", "provider": prov,
                               "gpu": gpu_id, "from": old_gpus[gpu_id].get("price_per_gpu_hr")})

    events.sort(key=lambda e: -abs(e.get("change_pct") or 0))
    log = [e for e in (data.get("changelog") or []) if e.get("date") != stamp]
    data["changelog"] = (events + log)[:200]
    log_ok("Changelog", f"{len(events)} changes recorded for {stamp}")
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

    # Extract 52-week range from indicators
    indicators = r.get("indicators", {})
    quotes = indicators.get("quote", [{}])[0]
    highs = [h for h in (quotes.get("high") or []) if h is not None]
    lows = [l for l in (quotes.get("low") or []) if l is not None]

    week52_high = meta.get("fiftyTwoWeekHigh") or (max(highs) if highs else None)
    week52_low = meta.get("fiftyTwoWeekLow") or (min(lows) if lows else None)

    # Calculate YTD change (approximate from first trading day data)
    timestamps = r.get("timestamp", [])
    closes = quotes.get("close") or []

    # Prior trading day's close. NOTE: meta.chartPreviousClose is the close
    # immediately *before the requested range* (i.e. ~1 year ago here), not the
    # prior session -- using it made previous_close wildly wrong. meta.previousClose
    # is absent on range=1y responses, so derive it from the daily close series.
    prior_closes = [c for c in closes if c is not None]
    prev_close = prior_closes[-2] if len(prior_closes) >= 2 else meta.get("previousClose")

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
        "current_price": round(current_price, 2) if current_price else None,
        "previous_close": round(prev_close, 2) if prev_close else None,
        "52_week_high": round(week52_high, 2) if week52_high else None,
        "52_week_low": round(week52_low, 2) if week52_low else None,
        "ytd_change_pct": ytd_change,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    log_ok(f"Stock/{ticker}", f"${current_price:.2f}" if current_price else "price unavailable")
    return stock_data


def merge_stocks_into_data(data, stocks):
    """Merge stock data into the indicators section, matching existing structure.

    The stock_data dict uses keys '52_week_high', '52_week_low', 'ytd_change_pct';
    we populate all dashboard-consumed fields (current, ytd_change, ytd_pct, 52w_*).
    """
    if not stocks:
        return data
    data["stocks"] = {s["ticker"]: s for s in stocks}
    indicators = data.get("indicators", {})

    def _apply(entry, s):
        if s.get("current_price") is not None:
            entry["current"] = s["current_price"]
        if s.get("ytd_change_pct") is not None:
            entry["ytd_change"] = s["ytd_change_pct"]
            entry["ytd_pct"] = s["ytd_change_pct"]
        hi, lo = s.get("52_week_high"), s.get("52_week_low")
        if hi is not None:
            entry["52w_high"] = hi
        if lo is not None:
            entry["52w_low"] = lo
        if hi is not None and lo is not None:
            entry["52w_range"] = f"${lo:.2f}-${hi:.2f}"
        entry["last_updated"] = s.get("last_updated")

    for s in stocks:
        tk = s["ticker"]
        if tk == "NVDA":
            entry = indicators.setdefault("nvidia_stock", {"ticker": "NVDA"})
            _apply(entry, s)
            indicators["nvda_price"] = s.get("current_price")
            indicators["nvda_ytd_change"] = s.get("ytd_change_pct")
        elif tk == "AMD":
            entry = indicators.setdefault("amd_stock", {"ticker": "AMD"})
            _apply(entry, s)
            indicators["amd_price"] = s.get("current_price")
            indicators["amd_ytd_change"] = s.get("ytd_change_pct")

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


def _get_reddit_token():
    """Get an OAuth access token via client_credentials. Returns None if no creds.

    Reads REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET from environment (or .env)
    or from config.py. Register an app at https://www.reddit.com/prefs/apps
    (type: "script") to get these.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not (client_id and client_secret):
        try:
            cfg_globals = {}
            with open(CONFIG_PY) as f:
                exec(f.read(), cfg_globals)
            client_id = client_id or cfg_globals.get("REDDIT_CLIENT_ID")
            client_secret = client_secret or cfg_globals.get("REDDIT_CLIENT_SECRET")
        except Exception:
            pass
    if not (client_id and client_secret):
        return None

    if not _HAS_REQUESTS:
        return None  # urllib basic-auth path not implemented; requests required for OAuth
    try:
        resp = _req.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": "GPUDashboard/1.0 (market research)"},
            timeout=15,
        )
        if resp.status_code != 200:
            log_info(f"Reddit OAuth returned HTTP {resp.status_code}: {resp.text[:120]}")
            return None
        return resp.json().get("access_token")
    except Exception as exc:
        log_info(f"Reddit OAuth error: {exc}")
        return None


def fetch_reddit_sentiment():
    """Fetch GPU mention counts and sentiment from Reddit OAuth API.

    Reddit blocks unauthenticated traffic site-wide as of 2024+, so OAuth is
    required. Without credentials, this returns {} (no data) and the merge
    step preserves existing values.
    """
    log_info("Fetching Reddit sentiment...")
    token = _get_reddit_token()
    if not token:
        log_info("No Reddit OAuth token (set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET). Skipping.")
        return {}

    results = {}
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "GPUDashboard/1.0 (market research)",
    }
    base = "https://oauth.reddit.com"

    for gpu_id, terms in _GPU_SEARCH_TERMS.items():
        total_mentions = 0
        total_score = 0
        total_upvote_ratio = 0.0
        post_count = 0
        hot_topics = []

        for term in terms:
            for sub in _REDDIT_SUBREDDITS:
                try:
                    url = f"{base}/r/{sub}/search"
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
                    time.sleep(1.1)  # OAuth limit: 60 req/min = 1 req per 1.1s
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


def _build_market_snapshot(data):
    """Build a compact market snapshot dict for LLM prompts."""
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

    for gpu_id in price_summary:
        price_summary[gpu_id].sort(key=lambda x: x["price"])

    return {
        "gpu_pricing": {k: v[:5] for k, v in price_summary.items()},
        "stocks": data.get("stocks", {}),
        "recent_headlines": [a.get("headline", "") for a in data.get("news", [])[:10]],
        "historical": {k: dict(list(v.items())[-6:]) for k, v in data.get("historical", {}).items()},
        "sentiment": {k: {"score": v.get("score"), "adoption": v.get("adoption"), "mentions_30d": v.get("mentions_30d")} for k, v in data.get("sentiment", {}).items()},
        "forecasts": {k: {"trend": v.get("trend"), "next_month": v.get("next_month")} for k, v in data.get("forecasts", {}).items()} if data.get("forecasts") else {},
        "matrix": [{"gpu_id": m["gpu_id"], "cheapest_price": m.get("cheapest_price"), "cheapest_provider": m.get("cheapest_provider"), "monthly_change_pct": m.get("monthly_change_pct"), "flops_per_dollar": m.get("flops_per_dollar")} for m in data.get("matrix", [])[:15]],
        "spot": {k: {"on_demand_avg": v.get("on_demand_avg"), "quarterly_trend": v.get("quarterly_trend")} for k, v in data.get("spot", {}).items()},
        "specs": {k: {"vram_gb": v.get("vram_gb"), "fp16_tflops": v.get("fp16_tflops"), "tdp_watts": v.get("tdp_watts")} for k, v in data.get("specs", {}).items()},
        # Traded forward prices from Kalshi/Polymarket. The forecast section is
        # told to anchor on these rather than invent a target range.
        "prediction_markets": {
            g.get("label", g.get("gpu")): {
                "market_implied_curve": [
                    {"month": p.get("horizon_label"), "implied_usd_per_gpu_hr": p.get("implied_price")}
                    for p in (g.get("curve") or [])
                ],
                "cheapest_listed_now": g.get("listed_cheapest"),
            }
            for g in ((data.get("prediction_markets") or {}).get("gpus") or [])
        },
        "date": datetime.now().strftime("%Y-%m-%d"),
    }


def _call_llm(config, system_prompt, user_prompt, max_tokens=3000, max_attempts=3):
    """Call the LLM API and return the response content string.

    The configured model is a reasoning model, so its output budget is shared
    between the hidden reasoning trace and the answer. A section whose reasoning
    ran long used to come back with `content: null` and `finish_reason: length`,
    which surfaced as a hard "LLM returned empty content" failure and left that
    section frozen at whatever it last said. Retry with a bigger budget instead.
    """
    url = f"{config['base']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['key']}",
    }

    budget = max_tokens
    for attempt in range(1, max_attempts + 1):
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": budget,
            "temperature": 0.3,
        }
        status, body = http_post(url, payload, headers=headers, timeout=180)
        if status != 200:
            raise RuntimeError(f"LLM API returned HTTP {status}: {body[:200]}")

        choice = (json.loads(body).get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if content.strip():
            return content

        if choice.get("finish_reason") != "length" or attempt == max_attempts:
            raise RuntimeError(
                f"LLM returned empty content (finish_reason={choice.get('finish_reason')}, "
                f"max_tokens={budget})"
            )

        budget *= 2
        log_info(f"LLM spent its budget on reasoning; retrying with max_tokens={budget}")

    raise RuntimeError("LLM returned empty content")


# Section definitions: key -> (type, system_prompt, user_prompt_template)
# The user_prompt_template receives the snapshot JSON string as {snapshot}.
_AI_SECTIONS = {
    "summary": {
        "type": "quick_summary",
        "system": (
            "You are a GPU compute market analyst. Provide a concise market brief "
            "(300-500 words) covering: current GPU pricing trends, key stock movements, "
            "notable news, and actionable insights for organizations planning GPU procurement. "
            "Use markdown formatting. Be specific with numbers and prices."
        ),
        "user": "Generate a market brief based on this data:\n\n{snapshot}",
    },
    "trends": {
        "type": "market_trends",
        "system": (
            "You are a GPU compute market analyst specializing in pricing trends. "
            "Analyze GPU pricing trends using the historical data, monthly changes, and spot market data. "
            "Include a table of price changes, identify key dynamics (supply loosening, new GPU adoption, "
            "AMD competition, regional premiums), and list what to watch. Use markdown. Be specific with numbers."
        ),
        "user": "Analyze GPU pricing trends from this market data:\n\n{snapshot}",
    },
    "regional": {
        "type": "regional_analysis",
        "system": (
            "You are a GPU compute market analyst specializing in regional pricing. "
            "Provide a regional GPU pricing guide covering price premiums by region, "
            "fastest growing markets, and regional recommendations for cost-sensitive, "
            "compliance-required, and global inference workloads. Use markdown tables."
        ),
        "user": "Generate a regional pricing analysis from this data:\n\n{snapshot}",
    },
    "investment": {
        "type": "investment_outlook",
        "system": (
            "You are a GPU procurement advisor. Provide a GPU procurement guide covering: "
            "when to commit to reserved instances vs spot/on-demand, best deals right now with "
            "specific providers and prices, migration timing recommendations (e.g. A100->H100, H100->B200), "
            "and a provider comparison table. Use markdown. Be specific with prices and provider names."
        ),
        "user": "Generate a GPU procurement guide from this data:\n\n{snapshot}",
    },
    "notes": {
        "type": "market_notes",
        "system": (
            "You are a GPU market analyst writing a weekly market snapshot. Cover: "
            "prices moving this week with specific numbers and MoM changes, what changed in the market, "
            "quick buy/wait/sell recommendations, and 90-day price targets in a table. "
            "Use markdown. Keep it concise and actionable."
        ),
        "user": "Generate a market snapshot from this data:\n\n{snapshot}",
    },
    "efficiency": {
        "type": "efficiency_optimization",
        "system": (
            "You are a GPU compute optimization expert. Provide an optimization checklist covering: "
            "right-sizing guide (common mistakes and workload-to-GPU matching table), "
            "reducing idle time strategies, provider efficiency comparison, and quick wins "
            "with specific dollar savings. Use markdown tables. Be specific with prices."
        ),
        "user": "Generate GPU efficiency optimization advice from this data:\n\n{snapshot}",
    },
    "forecast": {
        "type": "price_forecasts",
        "system": (
            "You are a GPU market forecaster. Provide a price outlook covering: "
            "90-day forecast table (GPU, current price, target range, change %, confidence), "
            "12-month outlook table, factors driving prices up and down, "
            "and when to act (buy now / wait). Use markdown tables. Be specific with numbers. "
            "Where the data includes a prediction_markets block, those are traded forward prices "
            "from Kalshi and Polymarket: anchor your targets to that curve, cite it as the source, "
            "and say explicitly when your view departs from it and why. Do not present a number as "
            "a forecast when the market already quotes one. Note that the market settles on the "
            "Ornn neocloud index, a different basket from the provider list prices, so compare "
            "directions of travel rather than levels."
        ),
        "user": "Generate GPU price forecasts from this data:\n\n{snapshot}",
    },
    "sustainability": {
        "type": "sustainability_risk",
        "system": (
            "You are a GPU supply chain and sustainability analyst. Cover: "
            "GPU availability status table (lead times, status, trend), key supply chain risks "
            "(TSMC, HBM memory, CoWoS packaging), regulatory risks (export controls, EU AI Act), "
            "geopolitical risks, and green compute provider comparison. Use markdown tables."
        ),
        "user": "Generate a supply chain and sustainability analysis from this data:\n\n{snapshot}",
    },
}


def _get_gpu_section_key(existing_ai):
    """Find gpu_* section keys in existing AI analysis."""
    return [k for k in existing_ai if k.startswith("gpu_")]


def _build_gpu_section_prompt(gpu_id, snapshot_str):
    """Build system/user prompts for a per-GPU deep dive."""
    system = (
        f"You are a GPU market analyst. Provide a detailed market report for the {gpu_id} GPU. "
        "Include: specs at a glance table, price trend analysis, best providers table with prices, "
        "who should use it, and a buy/wait/sell recommendation. Use markdown. Be specific with numbers."
    )
    user = f"Generate a detailed market report for {gpu_id} based on this data:\n\n{snapshot_str}"
    return system, user


def update_ai_analysis(data, existing_ai):
    """Update ai_analysis.json, regenerating ALL sections via LLM."""
    now = datetime.now(timezone.utc).isoformat()
    config = load_config()

    if not config:
        log_info("No LLM config found -- keeping existing ai_analysis.json unchanged")
        return existing_ai

    snapshot = _build_market_snapshot(data)
    snapshot_str = json.dumps(snapshot, indent=2, default=str)

    # Regenerate each standard section
    for section_key, section_def in _AI_SECTIONS.items():
        try:
            log_info(f"Regenerating AI section: {section_key}...")
            user_prompt = section_def["user"].format(snapshot=snapshot_str)
            content = _call_llm(config, section_def["system"], user_prompt)
            existing_ai[section_key] = {
                "analysis": content,
                "type": section_def["type"],
                "timestamp": now,
            }
            log_ok(f"AI/{section_key}", f"{len(content)} chars")
        except Exception as exc:
            log_fail(f"AI/{section_key}", str(exc))

    # Regenerate per-GPU sections (e.g. gpu_H100-SXM)
    gpu_keys = _get_gpu_section_key(existing_ai)
    for gk in gpu_keys:
        gpu_id = gk.replace("gpu_", "", 1)
        try:
            log_info(f"Regenerating AI section: {gk}...")
            sys_prompt, usr_prompt = _build_gpu_section_prompt(gpu_id, snapshot_str)
            content = _call_llm(config, sys_prompt, usr_prompt)
            existing_ai[gk] = {
                "analysis": content,
                "gpu_id": gpu_id,
                "timestamp": now,
            }
            log_ok(f"AI/{gk}", f"{len(content)} chars")
        except Exception as exc:
            log_fail(f"AI/{gk}", str(exc))

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

    _snapshot_before = json.loads(json.dumps(data)) if data else None

    existing_ai = load_existing(AI_ANALYSIS_JSON, EMBEDDED_AI_JSON)
    if not existing_ai:
        existing_ai = {}

    # Track update timestamp
    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # ---- 1. GPU Cloud Pricing ----
    print("[1/11] GPU Cloud Pricing")
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

    azure_prices = None
    try:
        azure_prices = fetch_azure_pricing()
    except Exception as exc:
        log_fail("Azure", str(exc))

    lambda_prices = None
    try:
        lambda_prices = fetch_lambda_pricing()
    except Exception as exc:
        log_fail("Lambda", str(exc))

    coreweave_prices = None
    try:
        coreweave_prices = fetch_coreweave_pricing()
    except Exception as exc:
        log_fail("CoreWeave", str(exc))

    together_prices = None
    try:
        together_prices = fetch_together_pricing()
    except Exception as exc:
        log_fail("Together", str(exc))

    aws_prices = None
    try:
        aws_prices = fetch_aws_pricing()
    except Exception as exc:
        log_fail("AWS", str(exc))

    gcp_prices = None
    try:
        gcp_prices = fetch_gcp_pricing()
    except Exception as exc:
        log_fail("GCP", str(exc))

    data = merge_live_pricing_into_data(
        data,
        vastai_prices,
        runpod_prices,
        azure_prices=azure_prices,
        lambda_prices=lambda_prices,
        coreweave_prices=coreweave_prices,
        together_prices=together_prices,
        aws_prices=aws_prices,
        gcp_prices=gcp_prices,
    )
    print()

    # ---- 2. Recalculate derived data (matrix, historical, spot) ----
    print("[2/11] Recalculate Historical")
    print("-" * 40)
    try:
        data = update_historical(data)
    except Exception as exc:
        log_fail("Historical", str(exc))
    print()

    print("[3/11] Recalculate Spot")
    print("-" * 40)
    try:
        data = update_spot(data)
    except Exception as exc:
        log_fail("Spot", str(exc))
    print()

    print("[4/11] Recalculate Matrix")
    print("-" * 40)
    try:
        data = recalculate_matrix(data)
    except Exception as exc:
        log_fail("Matrix", str(exc))

    try:
        data = refresh_workload_recs(data)
    except Exception as exc:
        log_fail("Workload Recs", str(exc))

    try:
        data = refresh_tco(data)
    except Exception as exc:
        log_fail("TCO", str(exc))

    try:
        data = refresh_lead_times(data)
    except Exception as exc:
        log_fail("Lead Times", str(exc))

    try:
        data = refresh_inference_market(data, log_info=log_info, log_ok=log_ok)
    except Exception as exc:
        log_fail("Inference Market", str(exc))
    print()

    # ---- 5. Stock Prices ----
    print("[5/11] Stock Prices")
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
    print("[6/11] News Headlines")
    print("-" * 40)

    try:
        articles = fetch_news_rss()
        data = merge_news_into_data(data, articles)
    except Exception as exc:
        log_fail("Google News RSS", str(exc))
    print()

    # ---- 7. Community Sentiment (Reddit + HuggingFace + GitHub) ----
    print("[7/11] Reddit Sentiment")
    print("-" * 40)

    reddit_data = {}
    try:
        reddit_data = fetch_reddit_sentiment()
    except Exception as exc:
        log_fail("Reddit Sentiment", str(exc))
    print()

    print("[8/11] HuggingFace + GitHub")
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

    # ---- 9. Derived analytics: regional, volatility, model fit ----
    print("[9/11] Regional / Volatility / Model Fit")
    print("-" * 40)

    try:
        data = build_regional(data)
    except Exception as exc:
        log_fail("Regional", str(exc))

    try:
        data = compute_volatility(data)
    except Exception as exc:
        log_fail("Volatility", str(exc))

    try:
        data = build_modelfit(data)
    except Exception as exc:
        log_fail("Model Fit", str(exc))

    # Retired: per-provider utilization was unsourceable, and a 6-month point
    # forecast off ~12 monthly observations was not defensible. compute_volatility
    # above describes what the history actually supports.
    data.pop("utilization", None)
    print()

    # ---- 10. Prediction markets (Kalshi + Polymarket) ----
    print("[10/11] Prediction Markets")
    print("-" * 40)

    kalshi_curves, poly_markets = [], []
    try:
        log_info("Fetching Kalshi GPU compute ladders...")
        kalshi_curves = fetch_kalshi_gpu_markets()
        log_ok("Kalshi", f"{len(kalshi_curves)} GPU curves")
    except Exception as exc:
        log_fail("Kalshi", str(exc))

    try:
        log_info("Fetching Polymarket GPU rental brackets...")
        poly_markets = fetch_polymarket_gpu_markets()
        log_ok("Polymarket", f"{len(poly_markets)} open markets")
    except Exception as exc:
        log_fail("Polymarket", str(exc))

    if kalshi_curves or poly_markets:
        try:
            data = build_prediction_markets(data, kalshi_curves, poly_markets)
        except Exception as exc:
            log_fail("Prediction Markets", str(exc))
    else:
        # Never serve a stale order book as if it were live.
        data.pop("prediction_markets", None)
    print()

    # ---- 11. AI Analysis ----
    print("[11/11] AI Analysis")
    print("-" * 40)

    existing_ai = update_ai_analysis(data, existing_ai)
    print()

    # ---- Rebuild the summary block last, from everything above ----
    try:
        data = refresh_summary(data)
    except Exception as exc:
        log_fail("Summary", str(exc))
    print()

    try:
        data = build_changelog(data, _snapshot_before)
    except Exception as exc:
        log_fail("Changelog", str(exc))

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
