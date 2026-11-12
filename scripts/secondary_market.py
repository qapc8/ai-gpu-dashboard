"""Secondary-market (purchase) prices for GPU hardware -> data["secondary_market"].

Two kinds of numbers, never mixed up:

  live      eBay Browse API (official, https://developer.ebay.com/api-docs/buy/browse/)
            fixed-price listings, aggressively filtered, reduced to a median.
            Needs EBAY_CLIENT_ID / EBAY_CLIENT_SECRET (client-credentials OAuth).
            Without them the source is skipped and reported as such. eBay's public
            search pages return HTTP 403 to server-side clients, so they are not
            scraped.

  curated   scripts/curated/hardware_prices.json: dated, cited reference prices
            (vendor list prices, figures quoted in published reports). Every item
            built from it carries curated=true and the report's own as_of date.

Filtering method for eBay: per-GPU title must-match / must-not-match regexes
(VRAM and form factor are checked so a "40GB" card cannot land in an 80GB
bucket), an accessory/parts/rental blacklist, a per-GPU price floor that rejects
fans/brackets/boxes, then a 0.5x-2x band around the first-pass median; a bucket
is published only with >= MIN_LISTINGS surviving listings.

Asking prices are not transaction prices. rent_vs_buy is purchase-only:
power, hosting, networking and staff costs are excluded on purpose (the
dashboard's TCO section models those with its own assumptions).

Stdlib only.
"""

import base64
import json
import os
import re
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

TIMEOUT = 30
USER_AGENT = "ai-gpu-dashboard/1.0 (market research)"
CURATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "curated", "hardware_prices.json")

SRC_EBAY = "eBay Browse API"
SRC_CURATED = "Curated hardware prices"
SOURCES = [SRC_EBAY, SRC_CURATED]

EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_SCOPE = "https://api.ebay.com/oauth/api_scope"
EBAY_CAT_GPU = "27386"      # Computers/Tablets & Networking > Graphics/Video Cards
EBAY_CAT_SERVER = "11211"   # Computers/Tablets & Networking > Servers, Clients & Terminals > Servers

MIN_LISTINGS = 5
HISTORY_DAYS = 400
HOURS_PER_MONTH = 730.0
UTIL = 0.70

# condition buckets -> eBay conditionIds
CONDITION_IDS = {
    "new": "1000",
    # used + refurbished grades (certified, excellent, very good, good, seller)
    "used": "2000|2010|2020|2030|2500|3000",
}

_EXCLUDE = re.compile(
    r"\b(fan|fans|bracket|heat ?sink|cooler|cooling|shroud|backplate|water ?block|"
    r"cable|riser|adapter|box only|empty box|for parts|parts only|not working|"
    r"broken|defective|as[- ]is|untested|repair|rent|rental|renting|per hour|/hr|"
    r"cloud|hosting|lease|access|instance|dummy|replica|mock|sticker|manual|"
    r"laptop|mobile|notebook|bios chip|memory chip|pcb only|board only|"
    r"water ?cooled kit|mining rig frame|wanted|wtb)\b",
    re.I,
)

# (gpu_id, form) -> (search query, category, must-match regexes, must-not regexes, price floor USD)
BUCKETS = [
    ("H100-PCIe", "card", "nvidia h100 pcie 80gb", EBAY_CAT_GPU,
     [r"\bh100\b", r"pcie", r"\b80 ?gb\b"], [r"\bsxm\b", r"\bnvl\b", r"\bh100 ?nvl\b", r"\bh800\b"], 8000),
    ("H100-SXM", "card", "nvidia h100 sxm5 80gb", EBAY_CAT_GPU,
     [r"\bh100\b", r"\bsxm"], [r"\bpcie\b", r"\bh800\b", r"\bhgx\b", r"\b8x\b", r"\bserver\b"], 8000),
    ("H200", "card", "nvidia h200 141gb", EBAY_CAT_GPU,
     [r"\bh200\b"], [r"\bh20\b(?!0)", r"\bhgx\b", r"\b8x\b", r"\bserver\b"], 10000),
    ("A100-80GB", "card", "nvidia a100 80gb", EBAY_CAT_GPU,
     [r"\ba100\b", r"\b80 ?gb\b"], [r"\b40 ?gb\b", r"\ba800\b", r"\bhgx\b", r"\b[48]x\b", r"\bserver\b"], 3000),
    ("A100-40GB", "card", "nvidia a100 40gb", EBAY_CAT_GPU,
     [r"\ba100\b", r"\b40 ?gb\b"], [r"\b80 ?gb\b", r"\ba800\b", r"\bhgx\b", r"\b[48]x\b", r"\bserver\b"], 1500),
    ("RTX-4090", "card", "nvidia geforce rtx 4090 24gb", EBAY_CAT_GPU,
     [r"\b4090\b"], [r"\b4090 ?d\b", r"\b48 ?gb\b", r"\b4080\b", r"\b4070\b"], 900),
    ("RTX-5090", "card", "nvidia geforce rtx 5090 32gb", EBAY_CAT_GPU,
     [r"\b5090\b"], [r"\b5090 ?d\b", r"\b5080\b", r"\b5070\b"], 1200),
    ("L40S", "card", "nvidia l40s 48gb", EBAY_CAT_GPU,
     [r"\bl40s\b"], [r"\bl40\b(?!s)"], 3000),
    ("MI300X", "card", "amd instinct mi300x", EBAY_CAT_GPU,
     [r"\bmi300x\b"], [r"\bmi300a\b"], 3000),
    ("H100-SXM", "server_8x", "hgx h100 8x 80gb server", EBAY_CAT_SERVER,
     [r"\bh100\b", r"(\b8x\b|\b8 ?x ?h100|8-gpu|8 gpu|\bhgx\b)"], [r"\bpcie\b", r"\b[1-4]x\b"], 80000),
    ("H200", "server_8x", "hgx h200 8x server", EBAY_CAT_SERVER,
     [r"\bh200\b", r"(\b8x\b|8-gpu|8 gpu|\bhgx\b)"], [r"\b[1-4]x\b"], 120000),
    ("B200", "server_8x", "hgx b200 8x server", EBAY_CAT_SERVER,
     [r"\bb200\b", r"(\b8x\b|8-gpu|8 gpu|\bhgx\b|\bdgx\b)"], [r"\b[1-4]x\b"], 200000),
]


def _now():
    return datetime.now(timezone.utc)


def _round(x, nd=0):
    if x is None:
        return None
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


# --------------------------------------------------------------------------
# eBay Browse API
# --------------------------------------------------------------------------

def _ebay_token(client_id, client_secret):
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({"grant_type": "client_credentials",
                                   "scope": EBAY_SCOPE}).encode()
    req = urllib.request.Request(EBAY_TOKEN_URL, data=body, method="POST", headers={
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        tok = json.loads(resp.read().decode("utf-8")).get("access_token")
    if not tok:
        raise RuntimeError("eBay OAuth returned no access_token")
    return tok


def _ebay_search(token, query, category, condition_ids, limit=200):
    params = {
        "q": query,
        "category_ids": category,
        "limit": str(limit),
        "filter": ",".join([
            "buyingOptions:{FIXED_PRICE}",
            f"conditionIds:{{{condition_ids}}}",
            "priceCurrency:USD",
            "itemLocationCountry:US",
        ]),
    }
    url = EBAY_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8")).get("itemSummaries") or []


def _filter_prices(items, must, must_not, floor):
    prices = []
    for it in items:
        title = (it.get("title") or "").lower()
        try:
            price = float((it.get("price") or {}).get("value"))
        except (TypeError, ValueError):
            continue
        if (it.get("price") or {}).get("currency", "USD") != "USD":
            continue
        if price < floor or _EXCLUDE.search(title):
            continue
        if not all(re.search(p, title) for p in must):
            continue
        if any(re.search(p, title) for p in must_not):
            continue
        # multi-card lots inflate a per-card bucket
        if re.search(r"\b(lot of|lot|qty|quantity|pcs|pieces)\b", title):
            continue
        prices.append(price)
    if not prices:
        return []
    med = statistics.median(prices)
    return sorted(p for p in prices if 0.5 * med <= p <= 2.0 * med)


def fetch_ebay(client_id, client_secret, log_info):
    """Return (items, n_buckets_tried). Raises if auth or every search fails."""
    token = _ebay_token(client_id, client_secret)
    today = _now().date().isoformat()
    out, errors = [], 0
    for gpu_id, form, query, cat, must, must_not, floor in BUCKETS:
        for cond, cond_ids in CONDITION_IDS.items():
            try:
                raw = _ebay_search(token, query, cat, cond_ids)
            except Exception as e:  # one bucket failing must not kill the rest
                errors += 1
                log_info(f"eBay search failed for {gpu_id}/{form}/{cond}: {e}")
                continue
            prices = _filter_prices(raw, must, must_not, floor)
            if len(prices) < MIN_LISTINGS:
                continue
            out.append({
                "gpu_id": gpu_id, "form": form, "condition": cond,
                "median_usd": _round(statistics.median(prices)),
                "low_usd": _round(prices[0]), "high_usd": _round(prices[-1]),
                "n_listings": len(prices),
                "source": SRC_EBAY,
                "url": "https://www.ebay.com/sch/i.html?" + urllib.parse.urlencode({"_nkw": query}),
                "as_of": today, "curated": False,
                "price_basis": "median of filtered fixed-price US asking prices",
            })
    if errors == len(BUCKETS) * len(CONDITION_IDS):
        raise RuntimeError("every eBay search request failed")
    return out


# --------------------------------------------------------------------------
# curated
# --------------------------------------------------------------------------

def load_curated(path=CURATED_PATH):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f).get("entries") or []
    out = []
    for e in entries:
        for k in ("gpu_id", "form", "condition", "source", "url", "as_of"):
            if not e.get(k):
                raise ValueError(f"curated entry missing {k}: {e}")
        lo, hi, val = e.get("low_usd"), e.get("high_usd"), e.get("value_usd")
        if val is not None:
            med, basis = val, e.get("basis") or "single published figure"
        elif lo is not None and hi is not None:
            med = (lo + hi) / 2.0
            basis = f"midpoint of published range ${lo:,.0f}-${hi:,.0f}; {e.get('basis', '')}".rstrip("; ")
        else:
            raise ValueError(f"curated entry has no price: {e}")
        out.append({
            "gpu_id": e["gpu_id"], "form": e["form"], "condition": e["condition"],
            "median_usd": _round(med), "low_usd": _round(lo), "high_usd": _round(hi),
            "n_listings": None, "source": e["source"], "url": e["url"],
            "as_of": e["as_of"], "curated": True, "price_basis": basis,
        })
    if not out:
        raise ValueError("curated file has no entries")
    return out


# --------------------------------------------------------------------------
# derived
# --------------------------------------------------------------------------

def _matrix_medians(data):
    out = {}
    for row in data.get("matrix") or []:
        gid, med = row.get("gpu_id"), row.get("median_price")
        if gid and isinstance(med, (int, float)) and med > 0:
            out[gid] = (float(med), row.get("num_providers"))
    return out


# curated ids that don't name a variant -> which rental row to compare against
_RENT_ALIAS = {"H100": "H100-SXM"}


def _rent_vs_buy(items, data):
    rents = _matrix_medians(data)

    def rank(it):
        # prefer live over curated, used over new, card over server
        return (it["curated"], it["condition"] != "used", it["form"] != "card")

    best = {}
    for it in sorted(items, key=rank):
        if it.get("median_usd") and it["gpu_id"] not in best:
            best[it["gpu_id"]] = it
    out = {}
    for gid, it in sorted(best.items()):
        rent_id = _RENT_ALIAS.get(gid, gid)
        if rent_id not in rents:
            continue
        rent, nprov = rents[rent_id]
        per_gpu = it["median_usd"] / (8 if it["form"] == "server_8x" else 1)
        hours = per_gpu / rent
        entry = {
            "buy_usd_per_gpu": _round(per_gpu),
            "buy_basis": (f"{it['condition']} {it['form']}, {it['source']} ({it['as_of']})"
                          + ("; whole 8-GPU server / 8, includes chassis/CPU/RAM"
                             if it["form"] == "server_8x" else "")),
            "rent_hr": round(rent, 2),
            "rent_basis": "median on-demand across providers",
            "rent_gpu_id": rent_id,
            "rent_n_providers": nprov,
            "breakeven_hours": _round(hours),
            "breakeven_months_at_70pct_util": round(hours / (HOURS_PER_MONTH * UTIL), 1),
            "curated_buy_price": it["curated"],
            "note": "purchase price only; power, hosting, networking and staff excluded",
        }
        out[gid] = entry
    return out


def _update_history(prev_hist, items, today):
    hist = {k: list(v) for k, v in (prev_hist or {}).items()}
    cutoff = (datetime.fromisoformat(today) - timedelta(days=HISTORY_DAYS)).date().isoformat()
    for it in items:
        if it["curated"]:
            continue  # curated reference prices are not a daily series
        key = f"{it['gpu_id']}|{it['form']}|{it['condition']}"
        pts = [p for p in hist.get(key, []) if p.get("date") != today]
        pts.append({"date": today, "median_usd": it["median_usd"], "n": it["n_listings"]})
        hist[key] = sorted(pts, key=lambda p: p["date"])
    for k in list(hist):
        hist[k] = [p for p in hist[k] if p.get("date", "") >= cutoff]
        if not hist[k]:
            del hist[k]
    return hist


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_secondary_market(data, log_info=print, log_ok=print, log_fail=print):
    try:
        prev = data.get("secondary_market") or {}
        prev_items = prev.get("items") or []
        now = _now()
        today = now.date().isoformat()
        sources, items = {}, []

        # --- eBay (live) ---
        cid, csec = os.environ.get("EBAY_CLIENT_ID"), os.environ.get("EBAY_CLIENT_SECRET")
        ebay_url = "https://developer.ebay.com/api-docs/buy/browse/resources/item_summary/methods/search"
        if not (cid and csec):
            log_info(f"{SRC_EBAY}: skipped, EBAY_CLIENT_ID/EBAY_CLIENT_SECRET not set "
                     "(public eBay search pages return 403 to server-side clients)")
            sources[SRC_EBAY] = {"ok": False, "url": ebay_url,
                                 "note": "not configured: add EBAY_CLIENT_ID and EBAY_CLIENT_SECRET "
                                         "secrets (free eBay developer keyset) to enable live listing medians"}
        else:
            try:
                live = fetch_ebay(cid, csec, log_info)
                items.extend(live)
                sources[SRC_EBAY] = {"ok": True, "url": ebay_url,
                                     "note": f"{len(live)} buckets with >= {MIN_LISTINGS} filtered listings"}
                log_ok(SRC_EBAY, f"{len(live)} price buckets")
            except Exception as e:
                log_fail(SRC_EBAY, str(e)[:200])
                kept = [dict(it, stale=True) for it in prev_items if it.get("source") == SRC_EBAY]
                items.extend(kept)
                sources[SRC_EBAY] = {"ok": False, "url": ebay_url,
                                     "note": f"fetch failed ({str(e)[:120]}); kept {len(kept)} previous buckets as stale"}

        # --- curated ---
        try:
            cur = load_curated()
            items.extend(cur)
            sources[SRC_CURATED] = {"ok": True, "url": "scripts/curated/hardware_prices.json",
                                    "note": f"{len(cur)} cited reference prices (curated, not live)"}
            log_ok(SRC_CURATED, f"{len(cur)} entries")
        except Exception as e:
            log_fail(SRC_CURATED, str(e)[:200])
            kept = [dict(it, stale=True) for it in prev_items if it.get("curated")]
            items.extend(kept)
            sources[SRC_CURATED] = {"ok": False, "url": "scripts/curated/hardware_prices.json",
                                    "note": f"load failed ({str(e)[:120]}); kept previous entries"}

        items.sort(key=lambda it: (it["gpu_id"], it["form"], it["condition"], it["curated"]))
        data["secondary_market"] = {
            "as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": sources,
            "method": ("eBay: fixed-price US listings filtered by per-GPU title/VRAM regexes, an "
                       "accessory/parts/rental blacklist and a price floor, trimmed to 0.5x-2x of the "
                       f"first-pass median, published only with >= {MIN_LISTINGS} listings; curated "
                       "items are dated, cited reference prices, not medians."),
            "items": items,
            "history": _update_history(prev.get("history"), items, today),
            "rent_vs_buy": _rent_vs_buy(items, data),
        }
    except Exception as e:  # must never raise
        log_fail("secondary_market", f"build crashed: {e}")
    return data


if __name__ == "__main__":
    src = "/Users/quentinreyes/ai-gpu-dashboard-rm/data.json"
    out = ("/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/"
           "scratchpad/secondary_market_sample.json")
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    d = build_secondary_market(
        d,
        log_info=lambda m: print(f"  [INFO]  {m}"),
        log_ok=lambda s, m="": print(f"  [ OK ]  {s}: {m}"),
        log_fail=lambda s, m="": print(f"  [FAIL]  {s}: {m}"),
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"secondary_market": d.get("secondary_market")}, f, indent=2)
    print(f"wrote {out}")
