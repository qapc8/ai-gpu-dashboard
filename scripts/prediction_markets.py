"""GPU-price prediction markets: Kalshi and Polymarket.

Both venues now list contracts on the *rental* price of specific GPUs, and both
settle against the same public index (Ornn, https://dashboard.ornnai.com/compute).
That makes them the only forward-looking GPU price signal on this dashboard that
somebody is actually risking money on -- as opposed to an extrapolation of our
own history, which this project deliberately does not publish.

Two contract shapes are read:

  Kalshi     a ladder of "average hourly price above $K" binaries per month.
             Quote mids give P(price > K), i.e. a survival function; the implied
             median is where it crosses 0.50. Successive months form a forward
             curve.

  Polymarket mutually-exclusive price brackets ("$2.75-$3.00") per settlement
             date. Outcome prices are the bracket probabilities directly.

Everything published here is a market quote or a number derived from one. No
figure in this module is modelled by us.
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"

SETTLEMENT_SOURCE = {
    "name": "Ornn GPU compute index",
    "url": "https://dashboard.ornnai.com/compute",
}

# internal GPU id -> (Kalshi monthly-average series, display label)
KALSHI_SERIES = [
    ("H100-SXM", "KXH100MS", "H100 SXM"),
    ("H200", "KXH200MS", "H200"),
    ("B200", "KXB200MS", "B200"),
    ("RTX-5090", "KXRTX5090MS", "RTX 5090"),
]

# Polymarket titles name the GPU in parentheses.
POLY_GPU_PATTERNS = [
    ("H100-SXM", r"\(H100\)"),
    ("H200", r"\(H200\)"),
    ("B200", r"\(B200\)"),
    ("A100-80GB", r"\(A100\)"),
    ("RTX-5090", r"\(RTX\s*5090\)"),
]

# Display names for GPUs that only Polymarket lists (Kalshi supplies its own).
GPU_LABELS = {
    "H100-SXM": "H100 SXM",
    "H200": "H200",
    "B200": "B200",
    "B300": "B300",
    "A100-80GB": "A100 80GB",
    "RTX-5090": "RTX 5090",
}

_UA = {"User-Agent": "Mozilla/5.0 (gpu-dashboard prediction-markets)", "Accept": "application/json"}


def _get_json(url, params=None, timeout=30):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Turning quotes into an implied price
# ---------------------------------------------------------------------------

def _survival_from_ladder(points):
    """points: [(strike, probability_above)] -> cleaned, sorted, monotone list.

    Quotes on adjacent strikes are set independently and can cross (a $3.00
    binary bid above the $2.75 one). P(X > K) must be non-increasing in K, so
    clamp rather than publish a distribution with negative mass in a bucket.
    """
    pts = sorted((s, p) for s, p in points if s is not None and p is not None)
    cleaned = []
    ceiling = 1.0
    for strike, prob in pts:
        prob = max(0.0, min(1.0, prob))
        prob = min(prob, ceiling)
        ceiling = prob
        cleaned.append((strike, prob))
    return cleaned


def _implied_median(survival):
    """Price where P(above) crosses 0.50, linearly interpolated between strikes."""
    if not survival:
        return None
    # Entirely above the top strike / below the bottom one: the ladder does not
    # bracket the median, so say so instead of pinning it to an endpoint.
    if survival[0][1] < 0.5:
        return None
    if survival[-1][1] > 0.5:
        return None
    for (k0, p0), (k1, p1) in zip(survival, survival[1:]):
        if p0 >= 0.5 >= p1:
            if p0 == p1:
                return round((k0 + k1) / 2, 3)
            frac = (p0 - 0.5) / (p0 - p1)
            return round(k0 + frac * (k1 - k0), 3)
    return None


def _quantile(survival, q):
    """Price where P(above) crosses (1 - q). Used for an 80% band."""
    target = 1.0 - q
    if not survival or survival[0][1] < target or survival[-1][1] > target:
        return None
    for (k0, p0), (k1, p1) in zip(survival, survival[1:]):
        if p0 >= target >= p1:
            if p0 == p1:
                return round((k0 + k1) / 2, 3)
            frac = (p0 - target) / (p0 - p1)
            return round(k0 + frac * (k1 - k0), 3)
    return None


def _buckets_from_survival(survival):
    """Bucket probabilities between adjacent strikes, plus both tails."""
    if not survival:
        return []
    out = [{
        "label": f"under ${survival[0][0]:g}",
        "low": None,
        "high": survival[0][0],
        "prob": round(1.0 - survival[0][1], 4),
    }]
    for (k0, p0), (k1, p1) in zip(survival, survival[1:]):
        out.append({
            "label": f"${k0:g}-${k1:g}",
            "low": k0,
            "high": k1,
            "prob": round(p0 - p1, 4),
        })
    out.append({
        "label": f"over ${survival[-1][0]:g}",
        "low": survival[-1][0],
        "high": None,
        "prob": round(survival[-1][1], 4),
    })
    return [b for b in out if b["prob"] > 0.0005]


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _kalshi_horizon(event_ticker):
    """KXH100MS-26SEP -> ('2026-09', 'Sep 2026')."""
    m = re.search(r"-(\d{2})([A-Z]{3})$", event_ticker or "")
    if not m:
        return None, None
    year = 2000 + int(m.group(1))
    month = _MONTHS.get(m.group(2))
    if not month:
        return None, None
    label = datetime(year, month, 1).strftime("%b %Y")
    return f"{year:04d}-{month:02d}", label


def fetch_kalshi_gpu_markets(max_horizons=8):
    """Monthly-average GPU compute price ladders, newest month first."""
    curves = []
    for gpu_id, series, label in KALSHI_SERIES:
        try:
            events = _get_json(
                f"{KALSHI_API}/events",
                {"series_ticker": series, "status": "open", "limit": 200},
            ).get("events", [])
        except Exception:
            continue

        dated = []
        for ev in events:
            horizon, hlabel = _kalshi_horizon(ev.get("event_ticker"))
            if horizon:
                dated.append((horizon, hlabel, ev))
        dated.sort(key=lambda row: row[0])

        points = []
        for horizon, hlabel, ev in dated[:max_horizons]:
            ticker = ev.get("event_ticker")
            try:
                markets = _get_json(
                    f"{KALSHI_API}/markets",
                    {"event_ticker": ticker, "limit": 200},
                ).get("markets", [])
            except Exception:
                continue

            ladder = []
            open_interest = 0.0
            volume = 0.0
            for mkt in markets:
                strike = _num(mkt.get("floor_strike"))
                bid = _num(mkt.get("yes_bid_dollars"))
                ask = _num(mkt.get("yes_ask_dollars"))
                last = _num(mkt.get("last_price_dollars"))
                if bid is not None and ask is not None and ask > 0:
                    prob = (bid + ask) / 2
                elif last is not None and last > 0:
                    prob = last
                else:
                    continue
                ladder.append((strike, prob))
                open_interest += _num(mkt.get("open_interest_fp")) or 0.0
                volume += _num(mkt.get("volume_fp")) or 0.0

            survival = _survival_from_ladder(ladder)
            median = _implied_median(survival)
            if median is None:
                continue
            points.append({
                "horizon": horizon,
                "horizon_label": hlabel,
                "implied_price": median,
                "p10": _quantile(survival, 0.10),
                "p90": _quantile(survival, 0.90),
                "strikes": len(survival),
                "open_interest": round(open_interest),
                "volume": round(volume),
                "buckets": _buckets_from_survival(survival),
                "event_ticker": ticker,
                "url": f"https://kalshi.com/markets/{series.lower()}",
                "question": ev.get("title") or "",
            })

        if points:
            curves.append({
                "gpu": gpu_id,
                "label": label,
                "venue": "kalshi",
                "series": series,
                "points": points,
            })
    return curves


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

def _poly_bracket_bounds(title):
    """'$2.75-$3.00' / '<$2.25' / '$3.25+' -> (low, high)."""
    if not title:
        return None, None
    nums = [float(x) for x in re.findall(r"\$?(\d+(?:\.\d+)?)", title)]
    if not nums:
        return None, None
    if title.strip().startswith("<") or "under" in title.lower():
        return None, nums[0]
    if title.strip().endswith("+") or "above" in title.lower() or "over" in title.lower():
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    return nums[0], nums[0]


def _poly_median(brackets):
    """Implied median from mutually-exclusive brackets, interpolated in-bracket."""
    total = sum(b["prob"] for b in brackets)
    if total <= 0:
        return None
    ordered = sorted(
        brackets,
        key=lambda b: (b["low"] if b["low"] is not None else (b["high"] or 0) - 1),
    )
    cumulative = 0.0
    for b in ordered:
        prob = b["prob"] / total
        if cumulative + prob >= 0.5:
            low, high = b["low"], b["high"]
            if low is None or high is None:
                # Open-ended tail: report the edge rather than inventing a width.
                return round(high if low is None else low, 3)
            frac = (0.5 - cumulative) / prob if prob else 0.5
            return round(low + frac * (high - low), 3)
        cumulative += prob
    return None


def fetch_polymarket_gpu_markets():
    """Open Polymarket GPU rental-price bracket events."""
    try:
        payload = _get_json(
            f"{POLYMARKET_GAMMA}/public-search",
            {"q": "GPU rental prices", "limit_per_type": 40},
        )
    except Exception:
        return []

    out = []
    for ev in payload.get("events", []):
        if ev.get("closed"):
            continue
        title = ev.get("title") or ""
        gpu_id = next(
            (gid for gid, pat in POLY_GPU_PATTERNS if re.search(pat, title, re.IGNORECASE)),
            None,
        )
        if not gpu_id:
            continue
        # "hit___ in 2026" contracts are touch/barrier markets -- they ask
        # whether a level is ever reached, not where the price settles, so they
        # must not be averaged into a level estimate. (Note the title has no
        # word boundary after "hit": it reads "hit___ in 2026?".)
        if re.search(r"\bhit_*", title, re.IGNORECASE):
            continue

        brackets = []
        for mkt in ev.get("markets", []) or []:
            if mkt.get("closed"):
                continue
            try:
                prices = json.loads(mkt.get("outcomePrices") or "[]")
                outcomes = json.loads(mkt.get("outcomes") or "[]")
            except (TypeError, ValueError):
                continue
            if not prices or not outcomes:
                continue
            yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "yes"), 0)
            prob = _num(prices[yes_idx]) if yes_idx < len(prices) else None
            if prob is None:
                continue
            label = mkt.get("groupItemTitle") or mkt.get("question") or ""
            low, high = _poly_bracket_bounds(label)
            if low is None and high is None:
                continue
            brackets.append({
                "label": label,
                "low": low,
                "high": high,
                "prob": round(prob, 4),
                "volume": round(_num(mkt.get("volume")) or 0.0),
            })

        if len(brackets) < 2:
            continue

        end = ev.get("endDate") or ""
        horizon = end[:7] if len(end) >= 7 else None
        total_prob = sum(b["prob"] for b in brackets) or 1.0
        out.append({
            "gpu": gpu_id,
            "venue": "polymarket",
            "question": title,
            "horizon": horizon,
            "horizon_label": _poly_horizon_label(end),
            "end_date": end,
            "implied_price": _poly_median(brackets),
            "volume": round(_num(ev.get("volume")) or 0.0),
            "liquidity": round(_num(ev.get("liquidity")) or 0.0),
            "url": f"https://polymarket.com/event/{ev.get('slug')}",
            "buckets": [
                {**b, "prob": round(b["prob"] / total_prob, 4)} for b in brackets
            ],
        })

    out.sort(key=lambda m: (m["gpu"], m.get("end_date") or ""))
    return out


def _poly_horizon_label(end_iso):
    try:
        return datetime.fromisoformat((end_iso or "").replace("Z", "+00:00")).strftime("%d %b %Y")
    except ValueError:
        return end_iso or ""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_prediction_markets(data, kalshi_curves, polymarket_markets):
    """Attach a prediction_markets block, joined to our own listed prices."""
    now = datetime.now(timezone.utc).isoformat()
    providers = data.get("providers", {})

    def cheapest_listed(gpu_id):
        prices = [
            gpu["price_per_gpu_hr"]
            for prov in providers.values()
            for gid, gpu in (prov.get("gpus") or {}).items()
            if gid == gpu_id and gpu.get("price_per_gpu_hr")
        ]
        if not prices:
            return None, None
        cheapest = min(prices)
        return round(cheapest, 3), round(sum(prices) / len(prices), 3)

    gpus = {}
    for curve in kalshi_curves:
        entry = gpus.setdefault(curve["gpu"], {
            "gpu": curve["gpu"],
            "label": curve["label"],
            "curve": [],
            "markets": [],
        })
        entry["curve"] = curve["points"]
        entry["kalshi_series"] = curve["series"]
        entry["kalshi_url"] = f"https://kalshi.com/markets/{curve['series'].lower()}"

    for mkt in polymarket_markets:
        entry = gpus.setdefault(mkt["gpu"], {
            "gpu": mkt["gpu"],
            "label": GPU_LABELS.get(mkt["gpu"], mkt["gpu"]),
            "curve": [],
            "markets": [],
        })
        entry["markets"].append(mkt)

    for gpu_id, entry in gpus.items():
        listed_min, listed_avg = cheapest_listed(gpu_id)
        entry["listed_cheapest"] = listed_min
        entry["listed_average"] = listed_avg

        # Kalshi's monthly ladder is the primary curve. A GPU only Polymarket
        # lists (A100) still needs near/far, so fall back to its dated markets.
        points = entry["curve"]
        if not points:
            points = [
                {"implied_price": m["implied_price"], "horizon_label": m["horizon_label"]}
                for m in sorted(entry["markets"], key=lambda m: m.get("end_date") or "")
                if m.get("implied_price") is not None
            ]
        near = points[0] if points else None
        entry["near_implied"] = near["implied_price"] if near else None
        entry["near_horizon"] = near["horizon_label"] if near else None
        far = points[-1] if points else None
        entry["far_implied"] = far["implied_price"] if far else None
        entry["far_horizon"] = far["horizon_label"] if far else None
        if near and far and near is not far and near["implied_price"]:
            entry["curve_change_pct"] = round(
                (far["implied_price"] - near["implied_price"]) / near["implied_price"] * 100, 1
            )
        else:
            entry["curve_change_pct"] = None

    data["prediction_markets"] = {
        "last_updated": now,
        "settlement_source": SETTLEMENT_SOURCE,
        "venues": [
            {
                "key": "kalshi",
                "name": "Kalshi",
                "detail": "CFTC-regulated US event exchange",
                "url": "https://kalshi.com",
                "contract": "Monthly average $/GPU-hour, ladder of 'above $K' binaries",
            },
            {
                "key": "polymarket",
                "name": "Polymarket",
                "detail": "Onchain prediction market",
                "url": "https://polymarket.com",
                "contract": "Settlement-date price brackets",
            },
        ],
        "gpus": [gpus[k] for k in sorted(gpus, key=lambda g: -(gpus[g].get("near_implied") or 0))],
    }

    # Keep the Data tab's provenance table honest about this section.
    meta_sections = data.setdefault("_meta", {}).setdefault("sections", {})
    meta_sections["prediction_markets"] = {
        "basis": "measured",
        "detail": (
            "Live order books: Kalshi trade API and Polymarket Gamma API. "
            "Implied prices are read off the quoted ladder/brackets "
            "(median of the market-implied distribution) -- no model of ours. "
            "Both venues settle against the Ornn GPU compute index, which is "
            "not the same basket as our provider list-price average, so the "
            "two series are related but not directly comparable."
        ),
    }
    return data
