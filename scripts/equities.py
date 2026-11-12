"""Compute-complex equity basket, and how it moves against GPU rental prices.

The companies that build, house, power and rent out AI accelerators are the
public market's running bet on GPU demand. This module tracks ten of them
(chips, foundry, memory, servers, power & cooling, neoclouds), builds an
equal-weighted index, and lines it up against the H100-SXM rental price this
dashboard already records, so the two can be read on one chart.

Sources
  Yahoo Finance v8 chart API (same endpoint and headers as
  `fetch_stock_price` in update_data.py): 1y of daily closes per ticker.
  Prices and the index are derived only from those closes.

  The rental side is read from data["price_history"]["days"] (per-provider
  published prices, one snapshot per day). Nothing is fetched for it here.

Index method
  Equal-weight, rebalanced daily: on each trading day the index moves by the
  arithmetic mean of the daily returns of every constituent that has a close
  on both that day and the previous one. It is set to 100 on the first date
  on which every ticker has a close. If some ticker's history starts more than
  MAX_LATE_DAYS after the others (a recent IPO), the base is instead the
  earliest date with at least MIN_CONSTITUENTS closes and the late ticker
  joins on its second trading day -- chain-linking means joining never makes
  the index jump. The method string published with the index says which.

Rental series and correlation
  `rental_hr` is the cross-provider median H100-SXM price per GPU-hour on
  each trading date. That median is composition-sensitive (providers joined
  the scrape over time), so it is NOT used for correlation. Instead the daily
  rental change is a matched panel: the mean of per-provider % changes
  among providers that quote H100-SXM on both dates, skipping any provider
  whose scraper changed method on that date (price_history.METHODOLOGY_CHANGES).
  Correlation is Pearson over the last 90 aligned trading-day pairs, only if
  at least MIN_CORR_PAIRS exist; otherwise null with a reason. Rental prices
  change rarely, so many daily rental changes are exactly zero; that is
  stated in the note, not smoothed away.

Failures: a ticker that cannot be fetched keeps its previous entry (marked
stale: true); the index is only rebuilt from fresh closes.
"""

import json
import math
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone

try:  # read-only use of the correction log maintained by price_history.py
    from price_history import METHODOLOGY_CHANGES
except Exception:  # pragma: no cover - module import path differs
    try:
        from scripts.price_history import METHODOLOGY_CHANGES
    except Exception:
        METHODOLOGY_CHANGES = []

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 20

SOURCE_YAHOO = "Yahoo Finance (equities)"
SOURCE_RENTAL = "Rental price history (internal)"
SOURCES = [SOURCE_YAHOO, SOURCE_RENTAL]

TICKERS = [
    ("NVDA", "GPUs"),
    ("AMD", "GPUs"),
    ("TSM", "Foundry"),
    ("AVGO", "Custom accelerators & networking"),
    ("MU", "HBM memory"),
    ("SMCI", "Servers"),
    ("VRT", "Power & cooling"),
    ("CRWV", "Neocloud"),
    ("NBIS", "Neocloud"),
    ("ORCL", "Hyperscale GPU cloud"),
]

RENTAL_GPU = "H100-SXM"
MAX_LATE_DAYS = 60
MIN_CONSTITUENTS = 5
CORR_WINDOW = 90
MIN_CORR_PAIRS = 30


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def fetch_ticker(ticker):
    """Return {"name", "price", "history": [{"date", "close"}]} or raise."""
    url = YAHOO_URL.format(ticker=ticker) + "?" + urllib.parse.urlencode(
        {"range": "1y", "interval": "1d"})
    req = urllib.request.Request(url, headers=YAHOO_HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        body = json.loads(resp.read().decode("utf-8", "replace"))
    result = (body.get("chart") or {}).get("result") or []
    if not result:
        err = (body.get("chart") or {}).get("error")
        raise RuntimeError(f"no chart data ({err})")
    r = result[0]
    meta = r.get("meta") or {}
    stamps = r.get("timestamp") or []
    closes = ((r.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    # Prefer adjusted closes (splits/dividends) for return maths when present.
    adj = ((r.get("indicators") or {}).get("adjclose") or [{}])
    adj = adj[0].get("adjclose") if adj else None
    series = adj if adj and len(adj) == len(stamps) else closes
    tz_off = meta.get("gmtoffset") or 0
    by_date = {}
    for ts, c in zip(stamps, series):
        if c is None:
            continue
        d = datetime.fromtimestamp(ts + tz_off, tz=timezone.utc).strftime("%Y-%m-%d")
        by_date[d] = round(float(c), 4)
    if len(by_date) < 2:
        raise RuntimeError("fewer than 2 closes")
    price = meta.get("regularMarketPrice")
    if price is None:
        price = by_date[max(by_date)]
    history = [{"date": d, "close": by_date[d]} for d in sorted(by_date)]
    return {
        "name": meta.get("longName") or meta.get("shortName") or ticker,
        "price": round(float(price), 2),
        "history": history,
    }


# --------------------------------------------------------------------------
# derived metrics
# --------------------------------------------------------------------------

def _pct(a, b):
    if a is None or b in (None, 0):
        return None
    return round((a / b - 1) * 100, 2)


def _ticker_stats(info, today):
    hist = info["history"]
    closes = [p["close"] for p in hist]
    last_date = hist[-1]["date"]
    price = info["price"]
    # 1d: live price vs previous session close. If the last bar is today's
    # (in-progress) session, the previous close is the bar before it.
    prev = closes[-2] if len(closes) >= 2 else None
    if last_date < today.strftime("%Y-%m-%d") and abs(price - closes[-1]) / closes[-1] > 1e-6:
        prev = closes[-1]
    # 30d: close on or before 30 calendar days ago.
    target30 = _shift_date(last_date, -30)
    c30 = next((p["close"] for p in reversed(hist) if p["date"] <= target30), None)
    # YTD: last close of the previous year, else first close of this year.
    year = last_date[:4]
    prior = [p["close"] for p in hist if p["date"] < f"{year}-01-01"]
    ytd_base = prior[-1] if prior else next((p["close"] for p in hist if p["date"][:4] == year), None)
    return {
        "chg_1d_pct": _pct(price, prev),
        "chg_30d_pct": _pct(price, c30),
        "ytd_pct": _pct(price, ytd_base),
    }


def _shift_date(iso, days):
    from datetime import timedelta
    return (datetime.strptime(iso, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def _build_index(histories):
    """Equal-weight daily-rebalanced chain index. Returns (method, base, points)."""
    closes = {t: {p["date"]: p["close"] for p in h} for t, h in histories.items()}
    all_dates = sorted(set().union(*[set(c) for c in closes.values()]))
    if not all_dates:
        return None, None, []
    first = {t: min(c) for t, c in closes.items()}
    common = max(first.values())
    earliest = min(first.values())
    late = [t for t, f in first.items() if f > _shift_date(earliest, MAX_LATE_DAYS)]
    if late:
        base = next(d for d in all_dates
                    if sum(1 for c in closes.values() if d in c) >= min(MIN_CONSTITUENTS, len(closes)))
        method = (f"Equal-weight, daily-rebalanced chain index of {len(closes)} tickers; base 100 on "
                  f"{base}. Late listings ({', '.join(sorted(late))}) join on their second trading day; "
                  "each day's move is the mean daily return of constituents with closes on both days.")
    else:
        base = common
        method = (f"Equal-weight, daily-rebalanced chain index of {len(closes)} tickers; base 100 on "
                  f"{base}, the first date every ticker has a close. Each day's move is the mean daily "
                  "return of constituents with closes on both days (adjusted closes).")
    dates = [d for d in all_dates if d >= base]
    points = [{"date": base, "value": 100.0}]
    level = 100.0
    for prev_d, d in zip(dates, dates[1:]):
        rets = [c[d] / c[prev_d] - 1 for c in closes.values() if d in c and prev_d in c and c[prev_d]]
        if rets:
            level *= 1 + sum(rets) / len(rets)
        points.append({"date": d, "value": round(level, 2)})
    return method, base, points


def _corrected(provider, gpu, date):
    for ch in METHODOLOGY_CHANGES:
        if ch.get("date") != date:
            continue
        if ch.get("providers") and provider not in ch["providers"]:
            continue
        if ch.get("gpus") and gpu not in ch["gpus"]:
            continue
        return True
    return False


def _rental_vs_index(data, index_points):
    days = ((data.get("price_history") or {}).get("days")) or {}
    if not days:
        raise RuntimeError("data['price_history']['days'] missing")
    rental_dates = sorted(days)
    idx = {p["date"]: p["value"] for p in index_points}
    # Drop dates before any market-wide re-measurement ("scope": "all"): before
    # it the prices were seed values, not comparable with anything after.
    cutoff = max([c["date"] for c in METHODOLOGY_CHANGES if c.get("scope") == "all"] or [""])
    aligned = [d for d in sorted(idx) if d in days and d >= cutoff]

    def quotes(d):
        return {prov: g[RENTAL_GPU] for prov, g in (days.get(d) or {}).items()
                if isinstance(g, dict) and isinstance(g.get(RENTAL_GPU), (int, float)) and g[RENTAL_GPU] > 0}

    points = []
    for d in aligned:
        q = quotes(d)
        if q:
            points.append({"date": d, "index": idx[d], "rental_hr": round(statistics.median(q.values()), 4),
                           "n_providers": len(q)})

    pairs = []
    for a, b in zip(points, points[1:]):
        qa, qb = quotes(a["date"]), quotes(b["date"])
        chg = []
        for prov in set(qa) & set(qb):
            # skip a provider whose scraper changed method anywhere in (a, b]
            if any(_corrected(prov, RENTAL_GPU, d) for d in rental_dates if a["date"] < d <= b["date"]):
                continue
            chg.append(qb[prov] / qa[prov] - 1)
        if not chg:
            continue
        r_chg = statistics.fmean(chg)
        i_chg = b["index"] / a["index"] - 1
        pairs.append((i_chg, r_chg, a["date"], b["date"]))
    window = pairs[-CORR_WINDOW:]
    zero = sum(1 for w in window if w[1] == 0)
    basis = (f"rental_hr = cross-provider median {RENTAL_GPU} on-demand $/GPU-hr from this dashboard's "
             f"daily per-provider snapshots (price_history, from {aligned[0] if aligned else rental_dates[0]}), sampled on index "
             "trading dates. Correlation uses matched-panel daily changes (mean per-provider % change "
             "among providers quoting both days, excluding scraper-method changes), not the raw median.")
    if len(window) < MIN_CORR_PAIRS:
        return basis, points, None, (f"Only {len(window)} overlapping trading-day pairs; need "
                                     f"{MIN_CORR_PAIRS} for a 90-day correlation.")
    xs = [w[0] for w in window]
    ys = [w[1] for w in window]
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return basis, points, None, (f"Rental series did not change over the last {len(window)} "
                                     "trading-day pairs; correlation undefined.")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(window) * sx * sy)
    note = (f"Pearson correlation of daily % changes over the last {len(window)} aligned trading-day pairs "
            f"({window[0][2]} to {window[-1][3]}). {zero} of {len(window)} days had zero rental change -- on-demand list "
            "prices move in steps, so a low value mostly reflects that, not independence.")
    return basis, points, round(corr, 3) if not math.isnan(corr) else None, note


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_equities(data, log_info=print, log_ok=lambda *a: None, log_fail=lambda *a: None):
    try:
        return _build(data, log_info, log_ok, log_fail)
    except Exception as e:  # never raise
        log_fail(SOURCE_YAHOO, f"unexpected error: {e}")
        return data


def _build(data, log_info, log_ok, log_fail):
    now = datetime.now(timezone.utc)
    prev = data.get("equities") or {}
    prev_tickers = prev.get("tickers") or {}
    tickers, fresh, failed = {}, {}, []
    for t, role in TICKERS:
        try:
            info = fetch_ticker(t)
            stats = _ticker_stats(info, now)
            tickers[t] = {"name": info["name"], "role": role, "price": info["price"], **stats,
                          "history": info["history"], "stale": False}
            fresh[t] = info["history"]
        except Exception as e:
            failed.append(f"{t}: {e}")
            if t in prev_tickers:
                tickers[t] = dict(prev_tickers[t], stale=True)
    sources = {}
    if fresh:
        log_ok(SOURCE_YAHOO, f"{len(fresh)}/{len(TICKERS)} tickers" + (f"; failed {failed}" if failed else ""))
        sources[SOURCE_YAHOO] = {"ok": True, "url": "https://finance.yahoo.com/",
                                 "note": f"v8 chart API, 1y daily; {len(fresh)}/{len(TICKERS)} fresh"
                                         + (f"; failed: {'; '.join(failed)}" if failed else "")}
    else:
        log_fail(SOURCE_YAHOO, "; ".join(failed) or "no tickers")
        sources[SOURCE_YAHOO] = {"ok": False, "url": "https://finance.yahoo.com/",
                                 "note": "all tickers failed; previous values kept"}

    section = {"as_of": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
               "sources": sources, "tickers": tickers}

    if len(fresh) >= MIN_CONSTITUENTS:
        method, base, points = _build_index(fresh)
        if len(fresh) < len(TICKERS):
            method += f" Rebuilt from {len(fresh)} of {len(TICKERS)} tickers (others failed this run)."
        section["index"] = {"name": "Compute Complex", "method": method, "base_date": base, "points": points,
                            "constituents": sorted(fresh), "stale": False}
    elif prev.get("index"):
        section["index"] = dict(prev["index"], stale=True)
    else:
        section["index"] = None

    vs = None
    if section.get("index") and section["index"].get("points"):
        try:
            basis, pts, corr, note = _rental_vs_index(data, section["index"]["points"])
            vs = {"gpu_id": RENTAL_GPU, "basis": basis, "points": pts, "corr_90d": corr, "corr_note": note}
            log_ok(SOURCE_RENTAL, f"{len(pts)} aligned days, corr_90d={corr}")
            sources[SOURCE_RENTAL] = {"ok": True, "url": "data.json#price_history",
                                      "note": f"{len(pts)} aligned trading days"}
        except Exception as e:
            log_fail(SOURCE_RENTAL, str(e))
            sources[SOURCE_RENTAL] = {"ok": False, "url": "data.json#price_history", "note": str(e)}
            if prev.get("vs_rental"):
                vs = dict(prev["vs_rental"], stale=True)
    section["vs_rental"] = vs
    data["equities"] = section
    return data


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with open("/Users/quentinreyes/ai-gpu-dashboard-rm/data.json") as f:
        d = json.load(f)
    build_equities(d, print, lambda s, m="": print("OK  ", s, m), lambda s, m="": print("FAIL", s, m))
    out = "/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/equities_sample.json"
    with open(out, "w") as f:
        json.dump({"equities": d.get("equities")}, f, indent=1)
    print("wrote", out)
