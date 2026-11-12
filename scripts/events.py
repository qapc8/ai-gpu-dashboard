"""Calendar of market-moving events for AI accelerators (chart markers + look-ahead).

Window: ~18 months back (chart markers) to ~6 months ahead.

Sources
  Nasdaq          LIVE  next earnings date per ticker (api.nasdaq.com analyst
                        earnings-date, Zacks-fed). Nasdaq words a company-
                        confirmed date "is expected to report" and a projected
                        one "is estimated to report"; the latter is published
                        with date_confirmed=false. No date -> nothing emitted
                        (we never project a date ourselves).
  SEC EDGAR       LIVE  past earnings releases = 8-K filings carrying Item 2.02
                        (Results of Operations) for US filers.
  Nasdaq Surprise LIVE  past report dates for foreign filers that file 6-Ks
                        without item codes (TSM, NBIS): earnings-surprise table
                        (last 4 quarters only).
  Federal Register LIVE policy items via china_export.policy_events (reuses
                        data["china_export"] if it was built this run, else
                        fetches the Federal Register itself).
  Curated               scripts/curated/events.json: conferences, launches,
                        model releases and a few policy actions that never hit
                        the Federal Register; every entry has source + url.

A failing live source keeps its previous items from data["events"] and is
marked stale in `sources`. Impact is rule-based; the rule text is published in
the section as `impact_rule`.

Stdlib only.
"""

import json
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

try:  # imported as scripts.events or run from scripts/
    from china_export import fetch_federal_register, policy_events, POLICY_IMPACT_RULE
except ImportError:  # pragma: no cover
    from scripts.china_export import fetch_federal_register, policy_events, POLICY_IMPACT_RULE

UA = "ai-gpu-dashboard/1.0 (market research)"
SEC_UA = "ai-gpu-dashboard research bot admin@ai-gpu-dashboard.invalid"
TIMEOUT = 25

BACK_DAYS = 548    # ~18 months
AHEAD_DAYS = 210   # ~6-7 months (lets the early-June trade shows in)

TICKERS = ["NVDA", "AMD", "TSM", "SMCI", "AVGO", "MU", "CRWV", "NBIS",
           "ORCL", "MSFT", "GOOGL", "AMZN", "META"]
CIK = {"NVDA": 1045810, "AMD": 2488, "SMCI": 1375365, "AVGO": 1730168, "MU": 723125,
       "CRWV": 1769628, "ORCL": 1341439, "MSFT": 789019, "GOOGL": 1652044,
       "AMZN": 1018724, "META": 1326801}
FOREIGN_FILERS = ["TSM", "NBIS"]   # 6-K filers: no Item 2.02 tagging on EDGAR
NAMES = {"NVDA": "NVIDIA", "AMD": "AMD", "TSM": "TSMC", "SMCI": "Supermicro",
         "AVGO": "Broadcom", "MU": "Micron", "CRWV": "CoreWeave", "NBIS": "Nebius",
         "ORCL": "Oracle", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
         "META": "Meta"}

NASDAQ_DATE_URL = "https://api.nasdaq.com/api/analyst/{t}/earnings-date"
NASDAQ_SURPRISE_URL = "https://api.nasdaq.com/api/company/{t}/earnings-surprise"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
CURATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curated", "events.json")

SRC_NASDAQ = "Nasdaq Earnings Calendar"
SRC_SEC = "SEC EDGAR 8-K"
SRC_SURPRISE = "Nasdaq Earnings Surprise"
SRC_FR = "Federal Register"
SRC_CURATED = "Events (curated)"
SOURCES = [SRC_NASDAQ, SRC_SEC, SRC_SURPRISE, SRC_FR, SRC_CURATED]
SOURCE_URLS = {
    SRC_NASDAQ: "https://www.nasdaq.com/market-activity/earnings",
    SRC_SEC: "https://www.sec.gov/edgar/search/",
    SRC_SURPRISE: "https://www.nasdaq.com/market-activity/earnings",
    SRC_FR: "https://www.federalregister.gov/developers/documentation/api/v1",
    SRC_CURATED: "scripts/curated/events.json",
}
# id prefix -> producing source, used to carry over items when a source fails
ID_PREFIX = {SRC_NASDAQ: "earn-next-", SRC_SEC: "earn-8k-", SRC_SURPRISE: "earn-hist-",
             SRC_FR: "fr-", SRC_CURATED: None}

EARN_HIGH = {"NVDA", "MSFT", "GOOGL", "AMZN", "META"}
EARN_MED = {"AMD", "TSM", "AVGO", "MU", "ORCL", "CRWV", "NBIS"}
IMPACT_RULE = {
    "earnings": "high = NVDA (accelerator leader) and MSFT/GOOGL/AMZN/META (largest AI-capex "
                "spenders, whose capex guidance moves GPU demand); medium = AMD, TSM, AVGO, MU, "
                "ORCL, CRWV, NBIS (accelerator/HBM/foundry suppliers and GPU clouds); low = others (SMCI).",
    "conference": "tier 1 (NVIDIA GTC San Jose) = high; tier 2 (Computex, AMD Advancing AI, "
                  "AWS re:Invent, Google I/O) = medium; tier 3 (Hot Chips, CES) = low.",
    "launch": "high when the launch is an NVIDIA data-center platform milestone (NVDA in tickers); "
              "medium otherwise.",
    "model_release": "high = flagship (a new top-tier model generation, e.g. GPT-5/6, Gemini 3/4, "
                     "Claude Opus 4 / Fable 5, Grok 4, DeepSeek V4); medium = other frontier "
                     "point releases.",
    "policy": POLICY_IMPACT_RULE,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _get_json(url, ua=UA):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _mdY(s):
    for fmt in ("%m/%d/%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def _earn_item(id_, d, ticker, confirmed, source, url, when=""):
    return {"id": id_, "date": d.isoformat() if hasattr(d, "isoformat") else d,
            "date_confirmed": confirmed, "type": "earnings",
            "title": f"{NAMES.get(ticker, ticker)} ({ticker}) earnings" + (f" {when}" if when else ""),
            "tickers": [ticker], "source": source, "url": url}


# --------------------------------------------------------------------------
# fetchers (each raises on total failure)
# --------------------------------------------------------------------------

def fetch_nasdaq_upcoming(today):
    items, errors, empty = [], [], []
    for t in TICKERS:
        try:
            d = _get_json(NASDAQ_DATE_URL.format(t=t))["data"]
            text = re.sub(r"\s+", " ", d.get("reportText") or "")
            m = re.search(r"(?:expected|estimated)\*? to report earnings on (\d{1,2}/\d{1,2}/\d{4})", text)
            if not m:
                empty.append(t)
                continue
            day = _mdY(m.group(1))
            if not day or day < today:
                empty.append(t)
                continue
            confirmed = bool(re.search(r"\bexpected\*? to report", text)) and \
                "estimated" not in text.split("report earnings")[0]
            when = ""
            if "after market close" in text:
                when = "(after close)"
            elif "before market open" in text:
                when = "(before open)"
            items.append(_earn_item(f"earn-next-{t}-{day.isoformat()}", day, t, confirmed,
                                    "Nasdaq / Zacks",
                                    f"https://www.nasdaq.com/market-activity/stocks/{t.lower()}/earnings",
                                    when))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{t}: {e}")
        time.sleep(0.2)
    if len(errors) == len(TICKERS):
        raise RuntimeError("; ".join(errors)[:300])
    return items, errors, empty


def fetch_sec_8k_earnings(since):
    items, errors = [], []
    for t, cik in CIK.items():
        try:
            sub = _get_json(SEC_SUBMISSIONS_URL.format(cik=cik), ua=SEC_UA)
            r = sub["filings"]["recent"]
            for i, form in enumerate(r["form"]):
                if form not in ("8-K", "8-K/A") or "2.02" not in (r["items"][i] or ""):
                    continue
                fd = r["filingDate"][i]
                if fd < since:
                    continue
                acc = r["accessionNumber"][i]
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{acc.replace('-', '')}/{r['primaryDocument'][i]}")
                items.append(_earn_item(f"earn-8k-{t}-{fd}", fd, t, True, "SEC EDGAR 8-K (Item 2.02)", url))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{t}: {e}")
        time.sleep(0.15)   # SEC fair-access: stay well under 10 req/s
    if len(errors) == len(CIK):
        raise RuntimeError("; ".join(errors)[:300])
    return items, errors


def fetch_nasdaq_history(since):
    items, errors = [], []
    for t in FOREIGN_FILERS:
        try:
            rows = _get_json(NASDAQ_SURPRISE_URL.format(t=t))["data"]["earningsSurpriseTable"]["rows"]
            for row in rows:
                day = _mdY(row.get("dateReported") or "")
                if day and day.isoformat() >= since:
                    items.append(_earn_item(
                        f"earn-hist-{t}-{day.isoformat()}", day, t, True, "Nasdaq / Zacks",
                        f"https://www.nasdaq.com/market-activity/stocks/{t.lower()}/earnings",
                        f"(FQ end {row.get('fiscalQtrEnd')})" if row.get("fiscalQtrEnd") else ""))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{t}: {e}")
    if len(errors) == len(FOREIGN_FILERS):
        raise RuntimeError("; ".join(errors)[:300])
    return items, errors


def load_curated(path=CURATED_PATH):
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    items, dropped = [], 0
    for e in raw.get("items") or []:
        if not all(e.get(k) for k in ("id", "date", "type", "title", "source", "url")) or \
                not re.match(r"^\d{4}-\d{2}-\d{2}$", e["date"]) or not e["url"].startswith("http"):
            dropped += 1
            continue
        item = {k: e[k] for k in ("id", "date", "type", "title", "source", "url")}
        item["id"] = "cur-" + e["id"]
        item["date_confirmed"] = bool(e.get("date_confirmed", True))
        item["tickers"] = list(e.get("tickers") or [])
        if e.get("end_date"):
            item["end_date"] = e["end_date"]
        item["_tier"] = e.get("tier")
        item["_flagship"] = bool(e.get("flagship"))
        items.append(item)
    if not items:
        raise RuntimeError("no valid curated events")
    return items, dropped


# --------------------------------------------------------------------------
# impact + assembly
# --------------------------------------------------------------------------

def _impact(item):
    typ, tickers = item["type"], set(item.get("tickers") or [])
    if item.get("impact") in ("high", "medium", "low") and typ == "policy" and \
            item["id"].startswith("fr-"):
        return item["impact"]          # already set by china_export.policy_events
    if typ == "earnings":
        return "high" if tickers & EARN_HIGH else "medium" if tickers & EARN_MED else "low"
    if typ == "conference":
        return {1: "high", 2: "medium"}.get(item.get("_tier"), "low")
    if typ == "launch":
        return "high" if "NVDA" in tickers else "medium"
    if typ == "model_release":
        return "high" if item.get("_flagship") else "medium"
    if typ == "policy":   # curated policy actions: same rule as Federal Register items
        return "high" if re.search(r"export|licen|diffusion|H20|H200|tariff|semiconductor",
                                   item["title"], re.I) else "medium"
    return "low"


def _norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _finalize(items, start, end):
    out, seen_ids, seen_keys = [], set(), set()
    # earnings: one item per (ticker, date) -- prefer SEC (filed) over Nasdaq history/upcoming
    pri = {"earn-8k-": 0, "earn-hist-": 1, "earn-next-": 2}
    items = sorted(items, key=lambda i: next((v for k, v in pri.items() if i["id"].startswith(k)), 3))
    earn_days = {}
    for it in items:
        if not (start <= it["date"] <= end):
            continue
        if it["id"] in seen_ids:
            continue
        if it["type"] == "earnings":
            t = it["tickers"][0]
            key = ("earnings", t, it["date"])
            # SEC filing dates can trail the release by a day; treat +-1 day as the same report
            near = [d for d in earn_days.get(t, []) if abs((datetime.fromisoformat(d) -
                                                           datetime.fromisoformat(it["date"])).days) <= 1]
            if near:
                continue
            earn_days.setdefault(t, []).append(it["date"])
        else:
            key = (it["type"], it["date"], _norm_title(it["title"]))
        if key in seen_keys:
            continue
        seen_ids.add(it["id"])
        seen_keys.add(key)
        it["impact"] = _impact(it)
        clean = {k: it[k] for k in ("id", "date", "date_confirmed", "type", "title", "tickers",
                                    "impact", "source", "url")}
        if it.get("end_date"):
            clean["end_date"] = it["end_date"]
        out.append(clean)
    out.sort(key=lambda i: (i["date"], {"high": 0, "medium": 1, "low": 2}[i["impact"]], i["id"]))
    return out


def build_events(data, log_info=lambda m: None, log_ok=lambda a, b="": None,
                 log_fail=lambda a, b="": None):
    try:
        now = datetime.now(timezone.utc)
        today = now.date()
        start = (today - timedelta(days=BACK_DAYS)).isoformat()
        end = (today + timedelta(days=AHEAD_DAYS)).isoformat()
        prev_items = (data.get("events") or {}).get("items") or []
        sources, items = {}, []

        def carry(src):
            pref = ID_PREFIX.get(src)
            if src == SRC_CURATED:
                return [i for i in prev_items if i["id"].startswith("cur-")]
            return [dict(i, _prev=True) for i in prev_items if pref and i["id"].startswith(pref)]

        def ok(src, note):
            sources[src] = {"ok": True, "url": SOURCE_URLS[src], "note": note}
            log_ok(src, note)

        def fail(src, e):
            kept = carry(src)
            items.extend(kept)
            sources[src] = {"ok": False, "url": SOURCE_URLS[src], "stale": bool(kept),
                            "note": f"failed ({str(e)[:200]}); kept {len(kept)} previous items"}
            log_fail(src, str(e)[:200])

        try:
            got, errs, empty = fetch_nasdaq_upcoming(today)
            items.extend(got)
            note = f"{len(got)} upcoming dates ({sum(1 for g in got if not g['date_confirmed'])} estimated)"
            if empty:
                note += f"; no date published yet for {', '.join(empty)}"
            if errs:
                note += f"; {len(errs)} tickers failed"
                log_info("Nasdaq earnings-date errors: " + "; ".join(errs)[:300])
            ok(SRC_NASDAQ, note)
        except Exception as e:  # noqa: BLE001
            fail(SRC_NASDAQ, e)

        try:
            got, errs = fetch_sec_8k_earnings(start)
            items.extend(got)
            ok(SRC_SEC, f"{len(got)} Item 2.02 earnings 8-Ks since {start}"
               + (f"; {len(errs)} CIKs failed" if errs else ""))
        except Exception as e:  # noqa: BLE001
            fail(SRC_SEC, e)

        try:
            got, errs = fetch_nasdaq_history(start)
            items.extend(got)
            ok(SRC_SURPRISE, f"{len(got)} past report dates for {', '.join(FOREIGN_FILERS)} "
               "(last 4 quarters available)" + (f"; {len(errs)} failed" if errs else ""))
        except Exception as e:  # noqa: BLE001
            fail(SRC_SURPRISE, e)

        try:
            sec = data.get("china_export") or {}
            fresh = sec.get("rules") and not sec.get("rules_stale")
            if not fresh:
                rules, _ = fetch_federal_register(today)
                sec = {"rules": rules}
            pol = policy_events(sec, today)
            items.extend(pol)
            ok(SRC_FR, f"{len(pol)} policy items from {len(sec['rules'])} documents"
               + ("" if not fresh else " (via china_export)"))
        except Exception as e:  # noqa: BLE001
            fail(SRC_FR, e)

        try:
            cur, dropped = load_curated()
            items.extend(cur)
            ok(SRC_CURATED, f"{len(cur)} curated conference/launch/model/policy events"
               + (f"; {dropped} dropped (missing citation)" if dropped else ""))
        except Exception as e:  # noqa: BLE001
            fail(SRC_CURATED, e)

        final = _finalize(items, start, end)
        data["events"] = {
            "as_of": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "window": {"start": start, "end": end},
            "sources": sources,
            "impact_rule": IMPACT_RULE,
            "items": final,
        }
    except Exception as e:  # noqa: BLE001 -- contract: never raise
        log_fail(SRC_CURATED, f"build_events crashed: {e}"[:200])
    return data


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    repo = "/Users/quentinreyes/ai-gpu-dashboard-rm"
    out_path = ("/private/tmp/claude-501/-Users-quentinreyes/"
                "1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/events_sample.json")
    with open(os.path.join(repo, "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    t0 = time.time()
    d = build_events(d, log_info=lambda m: print("[INFO]", m),
                     log_ok=lambda s, m="": print("[ OK ]", s, m),
                     log_fail=lambda s, m="": print("[FAIL]", s, m))
    sec = d.get("events", {})
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"events": sec}, fh, indent=2)
    print(f"{len(sec.get('items', []))} items in {time.time() - t0:.1f}s -> {out_path}")
