"""Hyperscaler & neocloud capex, plus GPU-financing / credit-stress signals.

Builds ``data["capex"]``:

* **Capex (live)** -- SEC EDGAR XBRL ``companyfacts`` for MSFT, GOOGL, AMZN,
  META, ORCL and CRWV (CoreWeave).  The line is the cash-flow statement's
  purchases of property & equipment: ``PaymentsToAcquirePropertyPlantAndEquipment``
  (Amazon tags its "Purchases of property and equipment" as
  ``PaymentsToAcquireProductiveAssets``).  Where a company also reports
  ``RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability`` for the same
  quarter (MSFT, AMZN, ...) it is shown separately as ``finance_leases_bn`` and
  in ``capex_incl_finance_leases_bn``; ``capex_usd_bn`` and every total stay on
  the cash line so all companies are on the same basis.
  10-Q cash-flow values are year-to-date, so discrete quarters are derived:
  a ~3-month duration fact is used directly, otherwise Q = YTD(end) - YTD(prior
  quarter end) for two facts that share a start date (so Q4 = FY - 9M).
  The latest-filed value wins for each (start, end) pair (restatements).
  Quarters are labelled by calendar quarter of the period's middle month (Oracle's
  Jun-Aug quarter is 2026-Q3); ``fiscal_period`` keeps the company's own label.
  NBIS (Nebius) files 20-F/6-K and its XBRL in companyfacts stops at FY2023, so it
  is excluded.
* **Neocloud debt (live)** -- CoreWeave total debt from the same XBRL facts
  (``LongTermDebt`` incl. current portion, or ``DebtInstrumentCarryingAmount``
  when that is the only tag carried to the latest balance-sheet date).
* **Credit stress (live)** -- ICE BofA US High Yield OAS (FRED BAMLH0A0HYM2,
  keyless CSV), last ~2 years, weekly (Friday or last obs of the week).
* **Curated** -- ``scripts/curated/capex_guidance.json`` (latest capex guidance
  per company) and ``scripts/curated/gpu_financing_deals.json`` (GPU-backed /
  AI-datacenter debt deals).  Every entry carries source, url and as_of/date.

Any fetch failure raises inside the fetcher; the build keeps the previous
contents of that sub-part (marked ``stale``) and logs a failure.  Nothing is
ever guessed or zero-filled.  Stdlib only.
"""

import csv
import io
import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

TIMEOUT = 45
SEC_UA = "ai-gpu-dashboard research bot admin@ai-gpu-dashboard.invalid"
UA = "ai-gpu-dashboard/1.0 (market research)"
HERE = os.path.dirname(os.path.abspath(__file__))
CURATED_DIR = os.path.join(HERE, "curated")

EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EDGAR_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q"
FRED_HY_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2&cosd={start}"
FRED_HY_PAGE = "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"

COMPANIES = {
    # ticker: (cik, display name, capex tags in priority order)
    "MSFT": (789019, "Microsoft", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "GOOGL": (1652044, "Alphabet", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "AMZN": (1018724, "Amazon", ["PaymentsToAcquireProductiveAssets",
                                 "PaymentsToAcquirePropertyPlantAndEquipment"]),
    "META": (1326801, "Meta Platforms", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "ORCL": (1341439, "Oracle", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
    "CRWV": (1769628, "CoreWeave", ["PaymentsToAcquirePropertyPlantAndEquipment"]),
}
FINANCE_LEASE_TAG = "RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability"
DEBT_TAGS = ["LongTermDebt", "DebtInstrumentCarryingAmount"]
N_QUARTERS = 12
FILING_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}

SOURCE_EDGAR = "SEC EDGAR XBRL (capex)"
SOURCE_DEBT = "SEC EDGAR XBRL (CoreWeave debt)"
SOURCE_FRED = "FRED ICE BofA HY OAS"
SOURCE_GUIDANCE = "Curated capex guidance"
SOURCE_DEALS = "Curated GPU financing deals"
SOURCES = [SOURCE_EDGAR, SOURCE_DEBT, SOURCE_FRED, SOURCE_GUIDANCE, SOURCE_DEALS]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _get(url, ua=UA, retries=2):
    headers = {"Accept": "*/*", "Accept-Encoding": "identity"}
    if ua:
        headers["User-Agent"] = ua
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                if getattr(resp, "status", 200) != 200:
                    raise RuntimeError(f"HTTP {resp.status} from {url}")
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - retried then re-raised
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


_last_sec_call = [0.0]


def sec_companyfacts(cik):
    """Fetch EDGAR companyfacts JSON; SEC asks for <=10 req/s and a descriptive UA."""
    wait = 0.15 - (time.time() - _last_sec_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_sec_call[0] = time.time()
    text = _get(EDGAR_FACTS.format(cik=cik), ua=SEC_UA)
    try:
        return json.loads(text)
    except ValueError as e:
        raise RuntimeError(f"invalid JSON from EDGAR CIK {cik}: {e}") from e


# --------------------------------------------------------------------------
# XBRL quarter derivation (shared logic; supply_signals has its own copy)
# --------------------------------------------------------------------------

def _d(s):
    return date.fromisoformat(s)


def _nominal_month_end(end):
    """52/53-week fiscal periods end a few days into the next month (e.g. Sep 3)."""
    if end.day <= 7:
        end = end.replace(day=1) - timedelta(days=1)
    return end


def calendar_quarter(end):
    """Calendar quarter of the middle month of the 3-month period ending at ``end``."""
    e = _nominal_month_end(end)
    mid_month = (e.month - 2) % 12 + 1  # January end -> December middle month
    year = e.year - 1 if e.month == 1 else e.year
    return f"{year}-Q{(mid_month - 1) // 3 + 1}"


def _fiscal_year_end_month(facts_list):
    months = {}
    for f in facts_list:
        if f.get("fp") == "FY" and "start" in f:
            span = (_d(f["end"]) - _d(f["start"])).days
            if 350 <= span <= 380:
                m = _nominal_month_end(_d(f["end"])).month
                months[m] = months.get(m, 0) + 1
    return max(months, key=months.get) if months else 12


def fiscal_label(end, fye_month):
    e = _nominal_month_end(end)
    q = ((e.month - fye_month - 1) % 12) // 3 + 1
    fy = e.year if e.month <= fye_month else e.year + 1
    return f"FY{fy} Q{q}"


def _usd_facts(gaap, tags):
    """Merge duration facts across tags (priority order) -> {(start,end): fact}."""
    out = {}
    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        for f in node.get("units", {}).get("USD", []):
            if "start" not in f or f.get("form") not in FILING_FORMS:
                continue
            key = (f["start"], f["end"])
            prev = out.get(key)
            if prev is None:
                out[key] = dict(f, tag=tag)
            elif prev["tag"] == tag and f.get("filed", "") > prev.get("filed", ""):
                out[key] = dict(f, tag=tag)  # restated / later filing wins
    return out


def discrete_quarters(gaap, tags):
    """{end: {"value", "start", "form", "filed", "tag", "derived"}} for ~3-month periods."""
    facts = _usd_facts(gaap, tags)
    by_start = {}
    for (s, e), f in facts.items():
        by_start.setdefault(s, []).append((e, f))
    quarters = {}
    # 1) direct quarterly facts
    for (s, e), f in facts.items():
        span = (_d(e) - _d(s)).days
        if 80 <= span <= 100:
            quarters[e] = {"value": f["val"], "start": s, "form": f["form"],
                           "filed": f["filed"], "tag": f["tag"], "derived": False}
    # 2) YTD differences for ends without a direct quarterly fact
    for s, items in by_start.items():
        items.sort()
        for e, f in items:
            if e in quarters:
                continue
            span = (_d(e) - _d(s)).days
            if span <= 100:
                continue
            for e2, f2 in items:
                gap = (_d(e) - _d(e2)).days
                if 80 <= gap <= 100:
                    quarters[e] = {"value": f["val"] - f2["val"],
                                   "start": (_d(e2) + timedelta(days=1)).isoformat(),
                                   "form": f["form"], "filed": f["filed"],
                                   "tag": f["tag"], "derived": True}
                    break
    return quarters


# --------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------

def fetch_company_capex(ticker):
    cik, name, tags = COMPANIES[ticker]
    doc = sec_companyfacts(cik)
    gaap = doc.get("facts", {}).get("us-gaap", {})
    capq = discrete_quarters(gaap, tags)
    if not capq:
        raise RuntimeError(f"{ticker}: no capex facts for tags {tags}")
    all_facts = []
    for t in tags:
        all_facts += gaap.get(t, {}).get("units", {}).get("USD", [])
    fye = _fiscal_year_end_month(all_facts)
    leaseq = discrete_quarters(gaap, [FINANCE_LEASE_TAG]) if FINANCE_LEASE_TAG in gaap else {}

    rows = []
    for end in sorted(capq):
        q = capq[end]
        if q["value"] < 0:
            raise RuntimeError(f"{ticker}: negative derived capex at {end}")
        row = {
            "period": calendar_quarter(_d(end)),
            "fiscal_period": fiscal_label(_d(end), fye),
            "start": q["start"],
            "end": end,
            "capex_usd_bn": round(q["value"] / 1e9, 3),
            "form": q["form"],
            "filed": q["filed"],
            "derived_from_ytd": q["derived"],
        }
        lq = leaseq.get(end)
        if lq is not None and lq["value"] >= 0:
            row["finance_leases_bn"] = round(lq["value"] / 1e9, 3)
            row["capex_incl_finance_leases_bn"] = round((q["value"] + lq["value"]) / 1e9, 3)
        rows.append(row)
    # one row per calendar quarter (latest end wins if a fiscal calendar shifts)
    dedup = {}
    for r in rows:
        dedup[r["period"]] = r
    rows = [dedup[k] for k in sorted(dedup)][-N_QUARTERS:]
    return {"cik": cik, "name": name, "tags": tags, "fiscal_year_end_month": fye,
            "quarters": rows, "_gaap": gaap}


def coreweave_debt(gaap):
    """Latest total debt (incl. current portion) from CoreWeave's XBRL facts."""
    best = None
    for tag in DEBT_TAGS:
        pts = {}
        for f in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if "start" in f or f.get("form") not in FILING_FORMS:
                continue
            if f["end"] not in pts or f["filed"] > pts[f["end"]]["filed"]:
                pts[f["end"]] = f
        if not pts:
            continue
        last_end = max(pts)
        if best is None or last_end > best["as_of"]:
            hist = [{"date": e, "value_bn": round(pts[e]["val"] / 1e9, 3)} for e in sorted(pts)][-8:]
            f = pts[last_end]
            best = {"total_debt_bn": round(f["val"] / 1e9, 3), "as_of": last_end,
                    "tag": tag, "form": f["form"], "filed": f["filed"], "history": hist}
    if best is None:
        raise RuntimeError("CRWV: no debt facts")
    return best


def fetch_hy_oas(years=2):
    start = (date.today() - timedelta(days=int(365 * years) + 7)).isoformat()
    url = FRED_HY_CSV.format(start=start)
    # FRED's CDN drops requests carrying some custom User-Agents (HTTP/2 reset /
    # timeout), so try the library default UA first, then our descriptive one.
    text, err = None, None
    for ua in (None, UA):
        try:
            text = _get(url, ua=ua, retries=1)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if text is None:
        raise RuntimeError(f"FRED fetch failed: {err}")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][-1] != "BAMLH0A0HYM2":
        raise RuntimeError("unexpected FRED CSV header")
    daily = []
    for r in rows[1:]:
        if len(r) < 2 or r[1] in ("", "."):
            continue
        daily.append((r[0], float(r[1])))
    if len(daily) < 50:
        raise RuntimeError(f"only {len(daily)} HY OAS observations")
    weekly = {}
    for ds, v in daily:  # last observation of each ISO week
        y, w, _ = _d(ds).isocalendar()
        weekly[(y, w)] = (ds, v)
    pts = [{"date": ds, "value": v} for ds, v in sorted(weekly.values())]
    if pts[-1]["date"] != daily[-1][0]:
        pts.append({"date": daily[-1][0], "value": daily[-1][1]})
    return pts


def load_curated(name):
    path = os.path.join(CURATED_DIR, name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _check_cited(entry, date_key):
    for k in ("source", "url", date_key):
        if not entry.get(k):
            raise RuntimeError(f"curated entry missing {k}: {entry}")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_capex(data, log_info=lambda m: None, log_ok=lambda a, b="": None,
                log_fail=lambda a, b="": None):
    try:
        return _build(data, log_info, log_ok, log_fail)
    except Exception as e:  # noqa: BLE001 - must never raise
        log_fail(SOURCE_EDGAR, f"capex build crashed: {e}")
        return data


def _build(data, log_info, log_ok, log_fail):
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prev = data.get("capex") or {}
    prev_companies = prev.get("companies") or {}
    prev_fin = prev.get("financing") or {}
    sources = {}

    # ---- guidance (curated) ----
    guidance = {}
    try:
        g = load_curated("capex_guidance.json")
        for tk, entry in (g.get("guidance") or {}).items():
            _check_cited(entry, "as_of")
            guidance[tk] = {k: entry[k] for k in ("text", "source", "url", "as_of")}
        sources[SOURCE_GUIDANCE] = {"ok": True, "url": "scripts/curated/capex_guidance.json",
                                    "note": f"{len(guidance)} companies, hand-verified with citations"}
        log_ok(SOURCE_GUIDANCE, f"{len(guidance)} companies")
    except Exception as e:  # noqa: BLE001
        sources[SOURCE_GUIDANCE] = {"ok": False, "url": "scripts/curated/capex_guidance.json", "note": str(e)}
        log_fail(SOURCE_GUIDANCE, str(e))
        guidance = {tk: c.get("guidance") for tk, c in prev_companies.items() if c.get("guidance")}

    # ---- capex per company (EDGAR) ----
    companies, failed, crwv_gaap = {}, [], None
    for tk in COMPANIES:
        try:
            res = fetch_company_capex(tk)
            if tk == "CRWV":
                crwv_gaap = res["_gaap"]
            qs = res["quarters"]
            companies[tk] = {
                "name": res["name"],
                "cik": res["cik"],
                "xbrl_tags": res["tags"],
                "quarters": qs,
                "stale": False,
                "url": EDGAR_BROWSE.format(cik=res["cik"]),
            }
        except Exception as e:  # noqa: BLE001
            failed.append(f"{tk}: {e}")
            if tk in prev_companies:
                companies[tk] = dict(prev_companies[tk], stale=True)
        if tk in companies:
            companies[tk]["guidance"] = guidance.get(tk)
            _company_stats(companies[tk])
    ok_n = sum(1 for c in companies.values() if not c.get("stale"))
    note = f"{ok_n}/{len(COMPANIES)} companies live"
    if failed:
        note += "; failed: " + "; ".join(failed)
    sources[SOURCE_EDGAR] = {"ok": ok_n > 0 and not failed,
                             "url": "https://data.sec.gov/api/xbrl/companyfacts/",
                             "note": note + ". Cash purchases of PP&E, discrete quarters derived from YTD."}
    (log_ok if not failed else log_fail)(SOURCE_EDGAR, note)

    total = _total_quarters(companies)

    # ---- CoreWeave debt ----
    neocloud_debt = dict(prev_fin.get("neocloud_debt") or {})
    try:
        if crwv_gaap is None:
            crwv_gaap = sec_companyfacts(COMPANIES["CRWV"][0]).get("facts", {}).get("us-gaap", {})
        debt = coreweave_debt(crwv_gaap)
        debt.update({"source": f"SEC EDGAR XBRL ({debt['tag']}, {debt['form']} filed {debt['filed']})",
                     "url": EDGAR_BROWSE.format(cik=COMPANIES["CRWV"][0]), "stale": False})
        neocloud_debt["CRWV"] = debt
        sources[SOURCE_DEBT] = {"ok": True, "url": EDGAR_FACTS.format(cik=COMPANIES["CRWV"][0]),
                                "note": f"{debt['total_debt_bn']}bn at {debt['as_of']} ({debt['tag']})"}
        log_ok(SOURCE_DEBT, sources[SOURCE_DEBT]["note"])
    except Exception as e:  # noqa: BLE001
        if "CRWV" in neocloud_debt:
            neocloud_debt["CRWV"] = dict(neocloud_debt["CRWV"], stale=True)
        sources[SOURCE_DEBT] = {"ok": False, "url": EDGAR_FACTS.format(cik=COMPANIES["CRWV"][0]), "note": str(e)}
        log_fail(SOURCE_DEBT, str(e))

    # ---- HY OAS ----
    hy_prev = prev_fin.get("hy_oas")
    try:
        pts = fetch_hy_oas()
        hy = {"label": "ICE BofA US High Yield Index Option-Adjusted Spread",
              "unit": "%", "source": "ICE Data Indices via FRED (BAMLH0A0HYM2)",
              "url": FRED_HY_PAGE, "freq": "weekly (last obs of week)",
              "points": pts, "stale": False}
        sources[SOURCE_FRED] = {"ok": True, "url": FRED_HY_PAGE,
                                "note": f"{len(pts)} weekly points, latest {pts[-1]['date']} = {pts[-1]['value']}%"}
        log_ok(SOURCE_FRED, sources[SOURCE_FRED]["note"])
    except Exception as e:  # noqa: BLE001
        hy = dict(hy_prev, stale=True) if hy_prev else None
        sources[SOURCE_FRED] = {"ok": False, "url": FRED_HY_PAGE, "note": str(e)}
        log_fail(SOURCE_FRED, str(e))

    # ---- deals (curated) ----
    try:
        raw = load_curated("gpu_financing_deals.json")
        deals = []
        for d in raw.get("deals", []):
            _check_cited(d, "date")
            deals.append({k: d.get(k) for k in
                          ("date", "company", "amount_usd_bn", "type", "collateral", "source", "url")})
        deals.sort(key=lambda d: d["date"], reverse=True)
        sources[SOURCE_DEALS] = {"ok": True, "url": "scripts/curated/gpu_financing_deals.json",
                                 "note": f"{len(deals)} deals, hand-verified with citations"}
        log_ok(SOURCE_DEALS, f"{len(deals)} deals")
    except Exception as e:  # noqa: BLE001
        deals = prev_fin.get("deals") or []
        sources[SOURCE_DEALS] = {"ok": False, "url": "scripts/curated/gpu_financing_deals.json", "note": str(e)}
        log_fail(SOURCE_DEALS, str(e))

    data["capex"] = {
        "as_of": now,
        "sources": sources,
        "method": ("capex_usd_bn = cash purchases of property & equipment from the cash-flow "
                   "statement (XBRL PaymentsToAcquirePropertyPlantAndEquipment; Amazon "
                   "PaymentsToAcquireProductiveAssets). Discrete quarters derived from YTD "
                   "filings (Q4 = FY - 9M). finance_leases_bn = right-of-use assets obtained "
                   "for finance-lease liabilities (lease-note XBRL tag), shown separately where "
                   "tagged; it can differ from management's own 'capex incl. finance leases' "
                   "figure, so totals use the cash line only. period = calendar quarter of the fiscal quarter's middle month. "
                   "yoy_pct = latest quarter vs same calendar quarter a year earlier; "
                   "ttm_yoy_pct = trailing 4 quarters vs prior 4."),
        "companies": companies,
        "total_quarters": total,
        "financing": {"hy_oas": hy, "deals": deals, "neocloud_debt": neocloud_debt},
    }
    return data


def _company_stats(c):
    qs = c.get("quarters") or []
    vals = {q["period"]: q["capex_usd_bn"] for q in qs}
    periods = sorted(vals)
    c["ttm_bn"] = c["yoy_pct"] = c["ttm_yoy_pct"] = None
    c["latest_period"] = periods[-1] if periods else None
    if not periods:
        return

    def shift(p, n):
        y, q = int(p[:4]), int(p[-1])
        idx = y * 4 + (q - 1) - n
        return f"{idx // 4}-Q{idx % 4 + 1}"

    last = periods[-1]
    ttm_keys = [shift(last, i) for i in range(4)]
    if all(k in vals for k in ttm_keys):
        c["ttm_bn"] = round(sum(vals[k] for k in ttm_keys), 2)
        prior = [shift(last, i) for i in range(4, 8)]
        if all(k in vals for k in prior):
            p = sum(vals[k] for k in prior)
            if p > 0:
                c["ttm_yoy_pct"] = round((c["ttm_bn"] / p - 1) * 100, 1)
    ya = shift(last, 4)
    if ya in vals and vals[ya] > 0:
        c["yoy_pct"] = round((vals[last] / vals[ya] - 1) * 100, 1)


def _total_quarters(companies):
    """Sum only calendar quarters where every included company has a value."""
    included = [c for c in companies.values() if c.get("quarters")]
    if not included:
        return []
    sets = [{q["period"]: q["capex_usd_bn"] for q in c["quarters"]} for c in included]
    common = set(sets[0])
    for s in sets[1:]:
        common &= set(s)
    return [{"period": p, "capex_usd_bn": round(sum(s[p] for s in sets), 2),
             "n_companies": len(sets), "companies": sorted(
                 tk for tk, c in companies.items() if c.get("quarters"))}
            for p in sorted(common)]


if __name__ == "__main__":
    repo = os.path.dirname(HERE)
    with open(os.path.join(repo, "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    t0 = time.time()
    build_capex(d, log_info=lambda m: print("INFO", m),
                log_ok=lambda a, b="": print("OK  ", a, b),
                log_fail=lambda a, b="": print("FAIL", a, b))
    out = "/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/capex_sample.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"capex": d.get("capex")}, fh, indent=2)
    print(f"wrote {out} in {time.time() - t0:.1f}s")
