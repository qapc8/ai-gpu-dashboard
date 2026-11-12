"""Upstream AI-hardware supply signals.

Builds ``data["supply_signals"]``:

* **Taiwan monthly revenue (live)** -- TSMC plus the AI-server ODMs Hon Hai
  (Foxconn), Quanta, Wistron and Wiwynn.  The latest month comes from the TWSE
  OpenAPI (``t187ap05_L``, listed-company monthly revenue, keyless JSON).  The
  OpenAPI only carries the latest month, so history is back-filled from the
  official MOPS monthly-revenue archive pages (one static Big5 HTML page per
  month covering every listed company) and accumulated inside the section; on
  later runs only months not yet held are fetched.  Values are NT$ billions;
  ``yoy_pct`` is the year-over-year change the companies themselves report.
* **Micron quarterly revenue (live)** -- SEC EDGAR XBRL companyfacts
  (``RevenueFromContractWithCustomerExcludingAssessedTax``); discrete quarters
  derived from YTD/annual facts (Q4 = FY - 9M).  ``period`` is the calendar
  quarter of the fiscal quarter's middle month; ``fiscal_period`` keeps Micron's.
* **Taiwan exports (live)** -- Ministry of Finance "exports by major commodity"
  CSV (data.gov.tw dataset 8380, keyless): electronic components (mostly ICs;
  the HS-8542 line itself needs the customs query portal) and ICT products
  (computers/servers), USD mn, YoY computed from the series itself.
* **Curated** -- ``scripts/curated/supply_signals.json``: SK hynix and Samsung
  DS quarterly revenue (from their earnings releases; Korea's DART API needs a
  key and returns filings rather than a clean series) and point estimates for
  TSMC CoWoS capacity and HBM market share.  Every entry carries source, url and
  as_of.

A failed source keeps the previous points for its series (``stale: true``) and
logs a failure; nothing is guessed or zero-filled.  Stdlib only.
"""

import html
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import date, datetime, timedelta, timezone

TIMEOUT = 45
UA = "ai-gpu-dashboard/1.0 (market research)"
SEC_UA = "ai-gpu-dashboard research bot admin@ai-gpu-dashboard.invalid"
HERE = os.path.dirname(os.path.abspath(__file__))
CURATED_PATH = os.path.join(HERE, "curated", "supply_signals.json")

TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
MOPS_URL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc}_{m}_0.html"
MOPS_PAGE = "https://mopsov.twse.com.tw/mops/web/t21sc04_ifrs"
EDGAR_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MU_CIK = 723125
MU_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
FILING_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}

TW_COMPANIES = {
    # series key: (TWSE code, label)
    "tsmc_revenue": ("2330", "TSMC monthly revenue"),
    "honhai_revenue": ("2317", "Hon Hai (Foxconn) monthly revenue"),
    "quanta_revenue": ("2382", "Quanta Computer monthly revenue"),
    "wistron_revenue": ("3231", "Wistron monthly revenue"),
    "wiwynn_revenue": ("6669", "Wiwynn monthly revenue"),
}
MONTHS_BACK = 25          # back-fill target (24-month research window + 1)
MAX_MONTHLY_POINTS = 36
MAX_MOPS_FETCHES = 30
MOPS_WORKERS = 6
MOPS_BUDGET_S = 75

SOURCE_TWSE = "TWSE OpenAPI monthly revenue"
SOURCE_MOPS = "MOPS monthly revenue archive"
SOURCE_MU = "SEC EDGAR XBRL (Micron revenue)"
SOURCE_MOF = "Taiwan MOF export statistics"
SOURCE_CURATED = "Curated supply estimates"
SOURCES = [SOURCE_TWSE, SOURCE_MOPS, SOURCE_MU, SOURCE_MOF, SOURCE_CURATED]

# Ministry of Finance "exports by major commodity" (data.gov.tw dataset 8380),
# keyless UTF-8 CSV.  ym/ymt are ROC year+month; codlst0 selects all columns.
MOF_URL = ("https://web02.mof.gov.tw/njswww/webMain.aspx?sys=220&ym={ym}&ymt={ymt}&kind=21&type=4"
           "&funid=i8121&cycle=1&outmode=12&compmode=00&outkind=1&fld0=1&codlst0=" + "1" * 120 + "&utf=1")
MOF_PAGE = "https://data.gov.tw/dataset/8380"
MOF_SERIES = {
    # series key: (column header suffix, label)
    "tw_exports_electronic_components": ("(1)電子零組件",
                                         "Taiwan exports: electronic components (mostly ICs)"),
    "tw_exports_ict_products": ("(4)資通與視聽產品",
                                "Taiwan exports: information, communication & AV products (incl. servers)"),
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _get_bytes(url, ua=UA, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                if getattr(resp, "status", 200) != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return resp.read()
        except Exception as e:  # noqa: BLE001 - retried then re-raised
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def _num(s):
    s = (s or "").replace(",", "").strip()
    if s in ("", "-", "N/A"):
        return None
    return float(s)


def _ym_add(ym, n):
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + n
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


# --------------------------------------------------------------------------
# Taiwan monthly revenue
# --------------------------------------------------------------------------

def fetch_twse_latest():
    """-> (period 'YYYY-MM', {code: {"value_k": thousand NTD, "yoy_pct": float}})."""
    rows = json.loads(_get_bytes(TWSE_URL).decode("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("TWSE OpenAPI returned no rows")
    want = {code for code, _ in TW_COMPANIES.values()}
    out, period = {}, None
    for r in rows:
        code = r.get("公司代號")
        if code not in want:
            continue
        ym = r.get("資料年月", "")
        if len(ym) < 4:
            continue
        p = f"{int(ym[:-2]) + 1911:04d}-{int(ym[-2:]):02d}"
        period = period or p
        if p != period:
            raise RuntimeError(f"mixed periods in TWSE data: {p} vs {period}")
        val = _num(r.get("營業收入-當月營收"))
        if val is None or val <= 0:
            continue
        out[code] = {"value_k": val, "yoy_pct": _num(r.get("營業收入-去年同月增減(%)"))}
    if "2330" not in out:
        raise RuntimeError("TSMC (2330) missing from TWSE OpenAPI response")
    return period, out


_MOPS_ROW = re.compile(
    r">\s*(\d{4})\s*</td>\s*<td[^>]*>[^<]*</td>"          # code, name
    r"\s*<td[^>]*>([^<]*)</td>"                             # current month
    r"\s*<td[^>]*>[^<]*</td>"                               # previous month
    r"\s*<td[^>]*>([^<]*)</td>"                             # same month last year
    r"\s*<td[^>]*>[^<]*</td>"                               # MoM %
    r"\s*<td[^>]*>([^<]*)</td>",                            # YoY %
    re.I)


def fetch_mops_month(period):
    """-> {code: {"value_k", "yoy_pct"}} for one month, or {} if not yet published."""
    y, m = int(period[:4]), int(period[5:7])
    raw = _get_bytes(MOPS_URL.format(roc=y - 1911, m=m))
    text = raw.decode("big5", errors="replace")
    want = {code for code, _ in TW_COMPANIES.values()}
    out = {}
    for code, cur, ly, yoy in _MOPS_ROW.findall(text):
        if code in want:
            v = _num(html.unescape(cur))
            if v is not None and v > 0:
                out[code] = {"value_k": v, "yoy_pct": _num(html.unescape(yoy))}
    return out


def fetch_mof_exports(months=36):
    """-> {series_key: [{"period", "value"}]} monthly USD mn from the MOF CSV."""
    import csv
    import io
    today = date.today()
    end = f"{today.year:04d}-{today.month:02d}"
    start = _ym_add(end, -(months + 12))

    def roc(ym):
        return f"{int(ym[:4]) - 1911}{ym[5:7]}"

    text = _get_bytes(MOF_URL.format(ym=roc(start), ymt=roc(end))).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 13:
        raise RuntimeError(f"MOF CSV too short ({len(rows)} rows)")
    header = rows[0]
    cols = {}
    for key, (suffix, _) in MOF_SERIES.items():
        idx = [i for i, h in enumerate(header) if h.startswith("按美元計算") and h.endswith(suffix)]
        if len(idx) != 1:
            raise RuntimeError(f"MOF column {suffix!r} not found exactly once")
        cols[key] = idx[0]
    out = {k: [] for k in MOF_SERIES}
    for r in rows[1:]:
        m = re.match(r"\s*(\d+)年\s*(\d+)月", r[0] if r else "")
        if not m:
            continue
        period = f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}"
        for key, i in cols.items():
            v = _num(r[i]) if i < len(r) else None
            if v is not None:
                out[key].append({"period": period, "value": round(v, 1)})
    for key, pts in out.items():
        if len(pts) < 13:
            raise RuntimeError(f"MOF {key}: only {len(pts)} months")
        byp = {p["period"]: p["value"] for p in pts}
        for p in pts:
            ya = byp.get(_ym_add(p["period"], -12))
            p["yoy_pct"] = round((p["value"] / ya - 1) * 100, 1) if ya else None
        out[key] = sorted(pts, key=lambda p: p["period"])[-months:]
    return out


# --------------------------------------------------------------------------
# EDGAR quarterly (Micron)
# --------------------------------------------------------------------------

def _d(s):
    return date.fromisoformat(s)


def _nominal_month_end(end):
    if end.day <= 7:  # 52/53-week years end a few days into the next month
        end = end.replace(day=1) - timedelta(days=1)
    return end


def calendar_quarter(end):
    e = _nominal_month_end(end)
    mid_month = (e.month - 2) % 12 + 1
    year = e.year - 1 if e.month == 1 else e.year
    return f"{year}-Q{(mid_month - 1) // 3 + 1}"


def fiscal_label(end, fye_month):
    e = _nominal_month_end(end)
    q = ((e.month - fye_month - 1) % 12) // 3 + 1
    fy = e.year if e.month <= fye_month else e.year + 1
    return f"FY{fy} Q{q}"


def edgar_quarterly(cik, tags):
    doc = json.loads(_get_bytes(EDGAR_FACTS.format(cik=cik), ua=SEC_UA).decode("utf-8"))
    gaap = doc.get("facts", {}).get("us-gaap", {})
    facts, fy_months = {}, {}
    for tag in tags:
        for f in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if "start" not in f or f.get("form") not in FILING_FORMS:
                continue
            key = (f["start"], f["end"])
            prev = facts.get(key)
            if prev is None or (prev["tag"] == tag and f["filed"] > prev["filed"]):
                facts[key] = dict(f, tag=tag)
            if f.get("fp") == "FY" and 350 <= (_d(f["end"]) - _d(f["start"])).days <= 380:
                mm = _nominal_month_end(_d(f["end"])).month
                fy_months[mm] = fy_months.get(mm, 0) + 1
    fye = max(fy_months, key=fy_months.get) if fy_months else 12
    by_start = {}
    for (s, e), f in facts.items():
        by_start.setdefault(s, []).append((e, f))
    q = {}
    for (s, e), f in facts.items():
        if 80 <= (_d(e) - _d(s)).days <= 100:
            q[e] = (f["val"], f)
    for s, items in by_start.items():
        for e, f in items:
            if e in q or (_d(e) - _d(s)).days <= 100:
                continue
            for e2, f2 in items:
                if 80 <= (_d(e) - _d(e2)).days <= 100:
                    q[e] = (f["val"] - f2["val"], f)
                    break
    if not q:
        raise RuntimeError(f"no quarterly facts for CIK {cik}")
    pts = {}
    for e in sorted(q):
        val, f = q[e]
        pts[calendar_quarter(_d(e))] = {
            "period": calendar_quarter(_d(e)), "fiscal_period": fiscal_label(_d(e), fye),
            "end": e, "value": round(val / 1e9, 3), "form": f["form"], "filed": f["filed"]}
    keys = sorted(pts)
    for k in keys:
        y, qq = int(k[:4]), k[-1]
        ya = pts.get(f"{y - 1}-Q{qq}")
        pts[k]["yoy_pct"] = (round((pts[k]["value"] / ya["value"] - 1) * 100, 1)
                             if ya and ya["value"] > 0 else None)
    return [pts[k] for k in keys][-12:]


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_supply_signals(data, log_info=lambda m: None, log_ok=lambda a, b="": None,
                         log_fail=lambda a, b="": None):
    try:
        return _build(data, log_info, log_ok, log_fail)
    except Exception as e:  # noqa: BLE001 - must never raise
        log_fail(SOURCE_TWSE, f"supply_signals build crashed: {e}")
        return data


def _build(data, log_info, log_ok, log_fail):
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    prev = data.get("supply_signals") or {}
    prev_series = prev.get("series") or {}
    sources, series = {}, {}

    # ---- Taiwan monthly revenue: previous points + TWSE latest + MOPS back-fill ----
    held = {key: {p["period"]: p for p in (prev_series.get(key) or {}).get("points", [])}
            for key in TW_COMPANIES}
    code_to_key = {code: key for key, (code, _) in TW_COMPANIES.items()}
    fresh = {key: False for key in TW_COMPANIES}

    def put(period, code, rec, src):
        key = code_to_key[code]
        held[key][period] = {"period": period, "value": round(rec["value_k"] / 1e6, 2),
                             "yoy_pct": None if rec["yoy_pct"] is None else round(rec["yoy_pct"], 1),
                             "src": src}
        fresh[key] = True

    latest = None
    try:
        latest, recs = fetch_twse_latest()
        for code, rec in recs.items():
            put(latest, code, rec, "twse")
        sources[SOURCE_TWSE] = {"ok": True, "url": TWSE_URL,
                                "note": f"{latest}: {len(recs)}/{len(TW_COMPANIES)} companies"}
        log_ok(SOURCE_TWSE, sources[SOURCE_TWSE]["note"])
    except Exception as e:  # noqa: BLE001
        sources[SOURCE_TWSE] = {"ok": False, "url": TWSE_URL, "note": str(e)}
        log_fail(SOURCE_TWSE, str(e))

    # back-fill months missing for TSMC (one MOPS page covers all companies)
    anchor = latest or max(held["tsmc_revenue"], default=None)
    if anchor is None:
        today = date.today()
        anchor = _ym_add(f"{today.year:04d}-{today.month:02d}", -1)
    wanted = [_ym_add(anchor, -i) for i in range(MONTHS_BACK)]
    missing = [p for p in wanted if p not in held["tsmc_revenue"]]
    fetched, errors, empty = 0, [], []
    # MOPS pages are slow (~15 s each), so fetch in parallel within a time budget.
    todo = sorted(missing, reverse=True)[:MAX_MOPS_FETCHES]
    results = {}
    if todo:
        pool = ThreadPoolExecutor(max_workers=MOPS_WORKERS)
        futs = {pool.submit(fetch_mops_month, p): p for p in todo}
        try:
            for fut in as_completed(futs, timeout=MOPS_BUDGET_S):
                p = futs[fut]
                try:
                    results[p] = fut.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{p}: {e}")
        except FuturesTimeout:
            pending = sorted(p for f, p in futs.items() if not f.done())
            log_info(f"MOPS: time budget hit, {len(pending)} months deferred to next run")
        pool.shutdown(wait=False, cancel_futures=True)
    for p in sorted(results):
        recs = results[p]
        if not recs:
            empty.append(p)
            continue
        for code, rec in recs.items():
            cur = held[code_to_key[code]].get(p)
            if cur is None or cur.get("src") != "twse":
                put(p, code, rec, "mops")
        fetched += 1
    mops_ok = not errors
    note = f"{fetched} month pages fetched, {len(missing)} missing before run"
    if empty:
        note += f"; not yet published/empty: {', '.join(sorted(empty))}"
    if errors:
        note += "; errors: " + "; ".join(errors[:3])
    sources[SOURCE_MOPS] = {"ok": bool(mops_ok), "url": MOPS_PAGE, "note": note}
    (log_ok if mops_ok else log_fail)(SOURCE_MOPS, note)

    for key, (code, label) in TW_COMPANIES.items():
        pts = [dict((k, v) for k, v in held[key][p].items() if k != "src") for p in sorted(held[key])]
        pts = pts[-MAX_MONTHLY_POINTS:]
        if not pts:
            continue
        series[key] = {
            "label": label, "unit": "NT$ bn", "freq": "monthly",
            "source": "TWSE OpenAPI (latest month) + MOPS monthly revenue archive",
            "url": TWSE_URL, "ticker": f"{code}.TW", "points": pts,
            "stale": not fresh[key],
        }

    # ---- Micron quarterly revenue ----
    try:
        mpts = edgar_quarterly(MU_CIK, MU_TAGS)
        series["micron_revenue"] = {
            "label": "Micron quarterly revenue", "unit": "USD bn", "freq": "quarterly",
            "source": "SEC EDGAR XBRL companyfacts (RevenueFromContractWithCustomerExcludingAssessedTax)",
            "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=723125&type=10-Q",
            "points": mpts, "stale": False,
            "note": "period = calendar quarter of the fiscal quarter's middle month",
        }
        last = mpts[-1]
        sources[SOURCE_MU] = {"ok": True, "url": EDGAR_FACTS.format(cik=MU_CIK),
                              "note": f"{len(mpts)} quarters, latest {last['fiscal_period']} = ${last['value']}bn"}
        log_ok(SOURCE_MU, sources[SOURCE_MU]["note"])
    except Exception as e:  # noqa: BLE001
        if "micron_revenue" in prev_series:
            series["micron_revenue"] = dict(prev_series["micron_revenue"], stale=True)
        sources[SOURCE_MU] = {"ok": False, "url": EDGAR_FACTS.format(cik=MU_CIK), "note": str(e)}
        log_fail(SOURCE_MU, str(e))

    # ---- Taiwan exports (MOF) ----
    try:
        mof = fetch_mof_exports()
        for key, pts in mof.items():
            series[key] = {"label": MOF_SERIES[key][1], "unit": "USD mn", "freq": "monthly",
                           "source": "Taiwan Ministry of Finance, exports by major commodity",
                           "url": MOF_PAGE, "points": pts, "stale": False}
        last = mof["tw_exports_electronic_components"][-1]
        sources[SOURCE_MOF] = {"ok": True, "url": MOF_PAGE,
                               "note": f"latest {last['period']}: electronic components ${last['value']}mn"}
        log_ok(SOURCE_MOF, sources[SOURCE_MOF]["note"])
    except Exception as e:  # noqa: BLE001
        for key in MOF_SERIES:
            if key in prev_series:
                series[key] = dict(prev_series[key], stale=True)
        sources[SOURCE_MOF] = {"ok": False, "url": MOF_PAGE, "note": str(e)}
        log_fail(SOURCE_MOF, str(e))

    # ---- curated: Korean memory makers + capacity / share estimates ----
    estimates = prev.get("estimates") or []
    try:
        with open(CURATED_PATH, encoding="utf-8") as fh:
            cur = json.load(fh)
        new_est = []
        for e in cur.get("estimates", []):
            for k in ("metric", "value", "unit", "period", "source", "url", "as_of"):
                if e.get(k) in (None, ""):
                    raise RuntimeError(f"estimate missing {k}: {e}")
            new_est.append({k: e[k] for k in ("metric", "value", "unit", "period", "source", "url", "as_of")}
                           | {k: e[k] for k in ("value_high", "note") if e.get(k) not in (None, "")})
        for key, s in (cur.get("quarterly") or {}).items():
            pts = []
            for p in s.get("points", []):
                for k in ("period", "value", "source", "url"):
                    if p.get(k) in (None, ""):
                        raise RuntimeError(f"{key} point missing {k}: {p}")
                pts.append(dict(p))
            pts.sort(key=lambda p: p["period"])
            byp = {p["period"]: p for p in pts}
            for p in pts:
                ya = byp.get(f"{int(p['period'][:4]) - 1}{p['period'][4:]}")
                if p.get("yoy_pct") is None:
                    p["yoy_pct"] = (round((p["value"] / ya["value"] - 1) * 100, 1)
                                    if ya and ya["value"] else None)
            series[key] = {"label": s["label"], "unit": s["unit"], "freq": "quarterly",
                           "source": s.get("source") or "Company earnings releases (curated)",
                           "url": pts[-1]["url"] if pts else "", "points": pts,
                           "stale": False, "curated": True,
                           "as_of": s.get("as_of")}
        estimates = new_est
        sources[SOURCE_CURATED] = {"ok": True, "url": "scripts/curated/supply_signals.json",
                                   "note": f"{len(new_est)} estimates, {len(cur.get('quarterly') or {})} curated series; each cited"}
        log_ok(SOURCE_CURATED, sources[SOURCE_CURATED]["note"])
    except Exception as e:  # noqa: BLE001
        for key, s in prev_series.items():
            if s.get("curated") and key not in series:
                series[key] = dict(s, stale=True)
        sources[SOURCE_CURATED] = {"ok": False, "url": "scripts/curated/supply_signals.json", "note": str(e)}
        log_fail(SOURCE_CURATED, str(e))

    data["supply_signals"] = {"as_of": now, "sources": sources, "series": series,
                              "estimates": estimates}
    return data


if __name__ == "__main__":
    repo = os.path.dirname(HERE)
    with open(os.path.join(repo, "data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    t0 = time.time()
    build_supply_signals(d, log_info=lambda m: print("INFO", m),
                         log_ok=lambda a, b="": print("OK  ", a, b),
                         log_fail=lambda a, b="": print("FAIL", a, b))
    out = "/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/supply_signals_sample.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"supply_signals": d.get("supply_signals")}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out} in {time.time() - t0:.1f}s")
