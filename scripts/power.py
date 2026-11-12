"""Power: the binding constraint on new GPU capacity.

Three parts, each sourced independently:

wholesale  Daily wholesale electricity prices ($/MWh) at the hubs that serve
           the big data-center clusters. Source: EIA's "Wholesale Electricity
           and Natural Gas Market Data" spreadsheets, which republish ICE
           next-day *peak-block* bilateral trades (weighted-average price per
           delivery day). No key needed. The current-year file is refreshed by
           EIA roughly every two weeks, so the series lags by ~1-3 weeks and
           only has trading days. It carries PJM Western Hub, CAISO SP15/NP15,
           Mid-C and Palo Verde. It does NOT carry an ERCOT hub, and ERCOT's
           own MIS/API is bot-protected and needs a registered account, so
           ERCOT prices are deliberately absent rather than approximated.
           Dominion (N. Virginia) zonal prices are likewise not in any free
           unauthenticated feed; PJM Western Hub is the nearest liquid proxy.

demand     Optional, needs EIA_API_KEY (free at eia.gov/opendata). EIA-930
           daily demand (GWh/day) for ERCOT, PJM and CAISO from API v2. EIA's
           API has no wholesale price route, so the key adds load, not prices.
           Without the key the part is skipped and the sources map says so.

projects   Epoch AI's "Frontier Data Centers" dataset (CC-BY 4.0), downloaded
           live as a zip of CSVs. Epoch builds per-site timelines of facility
           power from satellite imagery, permits and company disclosures, each
           row cited on its directory page. We publish, per site: planned MW
           (facility power at the last row of its timeline), online MW (latest timeline
           row on or before today), target date (first date the planned MW is
           reached), and a status derived from those two numbers:
             energized     online >= planned (fully built out)
             construction  a timeline row exists on/before today, online < planned
                           (many are partially energized: see mw_online)
             announced     no timeline row on/before today (rare: Epoch usually
                           starts a timeline at land clearing)
           Totals: energized_mw = sum of mw_online over all listed sites;
           construction_mw = sum of (mw - mw_online) for construction sites;
           announced_mw = sum of mw for announced sites.
           These are Epoch's estimates, not utility meter data.

queues     Curated, scripts/curated/power_queues.json: interconnection queue and
           large-load request figures (LBNL Queued Up, ERCOT, PJM IMM, Dominion),
           each with source, url and as_of verified by hand. Metrics differ by
           row and are not additive.

On a fetch failure each part keeps its previous contents from `data` and is
marked stale. No number in this module is modelled or guessed by us.
"""

import csv
import io
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta, timezone

UA = {"User-Agent": "ai-gpu-dashboard/1.0 (market research)"}
TIMEOUT = 60

EIA_WHOLESALE_PAGE = "https://www.eia.gov/electricity/wholesale/"
EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
EPOCH_ZIP = "https://epoch.ai/data/data_centers/data_centers.zip"
EPOCH_PAGE = "https://epoch.ai/data/ai-data-centers"
EPOCH_DIR = "https://epoch.ai/data/ai-data-centers/directory/"
CURATED_QUEUES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curated", "power_queues.json")

SRC_EIA_ICE = "EIA wholesale (ICE)"
SRC_EIA_API = "EIA API v2 (EIA-930 demand)"
SRC_EPOCH = "Epoch AI data centers"
SRC_QUEUES = "Curated interconnection queues"
SOURCES = [SRC_EIA_ICE, SRC_EIA_API, SRC_EPOCH, SRC_QUEUES]

# hub key -> (ICE hub name prefix in the EIA sheet, display label)
HUBS = [
    ("pjm_west", "PJM WH Real Time Peak", "PJM Western Hub (N. Virginia / Mid-Atlantic proxy), RT peak"),
    ("caiso_sp15", "SP15 EZ Gen DA LMP Peak", "CAISO SP15 (Southern California), DA peak"),
    ("caiso_np15", "NP15 EZ Gen DA LMP Peak", "CAISO NP15 (Northern California), DA peak"),
    ("mid_c", "Mid C Peak", "Mid-Columbia (Pacific Northwest), peak"),
    ("palo_verde", "Palo Verde Peak", "Palo Verde (Arizona / Phoenix), peak"),
]

# EIA-930 balancing authority -> (timezone facet, label)
DEMAND_BAS = [("ERCO", "Central", "ERCOT"), ("PJM", "Eastern", "PJM"), ("CISO", "Pacific", "CAISO")]

HISTORY_DAYS = 400
WHOLESALE_DAYS = 366
MIN_PROJECT_MW = 400  # planned facility power threshold for "major" projects

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
                "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
STATE_NAMES = {"mississippi": "MS", "texas": "TX", "virginia": "VA", "louisiana": "LA", "ohio": "OH",
               "indiana": "IN", "georgia": "GA", "wisconsin": "WI", "tennessee": "TN", "new mexico": "NM",
               "michigan": "MI", "pennsylvania": "PA", "north dakota": "ND", "oklahoma": "OK", "iowa": "IA",
               "nebraska": "NE", "arizona": "AZ", "kentucky": "KY", "alabama": "AL", "utah": "UT",
               "nevada": "NV", "oregon": "OR", "new york": "NY", "wyoming": "WY", "minnesota": "MN"}


# --------------------------------------------------------------------------
# http helpers
# --------------------------------------------------------------------------

def _get(url, params=None, timeout=TIMEOUT):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}")
        return resp.read()


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# EIA / ICE wholesale spreadsheets
# --------------------------------------------------------------------------

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_rows(blob):
    """Minimal stdlib reader for the first worksheet of an .xlsx. Yields lists of cell values."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(_NS + "t")))
    sheet = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))[0]
    for row in ET.fromstring(z.read(sheet)).iter(_NS + "row"):
        out = {}
        for c in row.findall(_NS + "c"):
            col = re.match(r"[A-Z]+", c.get("r", "")).group(0)
            v = c.find(_NS + "v")
            t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(_NS + "t"))
            elif v is None:
                val = None
            elif t == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
            out[col] = val
        yield out


def _excel_date(v):
    return (date(1899, 12, 30) + timedelta(days=int(float(v)))).isoformat()


def _ice_file_urls(today):
    """Find this year's and last year's ICE electric spreadsheet links on EIA's page."""
    page = _get(EIA_WHOLESALE_PAGE).decode("utf-8", "replace")
    links = re.findall(r'href="([^"]*ice_electric-(\d{4})(?:final)?\.xlsx)"', page)
    if not links:
        raise RuntimeError("no ice_electric-*.xlsx links on EIA wholesale page")
    want = {today.year, today.year - 1}
    urls = {}
    for href, year in links:
        y = int(year)
        if y in want and y not in urls:
            urls[y] = urllib.parse.urljoin(EIA_WHOLESALE_PAGE, href)
    if not urls:
        raise RuntimeError(f"no ICE files for {sorted(want)} on EIA page")
    return [urls[y] for y in sorted(urls)]


def fetch_ice_wholesale(today):
    """Return ({hub_key: [{"date","value"}]}, [file urls]). Raises on failure."""
    series = {k: {} for k, _, _ in HUBS}
    files = _ice_file_urls(today)
    for url in files:
        rows = _xlsx_rows(_get(url))
        header = next(rows)
        cols = {(v or "").strip().lower().replace("\n", " "): k for k, v in header.items()}
        c_hub = cols.get("price hub")
        c_del = next((k for h, k in cols.items() if h.startswith("delivery start")), None)
        c_avg = next((k for h, k in cols.items() if h.startswith("wtd avg price")), None)
        if not (c_hub and c_del and c_avg):
            raise RuntimeError(f"unexpected header in {url}: {list(cols)}")
        for r in rows:
            hub = (r.get(c_hub) or "").strip().lower()
            if not hub or r.get(c_avg) in (None, "") or r.get(c_del) in (None, ""):
                continue
            for key, prefix, _ in HUBS:
                if hub == prefix.lower():
                    try:
                        series[key][_excel_date(r[c_del])] = round(float(r[c_avg]), 2)
                    except (TypeError, ValueError):
                        pass
    cutoff = (today - timedelta(days=WHOLESALE_DAYS)).isoformat()
    out = {k: [{"date": d, "value": v} for d, v in sorted(s.items()) if d >= cutoff] for k, s in series.items()}
    if not any(out.values()):
        raise RuntimeError("no hub rows parsed from ICE spreadsheets")
    return out, files


# --------------------------------------------------------------------------
# EIA API v2 demand (optional key)
# --------------------------------------------------------------------------

def fetch_eia_demand(api_key, ba, tz, today):
    start = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    params = [("api_key", api_key), ("frequency", "daily"), ("data[0]", "value"),
              ("facets[respondent][]", ba), ("facets[type][]", "D"), ("facets[timezone][]", tz),
              ("start", start), ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
              ("length", "5000")]
    body = json.loads(_get(EIA_API_URL + "?" + urllib.parse.urlencode(params)))
    if "error" in body:
        raise RuntimeError(str(body["error"])[:200])
    rows = (body.get("response") or {}).get("data") or []
    pts = []
    for r in rows:
        try:
            pts.append({"date": r["period"], "value": round(float(r["value"]) / 1000.0, 1)})  # MWh -> GWh
        except (KeyError, TypeError, ValueError):
            continue
    if not pts:
        raise RuntimeError(f"no demand rows for {ba}")
    return pts


# --------------------------------------------------------------------------
# Epoch AI data centers
# --------------------------------------------------------------------------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _clean_tag(s):
    return ", ".join(p.split("#")[0].strip() for p in (s or "").split(",") if p.split("#")[0].strip())


def _region(country, address):
    if (country or "").strip() != "United States":
        return "Intl"
    addr = (address or "").replace("\n", " ")
    found = [m for m in re.findall(r"\b([A-Z]{2})\b", addr) if m in US_STATES]
    st = found[-1] if found else None
    if not st:
        low = addr.lower()
        st = next((code for name, code in STATE_NAMES.items() if name in low), None)
    return f"US-{st}" if st else "US"


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_epoch_projects(today):
    z = zipfile.ZipFile(io.BytesIO(_get(EPOCH_ZIP)))
    names = {os.path.basename(n): n for n in z.namelist()}
    if "data_centers.csv" not in names or "data_center_timelines.csv" not in names:
        raise RuntimeError(f"unexpected zip contents: {sorted(names)}")
    dcs = {r["Name"]: r for r in csv.DictReader(io.TextIOWrapper(z.open(names["data_centers.csv"]), "utf-8"))}
    tls = {}
    for r in csv.DictReader(io.TextIOWrapper(z.open(names["data_center_timelines.csv"]), "utf-8")):
        tls.setdefault(r["Data center"], []).append(r)
    t = today.isoformat()
    projects = []
    for name, rows in tls.items():
        rows.sort(key=lambda r: r["Date"])
        # Planned = the end state of the timeline (its last row). Fall back to the
        # peak if the last row is zero (e.g. a decommissioning note).
        planned = _num(rows[-1].get("Power (MW)")) or max(_num(r.get("Power (MW)")) for r in rows)
        if planned < MIN_PROJECT_MW:
            continue
        past = [r for r in rows if r["Date"] <= t]
        online = _num(past[-1].get("Power (MW)")) if past else 0.0
        target = next((r["Date"] for r in rows if _num(r.get("Power (MW)")) >= planned * 0.999), rows[-1]["Date"])
        if not past:
            status = "announced"
        elif online >= planned * 0.999:
            status = "energized"
        else:
            status = "construction"
        dc = dcs.get(name, {})
        projects.append({
            "name": name,
            "company": _clean_tag(dc.get("Owner")) or None,
            "users": _clean_tag(dc.get("Users")) or None,
            "location": (dc.get("Address") or "").replace("\n", " ").strip() or dc.get("Country") or None,
            "country": dc.get("Country") or None,
            "region": _region(dc.get("Country"), (dc.get("Address") or "") + " " + name),
            "project": _clean_tag(dc.get("Project")) or None,
            "mw": round(planned),
            "mw_online": round(online),
            "status": status,
            "target_date": target,
            "source": "Epoch AI, Frontier Data Centers (CC-BY 4.0)",
            "url": EPOCH_DIR + _slug(name),
            "as_of": past[-1]["Date"] if past else rows[0]["Date"],
        })
    if not projects:
        raise RuntimeError("no projects above threshold in Epoch data")
    projects.sort(key=lambda p: -p["mw"])
    return projects


def _totals(projects):
    return {
        "announced_mw": round(sum(p["mw"] for p in projects if p["status"] == "announced")),
        "construction_mw": round(sum(p["mw"] - p.get("mw_online", 0) for p in projects
                                     if p["status"] == "construction")),
        "energized_mw": round(sum(p.get("mw_online", p["mw"] if p["status"] == "energized" else 0)
                                  for p in projects)),
        "n_projects": len(projects),
        "method": ("energized_mw = online facility MW summed over all listed sites (incl. partially energized); "
                   "construction_mw = planned minus online for sites still building; announced_mw = planned MW "
                   f"of sites with no construction yet. Sites with planned facility power >= {MIN_PROJECT_MW} MW."),
    }


# --------------------------------------------------------------------------
# curated queues
# --------------------------------------------------------------------------

def load_queues():
    with open(CURATED_QUEUES) as f:
        entries = json.load(f)["entries"]
    out = []
    for e in entries:
        for k in ("iso", "metric", "value_gw", "as_of", "source", "url"):
            if e.get(k) in (None, ""):
                raise RuntimeError(f"curated queue entry missing {k}: {e}")
        out.append({k: e[k] for k in ("iso", "metric", "value_gw", "as_of", "source", "url")}
                   | {k: e[k] for k in ("published", "note") if e.get(k)})
    return out


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_power(data, log_info=print, log_ok=lambda *a: None, log_fail=lambda *a: None):
    try:
        return _build(data, log_info, log_ok, log_fail)
    except Exception as e:  # never raise
        log_fail(SRC_EIA_ICE, f"unexpected error in build_power: {e}")
        return data


def _build(data, log_info, log_ok, log_fail):
    now = _now()
    today = now.date()
    prev = data.get("power") or {}
    sources = {}
    section = {"as_of": now.isoformat(timespec="seconds").replace("+00:00", "Z"), "sources": sources}

    # ---- wholesale --------------------------------------------------------
    prev_ws = prev.get("wholesale") or {}
    wholesale = {}
    try:
        series, files = fetch_ice_wholesale(today)
        n_ok = 0
        for key, prefix, label in HUBS:
            pts = series.get(key) or []
            if pts:
                n_ok += 1
                wholesale[key] = {"label": label, "unit": "$/MWh", "source": "EIA / ICE next-day peak index",
                                  "url": EIA_WHOLESALE_PAGE, "basis": "ICE weighted-average peak-block price, "
                                  "by delivery date (trading days only)", "points": pts, "stale": False}
            elif key in prev_ws:
                wholesale[key] = dict(prev_ws[key], stale=True)
        latest = max((w["points"][-1]["date"] for w in wholesale.values() if w.get("points")), default=None)
        log_ok(SRC_EIA_ICE, f"{n_ok}/{len(HUBS)} hubs, latest delivery date {latest}")
        sources[SRC_EIA_ICE] = {"ok": True, "url": EIA_WHOLESALE_PAGE,
                                "note": f"{n_ok}/{len(HUBS)} hubs from {', '.join(os.path.basename(f) for f in files)}; "
                                        f"EIA refreshes ~biweekly, latest delivery date {latest}. No ERCOT hub in this "
                                        "file; ERCOT's own feeds are bot-protected / need registration, so ERCOT "
                                        "prices are omitted."}
    except Exception as e:
        log_fail(SRC_EIA_ICE, str(e))
        sources[SRC_EIA_ICE] = {"ok": False, "url": EIA_WHOLESALE_PAGE, "note": f"fetch failed: {e}; previous kept"}
        wholesale = {k: dict(v, stale=True) for k, v in prev_ws.items()}
    section["wholesale"] = wholesale

    # ---- demand (optional key) -------------------------------------------
    key = os.environ.get("EIA_API_KEY", "").strip()
    prev_dm = prev.get("demand") or {}
    demand = {}
    if not key:
        log_info("power: EIA_API_KEY not set; skipping EIA-930 demand")
        sources[SRC_EIA_API] = {"ok": False, "url": "https://www.eia.gov/opendata/",
                                "note": "skipped: EIA_API_KEY not set (free key). Adds daily demand for ERCOT/PJM/CAISO."}
        demand = {k: dict(v, stale=True) for k, v in prev_dm.items()}
    else:
        errs = []
        for ba, tz, label in DEMAND_BAS:
            try:
                pts = fetch_eia_demand(key, ba, tz, today)
                demand[label.lower()] = {"label": f"{label} daily demand", "unit": "GWh/day",
                                         "source": "EIA-930 via EIA API v2", "url": "https://www.eia.gov/opendata/",
                                         "points": pts, "stale": False}
            except Exception as e:
                errs.append(f"{ba}: {e}")
                if label.lower() in prev_dm:
                    demand[label.lower()] = dict(prev_dm[label.lower()], stale=True)
        n = sum(1 for v in demand.values() if not v.get("stale"))
        if n:
            log_ok(SRC_EIA_API, f"{n}/{len(DEMAND_BAS)} balancing authorities" + (f"; {errs}" if errs else ""))
        else:
            log_fail(SRC_EIA_API, "; ".join(errs))
        sources[SRC_EIA_API] = {"ok": bool(n), "url": "https://www.eia.gov/opendata/",
                                "note": f"{n}/{len(DEMAND_BAS)} BAs" + (f"; failed: {'; '.join(errs)}" if errs else "")}
    section["demand"] = demand

    # ---- queues (curated) -------------------------------------------------
    try:
        section["queues"] = load_queues()
        log_ok(SRC_QUEUES, f"{len(section['queues'])} entries")
        sources[SRC_QUEUES] = {"ok": True, "url": "scripts/curated/power_queues.json",
                               "note": "hand-verified figures; each row cites its source and as_of"}
    except Exception as e:
        log_fail(SRC_QUEUES, str(e))
        sources[SRC_QUEUES] = {"ok": False, "url": "scripts/curated/power_queues.json", "note": str(e)}
        section["queues"] = prev.get("queues") or []

    # ---- projects (Epoch AI) ----------------------------------------------
    try:
        projects = fetch_epoch_projects(today)
        section["projects"] = projects
        section["projects_stale"] = False
        log_ok(SRC_EPOCH, f"{len(projects)} sites >= {MIN_PROJECT_MW} MW planned")
        sources[SRC_EPOCH] = {"ok": True, "url": EPOCH_PAGE,
                              "note": "Epoch AI Frontier Data Centers dataset, CC-BY 4.0; facility-power estimates "
                                      "from satellite imagery and disclosures"}
    except Exception as e:
        log_fail(SRC_EPOCH, str(e))
        sources[SRC_EPOCH] = {"ok": False, "url": EPOCH_PAGE, "note": f"fetch failed: {e}; previous kept"}
        section["projects"] = prev.get("projects") or []
        section["projects_stale"] = True
    section["totals"] = _totals(section["projects"])

    data["power"] = section
    return data


if __name__ == "__main__":
    with open("/Users/quentinreyes/ai-gpu-dashboard-rm/data.json") as f:
        d = json.load(f)
    build_power(d, print, lambda s, m="": print("OK  ", s, m), lambda s, m="": print("FAIL", s, m))
    out = "/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/scratchpad/power_sample.json"
    with open(out, "w") as f:
        json.dump({"power": d.get("power")}, f, indent=1)
    print("wrote", out)
