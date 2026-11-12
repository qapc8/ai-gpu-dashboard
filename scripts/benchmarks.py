"""Third-party GPU rental price indices vs our own medians -> data["benchmarks"].

Sources
  Silicon Data   Public "Silicon Index" product pages (silicondata.com/products/
                 silicon-index/<gpu>) render a dated table of the last ~30 daily
                 readings server-side (e.g. ticker SDH100RT). We parse that table:
                 latest reading -> index value, all rows -> history. Live, no key.

  Ornn (OCPI)    Ornn's dashboard (dashboard.ornnai.com/compute) loads its series
                 from a JSON route the page itself calls. That route is not a
                 documented public API, so it is OFF by default and only used when
                 the owner sets BENCH_ORNN_DASHBOARD_API=1 after deciding Ornn's
                 terms allow it. Ornn's figures are otherwise visible only on the
                 dashboard, Bloomberg, or via the Kalshi/Polymarket contracts that
                 settle on it (see prediction_markets.py).

  SF Compute,    Not included: SF Compute's index page sits behind sign-in and
  SemiAnalysis   SemiAnalysis' public index page loads its numbers client-side from
                 a paid API ("talk to sales"). Both are listed in `sources` as
                 not available rather than guessed.

  Curated        scripts/curated/benchmark_indices.json (optional). Entries need
                 name/publisher/gpu_id/value_hr/as_of/url; emitted with fetched=false.

Comparison: ours_hr is the median on-demand price across providers from
data["matrix"]; diff_pct = (ours - theirs) / theirs. Index readings older than
30 days are flagged stale in the comparison.

Stdlib only.
"""

import html
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

TIMEOUT = 30
USER_AGENT = "ai-gpu-dashboard/1.0 (market research)"
HISTORY_DAYS = 400
STALE_DAYS = 30
CURATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "curated", "benchmark_indices.json")

SRC_SD = "Silicon Data"
SRC_ORNN = "Ornn"
SRC_SFC = "SF Compute"
SRC_SA = "SemiAnalysis"
SRC_CURATED = "Curated benchmark indices"
SOURCES = [SRC_SD, SRC_ORNN, SRC_CURATED]

SD_BASE = "https://www.silicondata.com/products/silicon-index/"
# page slug -> (our gpu_id, note on the mapping)
SD_PAGES = [
    ("h100", "H100-SXM", "publisher index is 'H100' (form factor not split); compared with our H100-SXM median"),
    ("h200", "H200", ""),
    ("a100", "A100-80GB", "publisher index is 'A100' (memory size not split); compared with our A100-80GB median"),
    ("b200", "B200", ""),
    ("b300", "B300", ""),
    ("mi300x", "MI300X", ""),
]

ORNN_URL = "https://dashboard.ornnai.com/compute"
ORNN_API = "https://dashboard.ornnai.com/api/gpu/{}/index-history"
ORNN_SERIES = [
    ("H100 SXM", "H100-SXM", ""),
    ("H200", "H200", ""),
    ("B200", "B200", ""),
    ("A100 SXM4", "A100-80GB", "publisher series is 'A100 SXM4' (memory size not split)"),
    ("RTX 5090", "RTX-5090", ""),
]


def _now():
    return datetime.now(timezone.utc)


def _get(url, accept="text/html"):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}")
        return resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="replace")


def _text_tokens(page):
    s = re.sub(r"<script\b.*?</script>", " ", page, flags=re.S | re.I)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "\x00", s)
    toks = [re.sub(r"\s+", " ", html.unescape(t)).strip() for t in s.split("\x00")]
    return [t for t in toks if t]


def _parse_date(s):
    return datetime.strptime(s.strip(), "%b %d, %Y").date().isoformat()


_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$")
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")


# --------------------------------------------------------------------------
# Silicon Data
# --------------------------------------------------------------------------

def parse_silicon_data(page):
    """Return (rows [(date, value)], ticker|None, caption) from an index page."""
    toks = _text_tokens(page)
    try:
        start = next(i for i, t in enumerate(toks) if t.startswith("Price (USD"))
    except StopIteration:
        raise RuntimeError("daily readings table not found")
    rows, i = [], start + 1
    while i + 1 < len(toks):
        if _DATE_RE.match(toks[i]) and _NUM_RE.match(toks[i + 1]):
            rows.append((_parse_date(toks[i]), float(toks[i + 1])))
            i += 2
        elif _DATE_RE.match(toks[i]) or toks[i] in ("/hr", ")"):
            i += 1
        else:
            break
    if len(rows) < 5:
        raise RuntimeError(f"only {len(rows)} readings parsed")
    caption = next((t for t in toks[i:i + 5] if "Source: Silicon Data" in t), "")
    ticker = None
    m = re.search(r"Daily readings of ticker ([A-Z0-9]+)", " ".join(toks))
    if m:
        ticker = m.group(1)
    rows.sort()
    for _, v in rows:
        if not (0.05 <= v <= 100):
            raise RuntimeError(f"implausible reading {v}")
    return rows, ticker, caption


def fetch_silicon_data(log_info):
    out = []
    for slug, gpu_id, map_note in SD_PAGES:
        url = SD_BASE + slug
        try:
            rows, ticker, caption = parse_silicon_data(_get(url))
        except Exception as e:
            log_info(f"Silicon Data {slug}: {e}")
            continue
        name = f"Silicon Data {ticker}" if ticker else f"Silicon Data {slug.upper()} index"
        note = "; ".join(x for x in [
            (caption or "daily standardized rental index (neo-cloud providers and marketplaces)").rstrip("."),
            map_note] if x)
        out.append({"name": name, "publisher": SRC_SD, "gpu_id": gpu_id,
                    "value_hr": round(rows[-1][1], 2), "as_of": rows[-1][0],
                    "url": url, "method_note": note, "fetched": True,
                    "_series": rows})
    if not out:
        raise RuntimeError("no Silicon Data index page could be parsed")
    return out


# --------------------------------------------------------------------------
# Ornn (opt-in)
# --------------------------------------------------------------------------

def fetch_ornn(log_info):
    end = _now().date()
    start = end - timedelta(days=85)  # Ornn's public tier exposes ~3 months
    out = []
    for series, gpu_id, map_note in ORNN_SERIES:
        q = urllib.parse.urlencode({"startDate": start.isoformat(), "endDate": end.isoformat()})
        url = ORNN_API.format(urllib.parse.quote(series)) + "?" + q
        try:
            js = json.loads(_get(url, accept="application/json"))
            pts = sorted((p["timestamp"][:10], float(p["index_value"]))
                         for p in (js.get("data") or [])
                         if p.get("timestamp") and p.get("index_value") is not None)
        except Exception as e:
            log_info(f"Ornn {series}: {e}")
            continue
        if not pts:
            continue
        name = f"Ornn OCPI {series}"
        out.append({"name": name, "publisher": SRC_ORNN, "gpu_id": gpu_id,
                    "value_hr": round(pts[-1][1], 2), "as_of": pts[-1][0],
                    "url": ORNN_URL,
                    "method_note": "; ".join(x for x in [
                        "Ornn Compute Price Index, daily, built from executed rental transactions "
                        "(per Ornn methodology)", map_note] if x),
                    "fetched": True, "_series": pts})
    if not out:
        raise RuntimeError("no Ornn series returned data")
    return out


# --------------------------------------------------------------------------
# curated
# --------------------------------------------------------------------------

def load_curated(path=CURATED_PATH):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f).get("entries") or []
    out = []
    for e in entries:
        for k in ("name", "publisher", "gpu_id", "value_hr", "as_of", "url"):
            if e.get(k) in (None, ""):
                raise ValueError(f"curated entry missing {k}: {e}")
        out.append({"name": e["name"], "publisher": e["publisher"], "gpu_id": e["gpu_id"],
                    "value_hr": float(e["value_hr"]), "as_of": e["as_of"], "url": e["url"],
                    "method_note": e.get("method_note", "curated reference value"),
                    "fetched": False})
    return out


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _matrix_medians(data):
    out = {}
    for row in data.get("matrix") or []:
        gid, med = row.get("gpu_id"), row.get("median_price")
        if gid and isinstance(med, (int, float)) and med > 0:
            out[gid] = (float(med), row.get("num_providers"))
    return out


def _merge_history(hist, name, series, cutoff):
    pts = {p["date"]: p["value_hr"] for p in hist.get(name, []) if p.get("date")}
    for d, v in series:
        pts[d] = round(v, 4)
    hist[name] = [{"date": d, "value_hr": v} for d, v in sorted(pts.items()) if d >= cutoff]


def build_benchmarks(data, log_info=print, log_ok=print, log_fail=print):
    try:
        prev = data.get("benchmarks") or {}
        prev_indices = prev.get("indices") or []
        now = _now()
        today = now.date()
        sources, indices = {}, []

        def keep_prev(publisher):
            return [dict(x, stale=True) for x in prev_indices if x.get("publisher") == publisher]

        # Silicon Data (live)
        try:
            sd = fetch_silicon_data(log_info)
            indices.extend(sd)
            sources[SRC_SD] = {"ok": True, "url": SD_BASE + "h100",
                               "note": f"{len(sd)} index pages parsed (latest daily reading + ~30-day table)"}
            log_ok(SRC_SD, ", ".join(f"{x['name']}={x['value_hr']}" for x in sd))
        except Exception as e:
            log_fail(SRC_SD, str(e)[:200])
            kept = keep_prev(SRC_SD)
            indices.extend(kept)
            sources[SRC_SD] = {"ok": False, "url": SD_BASE + "h100",
                               "note": f"fetch failed ({str(e)[:120]}); kept {len(kept)} previous values as stale"}

        # Ornn (opt-in)
        if os.environ.get("BENCH_ORNN_DASHBOARD_API") == "1":
            try:
                orn = fetch_ornn(log_info)
                indices.extend(orn)
                sources[SRC_ORNN] = {"ok": True, "url": ORNN_URL,
                                     "note": f"{len(orn)} series from the dashboard's data route"}
                log_ok(SRC_ORNN, ", ".join(f"{x['name']}={x['value_hr']}" for x in orn))
            except Exception as e:
                log_fail(SRC_ORNN, str(e)[:200])
                kept = keep_prev(SRC_ORNN)
                indices.extend(kept)
                sources[SRC_ORNN] = {"ok": False, "url": ORNN_URL,
                                     "note": f"fetch failed ({str(e)[:120]}); kept {len(kept)} previous values as stale"}
        else:
            log_info("Ornn: skipped (dashboard data route is undocumented; set BENCH_ORNN_DASHBOARD_API=1 to enable)")
            sources[SRC_ORNN] = {"ok": False, "url": ORNN_URL,
                                 "note": "disabled: no documented public API; values are on the Ornn dashboard, "
                                         "Bloomberg, and implied by the Kalshi/Polymarket contracts that settle on it"}

        sources[SRC_SFC] = {"ok": False, "url": "https://sfcompute.com/index",
                            "note": "not public: index page requires sign-in"}
        sources[SRC_SA] = {"ok": False, "url": "https://gpu-index.semianalysis.com/",
                           "note": "not public: page loads values client-side from a paid API (Bloomberg SAH100SC)"}

        # Curated (optional file)
        if os.path.exists(CURATED_PATH):
            try:
                cur = load_curated()
                indices.extend(cur)
                sources[SRC_CURATED] = {"ok": True, "url": "scripts/curated/benchmark_indices.json",
                                        "note": f"{len(cur)} curated reference values (not live)"}
                log_ok(SRC_CURATED, f"{len(cur)} entries")
            except Exception as e:
                log_fail(SRC_CURATED, str(e)[:200])
                indices.extend(dict(x, stale=True) for x in prev_indices if not x.get("fetched"))
                sources[SRC_CURATED] = {"ok": False, "url": "scripts/curated/benchmark_indices.json",
                                        "note": f"load failed ({str(e)[:120]})"}

        # history: publisher-dated readings, merged into what we already hold
        cutoff = (today - timedelta(days=HISTORY_DAYS)).isoformat()
        hist = {k: list(v) for k, v in (prev.get("history") or {}).items()}
        for x in indices:
            if x.get("stale"):
                continue
            series = x.pop("_series", None) or [(x["as_of"], x["value_hr"])]
            _merge_history(hist, x["name"], series, cutoff)
        for x in indices:
            x.pop("_series", None)

        # comparison
        ours = _matrix_medians(data)
        comparison = []
        for x in indices:
            if x["gpu_id"] not in ours:
                continue
            o, nprov = ours[x["gpu_id"]]
            theirs = x["value_hr"]
            try:
                age = (today - datetime.fromisoformat(x["as_of"]).date()).days
            except ValueError:
                age = None
            row = {"name": x["name"], "gpu_id": x["gpu_id"],
                   "theirs_hr": round(theirs, 2), "ours_hr": round(o, 2),
                   "ours_basis": f"median on-demand across providers ({nprov} providers)",
                   "diff_pct": round((o - theirs) / theirs * 100, 1) if theirs else None,
                   "their_as_of": x["as_of"],
                   "their_age_days": age,
                   "stale": bool(x.get("stale")) or (age is not None and age > STALE_DAYS)}
            comparison.append(row)

        data["benchmarks"] = {
            "as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": sources,
            "indices": indices,
            "comparison": comparison,
            "history": hist,
        }
    except Exception as e:  # must never raise
        log_fail("benchmarks", f"build crashed: {e}")
    return data


if __name__ == "__main__":
    src = "/Users/quentinreyes/ai-gpu-dashboard-rm/data.json"
    out = ("/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-f92320644371/"
           "scratchpad/benchmarks_sample.json")
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    d = build_benchmarks(
        d,
        log_info=lambda m: print(f"  [INFO]  {m}"),
        log_ok=lambda s, m="": print(f"  [ OK ]  {s}: {m}"),
        log_fail=lambda s, m="": print(f"  [FAIL]  {s}: {m}"),
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"benchmarks": d.get("benchmarks")}, f, indent=2)
    print(f"wrote {out}")
