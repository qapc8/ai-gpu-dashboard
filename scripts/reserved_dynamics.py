"""How reserved (committed) GPU prices move over time -> data["reserved_dynamics"].

Published 1- and 3-year rates are sticky price lists; most committed capacity
is negotiated privately. So this section reads dynamics from the places where
they are observable:

  term_curve      GetDeploying's weekly median price by contract length
                  (on-demand, 1/3/6/12/24/36 months), per GPU, with the number
                  of providers behind each point. Already fetched by
                  reserved_market; reshaped here into a curve over time.
  discounts       Our own daily record of each provider's committed rate
                  against the same provider's on-demand rate (price_history
                  term_days / days). A shrinking discount means providers need
                  to give less to win a commitment: the market is tightening.
  short_term      Daily history of short-term capacity prices (AWS Capacity
                  Blocks, GCP DWS, Lambda/Together reservations), accumulated
                  from reserved_market.short_term. AWS reprices Capacity
                  Blocks with supply and demand, so this is the most liquid
                  public reserved price.
  rate_changes    Every change in a committed or short-term rate we observed.
  backlog         Remaining performance obligations (contracted, not yet
                  delivered revenue) from SEC XBRL: CoreWeave (almost all GPU
                  capacity), Oracle and Microsoft (company-wide, includes
                  non-GPU business). Backlog growth is signed reserved demand.
  contracts       Curated log of disclosed multi-year compute contracts, with
                  an implied $/GPU-hr only where value, term and GPU count are
                  all disclosed (scripts/curated/compute_contracts.json).

Nothing is estimated here beyond the arithmetic stated on each field.
"""

import json
import os
from datetime import date, datetime, timedelta, timezone
from statistics import median

SOURCES = ["Reserved Dynamics", "SEC EDGAR XBRL (backlog)", "Curated compute contracts"]

TOP = ["B300", "GB200", "B200", "H200", "H100-SXM", "MI300X"]
CURVE_TERMS = [("on_demand", "On-demand", 0), ("reserved_1mo", "1 mo", 1), ("reserved_3mo", "3 mo", 3),
               ("reserved_6mo", "6 mo", 6), ("reserved_12mo", "12 mo", 12),
               ("reserved_24mo", "24 mo", 24), ("reserved_36mo", "36 mo", 36)]
BACKLOG_COS = {  # ticker: (cik, name, scope note)
    "CRWV": (1769628, "CoreWeave", "almost entirely GPU cloud capacity"),
    "ORCL": (1341439, "Oracle", "company-wide; OCI AI capacity is the main driver of recent growth"),
    "MSFT": (789019, "Microsoft", "company-wide commercial backlog, includes Azure and software"),
}
HISTORY_DAYS = 400
MIN_CHANGE = 0.005      # rate moves under 0.5% are rounding in the source, not repricing
MIN_PROVIDERS = 3       # a weekly median needs this many providers before its trend is reported
CURATED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curated", "compute_contracts.json")
EDGAR_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q"


def _trim(points):
    cutoff = (date.today() - timedelta(days=HISTORY_DAYS)).isoformat()
    return [p for p in points if p["date"] >= cutoff]


def _upsert(points, point):
    points = [p for p in points if p["date"] != point["date"]]
    points.append(point)
    return _trim(sorted(points, key=lambda p: p["date"]))


def _term_label(t):
    return {"1yr": "1-year", "3yr": "3-year", "5yr": "5-year", "short": "short-term"}.get(t, t)


def build_reserved_dynamics(data, log_info, log_ok, log_fail):
    prev = data.get("reserved_dynamics") or {}
    sec = {"as_of": datetime.now(timezone.utc).isoformat(), "sources": {}}
    try:
        _derived(data, prev, sec)
        sec["sources"]["Reserved Dynamics"] = {"ok": True, "url": "", "note": "derived from data already collected"}
        log_ok("Reserved Dynamics", f"{len(sec['term_curve'])} GPU term curves, "
                                    f"{len(sec['rate_changes'])} rate changes")
    except Exception as exc:
        for k in ("term_curve", "discounts", "short_term", "rate_changes", "summary"):
            if k in prev:
                sec[k] = prev[k]
        sec["sources"]["Reserved Dynamics"] = {"ok": False, "url": "", "note": str(exc)}
        log_fail("Reserved Dynamics", str(exc))
    try:
        sec["backlog"] = _backlog(prev.get("backlog") or {})
        sec["sources"]["SEC EDGAR XBRL (backlog)"] = {
            "ok": True, "url": "https://data.sec.gov/api/xbrl/companyfacts/",
            "note": "RevenueRemainingPerformanceObligation, quarter-end balances from 10-Q/10-K"}
        log_ok("SEC EDGAR XBRL (backlog)", ", ".join(f"{k} ${v['points'][-1]['value_bn']:.0f}bn"
                                                     for k, v in sec["backlog"].items() if v.get("points")))
    except Exception as exc:
        sec["backlog"] = {k: dict(v, stale=True) for k, v in (prev.get("backlog") or {}).items()}
        sec["sources"]["SEC EDGAR XBRL (backlog)"] = {"ok": False, "url": "https://data.sec.gov/", "note": str(exc)}
        log_fail("SEC EDGAR XBRL (backlog)", str(exc))
    try:
        with open(CURATED) as f:
            cur = json.load(f)
        items = [i for i in cur.get("items", []) if i.get("url") and i.get("date")]
        sec["contracts"] = {"as_of": cur.get("as_of"), "method": cur.get("method"),
                            "items": sorted(items, key=lambda i: i["date"], reverse=True)}
        sec["sources"]["Curated compute contracts"] = {"ok": True, "url": "", "note": f"{len(items)} cited contracts"}
        log_ok("Curated compute contracts", f"{len(items)} contracts")
    except Exception as exc:
        if prev.get("contracts"):
            sec["contracts"] = prev["contracts"]
        sec["sources"]["Curated compute contracts"] = {"ok": False, "url": "", "note": str(exc)}
        log_fail("Curated compute contracts", str(exc))
    data["reserved_dynamics"] = sec
    return data


def _derived(data, prev, sec):
    rm = data.get("reserved_market") or {}
    ph = data.get("price_history") or {}
    today = date.today().isoformat()

    # ---- term curve over time (GetDeploying weekly medians) ----
    gd = ((rm.get("indexes") or {}).get("getdeploying") or {}).get("series") or {}
    curve = {}
    for gpu, series in gd.items():
        weeks = sorted({p[0] for k, _, _ in CURVE_TERMS for p in (series.get(k) or [])})
        if not weeks:
            continue
        rows = []
        for w in weeks:
            row = {"week": w}
            for k, _, _ in CURVE_TERMS:
                hit = next((p for p in (series.get(k) or []) if p[0] == w), None)
                if hit:
                    row[k] = hit[1]
                    row[k + "_n"] = hit[2]
            od = row.get("on_demand")
            for k, _, _ in CURVE_TERMS[1:]:
                if od and row.get(k):
                    row[k + "_disc"] = round((1 - row[k] / od) * 100, 1)
            rows.append(row)
        curve[gpu] = rows
    sec["term_curve"] = curve
    sec["term_curve_terms"] = [{"key": k, "label": l, "months": m} for k, l, m in CURVE_TERMS]
    sec["term_curve_note"] = ("GetDeploying weekly median across whichever providers list each contract "
                              "length that week (n = providers). With 1-2 providers a point is one "
                              "provider's list price, so read thin terms with care. CC BY 4.0.")

    # ---- our daily committed-vs-on-demand discounts ----
    days, tdays = ph.get("days") or {}, ph.get("term_days") or {}
    disc = {}
    for d in sorted(tdays):
        for prov, gpus in (tdays[d] or {}).items():
            for gpu, terms in (gpus or {}).items():
                od = ((days.get(d) or {}).get(prov) or {}).get(gpu)
                for term, rate in (terms or {}).items():
                    if not rate:
                        continue
                    key = f"{prov}|{term}"
                    disc.setdefault(gpu, {}).setdefault(key, []).append({
                        "date": d, "rate": rate, "on_demand": od,
                        "discount_pct": round((1 - rate / od) * 100, 1) if od else None})
    sec["discounts"] = disc

    # ---- short-term price history (accumulated) ----
    st_prev = prev.get("short_term") or {}
    st = {k: list(v) for k, v in st_prev.items()}
    for r in rm.get("short_term") or []:
        key = "|".join([r["provider"], r.get("product") or "", r["gpu"], r.get("term") or ""])
        st[key] = _upsert(st.get(key, []), {"date": today, "rate": r["rate"],
                                            "on_demand": r.get("on_demand_same_provider"),
                                            "discount_pct": r.get("discount_pct")})
    sec["short_term"] = st

    # ---- rate changes ----
    changes = []
    for gpu, series in disc.items():
        for key, pts in series.items():
            prov, term = key.split("|", 1)
            for a, b in zip(pts, pts[1:]):
                if abs(b["rate"] / a["rate"] - 1) >= MIN_CHANGE:
                    changes.append({"date": b["date"], "kind": "committed", "provider": prov, "gpu": gpu,
                                    "term": _term_label(term), "from": a["rate"], "to": b["rate"],
                                    "change_pct": round((b["rate"] / a["rate"] - 1) * 100, 1)})
    for key, pts in st.items():
        prov, product, gpu, term = key.split("|", 3)
        for a, b in zip(pts, pts[1:]):
            if abs(b["rate"] / a["rate"] - 1) >= MIN_CHANGE:
                changes.append({"date": b["date"], "kind": "short_term", "provider": prov, "gpu": gpu,
                                "term": f"{product} ({term})" if product else term, "from": a["rate"],
                                "to": b["rate"], "change_pct": round((b["rate"] / a["rate"] - 1) * 100, 1)})
    changes.sort(key=lambda c: (c["date"], abs(c["change_pct"])), reverse=True)
    sec["rate_changes"] = changes[:150]

    # ---- per-GPU summary: where the discount is now and how it moved ----
    summary = {}
    for gpu in TOP:
        rows = curve.get(gpu) or []
        s = {}
        if rows:
            last = rows[-1]
            base = next((r for r in reversed(rows) if r["week"] <= _weeks_back(last["week"], 12)), None)
            s["week"] = last["week"]
            for k, label, _ in CURVE_TERMS[1:]:
                if last.get(k + "_disc") is not None:
                    s[k] = {"label": label, "price": last.get(k), "discount_pct": last[k + "_disc"],
                            "n": last.get(k + "_n"),
                            "discount_change_pp_12w": (round(last[k + "_disc"] - base[k + "_disc"], 1)
                                                       if base and base.get(k + "_disc") is not None
                                                       and min(last.get(k + "_n") or 0, base.get(k + "_n") or 0,
                                                               last.get("on_demand_n") or 0, base.get("on_demand_n") or 0) >= MIN_PROVIDERS
                                                       else None),
                            "thin": (last.get(k + "_n") or 0) < MIN_PROVIDERS}
        ours = []
        for key, pts in (disc.get(gpu) or {}).items():
            if pts and pts[-1].get("discount_pct") is not None:
                ours.append(pts[-1]["discount_pct"])
        s["our_providers_median_discount_pct"] = round(median(ours), 1) if ours else None
        s["n_changes_90d"] = sum(1 for c in changes if c["gpu"] == gpu and c["date"] >= _days_back(today, 90))
        summary[gpu] = s
    sec["summary"] = summary


def _weeks_back(iso, n):
    return (date.fromisoformat(iso) - timedelta(weeks=n)).isoformat()


def _days_back(iso, n):
    return (date.fromisoformat(iso) - timedelta(days=n)).isoformat()


def _backlog(prev):
    import capex  # shares SEC rate limiting and User-Agent
    out = {}
    for tic, (cik, name, scope) in BACKLOG_COS.items():
        try:
            facts = capex.sec_companyfacts(cik)
            node = facts["facts"]["us-gaap"]["RevenueRemainingPerformanceObligation"]["units"]["USD"]
            by_end = {}
            for f in node:
                if f.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A") or "start" in f:
                    continue
                if f["end"] not in by_end or f.get("filed", "") > by_end[f["end"]].get("filed", ""):
                    by_end[f["end"]] = f
            pts = [{"date": e, "value_bn": round(f["val"] / 1e9, 2), "form": f["form"], "filed": f.get("filed")}
                   for e, f in sorted(by_end.items())][-12:]
            for i, p in enumerate(pts):
                prior = next((q for q in pts[:i] if 350 <= (date.fromisoformat(p["date"]) - date.fromisoformat(q["date"])).days <= 380), None)
                p["yoy_pct"] = round((p["value_bn"] / prior["value_bn"] - 1) * 100, 1) if prior else None
            out[tic] = {"name": name, "scope": scope, "points": pts, "url": EDGAR_BROWSE.format(cik=cik),
                        "stale": False}
        except Exception:
            if tic in prev:
                out[tic] = dict(prev[tic], stale=True)
    if not out:
        raise RuntimeError("no backlog series fetched")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.json")
    d = json.load(open(path))
    pr = lambda *a: print(*a)
    build_reserved_dynamics(d, pr, pr, pr)
    out = d["reserved_dynamics"]
    print(json.dumps(out["summary"], indent=1)[:1500])
    print(out["rate_changes"][:5])
    print({k: v["points"][-1] for k, v in out["backlog"].items()})
