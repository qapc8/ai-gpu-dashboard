"""Derived console signals: per-GPU market read, alerts and anomaly flags.

Nothing here fetches anything. Every output is computed from sections the
pipeline already measured (price_history, availability, indicators, changelog,
providers), and every signal carries the inputs that produced it, so a reader
can check the call rather than trust it.

Sections written:
  data["signals"]   -- per-GPU tightening / stable / loosening, with reasons
  data["alerts"]    -- today's notable events (big moves, sell-outs, listings)
  data["anomalies"] -- prices far from the cross-provider median, and prices
                       that have not moved for a long time
"""

from datetime import datetime, timedelta, timezone
from statistics import median

SOURCES = ["Signals"]

PRICE_MOVE_PCT = 5.0        # matched-provider median move that counts as a signal
SUPPLY_MOVE_PCT = 35.0      # rentable marketplace GPUs; Vast sampling alone moves ±10-25%
ALERT_MOVE_PCT = 10.0       # single-provider day-over-day move worth an alert
OUTLIER_LOW = 0.5           # price below 50% of the GPU's median
OUTLIER_HIGH = 2.0          # price above 200% of the GPU's median
UNCHANGED_DAYS = 45         # same price this many observed days running

RULE = (
    "Three inputs each vote +1 (tightening), -1 (loosening) or 0: the 30-day change in the "
    f"median price across providers seen on both dates (|move| >= {PRICE_MOVE_PCT:g}%), "
    f"the 7-day change in rentable marketplace GPUs (|move| >= {SUPPLY_MOVE_PCT:g}%, fewer = tighter) "
    "or a sold-out status, and the vendor lead-time status (constrained = +1, available = -1). "
    "Total >= +2 is tightening, <= -2 is loosening, otherwise stable. "
    "With fewer than two inputs available the call is 'insufficient data'."
)


def _matched_change(days, gpu, back):
    """Median % change for one GPU between the latest day and `back` days
    earlier, over providers that priced it on both dates. Composition changes
    (a provider appearing or vanishing) don't count as moves."""
    dates = sorted(days)
    if not dates:
        return None, 0
    last = dates[-1]
    target = (datetime.fromisoformat(last) - timedelta(days=back)).date().isoformat()
    earlier = [d for d in dates if d <= target]
    if not earlier:
        return None, 0
    then = earlier[-1]
    a, b = [], []
    for prov, gpus in (days[last] or {}).items():
        p1 = (gpus or {}).get(gpu)
        p0 = ((days[then] or {}).get(prov) or {}).get(gpu)
        if p1 and p0:
            a.append(p0)
            b.append(p1)
    if not a:
        return None, 0
    return round((median(b) / median(a) - 1) * 100, 1), len(a)


def _supply_change(hist, back=7):
    pts = [p for p in (hist or []) if p.get("rentable_gpus") is not None]
    if len(pts) < 2:
        return None
    last = pts[-1]
    target = (datetime.fromisoformat(last["date"]) - timedelta(days=back)).date().isoformat()
    earlier = [p for p in pts[:-1] if p["date"] <= target]
    if not earlier or not earlier[-1]["rentable_gpus"]:
        return None
    return round((last["rentable_gpus"] / earlier[-1]["rentable_gpus"] - 1) * 100, 1)


def build_signals(data, log_info, log_ok, log_fail):
    try:
        _build(data)
        log_ok("Signals", f"{len(data['signals']['gpus'])} GPUs, {len(data['alerts']['items'])} alerts, "
                          f"{len(data['anomalies']['items'])} anomalies")
    except Exception as exc:  # derived data must never take the run down
        log_fail("Signals", str(exc))
    return data


def _build(data):
    now = datetime.now(timezone.utc)
    days = (data.get("price_history") or {}).get("days") or {}
    avail = data.get("availability") or {}
    lead = (data.get("indicators") or {}).get("gpu_lead_times") or {}
    matrix = {r["gpu_id"]: r for r in (data.get("matrix") or []) if r.get("gpu_id")}

    # ---- per-GPU signal ----
    gpus = {}
    for gpu, row in matrix.items():
        votes, reasons, inputs = [], [], {}

        p30, n30 = _matched_change(days, gpu, 30)
        p7, _ = _matched_change(days, gpu, 7)
        inputs.update(price_30d_pct=p30, price_7d_pct=p7, price_providers=n30)
        if p30 is not None:
            v = 1 if p30 >= PRICE_MOVE_PCT else -1 if p30 <= -PRICE_MOVE_PCT else 0
            votes.append(v)
            reasons.append(f"Median price {p30:+.1f}% over 30 days ({n30} provider{'s' if n30 != 1 else ''})")

        a = (avail.get("gpus") or {}).get(gpu) or {}
        s7 = _supply_change((avail.get("history") or {}).get(gpu))
        status = a.get("status")
        inputs.update(supply_7d_pct=s7, availability=status)
        if status == "sold_out":
            votes.append(1)
            reasons.append("Sold out on the marketplaces we track")
        elif s7 is not None:
            v = 1 if s7 <= -SUPPLY_MOVE_PCT else -1 if s7 >= SUPPLY_MOVE_PCT else 0
            votes.append(v)
            reasons.append(f"Rentable marketplace supply {s7:+.0f}% over 7 days")

        lt = lead.get(gpu) or {}
        inputs.update(lead_time_weeks=lt.get("weeks"), lead_time_status=lt.get("status"))
        if lt.get("status"):
            v = 1 if lt["status"] == "constrained" else -1 if lt["status"] == "available" else 0
            votes.append(v)
            wk = lt.get("weeks")
            reasons.append(f"Vendor lead time {wk if wk is not None else '?'} week{'' if wk == 1 else 's'} ({lt['status']})")

        score = sum(votes)
        if len(votes) < 2:
            call = "insufficient"
        elif score >= 2:
            call = "tightening"
        elif score <= -2:
            call = "loosening"
        else:
            call = "stable"
        gpus[gpu] = {"call": call, "score": score, "inputs_used": len(votes),
                     "reasons": reasons, "inputs": inputs, "name": row.get("name") or gpu}

    data["signals"] = {"as_of": now.isoformat(), "rule": RULE, "gpus": gpus}

    # ---- alerts ----
    items = []
    log = data.get("changelog") or []
    latest = max((e.get("date") for e in log if e.get("date")), default=None)
    for e in log:
        if e.get("date") != latest or e.get("methodology_change"):
            continue
        if e["type"] == "price" and abs(e.get("change_pct") or 0) >= ALERT_MOVE_PCT:
            up = e["change_pct"] > 0
            items.append({"date": latest, "kind": "price_up" if up else "price_down",
                          "severity": "high" if abs(e["change_pct"]) >= 25 else "medium",
                          "gpu": e["gpu"], "provider": e["provider"],
                          "text": f"{e['provider']} {e['gpu']} {e['change_pct']:+.1f}% "
                                  f"(${e['from']:.2f} → ${e['to']:.2f}/hr)",
                          "needs_review": bool(e.get("needs_review"))})
        elif e["type"] == "listed":
            items.append({"date": latest, "kind": "listed", "severity": "low", "gpu": e["gpu"],
                          "provider": e["provider"],
                          "text": f"{e['provider']} now lists {e['gpu']}"
                                  + (f" at ${e['to']:.2f}/hr" if e.get("to") else "")})
        elif e["type"] == "delisted":
            items.append({"date": latest, "kind": "delisted", "severity": "medium", "gpu": e["gpu"],
                          "provider": e["provider"], "text": f"{e['provider']} no longer lists {e['gpu']}"})

    for gpu, hist in ((avail.get("history") or {}).items()):
        if len(hist) >= 2 and hist[-1].get("status") != hist[-2].get("status"):
            new, old = hist[-1].get("status"), hist[-2].get("status")
            if new in (None, "unknown") or old in (None, "unknown"):
                continue
            items.append({"date": hist[-1]["date"], "kind": "availability",
                          "severity": "high" if new == "sold_out" else "medium", "gpu": gpu,
                          "text": f"{gpu} availability {old.replace('_', ' ')} → {new.replace('_', ' ')}"})

    # Reserved and short-term capacity rates rarely move, so any repricing is news.
    rd = data.get("reserved_dynamics") or {}
    rc = rd.get("rate_changes") or []
    rc_latest = max((c["date"] for c in rc), default=None)
    if rc_latest and latest and rc_latest >= latest:
        for c in rc:
            if c["date"] != rc_latest:
                continue
            up = c["change_pct"] > 0
            items.append({"date": c["date"], "kind": "reserved_up" if up else "reserved_down",
                          "severity": "high" if abs(c["change_pct"]) >= 10 else "medium",
                          "gpu": c["gpu"], "provider": c["provider"],
                          "text": f"{c['provider']} {c['gpu']} {c['term']} rate {c['change_pct']:+.1f}% "
                                  f"(${c['from']:.2f} → ${c['to']:.2f}/hr)"})

    sev = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda x: (sev.get(x["severity"], 3), x.get("gpu", "")))
    data["alerts"] = {"as_of": now.isoformat(), "date": latest,
                      "rule": f"Single-provider on-demand moves of {ALERT_MOVE_PCT:g}%+ since the previous run, "
                              "any change in a reserved or short-term capacity rate, "
                              "listings, delistings and availability status changes.",
                      "items": items}

    # ---- anomalies ----
    anomalies = []
    for prov, p in (data.get("providers") or {}).items():
        for gpu, g in (p.get("gpus") or {}).items():
            price, med = g.get("price_per_gpu_hr"), (matrix.get(gpu) or {}).get("median_price")
            if not price or not med or (matrix.get(gpu) or {}).get("num_providers", 0) < 3:
                continue
            ratio = price / med
            if ratio < OUTLIER_LOW or ratio > OUTLIER_HIGH:
                anomalies.append({"kind": "outlier_low" if ratio < 1 else "outlier_high",
                                  "provider": prov, "gpu": gpu, "price": price, "median": med,
                                  "ratio": round(ratio, 2),
                                  "text": f"{prov} {gpu} at ${price:.2f}/hr is {ratio:.0%} of the "
                                          f"${med:.2f} median"})
    moves = (data.get("price_history") or {}).get("moves") or {}
    dates = sorted(days)
    for prov, gpus_m in moves.items():
        for gpu, m in (gpus_m or {}).items():
            cur = m.get("current")
            if not cur or not dates:
                continue
            run = 0
            for d in reversed(dates):
                v = ((days[d] or {}).get(prov) or {}).get(gpu)
                if v is None:
                    continue
                if abs(v - cur) > 1e-6:
                    break
                run += 1
            if run >= UNCHANGED_DAYS:
                anomalies.append({"kind": "unchanged", "provider": prov, "gpu": gpu, "price": cur,
                                  "days": run,
                                  "text": f"{prov} {gpu} unchanged at ${cur:.2f}/hr for {run} observed days"})
    order = {"outlier_low": 0, "outlier_high": 1, "unchanged": 2}
    anomalies.sort(key=lambda x: (order[x["kind"]], -(x.get("days") or 0), x["provider"]))
    data["anomalies"] = {
        "as_of": now.isoformat(),
        "rule": (f"Outlier: a price below {OUTLIER_LOW:.0%} or above {OUTLIER_HIGH:.0%} of the GPU's "
                 "cross-provider median (GPUs with 3+ providers). Unchanged: the same price on "
                 f"{UNCHANGED_DAYS}+ consecutive observed days. List prices from big clouds often hold "
                 "for months, so unchanged is a prompt to check, not proof of staleness."),
        "items": anomalies}


if __name__ == "__main__":
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.json")
    d = json.load(open(path))
    pr = lambda *a: print(*a)
    build_signals(d, pr, pr, pr)
    out = {k: d[k] for k in ("signals", "alerts", "anomalies")}
    print(json.dumps({k: (v if k != "signals" else {g: x["call"] for g, x in v["gpus"].items()})
                      for k, v in out.items() if k == "signals"}, indent=1))
    print(len(out["alerts"]["items"]), "alerts;", len(out["anomalies"]["items"]), "anomalies")
    for a in out["alerts"]["items"][:5] + out["anomalies"]["items"][:6]:
        print(" -", a["text"])
