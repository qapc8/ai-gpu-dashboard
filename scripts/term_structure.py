"""On-demand vs committed pricing for the top-tier GPUs.

Three jobs:

1. ``fetch_aws_savings_plans`` -- AWS sells its H200 / B200 / B300 instances
   (p5en, p6-*) under Savings Plans only; there are no Reserved Instances for
   them, so the RI-only reader in update_data published no committed AWS rate
   for any Blackwell part, and for H100 only the 3-year one.
2. ``attach_terms`` -- every provider/GPU entry for a top-tier part gets a
   ``terms`` list: each published commitment (length, effective $/GPU-hr,
   payment, minimum cluster size, the provider's own label). The flat
   ``reserved_1yr_hr`` / ``reserved_3yr_hr`` fields stay as the headline rate
   for each length so older readers keep working.
3. ``record_term_history`` + ``build_term_structure`` -- the daily record of
   headline committed rates (backfilled from git for the months the flat
   fields existed) and the derived ``term_structure`` section: per GPU, the
   on-demand band, a ladder of commitment buckets with same-provider
   discounts and break-even utilisation, the spread over time, and a
   plain-language explanation.

A discount is always measured against the SAME provider's on-demand rate. The
cross-market gap (cheapest commitment anywhere vs cheapest on-demand anywhere)
is reported separately and labelled as such, because it mixes providers.
"""

import json
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone

TOP_TIER = ["B300", "GB200", "B200", "H200", "H100-SXM", "MI325X", "MI300X"]

TERM_ORDER = ["3d", "1w", "2w", "1mo", "3mo", "6mo", "1yr", "2yr", "3yr"]
TERM_LABEL = {"3d": "3 days", "1w": "1 week", "2w": "2 weeks", "1mo": "1 month",
              "3mo": "3 months", "6mo": "6 months", "1yr": "1 year", "2yr": "2 years",
              "3yr": "3 years"}

# Commitment lengths differ by provider (Together sells 3-30 days, Lambda
# 2 weeks-1 year, the hyperscalers 1 and 3 years), so the ladder groups them.
BUCKETS = [
    ("short", "Up to 1 month", ("3d", "1w", "2w", "1mo")),
    ("medium", "3-6 months", ("3mo", "6mo")),
    ("1yr", "1 year", ("1yr",)),
    ("3yr", "3 years", ("2yr", "3yr")),
]
_BUCKET_OF = {t: key for key, _label, terms in BUCKETS for t in terms}

_UA = {"User-Agent": "ai-gpu-dashboard/1.0"}

# ---------------------------------------------------------------------------
# AWS Savings Plans
# ---------------------------------------------------------------------------

_AWS_SP_INSTANCES = {
    "p5.48xlarge": ("H100-SXM", 8),
    "p5en.48xlarge": ("H200", 8),
    "p6-b200.48xlarge": ("B200", 8),
    "p6-b300.48xlarge": ("B300", 8),
}
_AWS_SP_INDEX = ("https://pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/"
                 "AWSComputeSavingsPlan/current/region_index.json")
_PAYMENT = {"No Upfront": "no_upfront", "Partial Upfront": "partial_upfront",
            "All Upfront": "all_upfront"}


def _get_json(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        return json.load(r)


def fetch_aws_savings_plans():
    """{gpu_id: [term, ...]} from the public AWS Savings Plans price list, us-east-1.

    EC2 Instance Savings Plans (family-locked, the deeper discount) are
    published for every payment option. Compute Savings Plans (any family) are
    added only for a length the Instance plan does not cover -- p6-b200 and
    p6-b300 have no 1-year Instance plan. Upfront payments are already
    amortised into AWS's effective hourly rate.
    """
    index = _get_json(_AWS_SP_INDEX, timeout=60)
    region = next((r for r in index.get("regions", []) if r.get("regionCode") == "us-east-1"), None)
    if not region:
        raise RuntimeError("AWS Savings Plans: us-east-1 not in region index")
    doc = _get_json("https://pricing.us-east-1.amazonaws.com" + region["versionUrl"], timeout=300)
    products = {p["sku"]: p.get("attributes", {}) for p in doc.get("products", [])}

    found = {}  # (gpu, family, years, payment) -> rate
    for term in (doc.get("terms") or {}).get("savingsPlan", []):
        attrs = products.get(term.get("sku")) or {}
        family = attrs.get("productFamily", "")
        years = (term.get("leaseContractLength") or {}).get("duration")
        payment = _PAYMENT.get(attrs.get("purchaseOption"))
        if family not in ("EC2InstanceSavingsPlans", "ComputeSavingsPlans") or years not in (1, 3) or not payment:
            continue
        for rate in term.get("rates", []):
            inst = rate.get("discountedInstanceType")
            if inst not in _AWS_SP_INSTANCES:
                continue
            # Linux, shared tenancy, the instance itself (not a dedicated host
            # or an unused-reservation line).
            if rate.get("discountedUsageType") != f"BoxUsage:{inst}" or rate.get("discountedOperation") != "RunInstances":
                continue
            try:
                price = float(rate["discountedRate"]["price"])
            except (KeyError, TypeError, ValueError):
                continue
            gpu, n = _AWS_SP_INSTANCES[inst]
            found[(gpu, family, years, payment, inst)] = price / n
    del doc

    out = {}
    for (gpu, family, years, payment, inst), per_gpu in found.items():
        instance_plan = family == "EC2InstanceSavingsPlans"
        if not instance_plan and any(k[0] == gpu and k[1] == "EC2InstanceSavingsPlans" and k[2] == years
                                     for k in found):
            continue  # an Instance plan covers this length; it is the like-for-like rate
        if not instance_plan and payment != "no_upfront":
            continue
        product = "EC2 Instance Savings Plan" if instance_plan else "Compute Savings Plan"
        out.setdefault(gpu, []).append({
            "term": f"{years}yr",
            "rate": round(per_gpu, 4),
            "payment": payment,
            "min_gpus": None,
            "product": product,
            "label": f"{product}, {years} year, {payment.replace('_', ' ')} ({inst})",
            "effective_from": None,
            "note": None if instance_plan else "No 1-year EC2 Instance Savings Plan is published for this family.",
        })
    if not out:
        raise RuntimeError("AWS Savings Plans: no GPU instance rates found")
    for rows in out.values():
        rows.sort(key=lambda t: (TERM_ORDER.index(t["term"]), t["rate"]))
    return out


# ---------------------------------------------------------------------------
# Attaching terms to provider entries
# ---------------------------------------------------------------------------

# Committed rates read by the main price fetchers into flat fields, described
# so they can sit in the same ladder as everything else.
_FLAT_PRODUCTS = {
    "Azure": ("Reserved VM Instance", "monthly"),
    "GCP": ("Committed use discount", "monthly"),
    "AWS": ("Reserved Instance (standard)", "no_upfront"),
    "DigitalOcean": ("12-month reserved", "unknown"),
}


def _accessible(t):
    """Headline preference: pay-as-you-go-like billing, smallest cluster, cheapest."""
    return (t.get("stale", False),
            t.get("payment") not in ("no_upfront", "monthly", "unknown"),
            t.get("min_gpus") or 1,
            t["rate"])


def headline(rows):
    """The one rate per commitment length we quote as 'the' price for it."""
    return min(rows, key=_accessible) if rows else None


def attach_terms(data, fetched, failed, now):
    """Write `terms` onto each top-tier provider/GPU entry.

    fetched: {provider_key: {gpu_id: [term, ...]}} from this run.
    failed:  provider keys whose term fetch failed this run -- their previous
             terms are kept, marked stale, rather than silently vanishing.
    """
    providers = data.get("providers") or {}
    # A commitment for a GPU the provider sells no on-demand plan for
    # (DigitalOcean B300 is 12-month reserved only) has no entry to hang on.
    for prov_key, prov in providers.items():
        gpus = prov.get("gpus") or {}
        if prov_key in fetched:
            only = {g: rows for g, rows in fetched[prov_key].items()
                    if g in TOP_TIER and g not in gpus and rows}
            if only:
                prov["terms_only"] = only
            else:
                prov.pop("terms_only", None)
    for prov_key, prov in providers.items():
        for gpu_id, entry in (prov.get("gpus") or {}).items():
            if gpu_id not in TOP_TIER:
                entry.pop("terms", None)
                continue
            rows = None
            if prov_key in fetched:
                rows = [dict(t) for t in (fetched[prov_key].get(gpu_id) or [])]
            elif prov_key in failed and entry.get("terms"):
                rows = [dict(t, stale=True) for t in entry["terms"]]
            else:
                rows = []
                product, payment = _FLAT_PRODUCTS.get(prov_key, ("Reserved", "unknown"))
                for years in (1, 3):
                    rate = entry.get(f"reserved_{years}yr_hr")
                    if rate:
                        rows.append({
                            "term": f"{years}yr", "rate": rate, "payment": payment,
                            "min_gpus": None, "product": product,
                            "label": f"{product}, {years} year", "effective_from": None, "note": None,
                        })
            for t in rows:
                t.setdefault("product", t.get("label", "").split(",")[0] or "Reserved")
                t.setdefault("min_gpus", None)
            rows.sort(key=lambda t: (TERM_ORDER.index(t["term"]) if t["term"] in TERM_ORDER else 99, t["rate"]))
            if rows:
                entry["terms"] = rows
                entry["terms_checked"] = now[:10]
            else:
                entry.pop("terms", None)
            # Keep the flat 1yr/3yr fields equal to the ladder's headline.
            for years in (1, 3):
                same = [t for t in rows if t["term"] == f"{years}yr" and not t.get("stale")]
                h = headline(same)
                if h:
                    entry[f"reserved_{years}yr_hr"] = round(h["rate"], 4)
                elif prov_key in fetched:
                    entry.pop(f"reserved_{years}yr_hr", None)
    return data


# ---------------------------------------------------------------------------
# History of committed rates
# ---------------------------------------------------------------------------

def _snapshot_terms(providers):
    """{provider: {gpu: {bucket: headline rate}}} for today."""
    snap = {}
    for prov_key, prov in (providers or {}).items():
        for gpu_id, entry in (prov.get("gpus") or {}).items():
            if gpu_id not in TOP_TIER:
                continue
            rows = [t for t in entry.get("terms") or [] if not t.get("stale")]
            if not rows:
                for years in (1, 3):  # older snapshots: flat fields only
                    if entry.get(f"reserved_{years}yr_hr"):
                        rows.append({"term": f"{years}yr", "rate": entry[f"reserved_{years}yr_hr"],
                                     "payment": "unknown"})
            by_bucket = {}
            for t in rows:
                b = _BUCKET_OF.get(t["term"])
                if b:
                    by_bucket.setdefault(b, []).append(t)
            day = {b: round(headline(ts)["rate"], 4) for b, ts in by_bucket.items()}
            if day:
                snap.setdefault(prov_key, {})[gpu_id] = day
    return snap


def record_term_history(data, repo_dir, log_info=lambda m: None, keep_days=400):
    """Add today's headline committed rates to price_history.term_days.

    The first run backfills from git: reserved_1yr_hr / reserved_3yr_hr have
    been in data.json since 2026-08, so the 1- and 3-year series start there
    rather than today. Shorter terms start today.
    """
    block = data.setdefault("price_history", {})
    days = block.setdefault("term_days", {})
    try:
        log = subprocess.run(["git", "log", "--max-count=400", "--format=%H %ad", "--date=short",
                              "--", "data.json"], cwd=repo_dir, capture_output=True,
                             text=True, timeout=60).stdout.strip()
    except Exception:
        log = ""
    added = 0
    seen = set()
    for line in log.split("\n") if log else []:
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, date = parts
        if date in seen or date in days:
            seen.add(date)
            continue
        seen.add(date)
        try:
            raw = subprocess.run(["git", "show", f"{sha}:data.json"], cwd=repo_dir,
                                 capture_output=True, text=True, timeout=60).stdout
            snap = _snapshot_terms(json.loads(raw).get("providers"))
        except Exception:
            continue
        if snap:
            days[date] = snap
            added += 1
    if added:
        log_info(f"Term history: backfilled {added} days from git")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = _snapshot_terms(data.get("providers"))
    if snap:
        days[today] = snap
    for old in sorted(days)[:-keep_days]:
        days.pop(old, None)
    return data


# ---------------------------------------------------------------------------
# The derived section
# ---------------------------------------------------------------------------

def _median(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return None
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _pct(x):
    return f"{abs(round(x))}%"


def _money(x):
    return f"${x:,.2f}"


def _is_cluster(row):
    """A reservation of a multi-node cluster (Lambda 1-Click Clusters: 16+ GPUs
    on InfiniBand) is a different product from a discount on the single-node
    instance. It is listed, but kept out of the discount aggregates."""
    return bool(row.get("min_gpus") and row["min_gpus"] > 8)


def _row(prov_key, name, t, od, reserved_only=False):
    return {
        "provider": name, "provider_key": prov_key, "term": t["term"],
        "term_max": t.get("term_max"), "rate": t["rate"], "payment": t.get("payment"),
        "min_gpus": t.get("min_gpus"), "product": t.get("product"), "label": t.get("label"),
        "note": t.get("note"), "effective_from": t.get("effective_from"),
        "stale": bool(t.get("stale")), "reserved_only": reserved_only,
        "on_demand_same_provider": od,
        "discount_pct": round((1 - t["rate"] / od) * 100, 1) if od else None,
    }


def build_term_structure(data, log_ok=lambda a, b="": None):
    providers = data.get("providers") or {}
    specs = data.get("specs") or {}
    term_days = (data.get("price_history") or {}).get("term_days") or {}
    od_days = (data.get("price_history") or {}).get("days") or {}
    out = {}

    for gpu_id in TOP_TIER:
        od_rows, spot_rows = [], []
        commit = {key: [] for key, _l, _t in BUCKETS}
        for prov_key, prov in providers.items():
            name = prov.get("provider_name", prov_key)
            for t in (prov.get("terms_only") or {}).get(gpu_id) or []:
                if _BUCKET_OF.get(t["term"]):
                    commit[_BUCKET_OF[t["term"]]].append(_row(prov_key, name, t, None, reserved_only=True))
            entry = (prov.get("gpus") or {}).get(gpu_id)
            if not entry:
                continue
            od = entry.get("price_per_gpu_hr")
            usable = bool(od and od > 0 and not entry.get("stale"))
            if usable:
                od_rows.append({"provider": name, "provider_key": prov_key, "rate": od,
                                "type": prov.get("type", "cloud"),
                                "cluster_size": entry.get("cluster_size"),
                                "instance": entry.get("instance")})
            if entry.get("spot_hr"):
                spot_rows.append({"provider": name, "rate": entry["spot_hr"],
                                  "discount_pct": round((1 - entry["spot_hr"] / od) * 100, 1) if usable else None})
            for t in entry.get("terms") or []:
                if _BUCKET_OF.get(t["term"]):
                    commit[_BUCKET_OF[t["term"]]].append(_row(prov_key, name, t, od if usable else None))

        if not od_rows and not any(commit.values()):
            continue

        cloud = [r for r in od_rows if r["type"] != "marketplace"]
        market = [r for r in od_rows if r["type"] == "marketplace"]
        cloud_rates = [r["rate"] for r in cloud]
        cloud_low = min(cloud, key=lambda r: r["rate"]) if cloud else None
        market_low = min(market, key=lambda r: r["rate"]) if market else None
        rec = {
            "name": (specs.get(gpu_id) or {}).get("name", gpu_id),
            "on_demand": {
                "providers": len(cloud),
                "low": cloud_low["rate"] if cloud_low else None,
                "low_provider": cloud_low["provider"] if cloud_low else None,
                "median": round(_median(cloud_rates), 3) if cloud_rates else None,
                "high": max(cloud_rates) if cloud_rates else None,
                "basis": "dedicated clouds (marketplaces listed separately)",
                "marketplace_low": market_low["rate"] if market_low else None,
                "marketplace_low_provider": market_low["provider"] if market_low else None,
                "rows": sorted(od_rows, key=lambda r: r["rate"]),
            },
            "spot": sorted(spot_rows, key=lambda r: r["rate"]),
            "buckets": [],
        }

        for key, label, _terms in BUCKETS:
            rows = sorted(commit[key], key=lambda r: (r["provider"], TERM_ORDER.index(r["term"]), r["rate"]))
            for r in rows:
                r["cluster_reservation"] = _is_cluster(r)
            # One headline per provider among like-for-like offers, so a
            # provider with nine payment / cluster variants counts once.
            heads = {}
            for r in rows:
                if r["stale"] or r["cluster_reservation"]:
                    continue
                cur = heads.get(r["provider"])
                if cur is None or _accessible(r) < _accessible(cur):
                    heads[r["provider"]] = r
            for r in rows:
                r["headline"] = heads.get(r["provider"]) is r
            hv = list(heads.values())
            with_od = [r for r in hv if r["discount_pct"] is not None]
            discounts = [r["discount_pct"] for r in with_od]
            ratios = [r["rate"] / r["on_demand_same_provider"] for r in with_od]
            low = min(hv, key=lambda r: r["rate"]) if hv else None
            clusters = [r for r in rows if r["cluster_reservation"] and not r["stale"]]
            rec["buckets"].append({
                "key": key, "label": label,
                "providers": len(hv),
                "providers_with_discount": sorted(r["provider"] for r in with_od),
                "rate_low": low["rate"] if low else None,
                "rate_low_provider": low["provider"] if low else None,
                "rate_median": round(_median([r["rate"] for r in hv]), 3) if hv else None,
                "median_discount_pct": round(_median(discounts), 1) if discounts else None,
                "discount_min_pct": min(discounts) if discounts else None,
                "discount_max_pct": max(discounts) if discounts else None,
                # Billed for every hour of the term: at utilisation u the cost
                # per used hour is rate/u, which equals on-demand at u = rate/od.
                "breakeven_utilization_pct": round(_median(ratios) * 100) if ratios else None,
                "vs_cheapest_cloud_on_demand_pct": (
                    round((1 - low["rate"] / cloud_low["rate"]) * 100, 1) if low and cloud_low else None),
                "cluster_reservations": [
                    {"provider": r["provider"], "rate": r["rate"], "min_gpus": r["min_gpus"],
                     "vs_own_on_demand_pct": r["discount_pct"]} for r in clusters],
                "rows": rows,
            })

        rec["history"] = _history(gpu_id, term_days, od_days)
        rec["explanation"] = _explain(gpu_id, rec)
        out[gpu_id] = rec

    data["term_structure"] = out
    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["term_structure"] = {
        "basis": "derived",
        "detail": (
            "On-demand and committed rates for the top-tier GPUs, read from each provider's "
            "published price list (AWS Savings Plans and Reserved Instances, Azure reservations, "
            "GCP committed use, Lambda 1-Click Cluster reservations, Together reserved clusters, "
            "DigitalOcean 12-month reserved). Discounts compare a commitment with the SAME "
            "provider's on-demand rate; multi-node cluster reservations are listed but kept out "
            "of the discount medians. The spread-over-time series uses a fixed set of providers. "
            "Providers that only quote commitments through sales (CoreWeave, Nebius, Crusoe, "
            "Oracle, RunPod, Vultr) are not shown as offering one."
        ),
    }
    log_ok("Term Structure", f"{len(out)} top-tier GPUs, "
                             f"{sum(len(b['rows']) for r in out.values() for b in r['buckets'])} committed rates")
    return data


def _changed_providers(since, gpu_id):
    try:
        from price_history import METHODOLOGY_CHANGES
    except Exception:
        return set()
    return {p for c in METHODOLOGY_CHANGES
            if c["date"] > since and c.get("kind") != "joined"
            and (not c.get("gpus") or gpu_id in c["gpus"]) for p in c["providers"]}


def _history(gpu_id, term_days, od_days):
    """Monthly same-provider discount per bucket over a FIXED set of providers.

    A provider counts toward a bucket's series only if it has an observation in
    every month of that series and we did not change how we read it inside the
    window -- otherwise a provider joining (DigitalOcean, the AWS 1-year
    Savings Plan) or being re-measured (Azure, 2026-11-09) would read as the
    market's discount moving.
    """
    obs = {}  # bucket -> provider -> month -> [discount]
    for date in sorted(term_days):
        month = date[:7]
        od_day = od_days.get(date) or {}
        for prov, gpus in (term_days[date] or {}).items():
            rates = (gpus or {}).get(gpu_id)
            od = (od_day.get(prov) or {}).get(gpu_id)
            if not rates or not od:
                continue
            for b, rate in rates.items():
                obs.setdefault(b, {}).setdefault(prov, {}).setdefault(month, []).append((1 - rate / od) * 100)
    od_all = {}
    for date, provs in od_days.items():
        for prov, gpus in (provs or {}).items():
            if (gpus or {}).get(gpu_id):
                od_all.setdefault(date[:7], []).append(gpus[gpu_id])

    months = sorted({m for provs in obs.values() for pm in provs.values() for m in pm})
    if not months:
        return []
    changed = _changed_providers(months[0] + "-01", gpu_id)
    rows = {m: {"month": m, "on_demand_median": round(_median(od_all[m]), 3) if od_all.get(m) else None}
            for m in months}
    for b, provs in obs.items():
        b_months = sorted({m for pm in provs.values() for m in pm})
        basket = sorted(p for p, pm in provs.items()
                        if all(m in pm for m in b_months) and p not in changed)
        for m in b_months:
            vals = [_median(provs[p][m]) for p in basket]
            if vals:
                rows[m][f"{b}_discount_pct"] = round(_median(vals), 1)
                rows[m][f"{b}_providers"] = basket
    return [rows[m] for m in months]


_ADJ = {"short": "short-term", "medium": "3-6 month", "1yr": "1-year", "3yr": "3-year"}
_PHRASE = {"short": "a commitment of up to a month", "medium": "a 3-6 month commitment",
           "1yr": "a 1-year commitment", "3yr": "a 3-year commitment"}


def _explain(gpu_id, rec):
    """Plain-language reading of the ladder, generated from the numbers."""
    od = rec["on_demand"]
    parts = []
    if od["providers"]:
        s = (f"On demand, {gpu_id} costs {_money(od['low'])}-{_money(od['high'])} per GPU-hour across "
             f"{od['providers']} dedicated clouds (median {_money(od['median'])}, cheapest {od['low_provider']})")
        if od.get("marketplace_low"):
            s += f"; marketplace hosts list it from {_money(od['marketplace_low'])} ({od['marketplace_low_provider']})"
        parts.append(s + ".")
    offered = [b for b in rec["buckets"] if b["providers"] or b["cluster_reservations"]]
    if not offered:
        parts.append("No provider publishes a committed rate for it; commitments are by sales quote only.")
        return " ".join(parts)
    for b in offered:
        phrase = _PHRASE[b["key"]]
        if b["providers"] and b["median_discount_pct"] is None:
            parts.append(f"{phrase[0].upper() + phrase[1:]} is published by {b['rate_low_provider']} at "
                         f"{_money(b['rate_low'])} (reserved only -- no on-demand plan to compare with).")
        elif b["providers"]:
            n = b["providers"]
            if b["median_discount_pct"] >= 0:
                s = f"{phrase[0].upper() + phrase[1:]} saves a median {_pct(b['median_discount_pct'])} on the same provider's on-demand rate"
            else:
                s = f"{phrase[0].upper() + phrase[1:]} costs a median {_pct(b['median_discount_pct'])} more than the same provider's on-demand rate"
            who = b["providers_with_discount"]
            if len(who) > 1 and b["discount_min_pct"] != b["discount_max_pct"]:
                s += f" ({round(b['discount_min_pct'])}-{round(b['discount_max_pct'])}% across {len(who)} providers)"
            elif len(who) > 1:
                s += f" ({len(who)} providers)"
            else:
                s += f" ({who[0]} only)"
            be = b["breakeven_utilization_pct"]
            if be is not None and 0 < be < 100:
                s += f", so it beats paying on demand only if the GPUs are busy more than {be}% of the time"
            parts.append(s + ".")
        for c in b["cluster_reservations"][:1]:
            v = c["vs_own_on_demand_pct"]
            rel = (f"{_pct(v)} {'below' if v >= 0 else 'above'} its single-node on-demand rate"
                   if v is not None else "")
            parts.append(f"{c['provider']} also reserves multi-node clusters ({c['min_gpus']}+ GPUs, InfiniBand) "
                         f"from {_money(min(x['rate'] for x in b['cluster_reservations']))}"
                         f"{', ' + rel if rel else ''} -- a different product, kept out of the medians.")
    longest = [b for b in offered if b["providers"]][-1:] 
    if longest and longest[0]["vs_cheapest_cloud_on_demand_pct"] is not None:
        b = longest[0]
        gap = b["vs_cheapest_cloud_on_demand_pct"]
        parts.append(
            f"Between providers the gap is different: the cheapest {_ADJ[b['key']]} rate "
            f"({_money(b['rate_low'])}, {b['rate_low_provider']}) is {_pct(gap)} "
            f"{'below' if gap >= 0 else 'above'} the cheapest dedicated-cloud on-demand rate "
            f"({_money(od['low'])}, {od['low_provider']}).")
    hist = [h for h in rec.get("history") or [] if h.get("1yr_discount_pct") is not None]
    if len(hist) >= 2:
        a, z = hist[0], hist[-1]
        who = ", ".join(z.get("1yr_providers") or [])
        delta = z["1yr_discount_pct"] - a["1yr_discount_pct"]
        if abs(delta) >= 1:
            parts.append(f"Over time, the 1-year discount at {who} has {'widened' if delta > 0 else 'narrowed'} "
                         f"from {_pct(a['1yr_discount_pct'])} ({a['month']}) to {_pct(z['1yr_discount_pct'])} ({z['month']}).")
        else:
            parts.append(f"Over time, the 1-year discount at {who} has held near {_pct(z['1yr_discount_pct'])} "
                         f"since {a['month']}.")
    else:
        parts.append("Commitment history for this GPU is too short to show a trend yet.")
    return " ".join(parts)
