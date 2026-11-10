"""The reserved / committed GPU market, built daily into data["reserved_market"].

Sections (all per GPU-hour, USD):

* short_term   -- capacity bought for days to months: AWS EC2 Capacity Blocks,
                  GCP DWS Flex-start and Calendar mode, Together reserved
                  clusters, Lambda 1-Click Cluster reservations. Each is set
                  against the same provider's on-demand rate.
* discounts    -- the hyperscalers' published discount schedule (1/3/5 years,
                  by product and payment option), us-east / us-central.
* regional     -- where you buy: AWS on-demand by region (AWS applies the same
                  discount % in every region, so committed rates scale with
                  it) and Azure on-demand plus 1/3/5-year reservations by region.
* indexes      -- the Ornn transaction index (daily, accumulated here) and
                  GetDeploying's weekly median on-demand vs reserved prices
                  (CC BY 4.0).
* evidence     -- curated, dated reports (reference/reserved_market_evidence.json).
* takeaways    -- sentences generated from the numbers above.

Every fetcher raises on failure; the caller keeps the previous section and
marks it stale rather than publishing a guess.
"""

import csv
import gzip
import io
import json
import os
import re
import html as _html
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

TOP = ["B300", "GB200", "B200", "H200", "H100-SXM", "MI300X"]
_UA = {"User-Agent": "ai-gpu-dashboard/1.0"}
EVIDENCE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "reference", "reserved_market_evidence.json")


def _get(url, timeout=60, binary=False):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw if binary else raw.decode("utf-8", "replace")


def _median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


# ---------------------------------------------------------------------------
# AWS EC2 Capacity Blocks for ML
# ---------------------------------------------------------------------------

CB_URL = "https://aws.amazon.com/ec2/capacityblocks/pricing/"
_CB_INSTANCES = {
    "p5.48xlarge": "H100-SXM", "p5e.48xlarge": "H200", "p5en.48xlarge": "H200",
    "p6-b200.48xlarge": "B200", "p6-b300.48xlarge": "B300",
    "u-p6e-gb200x36": "GB200", "u-p6e-gb200x72": "GB200",
}


def fetch_aws_capacity_blocks():
    """Capacity Block rates as published on the pricing page.

    The page embeds its price table as JSON. One hourly rate per instance type;
    AWS states it is the same in every commercial region, set by supply and
    demand and revised periodically, and charged upfront at the rate prevailing
    when the block is bought. The EC2 price-list files carry Capacity Block
    SKUs at $0.00 (the charge is the reservation fee), so the page is the only
    public source.
    """
    page = _get(CB_URL, timeout=45)
    rows = []
    for m in re.finditer(r'<script type="application/json">(.*?)</script>', page, re.S):
        try:
            doc = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for it in (doc.get("data") or {}).get("items", []):
            fl = it.get("fields", {})
            if "itemTableData" not in fl or "Hourly" not in fl.get("itemTableColumns", ""):
                continue
            groups = {g["id"]: g["label"] for g in json.loads(fl.get("itemTableRowGroups", "[]"))}
            for r in json.loads(fl["itemTableData"]):
                inst = groups.get(r.get("idProperty"))
                loc = _html.unescape(re.sub("<[^>]+>", "", r.get("2", ""))).strip()
                price = _html.unescape(re.sub("<[^>]+>", "", r.get("3", ""))).replace("\xa0", " ")
                nums = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,.]+)", price)]
                acc = re.sub("<[^>]+>", "", r.get("4", "")).strip()
                rows.append((inst, loc, nums, acc))
    out = {}
    for inst, loc, nums, acc in rows:
        gpu = _CB_INSTANCES.get(inst)
        if not gpu or len(nums) < 2 or "GovCloud" in loc:
            continue
        per_gpu = nums[1]  # the page's per-accelerator hourly column
        e = out.setdefault(inst, {"instance": inst, "gpu": gpu, "rates": set(), "regions": []})
        e["rates"].add(round(per_gpu, 4))
        e["regions"].append(loc)
    if not out:
        raise RuntimeError("Capacity Blocks: price table not found on page")
    for e in out.values():
        rates = sorted(e.pop("rates"))
        e["rate"] = rates[0]
        e["rate_varies_by_region"] = len(rates) > 1
        e["regions"] = sorted(set(e["regions"]))
    return out


# ---------------------------------------------------------------------------
# AWS on-demand by region (small per-region files behind the pricing page)
# ---------------------------------------------------------------------------

_AWS_OD_URL = ("https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/"
               "ec2-ondemand-without-sec-sel/{region}/Linux/index.json")
# Display names the pricing files use, for regions that list a GPU instance we
# track. Metadata only -- every price comes from the file.
AWS_REGIONS = [
    "US East (N. Virginia)", "US East (Ohio)", "US West (N. California)", "US West (Oregon)",
    "Canada (Central)", "South America (Sao Paulo)", "EU (Frankfurt)", "EU (Stockholm)",
    "EU (Ireland)", "EU (London)", "EU (Paris)", "EU (Spain)", "Israel (Tel Aviv)",
    "Middle East (UAE)", "Asia Pacific (Tokyo)", "Asia Pacific (Seoul)", "Asia Pacific (Mumbai)",
    "Asia Pacific (Hyderabad)", "Asia Pacific (Singapore)", "Asia Pacific (Sydney)",
    "Asia Pacific (Jakarta)",
]
_AWS_OD_INSTANCES = {
    "p5.48xlarge": ("H100-SXM", 8), "p5en.48xlarge": ("H200", 8),
    "p6-b200.48xlarge": ("B200", 8), "p6-b300.48xlarge": ("B300", 8),
}


def fetch_aws_regional_on_demand():
    """{instance: {region: per-GPU on-demand}} across AWS regions."""
    out = {}
    ok = 0
    for region in AWS_REGIONS:
        try:
            doc = json.loads(_get(_AWS_OD_URL.format(region=urllib.parse.quote(region)), timeout=40))
        except Exception:
            continue
        ok += 1
        for _name, row in ((doc.get("regions") or {}).get(region) or {}).items():
            inst = row.get("Instance Type")
            if inst not in _AWS_OD_INSTANCES or row.get("Operating System") != "Linux":
                continue
            if row.get("Pre Installed S/W", "NA") != "NA":
                continue
            try:
                price = float(row["price"])
            except (KeyError, ValueError):
                continue
            if price > 0:
                out.setdefault(inst, {})[region] = round(price / _AWS_OD_INSTANCES[inst][1], 4)
    if not ok or not out:
        raise RuntimeError("AWS regional on-demand: no region files readable")
    return out


# ---------------------------------------------------------------------------
# Azure on-demand + reservations by region
# ---------------------------------------------------------------------------

_AZURE_SKUS = {
    "Standard_ND96isr_H100_v5": ("H100-SXM", 8),
    "Standard_ND96isr_H200_v5": ("H200", 8),
    "Standard_ND128isr_NDR_GB200_v6": ("GB200", 4),
    "Standard_ND96isr_MI300X_v5": ("MI300X", 8),
}


def fetch_azure_regional():
    """{sku: {region: {on_demand, spot, 1yr, 3yr, 5yr}}}, Linux, per GPU-hour."""
    base = "https://prices.azure.com/api/retail/prices"
    out = {}
    for sku, (_gpu, n) in _AZURE_SKUS.items():
        url = base + "?" + urllib.parse.urlencode({"$filter": f"armSkuName eq '{sku}'"})
        pages = 0
        while url and pages < 20:
            pages += 1
            doc = json.loads(_get(url, timeout=40))
            for it in doc.get("Items", []):
                product = (it.get("productName") or "").lower()
                meter = (it.get("meterName") or "").lower()
                if "windows" in product or "dedicated host" in product:
                    continue
                price = it.get("retailPrice") or 0
                if price <= 0:
                    continue
                reg = it.get("armRegionName")
                rec = out.setdefault(sku, {}).setdefault(reg, {})
                term = (it.get("reservationTerm") or "").strip()
                if term:
                    years = {"1 Year": 1, "3 Years": 3, "5 Years": 5}.get(term)
                    if years:
                        rec[f"{years}yr"] = round(price / (years * 8760) / n, 4)
                elif it.get("type") == "Consumption" and "low priority" not in meter:
                    key = "spot" if "spot" in meter else "on_demand"
                    rec[key] = round(min(rec.get(key, 1e9), price / n), 4)
            url = doc.get("NextPageLink")
    out = {s: {r: v for r, v in regs.items() if v.get("on_demand")} for s, regs in out.items()}
    if not any(out.values()):
        raise RuntimeError("Azure regional: no prices returned")
    return out


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

ORNN_URL = "https://data.ornn.com/api/public-index/daily-index/all"
_ORNN_GPU = {"H100 SXM": "H100-SXM", "H200": "H200", "B200": "B200", "B300": "B300",
             "A100 SXM4": "A100-80GB", "GB200": "GB200", "MI300X": "MI300X"}


def fetch_ornn():
    """Ornn's public daily index (robots.txt allows this endpoint). The public
    endpoint returns only the latest day, so history is accumulated here."""
    doc = json.loads(_get(ORNN_URL, timeout=30))
    if not doc.get("success"):
        raise RuntimeError("Ornn: unsuccessful response")
    out = {}
    for x in doc.get("data") or []:
        gpu = _ORNN_GPU.get(x.get("gpu_type"))
        if gpu and x.get("index_value"):
            out[gpu] = {"date": (x.get("date") or "")[:10], "value": float(x["index_value"])}
    if not out:
        raise RuntimeError("Ornn: no GPU values")
    return out


GETDEPLOYING_URL = "https://getdeploying.com/dataset/gpu-prices/weekly.csv"
# GetDeploying's "nvidia-h100" groups SXM and PCIe listings; it is the H100
# market band, labelled as such on the page.
_GD_GPU = {"nvidia-h100": "H100-SXM", "nvidia-h200": "H200",
           "nvidia-b200": "B200", "nvidia-b300": "B300", "nvidia-gb200": "GB200",
           "amd-mi300x": "MI300X"}
_GD_MONTHS = (1, 3, 6, 12, 24, 36)


def fetch_getdeploying(weeks=26):
    """Weekly median on-demand and reserved prices per GPU (CC BY 4.0).

    Returns {gpu: {"on_demand": [(week, median, providers)], "reserved_<m>mo": [...]}}.
    Medians are over whichever providers list that week, so composition can
    change week to week -- this is a market band, not a fixed basket.
    """
    text = _get(GETDEPLOYING_URL, timeout=60)
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        gpu = _GD_GPU.get(r.get("gpu_slug"))
        if not gpu:
            continue
        bt = (r.get("billing_type") or "").upper()
        if bt == "ON_DEMAND":
            key = "on_demand"
        elif bt == "SPOT":
            key = "spot"
        elif bt == "RESERVATION" and r.get("reservation_months"):
            months = int(float(r["reservation_months"]))
            if months not in _GD_MONTHS:
                continue
            key = f"reserved_{months}mo"
        else:
            continue
        try:
            med = float(r["median_price"])
        except (KeyError, ValueError):
            continue
        out.setdefault(gpu, {}).setdefault(key, []).append(
            (r["date"], round(med, 3), int(float(r.get("provider_count") or 0))))
    if not out:
        raise RuntimeError("GetDeploying: no rows for tracked GPUs")
    for series in out.values():
        for key in list(series):
            series[key] = sorted(series[key])[-weeks:]
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_reserved_market(data, fetched, failed, gcp_rates, log_ok=lambda a, b="": None):
    """Assemble data["reserved_market"].

    fetched: {"cb": ..., "aws_regional": ..., "azure_regional": ..., "ornn": ..., "getdeploying": ...}
    failed:  set of those keys that failed this run -- their previous section
             is kept and flagged with the date it was last refreshed.
    gcp_rates: update_data.GCP_ALL_RATES (machine type -> per-GPU rates).
    """
    prev = data.get("reserved_market") or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    providers = data.get("providers") or {}
    rm = {"as_of": today, "sources": {}}

    def stamp(key, ok):
        old = (prev.get("sources") or {}).get(key) or {}
        rm["sources"][key] = {"refreshed": today if ok else old.get("refreshed"), "ok": ok}

    def od_of(prov_key, gpu):
        e = ((providers.get(prov_key) or {}).get("gpus") or {}).get(gpu) or {}
        return e.get("price_per_gpu_hr") if not e.get("stale") else None

    # ---- short-term capacity ----
    short = []
    if "cb" in fetched:
        seen = set()
        aws_reg = fetched.get("aws_regional") or prev.get("_aws_regional_raw") or {}
        for inst, e in sorted(fetched["cb"].items()):
            if (e["gpu"], e["rate"]) in seen:
                continue  # GB200 UltraServer x36 and x72 carry the same per-GPU rate
            seen.add((e["gpu"], e["rate"]))
            # Only the SAME instance's on-demand rate is comparable (p5e is not p5en).
            od = (aws_reg.get(inst) or {}).get("US East (N. Virginia)")
            short.append({
                "provider": "AWS", "product": "EC2 Capacity Block", "gpu": e["gpu"],
                "instance": "GB200 UltraServer" if inst.startswith("u-") else inst,
                "term": "1-182 days, booked ahead, paid upfront", "rate": e["rate"],
                "on_demand_same_provider": od,
                "discount_pct": round((1 - e["rate"] / od) * 100, 1) if od else None,
                "regions": len(e["regions"]),
                "note": "One rate in every commercial region; AWS revises it with supply and demand.",
                "source": CB_URL,
            })
    stamp("aws_capacity_blocks", "cb" in fetched)
    for mt, r in sorted((gcp_rates or {}).items()):
        if r.get("gpu") not in TOP:
            continue
        od = r.get("on_demand")
        for key, product, term in (("dws_flex", "DWS Flex-start", "runs when capacity frees up, up to 7 days"),
                                   ("dws_calendar", "DWS Calendar mode", "fixed dates, 1-90 days, booked ahead")):
            if r.get(key):
                short.append({
                    "provider": "GCP", "product": product, "gpu": r["gpu"], "instance": mt, "term": term,
                    "rate": r[key], "on_demand_same_provider": od,
                    "discount_pct": round((1 - r[key] / od) * 100, 1) if od else None,
                    "note": None if od else "No on-demand rate is published for this machine type.",
                    "source": "https://cloud.google.com/compute/gpus-pricing",
                })
    stamp("gcp_dws", bool(gcp_rates))
    for prov_key in ("Together", "Lambda"):
        prov = providers.get(prov_key) or {}
        for gpu, e in (prov.get("gpus") or {}).items():
            if gpu not in TOP:
                continue
            for t in e.get("terms") or []:
                if t.get("stale") or t.get("term") not in ("3d", "1w", "2w", "1mo", "3mo", "6mo"):
                    continue
                od = od_of(prov_key, gpu)
                short.append({
                    "provider": prov.get("provider_name", prov_key), "product": t.get("product") or "Reservation",
                    "gpu": gpu, "instance": None,
                    "term": t.get("label") or t["term"], "rate": t["rate"], "min_gpus": t.get("min_gpus"),
                    "on_demand_same_provider": od,
                    "discount_pct": round((1 - t["rate"] / od) * 100, 1) if od else None,
                    "note": t.get("note"), "source": None,
                })
    rm["short_term"] = sorted(short, key=lambda r: (TOP.index(r["gpu"]) if r["gpu"] in TOP else 99, r["rate"]))

    # ---- hyperscaler discount schedule ----
    disc = []
    for prov_key in ("AWS", "Azure", "GCP"):
        prov = providers.get(prov_key) or {}
        for gpu, e in (prov.get("gpus") or {}).items():
            od = e.get("price_per_gpu_hr")
            if gpu not in TOP or not od:
                continue
            for t in e.get("terms") or []:
                disc.append({"provider": prov_key, "gpu": gpu, "product": t.get("product"),
                             "term": t["term"], "payment": t.get("payment"), "rate": t["rate"],
                             "on_demand": od, "discount_pct": round((1 - t["rate"] / od) * 100, 1)})
    az = fetched.get("azure_regional") or prev.get("_azure_regional_raw") or {}
    for sku, regs in az.items():
        gpu, _n = _AZURE_SKUS[sku]
        r = regs.get("eastus") or regs.get("eastus2")
        if r and r.get("5yr") and r.get("on_demand"):
            disc.append({"provider": "Azure", "gpu": gpu, "product": "Reserved VM Instance",
                         "term": "5yr", "payment": "monthly", "rate": r["5yr"], "on_demand": r["on_demand"],
                         "discount_pct": round((1 - r["5yr"] / r["on_demand"]) * 100, 1)})
    for mt, r in (gcp_rates or {}).items():
        if r.get("gpu") in TOP and not r.get("on_demand"):
            for k in ("reserved_1yr", "reserved_3yr"):
                if r.get(k):
                    disc.append({"provider": "GCP", "gpu": r["gpu"], "product": "Committed use discount",
                                 "term": k[-3:], "payment": "monthly", "rate": r[k], "on_demand": None,
                                 "discount_pct": None, "note": "No on-demand rate published (commitment-only)."})
    rm["discounts"] = disc

    # ---- regional ----
    regional = {}
    aws_reg = fetched.get("aws_regional") or prev.get("_aws_regional_raw") or {}
    for inst, regs in aws_reg.items():
        gpu = _AWS_OD_INSTANCES[inst][0]
        vals = sorted(regs.items(), key=lambda kv: kv[1])
        regional.setdefault(gpu, {})["AWS"] = {
            "instance": inst, "regions": len(vals),
            "on_demand": {"low": vals[0][1], "low_region": vals[0][0],
                          "median": round(_median([v for _, v in vals]), 3),
                          "high": vals[-1][1], "high_region": vals[-1][0]},
            "by_region": [{"region": k, "on_demand": v} for k, v in vals],
            "note": "AWS applies the same Savings Plan / RI discount % in every region, "
                    "so committed rates vary across regions exactly as on-demand does.",
        }
    for sku, regs in az.items():
        gpu = _AZURE_SKUS[sku][0]
        rows = sorted(({"region": k, **v} for k, v in regs.items()), key=lambda r: r["on_demand"])
        if not rows:
            continue
        entry = {"instance": sku.replace("Standard_", ""), "regions": len(rows), "by_region": rows}
        for key in ("on_demand", "1yr", "3yr", "5yr"):
            vals = sorted((r[key], r["region"]) for r in rows if r.get(key))
            if vals:
                entry[key] = {"low": vals[0][0], "low_region": vals[0][1],
                              "median": round(_median([v for v, _ in vals]), 3),
                              "high": vals[-1][0], "high_region": vals[-1][1]}
        regional.setdefault(gpu, {})["Azure"] = entry
    rm["regional"] = regional
    rm["_aws_regional_raw"] = aws_reg
    rm["_azure_regional_raw"] = az
    stamp("aws_regional", "aws_regional" in fetched)
    stamp("azure_regional", "azure_regional" in fetched)

    # ---- indexes ----
    ornn_prev = ((prev.get("indexes") or {}).get("ornn") or {}).get("history") or {}
    ornn_hist = {g: dict(v) for g, v in ornn_prev.items()}
    if "ornn" in fetched:
        for gpu, x in fetched["ornn"].items():
            ornn_hist.setdefault(gpu, {})[x["date"]] = x["value"]
    for gpu in ornn_hist:
        ornn_hist[gpu] = dict(sorted(ornn_hist[gpu].items())[-400:])
    gd = fetched.get("getdeploying") or ((prev.get("indexes") or {}).get("getdeploying") or {}).get("series") or {}
    rm["indexes"] = {
        "ornn": {
            "history": ornn_hist,
            "latest": {g: (sorted(h.items())[-1] if h else None) for g, h in ornn_hist.items()},
            "source": ORNN_URL,
            "note": "Transaction-based daily index; the Kalshi and Polymarket GPU contracts settle on it. "
                    "History accumulates from 2026-11-10 (the public endpoint serves only the latest day).",
        },
        "getdeploying": {
            "series": gd,
            "source": GETDEPLOYING_URL,
            "attribution": "GetDeploying, GPU rental price history, https://getdeploying.com/gpus, CC BY 4.0",
            "note": "Weekly median over whichever providers list that week; provider mix varies.",
        },
    }
    stamp("ornn", "ornn" in fetched)
    stamp("getdeploying", "getdeploying" in fetched)

    # ---- evidence ----
    try:
        with open(EVIDENCE_FILE) as f:
            ev = json.load(f)
        rm["evidence"] = ev
    except Exception:
        rm["evidence"] = prev.get("evidence") or {"items": []}

    rm["takeaways"] = _takeaways(data, rm)
    rm["failed"] = sorted(failed)
    data["reserved_market"] = rm

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["reserved_market"] = {
        "basis": "mixed",
        "detail": ("Short-term capacity, hyperscaler discount schedules and regional prices are read "
                   "from provider price pages and APIs each run. The Ornn index and GetDeploying "
                   "weekly medians are third-party market data (GetDeploying CC BY 4.0). The evidence "
                   "timeline is curated by hand from dated public reports."),
    }
    log_ok("Reserved Market", f"{len(short)} short-term rates, {len(disc)} discount rows, "
                              f"{len(regional)} GPUs by region")
    return data


def _takeaways(data, rm):
    out = []
    ts = data.get("term_structure") or {}
    h = ts.get("H100-SXM") or {}
    b1 = next((b for b in h.get("buckets") or [] if b["key"] == "1yr"), None)
    b3 = next((b for b in h.get("buckets") or [] if b["key"] == "3yr"), None)
    if b1 and b3 and b1.get("median_discount_pct") is not None and b3.get("median_discount_pct") is not None:
        out.append(f"Published commitments on H100 save a median {round(b1['median_discount_pct'])}% for 1 year and "
                   f"{round(b3['median_discount_pct'])}% for 3 years against the same provider's on-demand rate; "
                   f"they pay off above {b1['breakeven_utilization_pct']}% and {b3['breakeven_utilization_pct']}% utilisation.")
    if b3 and b3.get("rate_low") and (h.get("on_demand") or {}).get("low"):
        out.append(f"The cheapest published 3-year H100 rate (${b3['rate_low']:.2f}, {b3['rate_low_provider']}) sits "
                   f"{'below' if b3['rate_low'] < h['on_demand']['low'] else 'above'} the cheapest dedicated-cloud "
                   f"on-demand price (${h['on_demand']['low']:.2f}, {h['on_demand']['low_provider']}): where you buy "
                   f"moves the price as much as how long you commit.")
    cb = [r for r in rm.get("short_term") or [] if r["product"] == "EC2 Capacity Block" and r.get("discount_pct") is not None]
    if cb:
        lo = min(cb, key=lambda r: r["discount_pct"])
        hi = max(cb, key=lambda r: r["discount_pct"])
        out.append(f"Short-term blocks are priced close to on-demand: AWS Capacity Blocks run "
                   f"{lo['discount_pct']:.0f}% ({lo['gpu']}) to {hi['discount_pct']:.0f}% ({hi['gpu']}) below "
                   f"us-east-1 on-demand, against 37-62% for a 1-3 year Savings Plan.")
    reg = rm.get("regional") or {}
    spreads = []
    for gpu, provs in reg.items():
        for prov, e in provs.items():
            od = e.get("on_demand") or {}
            if od.get("low") and od.get("high") and e.get("regions", 0) > 2:
                spreads.append((od["high"] / od["low"], gpu, prov, od))
    if spreads:
        x, gpu, prov, od = max(spreads)
        out.append(f"Region matters: {prov} {gpu} on-demand runs ${od['low']:.2f} ({od['low_region']}) to "
                   f"${od['high']:.2f} ({od['high_region']}), {x:.1f}x apart, and committed rates scale with it.")
    ornn = (rm.get("indexes") or {}).get("ornn", {}).get("history") or {}
    for gpu in ("H100-SXM", "B200"):
        hist = sorted((ornn.get(gpu) or {}).items())
        if len(hist) >= 2:
            (d0, v0), (d1, v1) = hist[max(0, len(hist) - 31)], hist[-1]
            out.append(f"Ornn's transaction index for {gpu} moved {((v1 / v0) - 1) * 100:+.1f}% "
                       f"from {d0} to {d1} (${v0:.2f} to ${v1:.2f}).")
    gd = (rm.get("indexes") or {}).get("getdeploying", {}).get("series") or {}
    g = gd.get("H100-SXM") or {}
    if g.get("on_demand") and g.get("reserved_12mo"):
        od0, od1 = g["on_demand"][0], g["on_demand"][-1]
        r1 = g["reserved_12mo"][-1]
        out.append(f"Across ~{od1[2]} providers (GetDeploying), the median H100 on-demand price went "
                   f"${od0[1]:.2f} -> ${od1[1]:.2f} between {od0[0]} and {od1[0]}; the median 12-month "
                   f"reservation is ${r1[1]:.2f}.")
    ev = [i for i in (rm.get("evidence") or {}).get("items", []) if i.get("kind") == "list_change"]
    if ev:
        out.append("Announced list-price changes keep pointing up -- "
                   + "; ".join(f"{i['statement'].rstrip('.')} ({i['date']})" for i in ev[-3:]) + ".")
    return out
