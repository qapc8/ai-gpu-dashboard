"""GPU availability: can you actually rent this GPU right now, and how deep is supply?

A list price says nothing about whether capacity exists behind it. This module
reads the two public marketplaces that expose live inventory without a login,
plus (optionally) SF Compute's order book, and combines them with how many of
the dashboard's own providers currently publish a usable price.

Sources
  Vast.ai   public offer search (cloud.vast.ai/api/v0/bundles). Each offer row
            carries machine_id, num_gpus, gpu_frac (share of the machine) and a
            `rentable` flag. The endpoint returns at most 64 rows per query and
            silently samples when more match (its `truncated` flag is not
            reliable), so an exhaustive census is not possible from the public
            API. We run a fixed set of queries per GPU (rentable offers ordered
            two ways, plus one query for occupied machines) and publish the
            union. Counts are therefore LOWER BOUNDS, flagged `sample_capped`
            when any query came back near the cap. Verified, non-external hosts
            only, matching the Vast.ai prices used elsewhere on the dashboard.
  RunPod    public GraphQL `gpuTypes { lowestPrice(input:{gpuCount:1,
            secureCloud:true|false}) { stockStatus } }` per GPU and tier,
            RunPod's own High / Medium / Low label. Anonymous calls return an
            empty answer intermittently even for GPUs that are in stock a
            minute later, so an empty answer is published as null
            ("unconfirmed"), never as "None" / sold out. RunPod therefore only
            adds positive evidence of stock. A tier RunPod does not offer for
            that GPU is also null.
  SF Compute public order book (api.sfcompute.com/preview/v2/orderbook) needs a
            bearer token. Read only when SFCOMPUTE_API_KEY is set; otherwise
            skipped and reported as such.
  providers_listing  count of providers in data["providers"] with a positive,
            non-stale price for the GPU (same usability rule as the matrix).

Nothing here is estimated. A source that fails keeps its previous values from
`data` with `stale: true`.
"""

import json
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

USER_AGENT = "ai-gpu-dashboard/1.0 (market research)"
TIMEOUT = 45
HISTORY_DAYS = 400

VAST_URL = "https://cloud.vast.ai/api/v0/bundles/"
RUNPOD_URL = "https://api.runpod.io/graphql"
SFC_URL = "https://api.sfcompute.com/preview/v2/orderbook/windows"

SOURCES = ["Vast.ai availability", "RunPod availability", "SF Compute orderbook"]
_SRC_VAST, _SRC_RUNPOD, _SRC_SFC = SOURCES

GPU_IDS = [
    "B300", "B200", "GB200", "RTX-5090", "H200", "H100-SXM", "H100-PCIe",
    "L40S", "A100-80GB", "A100-40GB", "RTX-4090", "MI300X", "MI325X",
    "MI250X", "L4", "T4", "GH200", "A100-PCIe",
]

# (Vast gpu_name, gpu_id, min GB, max GB per GPU). Same table as
# update_data._VAST_MODELS: exact name plus a memory window, because the name
# alone does not separate 40GB from 80GB A100s.
VAST_MODELS = (
    ("RTX 4090", "RTX-4090", 23, 25),
    ("RTX 5090", "RTX-5090", 31, 33),
    ("L4", "L4", 22, 25),
    ("Tesla T4", "T4", 15, 17),
    ("L40S", "L40S", 44, 49),
    ("A100 SXM4", "A100-80GB", 78, 82),
    ("A100 SXM4", "A100-40GB", 38, 42),
    ("A100 PCIE", "A100-PCIe", 78, 82),
    ("H100 SXM", "H100-SXM", 78, 82),
    ("H100 PCIE", "H100-PCIe", 78, 82),
    ("H200", "H200", 138, 145),
    ("GH200 SXM", "GH200", 90, 100),
    ("B200", "B200", 175, 195),
    ("B300", "B300", 260, 300),
    ("GB200", "GB200", 180, 195),
    ("MI300X", "MI300X", 185, 195),
)
VAST_ROW_CAP = 64
VAST_CAPPED_AT = 30   # rows per query at/above which sampling likely kicked in

# RunPod gpuTypes.id -> gpu_id (exact ids, checked against the live list).
RUNPOD_TYPES = {
    "NVIDIA B300 SXM6 AC": "B300",
    "NVIDIA B200": "B200",
    "NVIDIA GeForce RTX 5090": "RTX-5090",
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100-SXM",
    "NVIDIA H100 PCIe": "H100-PCIe",
    "NVIDIA L40S": "L40S",
    "NVIDIA A100-SXM4-80GB": "A100-80GB",
    "NVIDIA A100-SXM4-40GB": "A100-40GB",
    "NVIDIA A100 80GB PCIe": "A100-PCIe",
    "NVIDIA GeForce RTX 4090": "RTX-4090",
    "AMD Instinct MI300X OAM": "MI300X",
    "NVIDIA L4": "L4",
}

# SF Compute accelerator requirement values (case-sensitive per their docs).
SFC_ACCELERATORS = {"H100-SXM": "h100", "H200": "h200", "B200": "b200"}

STATUS_RULE = (
    "available if Vast.ai shows at least 8 rentable GPUs or RunPod reports "
    "Medium/High stock; limited if Vast.ai shows 1-7 rentable or RunPod reports "
    "Low; sold_out if Vast.ai lists machines with this GPU but none is rentable "
    "and RunPod shows no stock; otherwise unknown (no marketplace carries it). "
    "Cloud list prices (providers_listing) are context only, since a list price "
    "is not proof of capacity."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _request(url, data=None, headers=None, method=None):
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            raise


# --------------------------------------------------------------------------
# Vast.ai
# --------------------------------------------------------------------------

def _vast_query(q):
    url = VAST_URL + "?q=" + urllib.parse.quote(json.dumps(q))
    time.sleep(0.3)  # stay under Vast's anonymous rate limit
    d = _request(url)
    offers = d.get("offers") if isinstance(d, dict) else d
    if not isinstance(offers, list):
        raise RuntimeError("Vast.ai: response has no offers list")
    return offers


def fetch_vast():
    """{gpu_id: {offers, gpus, rentable_gpus, min_price, median_price,
    sample_capped}}; raises if every query fails."""
    out = {}
    failures = 0
    for vast_name, gid, lo_gb, hi_gb in VAST_MODELS:
        base = {
            "gpu_name": {"eq": vast_name},
            "gpu_ram": {"gte": lo_gb * 1000, "lte": hi_gb * 1024},
            "verified": {"eq": True},
            "external": {"eq": False},
            "type": "on-demand",
            "limit": VAST_ROW_CAP,
        }
        queries = [
            {**base, "rentable": {"eq": True}, "order": [["dph_total", "asc"]]},
            {**base, "rentable": {"eq": True}, "order": [["machine_id", "asc"]]},
            {**base, "rentable": {"eq": False}, "order": [["machine_id", "asc"]]},
        ]
        rows = {}
        capped = False
        ok = 0
        for q in queries:
            try:
                res = _vast_query(q)
            except Exception:  # noqa: BLE001
                continue
            ok += 1
            if len(res) >= VAST_CAPPED_AT:
                capped = True
            for r in res:
                if r.get("id") is not None:
                    rows[r["id"]] = r
        if ok == 0:
            failures += 1
            continue
        machines = {}
        for r in rows.values():
            n = r.get("num_gpus") or 0
            ram_gb = (r.get("gpu_ram") or 0) / 1024
            if n < 1 or not (lo_gb <= ram_gb <= hi_gb) or r.get("machine_id") is None:
                continue
            frac = r.get("gpu_frac") or 1.0
            m = machines.setdefault(r["machine_id"],
                                    {"total": 0, "free": 0, "price": None})
            m["total"] = max(m["total"], int(round(n / frac)) if frac > 0 else n)
            if r.get("rentable") is True:
                m["free"] = max(m["free"], n)
                dph = r.get("dph_total")
                if dph and dph > 0:
                    p = dph / n
                    m["price"] = p if m["price"] is None else min(m["price"], p)
        free = [m for m in machines.values() if m["free"] > 0]
        prices = sorted(m["price"] for m in free if m["price"] is not None)
        out[gid] = {
            "offers": len(free),
            "gpus": sum(m["total"] for m in machines.values()),
            "rentable_gpus": sum(m["free"] for m in free),
            "min_price": round(prices[0], 3) if prices else None,
            "median_price": round(statistics.median(prices), 3) if prices else None,
            "sample_capped": capped,
            "partial": ok < len(queries),
        }
    if failures == len(VAST_MODELS):
        raise RuntimeError("Vast.ai: every query failed")
    return out


# --------------------------------------------------------------------------
# RunPod
# --------------------------------------------------------------------------

def _runpod(query):
    body = json.dumps({"query": query}).encode()
    d = _request(RUNPOD_URL, data=body, headers={"Content-Type": "application/json"})
    if d.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {d['errors'][0].get('message')}")
    return d.get("data") or {}


def _runpod_stock(type_id, secure):
    q = ('{ gpuTypes(input:{id:%s}) { lowestPrice(input:{gpuCount:1, secureCloud:%s})'
         ' { stockStatus } } }') % (json.dumps(type_id), "true" if secure else "false")
    for attempt in range(2):
        gts = _runpod(q).get("gpuTypes") or []
        lp = (gts[0].get("lowestPrice") if gts else None) or {}
        s = lp.get("stockStatus")
        if s in ("High", "Medium", "Low"):
            return s
        time.sleep(0.5)
    return None  # unconfirmed: anonymous API answers empty intermittently


def fetch_runpod():
    """{gpu_id: {"secure": "High|Medium|Low"|None, "community": ...}}.
    None = tier not offered, or no stock quote returned (unconfirmed)."""
    listing = _runpod("{ gpuTypes { id secureCloud communityCloud } }").get("gpuTypes")
    if not listing:
        raise RuntimeError("RunPod: empty gpuTypes")
    offered = {g["id"]: g for g in listing if g.get("id") in RUNPOD_TYPES}
    if not offered:
        raise RuntimeError("RunPod: none of the mapped GPU ids are listed (ids changed?)")
    out = {}
    for tid, g in offered.items():
        gid = RUNPOD_TYPES[tid]
        out[gid] = {
            "secure": _runpod_stock(tid, True) if g.get("secureCloud") else None,
            "community": _runpod_stock(tid, False) if g.get("communityCloud") else None,
        }
    return out


# --------------------------------------------------------------------------
# SF Compute (optional, needs SFCOMPUTE_API_KEY)
# --------------------------------------------------------------------------

def fetch_sfcompute(token):
    """Open order book over the next 7 days per accelerator. Prices are what
    SF Compute quotes: dollars per NODE-hour (not converted)."""
    start = (int(time.time()) // 60 + 1) * 60
    end = start + 7 * 86400
    out = {}
    for gid, acc in SFC_ACCELERATORS.items():
        params = urllib.parse.urlencode({
            "requirements": f"accelerator:{acc}",
            "range_start_at": start, "range_end_at": end,
        })
        d = _request(f"{SFC_URL}?{params}",
                     headers={"Authorization": f"Bearer {token}"})
        wins = d.get("data")
        if not isinstance(wins, list):
            raise RuntimeError("SF Compute: windows response has no data list")
        asks = [float(w["best_ask"]["dollars_per_node_hour"]) for w in wins
                if (w.get("best_ask") or {}).get("dollars_per_node_hour")]
        bids = [float(w["best_bid"]["dollars_per_node_hour"]) for w in wins
                if (w.get("best_bid") or {}).get("dollars_per_node_hour")]
        out[gid] = {
            "windows": len(wins),
            "ask_orders": sum(int(w.get("total_ask_order_count") or 0) for w in wins),
            "bid_orders": sum(int(w.get("total_bid_order_count") or 0) for w in wins),
            "best_ask_per_node_hr": round(min(asks), 4) if asks else None,
            "best_bid_per_node_hr": round(max(bids), 4) if bids else None,
            "range_days": 7,
        }
    return out


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------

def _providers_listing(data, gid):
    providers = data.get("providers") or {}
    n = 0
    for prov in providers.values():
        e = ((prov or {}).get("gpus") or {}).get(gid) or {}
        p = e.get("price_per_gpu_hr")
        if isinstance(p, (int, float)) and p > 0 and not e.get("stale"):
            n += 1
    return n, len(providers)


def _status(vast, runpod):
    rp = [s for s in (runpod or {}).values() if s is not None]
    v_free = (vast or {}).get("rentable_gpus") or 0
    v_listed = bool(vast) and (vast.get("gpus") or 0) > 0
    if v_free >= 8 or any(s in ("High", "Medium") for s in rp):
        return "available"
    if v_free > 0 or "Low" in rp:
        return "limited"
    if v_listed:
        return "sold_out"
    return "unknown"


def build_availability(data, log_info, log_ok, log_fail):
    try:
        prev = data.get("availability") or {}
        prev_gpus = prev.get("gpus") or {}
        now = _now()
        sources = {}

        def run(name, url, fn, note_ok):
            try:
                res = fn()
                sources[name] = {"ok": True, "url": url, "note": note_ok(res)}
                log_ok(name, note_ok(res))
                return res, True
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"[:300]
                sources[name] = {"ok": False, "url": url,
                                 "note": f"fetch failed, previous values kept: {msg}"}
                log_fail(name, msg)
                return None, False

        log_info("Availability: querying Vast.ai offers...")
        vast, vast_ok = run(
            _SRC_VAST, VAST_URL, fetch_vast,
            lambda r: (f"{len(r)} GPU types; {sum(v['rentable_gpus'] for v in r.values())} "
                       "rentable GPUs seen (lower bound: search API samples at 64 rows/query)"))
        log_info("Availability: querying RunPod stock...")
        runpod, runpod_ok = run(
            _SRC_RUNPOD, RUNPOD_URL, fetch_runpod,
            lambda r: f"stock status for {len(r)} GPU types (1-GPU pods, secure & community)")

        token = os.environ.get("SFCOMPUTE_API_KEY", "").strip()
        sfc, sfc_ok = None, False
        if token:
            sfc, sfc_ok = run(
                _SRC_SFC, SFC_URL, lambda: fetch_sfcompute(token),
                lambda r: f"open order book, next 7 days, {len(r)} accelerators ($/node-hr)")
        else:
            log_info("SF Compute orderbook skipped: SFCOMPUTE_API_KEY not set "
                     "(the order book API requires a bearer token)")
            sources[_SRC_SFC] = {"ok": False, "url": SFC_URL,
                                 "note": "skipped: SFCOMPUTE_API_KEY not set "
                                         "(order book requires authentication)"}

        def pick(gid, key, fresh, ok):
            if ok:
                return (fresh or {}).get(gid)
            old = (prev_gpus.get(gid) or {}).get(key)
            if isinstance(old, dict):
                old = dict(old, stale=True)
            return old

        gpus = {}
        for gid in GPU_IDS:
            v = pick(gid, "vast", vast, vast_ok)
            r = pick(gid, "runpod", runpod, runpod_ok)
            s = pick(gid, "sfcompute", sfc, sfc_ok) if token else None
            listing, total = _providers_listing(data, gid)
            if not vast_ok and not runpod_ok:
                status = (prev_gpus.get(gid) or {}).get("status", "unknown")
            else:
                status = _status(v, r)
            gpus[gid] = {"vast": v, "runpod": r, "sfcompute": s,
                         "providers_listing": listing, "providers_total": total,
                         "status": status}

        # history: one point per GPU per day, same-day reruns replace.
        today = now.date().isoformat()
        cutoff = (now.date().toordinal() - HISTORY_DAYS)
        history = {}
        for gid, g in gpus.items():
            old_pts = (prev.get("history") or {}).get(gid, [])
            same_day = next((p for p in old_pts if p.get("date") == today), {})
            pts = [p for p in old_pts if p.get("date") != today]
            pts.append({
                "date": today,
                # a failed Vast fetch must not erase a good same-day reading
                "vast_gpus": ((g["vast"] or {}).get("gpus") if vast_ok
                              else same_day.get("vast_gpus")),
                "rentable_gpus": ((g["vast"] or {}).get("rentable_gpus") if vast_ok
                                  else same_day.get("rentable_gpus")),
                "providers_listing": g["providers_listing"],
                "status": g["status"],
            })
            pts = [p for p in pts
                   if datetime.fromisoformat(p["date"]).toordinal() >= cutoff]
            history[gid] = sorted(pts, key=lambda p: p["date"])

        data["availability"] = {
            "as_of": now.isoformat(),
            "sources": sources,
            "gpus": gpus,
            "status_rule": STATUS_RULE,
            "history": history,
        }
    except Exception as e:  # noqa: BLE001 -- must never raise
        log_fail("availability", f"build crashed: {type(e).__name__}: {e}")
    return data


if __name__ == "__main__":
    path = "/Users/quentinreyes/ai-gpu-dashboard-rm/data.json"
    out = ("/private/tmp/claude-501/-Users-quentinreyes/1c7063bc-b4c1-4db6-b232-"
           "f92320644371/scratchpad/availability_sample.json")
    with open(path) as f:
        d = json.load(f)
    t0 = time.time()
    build_availability(d, lambda m: print("INFO", m),
                       lambda s, m: print("OK  ", s, "-", m),
                       lambda s, m: print("FAIL", s, "-", m))
    with open(out, "w") as f:
        json.dump({"availability": d.get("availability")}, f, indent=1)
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    for gid, g in (d.get("availability") or {}).get("gpus", {}).items():
        print(f"  {gid:10s} {g['status']:9s} listing={g['providers_listing']}/"
              f"{g['providers_total']} vast={g['vast']} runpod={g['runpod']}")
