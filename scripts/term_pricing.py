"""Published committed-term (reserved / contract / savings-plan) GPU prices.

Each ``fetch_*_terms()`` function hits the provider's *official* public price
page and returns::

    {gpu_id: [term, ...]}

where each term is::

    {"term": "3d"|"1w"|"2w"|"1mo"|"3mo"|"6mo"|"1yr"|"2yr"|"3yr",
     "term_max": <same codes> (only when the price covers a range of lengths),
     "rate": float,            # USD per GPU-hour, effective hourly
     "payment": "monthly"|"no_upfront"|"partial_upfront"|"all_upfront"|"unknown",
     "min_gpus": int|None,     # minimum cluster size for this rate
     "label": str,             # provider's own wording
     "effective_from": "YYYY-MM-DD"|None,
     "note": str|None}

``term`` is the SHORTEST commitment that qualifies for the price (e.g. Lambda's
"2 weeks - 1 year" -> term "2w", term_max "1yr").  "3d" exists only because
Together AI publishes a "3-30 days" reservation band.

Only the top-tier GPUs are covered (B300, GB200, B200, H200, H100-SXM, MI325X,
MI300X).  Committed rates are published exactly as listed, even when they are
HIGHER than the same provider's on-demand rate (e.g. Lambda's multi-node
InfiniBand 1-Click Clusters vs its single-node on-demand instances).

Every fetcher raises on fetch/parse failure or when it finds no term prices --
it never returns empty or guessed data.  No prices are hardcoded.

Providers checked that publish NO self-serve committed GPU price (2026-11-10),
and therefore have no fetcher here:
  * Nebius  -- "Commitment discounts ... request commitments" (sales only);
               price tables only have Preemptible + On-demand columns.
  * Crusoe  -- table has only "Current on-demand" / "New on-demand starting
               Dec 1, 2026"; reserved = "contact us".
  * RunPod  -- pricing page: "For reserved capacity and contract pricing, speak
               to our enterprise sales team".  The GraphQL gpuTypes fields
               threeMonthPrice/sixMonthPrice/oneYearPrice exist but equal
               securePrice for every GPU (no discount, placeholder values).
  * CoreWeave -- "up to 60% discounts ... for committed usage", contact sales.
  * Oracle OCI -- price-list API only has PAY_AS_YOU_GO models.
  * Vultr   -- plans-metal API has hourly/monthly/preemptible costs only; no
               explicit term (e.g. 36-month prepaid) rate is published.

Stdlib only.
"""

import re
from datetime import datetime

from extra_providers import _get, _html_tokens, _price, map_gpu_name

LAMBDA_URL = "https://lambda.ai/pricing"
DIGITALOCEAN_URL = "https://www.digitalocean.com/pricing/gpu-droplets"
TOGETHER_URL = "https://www.together.ai/pricing"
TOGETHER_CLUSTERS_URL = "https://www.together.ai/gpu-clusters"

TOP_GPU_IDS = {"B300", "GB200", "B200", "H200", "H100-SXM", "MI325X", "MI300X"}
TERM_ORDER = ["3d", "1w", "2w", "1mo", "3mo", "6mo", "1yr", "2yr", "3yr"]
PAYMENTS = {"monthly", "no_upfront", "partial_upfront", "all_upfront", "unknown"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _norm_term(qty, unit):
    """(2, 'weeks') -> '2w'; (31, 'days') -> '1mo'. Raise if no clean match."""
    unit = unit.lower().rstrip("s")
    if unit == "day":
        days = {3: "3d", 7: "1w", 14: "2w"}
        if qty in days:
            return days[qty]
        for lo, hi, code in ((28, 31, "1mo"), (89, 92, "3mo"), (180, 184, "6mo"),
                             (364, 366, "1yr")):
            if lo <= qty <= hi:
                return code
    elif unit == "week" and qty in (1, 2):
        return f"{qty}w"
    elif unit == "month" and qty in (1, 3, 6, 12, 24, 36):
        return {12: "1yr", 24: "2yr", 36: "3yr"}.get(qty, f"{qty}mo")
    elif unit in ("year", "yr") and qty in (1, 2, 3):
        return f"{qty}yr"
    raise RuntimeError(f"unrecognised commitment length: {qty} {unit}")


_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})")


def _parse_date(text):
    m = _DATE_RE.search(text)
    if not m:
        return None
    return datetime.strptime(" ".join(m.groups()), "%B %d %Y").strftime("%Y-%m-%d")


def _add(out, gid, term):
    if gid not in TOP_GPU_IDS:
        return
    out.setdefault(gid, []).append(term)


def _finish(provider, out):
    if not out:
        raise RuntimeError(f"{provider}: no committed-term GPU prices found")
    for gid, terms in out.items():
        for t in terms:
            if t["term"] not in TERM_ORDER or t.get("term_max", t["term"]) not in TERM_ORDER:
                raise RuntimeError(f"{provider}: bad term code for {gid}: {t}")
            if not isinstance(t["rate"], float) or not (0.05 <= t["rate"] <= 100):
                raise RuntimeError(f"{provider}: implausible rate for {gid}: {t['rate']!r}")
            if t["payment"] not in PAYMENTS:
                raise RuntimeError(f"{provider}: bad payment for {gid}: {t['payment']!r}")
        terms.sort(key=lambda t: (TERM_ORDER.index(t["term"]), t["min_gpus"] or 0))
    return out


# --------------------------------------------------------------------------
# Lambda -- 1-Click Clusters table on lambda.ai/pricing
#   rows: <name> | <duration> | <GPU count> | <$/GPU/hr or "—"> | Talk to our team
# --------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"^(\d+)\s*(day|week|month|year)s?\s*[–\-—to]+\s*(\d+)\s*(day|week|month|year)s?$", re.I)


def fetch_lambda_terms() -> dict:
    toks = _html_tokens(_get(LAMBDA_URL))
    try:
        start = next(i for i, t in enumerate(toks) if t == "1-Click Clusters"
                     and i + 1 < len(toks) and toks[i + 1].lower() == "pricing")
        end = next(i for i, t in enumerate(toks) if i > start and t == "Instances pricing")
    except StopIteration:
        raise RuntimeError("Lambda: 1-Click Clusters pricing section not found") from None
    sec = toks[start:end]

    # The page's own description of what a 1-Click Cluster is (explains why
    # these per-GPU rates can exceed single-node on-demand instance rates).
    desc = next((t for t in sec if t.lower().startswith("production-ready clusters")), None)
    # Effective-date notice: only applied if it sits inside the cluster section.
    eff = None
    for t in sec:
        if "take effect" in t.lower():
            eff = _parse_date(t)

    out = {}
    for i in range(len(sec) - 3):
        name, dur, count, price = sec[i], sec[i + 1], sec[i + 2], sec[i + 3]
        m = _RANGE_RE.match(dur)
        if not m:
            continue  # includes "1 year+" rows priced "—" (contact sales)
        rate = _price(price)
        cm = re.match(r"^(\d[\d,]*)(\+?)$", count)
        if rate is None or not cm:
            continue
        gid = map_gpu_name(name)
        if not gid:
            continue
        n = int(cm.group(1).replace(",", ""))
        note = "1-Click Cluster: dedicated multi-node InfiniBand cluster"
        if desc:
            note += f" (page: '{desc}')"
        note += ("; per-GPU rate can exceed Lambda's single-node on-demand instance rate."
                 " Price excludes sales tax/VAT/GST.")
        if eff is None:
            note += (" Page's 'These prices take effect ...' notice is attached to the"
                     " Instances tables, not this table.")
        _add(out, gid, {
            "term": _norm_term(int(m.group(1)), m.group(2)),
            "term_max": _norm_term(int(m.group(3)), m.group(4)),
            "rate": rate,
            "payment": "unknown",
            "min_gpus": n,
            "label": f"1-Click Cluster, {dur}, {n}{cm.group(2)} GPUs",
            "effective_from": eff,
            "note": note,
        })
    return _finish("Lambda", out)


# --------------------------------------------------------------------------
# DigitalOcean -- "12 Month Reserved Plans" cards
#   <name> | $x.xx | /GPU/hour | 12 Month Reserved Price*
# --------------------------------------------------------------------------

def fetch_digitalocean_terms() -> dict:
    toks = _html_tokens(_get(DIGITALOCEAN_URL))
    eff = None
    for t in toks:
        if "pricing is effective" in t.lower():
            eff = _parse_date(t)
            break
    footnote = next((t for t in toks if t.startswith("*Reserve via a contractual commitment")),
                    None)
    # GPUs that have an on-demand card (<name> | $x | /GPU/hour | On-Demand Price).
    od_names = {map_gpu_name(toks[i]) for i in range(len(toks) - 3)
                if toks[i + 3].lower().startswith("on-demand price")
                and _price(toks[i + 1]) is not None}
    od_start = bool(od_names)

    out = {}
    for i in range(len(toks) - 3):
        name, p, unit, label = toks[i], toks[i + 1], toks[i + 2], toks[i + 3]
        rate = _price(p)
        if rate is None or unit.replace(" ", "").lower() != "/gpu/hour":
            continue
        lab = label.lower().rstrip("*").strip()
        m = re.match(r"^(\d+)\s*month reserved price$", lab)
        if not m:
            continue
        gid = map_gpu_name(name)
        if not gid:
            continue
        note = footnote.lstrip("*") if footnote else None
        if od_start and gid not in od_names:
            note = ((note + " ") if note else "") + "Reserved-only: no on-demand plan listed."
        _add(out, gid, {
            "term": _norm_term(int(m.group(1)), "month"),
            "rate": rate,
            "payment": "unknown",
            "min_gpus": None,
            "label": f"GPU Droplet {name}, {label.rstrip('*')}",
            "effective_from": eff,
            "note": note,
        })
    return _finish("DigitalOcean", out)


# --------------------------------------------------------------------------
# Together AI -- "On-demand hourly rates and reserved capacity" table
#   columns: Preemptible | On-demand | Reserved 3-30 days | 31-90 days |
#            91-180 days | 181+ days ; rows: "NVIDIA" <model> <cells...>
# --------------------------------------------------------------------------

_DAYS_RE = re.compile(r"^(\d+)\s*[-–]\s*(\d+)\s*days$", re.I)


def fetch_together_terms() -> dict:
    toks = _html_tokens(_get(TOGETHER_URL))
    try:
        start = toks.index("On-demand hourly rates and reserved capacity")
    except ValueError:
        raise RuntimeError("Together: reserved-capacity table not found") from None
    sec = toks[start:start + 200]

    # Column headers: the run of "Pay as you go"/"<a>-<b> days"/"<n>+ days" tokens.
    hi = next((i for i, t in enumerate(sec) if t.lower() == "pay as you go"), None)
    if hi is None:
        raise RuntimeError("Together: table header not found")
    cols = []
    j = hi
    while j < len(sec) and (sec[j].lower() == "pay as you go" or sec[j].lower().endswith("days")):
        cols.append(sec[j])
        j += 1
    if len(cols) < 3 or not cols[2].lower().endswith("days"):
        raise RuntimeError(f"Together: unexpected columns {cols}")

    # Payment terms come from the GPU-clusters product page ("pay upfront").
    payment, pay_note = "unknown", None
    try:
        ct = " ".join(_html_tokens(_get(TOGETHER_CLUSTERS_URL)))
        pm = re.search(r"Reserved\s+Lower hourly rate\s+Commitment\s+([^.]*?pay upfront)", ct,
                       re.I)
        if pm:
            payment = "all_upfront"
            pay_note = f"gpu-clusters page: reserved commitment '{pm.group(1)}'"
    except Exception:  # noqa: BLE001 -- payment is metadata, the rates stand alone
        pass

    out = {}
    i = j
    while i < len(sec) - 1:
        if sec[i] != "NVIDIA":
            if sec[i] in ("Sandbox",):
                break
            i += 1
            continue
        model = sec[i + 1]
        cells = []
        k = i + 2
        while k < len(sec) and sec[k] != "NVIDIA" and len(cells) < len(cols):
            if sec[k] == "Sandbox":
                break
            cells.append(sec[k])
            k += 1
        i = k
        gid = map_gpu_name("NVIDIA " + model)
        if not gid or len(cells) != len(cols):
            continue  # reserved columns collapsed into one "Contact us" cell
        for col, cell in zip(cols[2:], cells[2:]):
            rate = _price(cell)
            dm = _DAYS_RE.match(col)
            if rate is None or not dm:
                continue  # "181+ days" -> "Contact us"
            note = "All prices are per GPU per hour (page)."
            if pay_note:
                note += f" {pay_note}."
            _add(out, gid, {
                "term": _norm_term(int(dm.group(1)), "days"),
                "term_max": _norm_term(int(dm.group(2)), "days"),
                "rate": rate,
                "payment": payment,
                "min_gpus": None,
                "label": f"GPU Clusters reserved, NVIDIA {model}, {col}",
                "effective_from": None,
                "note": note,
            })
    return _finish("Together AI", out)


# --------------------------------------------------------------------------

TERM_PROVIDERS = {
    "Lambda": fetch_lambda_terms,
    "DigitalOcean": fetch_digitalocean_terms,
    "Together": fetch_together_terms,  # key matches data.json providers
}


if __name__ == "__main__":
    import json
    for name, fn in TERM_PROVIDERS.items():
        try:
            res = fn()
            print(f"== {name} OK ({len(res)} GPUs)")
            print(json.dumps(res, indent=1, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            print(f"== {name} FAILED: {type(e).__name__}: {e}")
