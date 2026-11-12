"""Public on-demand GPU price fetchers for additional cloud providers.

Each ``fetch_*_pricing()`` function hits the provider's *official* public
price source (API or pricing page), and returns::

    {gpu_id: {"on_demand": float,          # USD per GPU-hour
              "instance": str,             # shape / plan / product name
              "gpus_per_instance": int,
              # optional, only when the provider publishes them explicitly:
              "spot": float, "reserved_1yr": float, "reserved_3yr": float}}

Every fetcher raises an Exception on any fetch/parse failure or when it finds
zero supported GPUs -- it never returns empty or guessed data.  No prices are
hardcoded here; the only static data is shape/plan metadata (names, GPU counts)
for providers whose price list is already quoted per GPU-hour.

Stdlib only.
"""

import html
import json
import re
import urllib.request

TIMEOUT = 30
USER_AGENT = "ai-gpu-dashboard/1.0"

VALID_GPU_IDS = {
    "B300", "B200", "GB200", "RTX-5090", "H200", "H100-SXM", "H100-PCIe",
    "L40S", "A100-80GB", "A100-40GB", "RTX-4090", "MI300X", "MI325X",
    "MI250X", "L4", "T4", "GH200", "A100-PCIe",
}

OCI_URL = "https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/?currencyCode=USD"
NEBIUS_URL = "https://nebius.com/prices"
DIGITALOCEAN_URL = "https://www.digitalocean.com/pricing/gpu-droplets"
VULTR_METAL_URL = "https://api.vultr.com/v2/plans-metal?per_page=500"
VULTR_VCG_URL = "https://api.vultr.com/v2/plans?type=vcg&per_page=500"
CRUSOE_URL = "https://crusoe.ai/cloud/pricing"
HYPERBOLIC_URL = "https://www.hyperbolic.ai/marketplace"
# Paperspace is part of DigitalOcean; its price list lives in DO's docs, which
# serve a Markdown rendering of every page (append index.html.md).
PAPERSPACE_URL = "https://docs.digitalocean.com/products/paperspace/pricing/index.html.md"
HOTAISLE_URL = "https://hotaisle.xyz/pricing/"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        status = getattr(resp, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP {status} from {url}")
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _get_json(url):
    text = _get(url)
    try:
        return json.loads(text)
    except ValueError as e:
        raise RuntimeError(f"invalid JSON from {url}: {e}") from e


def _html_tokens(page):
    """Flatten HTML into a list of non-empty text tokens (one per text node)."""
    s = re.sub(r"<script\b.*?</script>", " ", page, flags=re.S | re.I)
    s = re.sub(r"<style\b.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "\x00", s)
    out = []
    for tok in s.split("\x00"):
        tok = re.sub(r"\s+", " ", html.unescape(tok)).strip()
        if tok:
            out.append(tok)
    return out


_PRICE_RE = re.compile(r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _price(text):
    """Parse '$4.41' -> 4.41; return None if not a plain price."""
    m = _PRICE_RE.match(text.strip())
    return float(m.group(1)) if m else None


def map_gpu_name(name):
    """Map a provider's marketing GPU name to a dashboard gpu_id, or None.

    Conservative: anything ambiguous or outside the tracked set returns None.
    """
    n = name.upper().replace("™", "").replace("®", "")
    n = re.sub(r"[_\-]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    def has(tok):
        return re.search(r"(?<![A-Z0-9])" + tok + r"(?![A-Z0-9])", n) is not None

    if has("GB300") or has("MI355X") or has("MI350X") or "RUBIN" in n:
        return None
    if has("GB200"):
        return "GB200"
    if has("GH200"):
        return "GH200"
    if has("B300"):
        return "B300"
    if has("B200"):
        return "B200"
    if has("H200"):
        return None if has("NVL") else "H200"
    if has("H100"):
        if has("NVL"):
            return None
        if has("PCIE"):
            return "H100-PCIe"
        return "H100-SXM"
    if has("L40S"):
        return "L40S"
    if has("A100"):
        if has("40GB") or has("40G"):
            return "A100-40GB"
        if has("PCIE"):
            return "A100-PCIe"
        if has("80GB") or has("80G") or has("SXM"):
            return "A100-80GB"
        return None  # unknown A100 variant
    if has("MI325X"):
        return "MI325X"
    if has("MI300X"):
        return "MI300X"
    if has("MI250X"):
        return "MI250X"
    if has("RTX 5090"):
        return "RTX-5090"
    if has("RTX 4090"):
        return "RTX-4090"
    if has("L4"):
        return "L4"
    if has("T4"):
        return "T4"
    return None


def _prefer(new, old):
    """True if candidate `new` should replace `old` (prefer 8-GPU instances,
    then larger instances, then cheaper per-GPU)."""
    if old is None:
        return True

    def key(e):
        g = e["gpus_per_instance"]
        return (g == 8, g, -e["on_demand"])

    return key(new) > key(old)


def _finish(provider, out):
    if not out:
        raise RuntimeError(f"{provider}: no supported GPUs found")
    for gid, e in out.items():
        if gid not in VALID_GPU_IDS:
            raise RuntimeError(f"{provider}: invalid gpu_id {gid}")
        od = e.get("on_demand")
        if not isinstance(od, float) or not (0.05 <= od <= 100):
            raise RuntimeError(f"{provider}: implausible on-demand price for {gid}: {od!r}")
        if not isinstance(e.get("gpus_per_instance"), int) or e["gpus_per_instance"] < 1:
            raise RuntimeError(f"{provider}: bad gpus_per_instance for {gid}")
    return out


# --------------------------------------------------------------------------
# Oracle Cloud (OCI) -- public price-list API, priced per GPU-hour
# --------------------------------------------------------------------------

# OCI product token (from "... Compute - GPU - <token>") -> gpu_id.
# Unlisted tokens (A10, E3, E4, MI355X, GB300, RTX PRO 6000, H100T ...) are skipped.
_OCI_TOKENS = {
    "H100": "H100-SXM",
    "H200": "H200",
    "B200": "B200",
    "B300": "B300",
    "GB200": "GB200",
    "MI300X": "MI300X",
    "L40S": "L40S",
    "A100 - v2": "A100-80GB",
}
# Shape metadata (not prices): OCI bare-metal shape that each part number bills.
_OCI_SHAPES = {
    "H100-SXM": ("BM.GPU.H100.8", 8),
    "H200": ("BM.GPU.H200.8", 8),
    "B200": ("BM.GPU.B200.8", 8),
    "B300": ("BM.GPU.B300.8", 8),
    "GB200": ("BM.GPU.GB200.4", 4),
    "MI300X": ("BM.GPU.MI300X.8", 8),
    "L40S": ("BM.GPU.L40S.4", 4),
    "A100-80GB": ("BM.GPU.A100-v2.8", 8),
}
_OCI_NAME_RE = re.compile(
    r"^(?:OCI|Oracle Cloud Infrastructure)?\s*-?\s*Compute\s*-\s*GPU\s*-\s*(.+?)\s*$")


def fetch_oci_pricing() -> dict:
    data = _get_json(OCI_URL)
    items = data.get("items") if isinstance(data, dict) else None
    if not items:
        raise RuntimeError("OCI: price list has no items")
    out = {}
    for it in items:
        if it.get("serviceCategory") != "Compute - GPU":
            continue
        if (it.get("metricName") or "").strip() != "GPU Per Hour":
            continue
        m = _OCI_NAME_RE.match(it.get("displayName") or "")
        if not m:
            continue
        gid = _OCI_TOKENS.get(m.group(1))
        if not gid:
            continue
        price = None
        for loc in it.get("currencyCodeLocalizations") or []:
            if loc.get("currencyCode") != "USD":
                continue
            for p in loc.get("prices") or []:
                if p.get("model") == "PAY_AS_YOU_GO" and p.get("value") is not None:
                    price = float(p["value"])
        if price is None or price <= 0:
            continue
        if gid in out:
            # Same GPU listed twice: keep the first, but refuse silent conflicts.
            if abs(out[gid]["on_demand"] - price) > 1e-9:
                raise RuntimeError(f"OCI: conflicting prices for {gid}")
            continue
        shape, n = _OCI_SHAPES[gid]
        out[gid] = {"on_demand": price, "instance": shape, "gpus_per_instance": n}
    return _finish("OCI", out)


# --------------------------------------------------------------------------
# Nebius -- Next.js __NEXT_DATA__ embeds the page's CMS blocks as JSON
# --------------------------------------------------------------------------

def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def fetch_nebius_pricing() -> dict:
    page = _get(NEBIUS_URL)
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  page, re.S)
    if not m:
        raise RuntimeError("Nebius: __NEXT_DATA__ not found (page not machine-readable)")
    root = json.loads(m.group(1))

    # Page content is a JSON string nested inside __NEXT_DATA__; decode any such strings.
    tables = []

    def collect(obj):
        for d in _walk(obj):
            for v in d.values():
                if isinstance(v, str) and v.startswith("{") and '"blocks"' in v:
                    try:
                        collect(json.loads(v))
                    except ValueError:
                        pass
            if d.get("type") == "highlight-table-block":
                content = (d.get("table") or {}).get("content")
                if isinstance(content, list) and content:
                    tables.append(content)

    collect(root)

    out = {}
    for content in tables:
        header = [str(h).strip() for h in content[0]]
        od_col = next((i for i, h in enumerate(header)
                       if h.lower().startswith("on-demand") and "gpu-hour" in h.lower()), None)
        if od_col is None:
            continue
        for row in content[1:]:
            if len(row) <= od_col:
                continue
            name = str(row[0]).strip()
            gid = map_gpu_name(name)
            if not gid:
                continue
            cell = str(row[od_col]).replace(" ", " ").strip()
            pm = re.match(r"^(from\s+)?\$\s*([0-9]+(?:\.[0-9]+)?)\s*$", cell, re.I)
            if not pm:
                continue  # "Contact us", etc.
            price = float(pm.group(2))
            # HGX platforms are 8-GPU nodes; L40S is sold per single GPU VM.
            n = 8 if "HGX" in name.upper() else 1
            inst = f"{name} ({n}-GPU)" if n == 8 else name
            if pm.group(1):
                inst += " [from]"
            cand = {"on_demand": price, "instance": inst, "gpus_per_instance": n}
            old = out.get(gid)
            if old is None or _prefer(cand, old) or (
                    cand["gpus_per_instance"] == old["gpus_per_instance"]
                    and price < old["on_demand"]):
                out[gid] = cand
    return _finish("Nebius", out)


# --------------------------------------------------------------------------
# DigitalOcean -- server-rendered pricing cards
# --------------------------------------------------------------------------

def fetch_digitalocean_pricing() -> dict:
    toks = _html_tokens(_get(DIGITALOCEAN_URL))
    od, spot, res1 = {}, {}, {}
    names = {}
    gpus = {}
    i = 0
    while i < len(toks) - 3:
        name, p, unit, label = toks[i], toks[i + 1], toks[i + 2], toks[i + 3]
        price = _price(p)
        if price is None or unit.replace(" ", "").lower() != "/gpu/hour":
            i += 1
            continue
        gid = map_gpu_name(name)
        if gid:
            lab = label.lower()
            if lab.startswith("on-demand"):
                od[gid] = price
                names[gid] = name
                # "GPUs per Droplet" -> "Choose between 1 or 8 GPU's per droplet" | "1"
                for j in range(i + 4, min(i + 10, len(toks) - 1)):
                    if toks[j].lower() == "gpus per droplet":
                        counts = [int(x) for x in re.findall(r"\d+", toks[j + 1])]
                        if counts:
                            gpus[gid] = 8 if 8 in counts else max(counts)
                        break
            elif "reserved" in lab and "12 month" in lab:
                res1[gid] = price
            elif "spot" in lab:
                spot[gid] = price
        i += 4

    out = {}
    for gid, price in od.items():
        if gid not in gpus:
            raise RuntimeError(f"DigitalOcean: GPU count not found for {names[gid]}")
        n = gpus[gid]
        e = {"on_demand": price,
             "instance": f"GPU Droplet {names[gid]} ({n}-GPU)",
             "gpus_per_instance": n}
        if gid in spot:
            e["spot"] = spot[gid]
        if gid in res1:
            e["reserved_1yr"] = res1[gid]
        out[gid] = e
    return _finish("DigitalOcean", out)


# --------------------------------------------------------------------------
# Vultr -- public plans API (no auth). hourly_cost is USD per instance-hour.
# --------------------------------------------------------------------------

_VULTR_TYPES = {
    "NVIDIA_B200": "B200",
    "NVIDIA_B300": "B300",
    "NVIDIA_GB200": "GB200",
    "NVIDIA_H200": "H200",
    "NVIDIA_H100": "H100-SXM",
    "NVIDIA_H100_SXM": "H100-SXM",
    "NVIDIA_GH200": "GH200",
    "NVIDIA_A100_SXM": "A100-80GB",
    "NVIDIA_L40S": "L40S",
    "NVIDIA_L4": "L4",
    "AMD_MI300X": "MI300X",
    "AMD_MI325X": "MI325X",
}


def fetch_vultr_pricing() -> dict:
    plans = []
    metal = _get_json(VULTR_METAL_URL)
    if not isinstance(metal.get("plans_metal"), list):
        raise RuntimeError("Vultr: plans-metal response missing 'plans_metal'")
    plans += metal["plans_metal"]
    vcg = _get_json(VULTR_VCG_URL)
    if not isinstance(vcg.get("plans"), list):
        raise RuntimeError("Vultr: plans?type=vcg response missing 'plans'")
    plans += vcg["plans"]

    out = {}
    for p in plans:
        gid = _VULTR_TYPES.get(p.get("gpu_type"))
        if not gid:
            continue
        # Vultr's 8-GPU bare-metal plans (H100, B200, MI300X...) report
        # deploy_ondemand: false -- they are preemptible/contracted only, so
        # their hourly_cost is not an on-demand list price. Publish only plans
        # Vultr lets anyone deploy on demand.
        if p.get("deploy_ondemand") is not True:
            continue
        n = p.get("gpu_count")
        hourly = p.get("hourly_cost")
        if not isinstance(n, int) or n < 1 or not hourly or hourly <= 0:
            continue
        # Fractional vGPU plans would have gpu_vram_gb < one full GPU; require
        # whole-GPU plans by checking VRAM scales with the count.
        vram = p.get("gpu_vram_gb")
        if gid == "A100-80GB" and vram and round(vram / n) != 80:
            continue
        per_gpu = round(float(hourly) / n, 4)
        cand = {"on_demand": per_gpu, "instance": p.get("id", "?"),
                "gpus_per_instance": n}
        if _prefer(cand, out.get(gid)):
            out[gid] = cand
    return _finish("Vultr", out)


# --------------------------------------------------------------------------
# Crusoe -- Webflow page: GPU table is rendered as parallel column lists
# --------------------------------------------------------------------------

def fetch_crusoe_pricing() -> dict:
    page = _get(CRUSOE_URL)
    i = page.find("GPU instances pricing")
    j = page.find("CPU instances pricing", i + 1)
    if i < 0 or j < 0:
        raise RuntimeError("Crusoe: GPU pricing table not found (page layout changed?)")
    seg = page[i:j]

    # Column headers, to confirm which list is the current on-demand price.
    heads = [re.sub(r"\s+", " ", html.unescape(h)).strip()
             for h in re.findall(r'class="pricing-heading[^"]*">([^<]*)<', seg)]
    if len(heads) < 2 or heads[0].lower() != "gpu model" or \
            heads[1].lower() != "current on-demand":
        raise RuntimeError(f"Crusoe: unexpected table headers {heads}")

    lists = seg.split('class="w-dyn-list"')[1:]
    if len(lists) < 2:
        raise RuntimeError("Crusoe: expected name and price column lists")

    # Column 0: GPU names + tags (memory, form factor).
    rows = []
    for item in re.split(r'role="listitem"', lists[0])[1:]:
        h = re.search(r'<h4[^>]*>(.*?)</h4>', item, re.S)
        tags = [html.unescape(t).strip() for t in
                re.findall(r'class="pricing-tag[^"]*">([^<]*)<', item)]
        if h:
            rows.append((html.unescape(re.sub(r"<[^>]+>", "", h.group(1))).strip(),
                         [t for t in tags if t]))

    # Column 1: current on-demand price per row.
    cells = []
    for item in re.split(r'role="listitem"', lists[1])[1:]:
        text = " ".join(_html_tokens(item))
        pm = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*GPU-hr", text)
        cells.append(float(pm.group(1)) if pm else None)

    if not rows or len(rows) != len(cells):
        raise RuntimeError(f"Crusoe: column length mismatch ({len(rows)} names, "
                           f"{len(cells)} prices)")

    out = {}
    for (name, tags), price in zip(rows, cells):
        if price is None:
            continue  # "Contact sales"
        full = " ".join([name] + tags)
        gid = map_gpu_name(full)
        if not gid:
            continue
        form = {t.upper() for t in tags}
        n = 8 if form & {"HGX", "SXM"} or gid in ("MI300X", "MI325X") else 1
        inst = f"{full} ({n}-GPU)" if n == 8 else f"{full} (per GPU)"
        cand = {"on_demand": price, "instance": inst, "gpus_per_instance": n}
        if _prefer(cand, out.get(gid)):
            out[gid] = cand
    return _finish("Crusoe", out)


# --------------------------------------------------------------------------
# Hyperbolic -- marketplace page: "GPU Type | Starting From (per GPU hour)"
# --------------------------------------------------------------------------

def fetch_hyperbolic_pricing() -> dict:
    toks = _html_tokens(_get(HYPERBOLIC_URL))
    try:
        start = next(i for i, t in enumerate(toks)
                     if t.lower().startswith("starting from") and "per gpu hour" in t.lower())
    except StopIteration:
        raise RuntimeError("Hyperbolic: 'Starting From (per GPU hour)' table not found")
    if toks[start - 1].lower() != "gpu type":
        raise RuntimeError(f"Hyperbolic: unexpected table header {toks[start - 1:start + 1]}")
    out = {}
    i = start + 1
    # Rows are (name, "$3.19 / HR") pairs until the first non-price cell.
    while i + 1 < len(toks):
        name, cell = toks[i], toks[i + 1]
        pm = re.match(r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*HR$", cell, re.I)
        if not pm:
            break
        gid = map_gpu_name(name)
        if gid:
            # On-demand GPUs are rented per GPU (1..8); the table quotes the
            # lowest current per-GPU rate, refreshed weekly by Hyperbolic.
            out[gid] = {"on_demand": float(pm.group(1)),
                        "instance": f"{name} on-demand [from]",
                        "gpus_per_instance": 1}
        i += 2
    return _finish("Hyperbolic", out)


# --------------------------------------------------------------------------
# Paperspace (DigitalOcean) -- Markdown price tables in DO's docs
# --------------------------------------------------------------------------

# Paperspace machine-type names -> gpu_id. "A100" is the 40GB part (the
# 80GB card is listed separately as "A100-80G"); Paperspace H100s are HGX
# (SXM) boards.
_PAPERSPACE_TYPES = {
    "A100": "A100-40GB",
    "A100-80G": "A100-80GB",
    "H100": "H100-SXM",
}


def _md_table(md, heading):
    """Rows (list of cell lists) of the first Markdown table after `heading`."""
    i = md.find(heading)
    if i < 0:
        return None
    rows = []
    started = False
    for line in md[i:].splitlines()[1:]:
        line = line.strip()
        if line.startswith("|"):
            started = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            rows.append(cells)
        elif started:
            break
    return rows


def fetch_paperspace_pricing() -> dict:
    md = _get(PAPERSPACE_URL)
    single = _md_table(md, "### GPU Machines")
    multi = _md_table(md, "### Multi-GPU Machines")
    if not single or [c.lower() for c in single[0][:2]] != ["gpu machine", "hourly price"]:
        raise RuntimeError("Paperspace: GPU Machines table not found or header changed")

    def price(cell):
        m = re.match(r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*hr$", cell.strip(), re.I)
        return float(m.group(1)) if m else None

    # Multi-GPU rows: "A100-80G ×8" -> (name, count, instance price)
    multis = {}
    for row in (multi or [])[1:]:
        m = re.match(r"^(.+?)\s*[×x]\s*(\d+)$", row[0].strip())
        p = price(row[1]) if len(row) > 1 else None
        if m and p:
            multis[m.group(1).strip()] = (int(m.group(2)), p)

    out = {}
    for row in single[1:]:
        if len(row) < 2:
            continue
        gid = _PAPERSPACE_TYPES.get(row[0].strip())
        p = price(row[1])
        if not gid or p is None:
            continue
        e = {"on_demand": p, "instance": f"Paperspace {row[0].strip()}",
             "gpus_per_instance": 1}
        if row[0].strip() in multis:
            n, total = multis[row[0].strip()]
            # DO documents multi-GPU price = n x single-GPU rate; only adopt
            # the larger shape when the published numbers agree.
            if abs(total / n - p) <= 0.01:
                e = {"on_demand": p, "instance": f"Paperspace {row[0].strip()} x{n}",
                     "gpus_per_instance": n}
        out[gid] = e
    return _finish("Paperspace", out)


# --------------------------------------------------------------------------
# Hot Aisle -- AMD MI300X cloud; static pricing cards
# --------------------------------------------------------------------------

def fetch_hotaisle_pricing() -> dict:
    toks = _html_tokens(_get(HOTAISLE_URL))
    out = {}
    # Card layout: <tier> | <"1x" / "2x & 4x" / "8x"> | <GPU> | "$2.99/GPU/hr"
    for i in range(2, len(toks)):
        pm = re.match(r"^\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*GPU\s*/\s*hr$", toks[i], re.I)
        if not pm:
            continue
        counts_tok, gpu_tok = toks[i - 2], toks[i - 1]
        if not re.fullmatch(r"[\dx&,\s]+", counts_tok, re.I):
            continue  # summary banner, not a card
        counts = [int(c) for c in re.findall(r"(\d+)\s*x", counts_tok, re.I)]
        gid = map_gpu_name(gpu_tok)
        if not gid or not counts:
            continue
        n = max(counts)
        kind = "bare metal" if n == 8 else "VM"
        cand = {"on_demand": float(pm.group(1)),
                "instance": f"{n}x {gpu_tok} {kind}", "gpus_per_instance": n}
        if _prefer(cand, out.get(gid)):
            out[gid] = cand
    return _finish("Hot Aisle", out)


# --------------------------------------------------------------------------
# Considered and deliberately NOT fetched (no honest public price source as of
# 2026-11-12):
#   SF Compute   -- prices only via its order book, which requires a bearer
#                   token (api.sfcompute.com/preview/v2/orderbook -> 401);
#                   /pricing redirects to login.
#   Fluidstack   -- no public price list (pricing URLs 404; sales-led).
#   TensorWave   -- product pages show "Starting at $X /GPU HR" beside
#                   "Book a call"; no on-demand/term is stated, so it cannot
#                   be published as an on-demand rate.
#   Voltage Park -- every tier on /pricing says "Contact for pricing".


# --------------------------------------------------------------------------

PROVIDERS = {  # provider key -> (fetch fn, provider_name, source_tag)
    "Oracle": (fetch_oci_pricing, "Oracle Cloud (OCI)", "oci_price_api"),
    "Nebius": (fetch_nebius_pricing, "Nebius", "nebius_pricing_page"),
    "DigitalOcean": (fetch_digitalocean_pricing, "DigitalOcean", "digitalocean_pricing_page"),
    "Vultr": (fetch_vultr_pricing, "Vultr", "vultr_plans_api"),
    "Crusoe": (fetch_crusoe_pricing, "Crusoe", "crusoe_pricing_page"),
    "Hyperbolic": (fetch_hyperbolic_pricing, "Hyperbolic", "hyperbolic_marketplace_page"),
    "Paperspace": (fetch_paperspace_pricing, "Paperspace (DigitalOcean)", "paperspace_do_docs"),
    "Hot Aisle": (fetch_hotaisle_pricing, "Hot Aisle", "hotaisle_pricing_page"),
}


if __name__ == "__main__":
    for key, (fn, pname, tag) in PROVIDERS.items():
        try:
            res = fn()
            print(f"== {pname} [{tag}] OK ({len(res)} GPUs)")
            for gid in sorted(res):
                print(f"   {gid:10s} {res[gid]}")
        except Exception as e:  # noqa: BLE001
            print(f"== {pname} [{tag}] FAILED: {type(e).__name__}: {e}")
