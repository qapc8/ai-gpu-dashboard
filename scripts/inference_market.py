"""Inference market data: demand and price dispersion, from OpenRouter.

Two things this dashboard could not previously say about inference are now
measured rather than curated:

  demand      Which models are actually being used, in tokens and requests per
              day. The roster and its `tokens_7d` figures used to be copied by
              hand off openrouter.ai/rankings, so they aged the moment nobody
              got round to it. OpenRouter has no rankings API, but the rankings
              page ships its dataset in the page payload, and that is a
              published figure, not an estimate.

  dispersion  What the same model costs across the providers serving it.
              /api/v1/models/{slug}/endpoints lists every provider with its
              price, quantization, context window and uptime. Open-weights
              models run an 8x spread between cheapest and dearest; closed
              models sit near 1.1x because every route is reselling one API.
              That contrast is the story, and it mirrors the GPU price
              dispersion the rest of the dashboard is built on.

Nothing here is modelled. Every number is a published rate or a published
volume, or arithmetic over them.
"""

import json
import re
import urllib.request
from datetime import datetime, timezone

RANKINGS_URL = "https://openrouter.ai/rankings"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"

_UA = {"User-Agent": "Mozilla/5.0 (gpu-dashboard inference-market)"}

# Keep roughly a quarter of daily observations. Enough to show a trend without
# growing data.json without bound.
HISTORY_DAYS = 90


def _get(url, timeout=40, as_json=True):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    return json.loads(body) if as_json else body


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Demand: the rankings payload
# ---------------------------------------------------------------------------

def _extract_embedded_array(html, key):
    """Pull the ranking rows out of the escaped JSON in a Next.js RSC payload.

    Anchoring on a key name broke: the rows moved from `"rankingData":[...]`
    into a React Query dehydrated state (`"queries":[{"state":{"data":[...]}}]`)
    and the section went stale for four runs before the freshness stamp
    surfaced it. Locate the array by what it contains -- the first object with
    a `model_permaslug` -- and walk back to its opening bracket, so the next
    re-nesting does not matter.
    """
    marker = f'\\"{key}\\":['
    i = html.find(marker)
    if i >= 0:
        start = html.index("[", i + len(marker) - 1)
    else:
        probe = html.find('model_permaslug')
        if probe < 0:
            raise RuntimeError("OpenRouter rankings: no ranking rows in page payload")
        # Back up to the '[' that opens the array of row objects.
        brace = html.rfind("{", 0, probe)
        start = html.rfind("[", 0, brace)
        if start < 0:
            raise RuntimeError("OpenRouter rankings: could not locate the rows array")
    depth, j = 0, start
    while j < len(html):
        ch = html[j]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        j += 1
    raw = html[start:j + 1].replace('\\"', '"').replace("\\\\", "\\")
    return json.loads(raw)


def fetch_openrouter_rankings():
    """Daily per-model token and request volumes from the rankings page.

    Returns (as_of_date, [row]). The page serves one day of the top models; the
    pipeline runs daily, so history is accumulated on our side the same way the
    GPU price history is.
    """
    html = _get(RANKINGS_URL, timeout=60, as_json=False)
    rows = _extract_embedded_array(html, "rankingData")
    if not rows:
        raise RuntimeError("OpenRouter rankings: empty dataset")

    as_of = max(r.get("date", "")[:10] for r in rows if r.get("date"))
    out = []
    for r in rows:
        if r.get("date", "")[:10] != as_of:
            continue
        prompt = _num(r.get("total_prompt_tokens"))
        completion = _num(r.get("total_completion_tokens"))
        if prompt + completion <= 0:
            continue
        out.append({
            "permaslug": r.get("model_permaslug"),
            "variant": r.get("variant"),
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(prompt + completion),
            "reasoning_tokens": int(_num(r.get("total_native_tokens_reasoning"))),
            "cached_tokens": int(_num(r.get("total_native_tokens_cached"))),
            "requests": int(_num(r.get("count"))),
            "tool_calls": int(_num(r.get("total_tool_calls"))),
        })
    out.sort(key=lambda r: -r["total_tokens"])
    return as_of, out


def _base_slug(permaslug):
    """`deepseek/deepseek-v4-flash-20260731` -> `deepseek/deepseek-v4-flash`.

    Rankings carry a dated permaslug; the models API is keyed on the undated
    one. Strip a trailing -YYYYMMDD only, so a real version suffix survives.
    """
    return re.sub(r"-\d{8}$", "", permaslug or "")


# ---------------------------------------------------------------------------
# Price dispersion: per-model provider endpoints
# ---------------------------------------------------------------------------

def fetch_model_endpoints(slug):
    """Every provider serving `slug`, with price, quantization and uptime."""
    data = _get(ENDPOINTS_URL.format(slug=slug), timeout=30).get("data") or {}
    out = []
    for e in data.get("endpoints") or []:
        pricing = e.get("pricing") or {}
        inp = _num(pricing.get("prompt"), None) if pricing.get("prompt") is not None else None
        outp = _num(pricing.get("completion"), None) if pricing.get("completion") is not None else None
        if not outp or outp <= 0:
            # Free / BYOK routes do not describe the market price.
            continue
        out.append({
            "provider": e.get("provider_name"),
            "quantization": e.get("quantization") or "unknown",
            "input": round(inp * 1_000_000, 4) if inp else 0.0,
            "output": round(outp * 1_000_000, 4),
            "context_k": round(e["context_length"] / 1000) if e.get("context_length") else None,
            "uptime_1d": round(e["uptime_last_1d"], 1) if e.get("uptime_last_1d") is not None else None,
        })
    out.sort(key=lambda r: r["output"])
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2, 4)


def build_inference(data, as_of, rankings, catalog, endpoints_by_slug, log=lambda m: None):
    """Assemble the inference block from measured rankings + endpoint prices."""
    models = []
    total_tokens = sum(r["total_tokens"] for r in rankings) or 1

    # OpenRouter ranks dated snapshots separately, and two releases of the same
    # model are often served side by side (deepseek-v4-flash-20260731 alongside
    # -20260423). They share a base slug, so left alone they render as two rows
    # with the same name and collide in the history dict. Keep them distinct.
    base_counts = {}
    for r in rankings:
        b = _base_slug(r["permaslug"])
        base_counts[b] = base_counts.get(b, 0) + 1

    for rank, r in enumerate(rankings, start=1):
        slug = _base_slug(r["permaslug"])
        meta = catalog.get(slug) or {}
        eps = endpoints_by_slug.get(slug) or []
        outs = [e["output"] for e in eps if e["output"]]
        ins = [e["input"] for e in eps if e["input"]]

        pricing = meta.get("pricing") or {}
        or_in = round(_num(pricing.get("prompt")) * 1_000_000, 4) if pricing.get("prompt") else None
        or_out = round(_num(pricing.get("completion")) * 1_000_000, 4) if pricing.get("completion") else None

        name = meta.get("name") or slug
        if base_counts.get(slug, 0) > 1:
            stamp = re.search(r"-(\d{8})$", r["permaslug"] or "")
            if stamp:
                d8 = stamp.group(1)
                # The catalogue name describes whichever release the base slug
                # currently resolves to, so it can carry its own MMDD tag --
                # "DeepSeek V4 Flash 0423". Appending the real snapshot date to
                # that gives "V4 Flash 0423 (2026-07-31)", which contradicts
                # itself. Drop the trailing tag and state the date once.
                base = re.sub(r"\s+\d{4}$", "", name)
                name = f"{base} ({d8[:4]}-{d8[4:6]}-{d8[6:]})"

        entry = {
            "rank": rank,
            "slug": slug,
            "permaslug": r["permaslug"],
            "name": name,
            "author": slug.split("/")[0] if "/" in slug else None,
            # A HuggingFace id is OpenRouter's own marker that the weights are
            # published. It is a fact about the listing, not our judgement --
            # and it is the key the model-fit sizing reads architecture from.
            "open_weights": bool(meta.get("hugging_face_id")),
            "hf_id": meta.get("hugging_face_id"),
            "context_k": round(meta["context_length"] / 1000) if meta.get("context_length") else None,
            "total_tokens": r["total_tokens"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "reasoning_tokens": r["reasoning_tokens"],
            "cached_tokens": r["cached_tokens"],
            "requests": r["requests"],
            "tool_calls": r["tool_calls"],
            "share_pct": round(r["total_tokens"] / total_tokens * 100, 2),
            "tokens_per_request": round(r["total_tokens"] / r["requests"]) if r["requests"] else None,
            "output_share_pct": round(r["completion_tokens"] / r["total_tokens"] * 100, 1),
            "price_in": or_in,
            "price_out": or_out,
            "endpoint_count": len(eps),
            "endpoints": eps,
        }
        if outs:
            entry.update({
                "out_low": min(outs),
                "out_high": max(outs),
                "out_median": _median(outs),
                "spread_x": round(max(outs) / min(outs), 1) if min(outs) else None,
                "in_low": min(ins) if ins else None,
                "cheapest_provider": eps[0]["provider"],
                "dearest_provider": eps[-1]["provider"],
                "quantizations": sorted({e["quantization"] for e in eps if e["quantization"] != "unknown"}),
            })
        models.append(entry)

    # ---- provider roll-up across every model we priced ----
    prov = {}
    for m in models:
        for e in m["endpoints"]:
            p = prov.setdefault(e["provider"], {
                "name": e["provider"], "models": 0, "outputs": [], "uptimes": [], "cheapest_wins": 0,
            })
            p["models"] += 1
            p["outputs"].append(e["output"])
            if e["uptime_1d"] is not None:
                p["uptimes"].append(e["uptime_1d"])
        if m.get("cheapest_provider") and m["cheapest_provider"] in prov:
            prov[m["cheapest_provider"]]["cheapest_wins"] += 1
    providers = [{
        "name": p["name"],
        "models": p["models"],
        "median_output": _median(p["outputs"]),
        "median_uptime_1d": _median(p["uptimes"]),
        "cheapest_wins": p["cheapest_wins"],
    } for p in prov.values()]
    providers.sort(key=lambda p: (-p["models"], p["name"]))

    # ---- accumulate our own daily history ----
    previous = (data.get("inference") or {}).get("history") or {}
    history = dict(previous)
    # Keyed on permaslug: it is the only identifier unique per ranked row.
    history[as_of] = {m["permaslug"]: m["total_tokens"] for m in models}
    for day in sorted(history)[:-HISTORY_DAYS]:
        history.pop(day, None)

    open_tokens = sum(m["total_tokens"] for m in models if m["open_weights"])
    spreads = [m["spread_x"] for m in models if m.get("spread_x")]

    data["inference"] = {
        "as_of": as_of,
        "updated": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "providers": providers,
        "history": history,
        "totals": {
            "models": len(models),
            "tokens": total_tokens,
            "requests": sum(m["requests"] for m in models),
            "open_weights_token_share_pct": round(open_tokens / total_tokens * 100, 1),
            "median_spread_x": _median(spreads),
            "provider_count": len(providers),
        },
    }

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["inference"] = {
        "basis": "measured",
        "detail": (
            "OpenRouter. Token and request volumes are the published daily "
            "rankings figures; per-provider prices, quantization, context and "
            "uptime come from /api/v1/models/{slug}/endpoints. The roster was "
            "previously copied by hand and went stale between edits; it is now "
            "whatever the rankings say that day. Trend history is accumulated "
            "here one observation per run, so it starts short and lengthens."
        ),
    }
    log(f"{len(models)} models, {len(providers)} providers, {len(history)} days of history")
    return data


def refresh_inference_market(data, log_info=lambda m: None, log_ok=lambda a, b="": None):
    """Fetch everything and rebuild the inference block."""
    log_info("Fetching OpenRouter rankings...")
    as_of, rankings = fetch_openrouter_rankings()
    log_ok("OpenRouter Rankings", f"{len(rankings)} models for {as_of}")

    log_info("Fetching OpenRouter model catalog...")
    catalog = {m.get("id"): m for m in _get(MODELS_URL, timeout=40).get("data", [])}

    endpoints_by_slug, failed = {}, []
    for r in rankings:
        slug = _base_slug(r["permaslug"])
        if slug in endpoints_by_slug:
            continue
        try:
            endpoints_by_slug[slug] = fetch_model_endpoints(slug)
        except Exception:
            failed.append(slug)
    if failed:
        log_info(f"Inference: no endpoint listing for {', '.join(failed[:5])}")
    log_ok("OpenRouter Endpoints", f"{sum(len(v) for v in endpoints_by_slug.values())} provider listings")

    return build_inference(data, as_of, rankings, catalog, endpoints_by_slug,
                           log=lambda m: log_ok("Inference Market", m))
