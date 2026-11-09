#!/usr/bin/env python3
"""Pre-publish checks on data.json.

Errors (exit 1, nothing is published):
  * a required section is missing, or fewer than MIN_PAIRS provider/GPU prices
  * a price outside a plausible range for a GPU-hour

Warnings (printed as GitHub annotations, publish goes ahead):
  * a provider's price moved more than MOVE_WARN_PCT since the last commit
  * a cross-GPU ordering that should hold does not (e.g. A100 80GB median
    cheaper than A100 40GB) -- the signature of a mislabelled listing, which is
    how a 40GB Vast.ai offer became the site's "best value A100 80GB"
  * a section, or the AI commentary, has not refreshed for several runs

Usage: python scripts/sanity_check.py [data.json] [previous data.json]
With no previous file, the version at git HEAD is used if available.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

MIN_PAIRS = 20
PRICE_RANGE = (0.05, 100.0)          # USD per GPU-hour
MOVE_WARN_PCT = 40
AI_MAX_AGE_DAYS = 7
SECTION_STALE_RUNS = 3

# (cheaper part, dearer part, tolerance) on cross-provider median price.
ORDERING = (
    ("A100-40GB", "A100-80GB", 1.0),
    ("H100-PCIe", "H100-SXM", 0.85),
    ("H100-SXM", "H200", 0.9),
    ("H100-SXM", "B200", 1.0),
    ("B200", "B300", 0.9),
    ("L4", "L40S", 1.0),
    ("T4", "L4", 1.0),
)


def _warn(msg):
    print(f"::warning::{msg}")


def _prices(d):
    out = {}
    for prov, p in (d.get("providers") or {}).items():
        for gpu, g in (p.get("gpus") or {}).items():
            if g.get("price_per_gpu_hr"):
                out[(prov, gpu)] = g
    return out


def _previous(path_arg):
    if path_arg:
        with open(path_arg) as f:
            return json.load(f)
    try:
        raw = subprocess.run(["git", "show", "HEAD:data.json"], capture_output=True,
                             check=True).stdout
        return json.loads(raw)
    except Exception:
        return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data.json"
    with open(path) as f:
        d = json.load(f)
    prev = _previous(sys.argv[2] if len(sys.argv) > 2 else None)
    errors = []

    missing = [k for k in ("providers", "matrix", "specs", "inference") if not d.get(k)]
    if missing:
        errors.append(f"data.json is missing required sections: {missing}")

    cur = _prices(d)
    if len(cur) < MIN_PAIRS:
        errors.append(f"only {len(cur)} provider/GPU prices -- refusing to publish")

    lo, hi = PRICE_RANGE
    for (prov, gpu), g in cur.items():
        p = g["price_per_gpu_hr"]
        if not lo <= p <= hi:
            errors.append(f"{prov}/{gpu} price {p} outside {lo}-{hi} $/GPU-hr")

    if prev:
        old = _prices(prev)
        for key, g in cur.items():
            if key in old and not g.get("stale"):
                a, b = old[key]["price_per_gpu_hr"], g["price_per_gpu_hr"]
                if a and abs(b - a) / a * 100 > MOVE_WARN_PCT:
                    _warn(f"{key[0]}/{key[1]} moved {a} -> {b} ({(b - a) / a * 100:+.0f}%) since last publish")

    med = {r["gpu_id"]: r.get("median_price") for r in d.get("matrix") or []}
    for cheap, dear, tol in ORDERING:
        if med.get(cheap) and med.get(dear) and med[dear] < med[cheap] * tol:
            _warn(f"median {dear} (${med[dear]}) is below {cheap} (${med[cheap]}) -- check for a mislabelled listing")

    fresh = (d.get("_meta") or {}).get("freshness") or {}
    stale = sorted(k for k, v in fresh.items() if v.get("auto") and v.get("stale_runs", 0) >= SECTION_STALE_RUNS)
    if stale:
        _warn(f"sections stale for {SECTION_STALE_RUNS}+ runs: {', '.join(stale)}")
    ai = fresh.get("ai_analysis") or {}
    if ai.get("refreshed"):
        age = (datetime.now(timezone.utc).date() - datetime.fromisoformat(ai["refreshed"]).date()).days
        if age > AI_MAX_AGE_DAYS:
            _warn(f"AI commentary is {age} days old (last generated {ai['refreshed']})")

    flagged = sum(1 for g in cur.values() if g.get("stale"))
    if flagged:
        print(f"{flagged} price(s) flagged stale/fallback and excluded from derived figures")

    if errors:
        for e in errors:
            print(f"::error::{e}")
        sys.exit(1)
    print(f"OK: {len(cur)} provider/GPU prices, {len(d.get('matrix') or [])} GPUs")


if __name__ == "__main__":
    main()
