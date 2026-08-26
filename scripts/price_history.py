"""Per-provider daily price history, and the volatility measures built on it.

Until now nothing on this dashboard could answer "what did *this provider*
charge last week". `historical` holds per-GPU monthly aggregates (min/avg/max
across the whole market), which hides the thing that actually matters to a
buyer: one provider can move 25% in a week while the market average barely
twitches.

The change feed looked like a shortcut and is not. It only records moves of 3%
or more, so sub-threshold drift is invisible -- of 164 consecutive event links,
48 fail to join up, meaning the price moved between logged events without being
logged. Reconstructing a series from it would produce numbers that are wrong by
a few percent, which is fatal when the thing being measured is a few-percent
move.

The exact record was already on disk: data.json has been committed once or more
a day since 2026-03-10, and each commit carries every provider's published price
for that day. `backfill_from_git` reads them. After that the history maintains
itself -- one snapshot appended per run, no git archaeology required.

One honesty problem this creates. Several large moves in the recent record are
not market movement at all, they are this pipeline's own scraper fixes landing
(Together reading the wrong pricing table, RunPod mixing Secure and Community
tiers, an NVL quote published as SXM). METHODOLOGY_CHANGES records when each of
those landed, and any window spanning one is flagged rather than presented as
volatility -- the two ends were measured differently, so the move is ours.
"""

import json
import subprocess
from datetime import datetime, timedelta, timezone

HISTORY_DAYS = 120
DATA_FILE = "data.json"


def _run(args, cwd=None):
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=120).stdout


def backfill_from_git(history, repo_dir, log_info=lambda m: None, max_commits=400):
    """Fill `history` with per-provider prices from past commits of data.json.

    Only dates absent from `history` are read, so this is expensive once and
    almost free afterwards. Returns the number of dates added.
    """
    log = _run(["git", "log", f"--max-count={max_commits}", "--format=%H %ad",
                "--date=short", "--", DATA_FILE], cwd=repo_dir).strip()
    if not log:
        return 0

    # Newest commit for a given day wins; walking newest-first means the first
    # time we see a date is the one to keep.
    seen_dates = set()
    added = 0
    for line in log.split("\n"):
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, date = parts
        if date in seen_dates or date in history:
            seen_dates.add(date)
            continue
        seen_dates.add(date)

        raw = _run(["git", "show", f"{sha}:{DATA_FILE}"], cwd=repo_dir)
        if not raw:
            continue
        try:
            snap = json.loads(raw)
        except json.JSONDecodeError:
            continue

        day = {}
        for prov_key, prov in (snap.get("providers") or {}).items():
            for gpu_id, info in (prov.get("gpus") or {}).items():
                price = info.get("price_per_gpu_hr")
                if price and price > 0:
                    day.setdefault(prov_key, {})[gpu_id] = round(float(price), 4)
        if day:
            history[date] = day
            added += 1

    if added:
        log_info(f"Price history: backfilled {added} days from git")
    return added


# Dates on which this pipeline changed how it reads a provider, with the
# listings affected. A series that crosses one of these is not comparable
# across the boundary: the numbers either side were produced by different
# methods, so the "move" is ours, not the market's.
#
# This has to be an explicit record. The first attempt read annotations off the
# change feed, but that feed is rebuilt and capped every run, so the notes were
# gone within days -- and RunPod's MI300X then showed a 378% weekly rise that
# was really a $0.50 Community quote from the old reader being compared against
# the $2.39 Secure rate from the new one.
METHODOLOGY_CHANGES = [
    {
        "date": "2026-08-20",
        "providers": ["RunPod"],
        "gpus": None,  # every RunPod listing
        "reason": ("on-demand switched from the cheapest live host (usually Community "
                   "Cloud) to the published Secure Cloud rate; NVL parts stopped being "
                   "published as SXM"),
    },
    {
        "date": "2026-08-20",
        "providers": ["Together"],
        "gpus": None,
        "reason": "on-demand switched from the Dedicated Inference table to GPU Clusters",
    },
    {
        "date": "2026-08-21",
        "providers": ["Azure"],
        "gpus": None,
        "reason": "on-demand now excludes non-Consumption price types",
    },
    {
        "date": "2026-08-21",
        "providers": ["GCP"],
        "gpus": ["B200"],
        "reason": "withdrawn: Google publishes no on-demand rate for a4-highgpu-8g",
    },
]


def _correction_in_window(provider, gpu, since):
    """Reason a provider/GPU comparison since `since` is not like-for-like."""
    for change in METHODOLOGY_CHANGES:
        if change["date"] <= since:
            continue
        if provider not in change["providers"]:
            continue
        if change["gpus"] and gpu not in change["gpus"]:
            continue
        return change["reason"]
    return None


def _series(history, provider, gpu):
    """[(date, price)] for one provider/GPU pair, oldest first."""
    out = []
    for date in sorted(history):
        price = ((history[date] or {}).get(provider) or {}).get(gpu)
        if price:
            out.append((date, price))
    return out


def _nearest_on_or_before(series, target):
    """Last observation at or before `target`, or None."""
    best = None
    for date, price in series:
        if date <= target:
            best = (date, price)
        else:
            break
    return best


def build_price_history(data, repo_dir, log_info=lambda m: None, log_ok=lambda a, b="": None):
    """Append today's prices, backfill from git, and derive per-provider moves."""
    providers = data.get("providers") or {}
    if not providers:
        return data

    block = data.get("price_history") or {}
    history = block.get("days") or {}

    # One-time (per missing date) backfill of the exact published record.
    backfill_from_git(history, repo_dir, log_info=log_info)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot = {}
    for prov_key, prov in providers.items():
        for gpu_id, info in (prov.get("gpus") or {}).items():
            price = info.get("price_per_gpu_hr")
            if price and price > 0:
                snapshot.setdefault(prov_key, {})[gpu_id] = round(float(price), 4)
    if snapshot:
        history[today] = snapshot

    for old in sorted(history)[:-HISTORY_DAYS]:
        history.pop(old, None)

    week_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    moves = {}
    for prov_key, prov in providers.items():
        for gpu_id, info in (prov.get("gpus") or {}).items():
            now_price = info.get("price_per_gpu_hr")
            if not now_price:
                continue
            series = _series(history, prov_key, gpu_id)
            if len(series) < 2:
                continue

            rec = {"current": round(now_price, 4), "observations": len(series),
                   "first_observed": series[0][0]}

            prior = _nearest_on_or_before(series, week_ago)
            if prior:
                rec["week_ago"] = prior[1]
                rec["week_ago_date"] = prior[0]
                rec["change_7d_pct"] = round((now_price / prior[1] - 1) * 100, 1)
                # A methodology change inside the window means the two ends were
                # measured differently; say so instead of calling it a move.
                why = _correction_in_window(prov_key, gpu_id, prior[0])
                if why:
                    rec["corrected_in_window"] = True
                    rec["correction_reason"] = why

            window = [(d, p) for d, p in series if d >= month_ago]
            if len(window) >= 2:
                prices = [p for _, p in window]
                lo, hi = min(prices), max(prices)
                rec["low_30d"] = lo
                rec["high_30d"] = hi
                # Peak-to-trough spread over the month, as a share of the low.
                rec["range_30d_pct"] = round((hi - lo) / lo * 100, 1) if lo else None
                # Mean absolute day-over-day move: how jumpy this listing is,
                # independent of direction.
                steps = [abs(b / a - 1) for a, b in zip(prices, prices[1:]) if a]
                rec["avg_daily_move_pct"] = round(sum(steps) / len(steps) * 100, 2) if steps else 0.0
                rec["observations_30d"] = len(window)
                if _correction_in_window(prov_key, gpu_id, window[0][0]):
                    rec["range_30d_spans_correction"] = True

            moves.setdefault(prov_key, {})[gpu_id] = rec

    # Provider-level roll-up: which listings actually move.
    summary = []
    for prov_key, gpus in moves.items():
        jumpy = [g["avg_daily_move_pct"] for g in gpus.values()
                 if g.get("avg_daily_move_pct") is not None
                 and not g.get("range_30d_spans_correction")]
        weekly = [g["change_7d_pct"] for g in gpus.values()
                  if g.get("change_7d_pct") is not None and not g.get("corrected_in_window")]
        # A provider whose every listing spans one of our own methodology
        # changes still belongs in the table -- dropping it silently reads as a
        # bug. It carries no volatility figure and says why, and starts
        # reporting again once the window clears the change.
        blocked = next((g.get("correction_reason") for g in gpus.values()
                        if g.get("correction_reason")), None)
        row = {
            "provider": prov_key,
            "type": (providers.get(prov_key) or {}).get("type", "cloud"),
            "gpus_tracked": len(gpus),
            "avg_daily_move_pct": round(sum(jumpy) / len(jumpy), 2) if jumpy else None,
            "max_daily_move_pct": round(max(jumpy), 2) if jumpy else None,
            "avg_change_7d_pct": round(sum(weekly) / len(weekly), 1) if weekly else None,
            "movers_7d": sum(1 for w in weekly if abs(w) >= 1),
        }
        if not jumpy and blocked:
            row["measurement_changed"] = blocked
        summary.append(row)
    # Unmeasurable providers sort last rather than reading as perfectly stable.
    summary.sort(key=lambda s: (s["avg_daily_move_pct"] is None, -(s["avg_daily_move_pct"] or 0)))

    dates = sorted(history)
    data["price_history"] = {
        "days": history,
        "moves": moves,
        "providers": summary,
        "as_of": today,
        "first_date": dates[0] if dates else None,
        "day_count": len(dates),
    }

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["price_history"] = {
        "basis": "measured",
        "detail": (
            "Each provider's published price per GPU, per day. Backfilled from "
            "the git history of data.json -- every commit is an exact record of "
            "what was published that day -- and appended once per run "
            "thereafter. The change feed was not used: it only logs moves of 3% "
            "or more, so a series rebuilt from it drifts. Windows containing "
            "one of this pipeline's own scraper corrections are flagged, "
            "because those moves are not the market."
        ),
    }
    log_ok("Price History", f"{len(dates)} days, {sum(len(v) for v in moves.values())} provider/GPU series")
    return data
