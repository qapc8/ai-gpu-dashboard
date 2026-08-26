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

HISTORY_DAYS = 200  # ~6 months: enough to rebuild monthly history from
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
        # Commit d274895, "Add GCP scraper and prefer Together HGX cluster
        # rates". Before this the cloud prices were hardcoded seed values;
        # after, they are scraped. The step is enormous and entirely ours --
        # GCP's H100 went 3.40 -> 11.06 overnight, AWS 4.28 -> 6.88, Azure
        # 3.67 -> 11.61 -- so no month before May 2026 is comparable with any
        # month after it. `scope: all` trims the monthly series to the first
        # clean month rather than merely flagging it, because a 3x artefact at
        # the head of a six-month chart swamps everything real in it.
        "date": "2026-04-30",
        "providers": ["AWS", "GCP", "Azure", "Lambda", "CoreWeave", "Together", "RunPod", "Vast.ai"],
        "gpus": None,
        "scope": "all",
        "reason": ("live scraping replaced hardcoded seed prices for the cloud "
                   "providers; figures before this date are not comparable"),
    },
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


def rebuild_historical(data, history, log_ok=lambda a, b="": None):
    """Recompute the monthly price history from the daily per-provider record.

    `historical` was a splice of two incompatible things. Months up to 2026-03
    were hand-seeded in gpu_data.py; from 2026-04 the pipeline started
    computing them from live providers. The two averaged different provider
    sets -- the seeded era never included hyperscaler list prices -- so the
    join produced a cliff that was pure methodology:

        H100-SXM  2026-03  avg $2.86, max $4.28   (seeded)
                  2026-04  avg $5.45, max $11.61  (computed, Azure now in)

    Nothing about the market did that. But `volatility` reads this series, so
    the dashboard was reporting H100 up 202% year-on-year and six GPUs
    "tightening" while real H100 prices were falling -- the prediction markets
    on the next tab price it flat.

    Recomputing every month from the daily record fixes the basis: one
    provider set throughout, and every figure traceable to a published price.
    The series gets shorter -- it can only start where the daily record does --
    which is the honest length.
    """
    if not history:
        return data

    # Average over the provider set we track *today*, in every month. The daily
    # record carries whoever was tracked at the time -- Oracle through April,
    # FluidStack until August -- so averaging "whatever was there" would make a
    # provider joining or leaving look like a price move. Same set throughout
    # means month-to-month change is the market.
    current = set(data.get("providers") or {})

    # A change marked `scope: all` rewrote every provider's basis, so months
    # before it cannot be compared with months after. Start the series at the
    # first whole month following the most recent one.
    wholesale = [c["date"] for c in METHODOLOGY_CHANGES if c.get("scope") == "all"]
    first_month = None
    if wholesale:
        cut = max(wholesale)
        year, mon = int(cut[:4]), int(cut[5:7])
        year, mon = (year + 1, 1) if mon == 12 else (year, mon + 1)
        first_month = f"{year:04d}-{mon:02d}"

    by_month = {}
    month_provs = {}
    for date in sorted(history):
        month = date[:7]
        if first_month and month < first_month:
            continue
        for provider, gpus in (history[date] or {}).items():
            if provider not in current:
                continue
            for gpu_id, price in (gpus or {}).items():
                if price and price > 0:
                    by_month.setdefault(gpu_id, {}).setdefault(month, []).append(price)
                    month_provs.setdefault(month, set()).add(provider)

    historical = {}
    for gpu_id, months in by_month.items():
        series = {}
        for month, prices in sorted(months.items()):
            entry = {
                "avg": round(sum(prices) / len(prices), 2),
                "min": round(min(prices), 2),
                "max": round(max(prices), 2),
                "observations": len(prices),
                "providers": len(month_provs.get(month) or ()),
            }
            # A month in which we changed how a provider is read is not
            # comparable with its neighbours; say so rather than let the step
            # be read as market movement.
            changed = [c["reason"] for c in METHODOLOGY_CHANGES if c["date"][:7] == month
                       and (not c["gpus"] or gpu_id in c["gpus"])]
            if changed:
                entry["measurement_changed"] = changed[0]
            series[month] = entry
        if series:
            historical[gpu_id] = series

    if not historical:
        return data
    data["historical"] = historical

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    months = sorted({m for s in historical.values() for m in s})
    meta["historical"] = {
        "basis": "derived",
        "detail": (
            "Monthly min/average/max per GPU, aggregated from the daily "
            "per-provider price record over the provider set tracked today, so "
            "a provider joining or leaving cannot read as a price move. "
            "Replaces a series that spliced a hand-seeded era onto live "
            "figures: it showed H100 doubling in a month and drove a false "
            "202% year-on-year rise into the volatility panel, which then "
            "labelled six GPUs 'tightening' while prices were flat or falling. "
            f"It starts at {months[0]} because live scraping replaced "
            "hardcoded seed prices on 2026-04-30 and nothing before that is "
            "comparable -- short and true rather than long and spliced."
        ),
    }
    log_ok("Historical", f"{len(historical)} GPUs over {len(months)} months from the daily record")
    return data


def compute_volatility_daily(data, history, log_ok=lambda a, b="": None):
    """Realized volatility per GPU, from the daily market average.

    This used to run off the monthly series. Rebuilding that series on a single
    comparable basis left four months -- three month-over-month returns -- and
    an annualized volatility from three returns is noise dressed as a
    statistic. The daily record holds ~160 observations of the same thing, so
    use those and report how many stand behind each figure.
    """
    current = set(data.get("providers") or {})
    wholesale = [c["date"] for c in METHODOLOGY_CHANGES if c.get("scope") == "all"]
    floor = max(wholesale) if wholesale else None

    # Average over a basket that held one basis for the whole window.
    #
    # The first attempt averaged every current provider and then discarded any
    # GPU whose basket contained a change. Because RunPod, Together and Azure
    # all changed in the same week, and between them carry nearly every part,
    # that marked 15 of 17 GPUs "unmeasurable" -- destroying the panel to fix
    # three listings. Dropping just the unstable providers keeps a comparable
    # series for everything, and they rejoin by themselves once their change
    # falls out of the longest window.
    horizon = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    unstable = {p for c in METHODOLOGY_CHANGES if c["date"] >= horizon and c.get("scope") != "all"
                for p in c["providers"]}
    basket = sorted(current - unstable)
    if not basket:
        return data

    dates = [d for d in sorted(history) if not (floor and d <= floor)]
    if not dates:
        return data

    # A provider must carry the GPU on essentially every day to be in its
    # average. Otherwise the basket changes size mid-series and the average
    # steps for a reason that has nothing to do with price: Vast.ai's T4
    # listing disappeared on 2026-07-28 and came back on 08-14, so the "T4
    # market average" jumped 0.34 -> 0.53 -> 0.34 between a two-provider and a
    # one-provider basket, and reported 118% annualized volatility on a part
    # whose price never moved.
    coverage = {}
    for date in dates:
        for provider, gpus in (history[date] or {}).items():
            if provider not in basket:
                continue
            for gpu_id, price in (gpus or {}).items():
                if price and price > 0:
                    coverage.setdefault(gpu_id, {}).setdefault(provider, 0)
                    coverage[gpu_id][provider] += 1
    consistent = {
        gpu_id: {p for p, n in provs.items() if n >= 0.9 * len(dates)}
        for gpu_id, provs in coverage.items()
    }

    series = {}
    for date in dates:
        prices = {}
        for provider, gpus in (history[date] or {}).items():
            if provider not in basket:
                continue
            for gpu_id, price in (gpus or {}).items():
                if price and price > 0 and provider in consistent.get(gpu_id, ()):
                    prices.setdefault(gpu_id, []).append(price)
        for gpu_id, vals in prices.items():
            # Only average a full basket; a day missing a member is skipped
            # rather than averaged over a smaller set.
            if len(vals) == len(consistent.get(gpu_id, ())):
                series.setdefault(gpu_id, []).append((date, sum(vals) / len(vals)))

    # Which providers carry each GPU, so a GPU-level average can be checked for
    # methodology changes among its constituents. The market average is
    # dominated by whoever lists the part -- MI300X is essentially RunPod and
    # Vast.ai -- so RunPod's tier fix showed up as MI300X "rising 378% in a
    # week" with 1194% annualized volatility. That is our change, not a market.
    carriers = {}
    for date in history:
        for provider, gpus in (history[date] or {}).items():
            if provider in basket:
                for gpu_id in (gpus or {}):
                    carriers.setdefault(gpu_id, set()).add(provider)

    def _changed_since(gpu_id, since):
        for provider in carriers.get(gpu_id, ()):
            why = _correction_in_window(provider, gpu_id, since)
            if why:
                return f"{provider}: {why}"
        return None

    def _pct_change(pts, days_back, gpu_id):
        target = (datetime.strptime(pts[-1][0], "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
        prior = _nearest_on_or_before(pts, target)
        if not prior or not prior[1]:
            return None, None
        why = _changed_since(gpu_id, prior[0])
        if why:
            return None, why
        return round((pts[-1][1] / prior[1] - 1) * 100, 1), None

    out = {}
    for gpu_id, pts in series.items():
        if len(pts) < 10:
            continue
        values = [v for _, v in pts]
        current_price = values[-1]

        # Volatility over a window containing one of our own changes measures
        # the change, not the market.
        vol_since = pts[max(0, len(pts) - 90)][0]
        vol_blocked = _changed_since(gpu_id, vol_since)
        rets = [b / a - 1 for a, b in zip(values, values[1:]) if a]
        vol = median_move = None
        days_moved = None
        if len(rets) >= 20 and not vol_blocked:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            vol = round((var ** 0.5) * (365 ** 0.5) * 100, 1)
            # Annualized volatility alone misleads here. A marketplace quote is
            # the cheapest *available* offer, so when cheap stock runs out it
            # steps to a much dearer machine -- Vast.ai's RTX-4090 went $0.40 to
            # $2.00 in a day and back. Real quotes, but supply gaps rather than
            # repricing, and squaring them puts annualized volatility at 787%:
            # arithmetically right, useless to read. The plain median is no
            # better, because most days see no change at all and it lands on
            # zero. So report how often the price moves, and how far when it
            # does -- two facts that actually describe the listing.
            moved = [abs(r) for r in rets if abs(r) > 0.005]
            days_moved = round(len(moved) / len(rets) * 100)
            if moved:
                mv = sorted(moved)
                mid = len(mv) // 2
                median_move = round((mv[mid] if len(mv) % 2 else (mv[mid - 1] + mv[mid]) / 2) * 100, 1)

        peak = max(values)
        c7, why7 = _pct_change(pts, 7, gpu_id)
        c30, why30 = _pct_change(pts, 30, gpu_id)
        c90, why90 = _pct_change(pts, 90, gpu_id)

        # Classify on the 30-day move, with a dead band wide enough that noise
        # does not get called a trend. No comparable window means no verdict.
        regime = "stable"
        if c30 is None:
            regime = "unmeasurable" if why30 else "stable"
        else:
            regime = "tightening" if c30 > 3 else "falling" if c30 < -3 else "stable"

        rec = {
            "current": round(current_price, 2),
            "change_7d_pct": c7,
            "change_30d_pct": c30,
            "change_90d_pct": c90,
            "annualized_volatility_pct": vol,
            "days_moved_pct": days_moved,
            "typical_move_pct": median_move,
            "drawdown_from_peak_pct": round((current_price - peak) / peak * 100, 1) if peak else None,
            "daily_observations": len(pts),
            "first_observed": pts[0][0],
            "regime": regime,
            "providers_in_average": sorted(consistent.get(gpu_id, ())),
            "note": (f"{len(pts)} daily observations, averaged over the "
                     f"{len(consistent.get(gpu_id, ()))} providers that list this GPU "
                     f"every day since {pts[0][0]}"),
        }
        why = why30 or why7 or why90 or vol_blocked
        if why:
            rec["measurement_changed"] = why
            rec["note"] = ("A provider carrying this GPU changed how it is read within "
                           f"the window, so the comparison is not like-for-like -- {why}")
        out[gpu_id] = rec

    if not out:
        return data
    data["volatility"] = out
    data["volatility_basket"] = {
        "providers": basket,
        "excluded": sorted(unstable & current),
        "note": ("Averaged over providers whose basis held for the whole window. "
                 "Excluded providers changed how we read them within the last 90 "
                 "days and rejoin automatically once that falls out of the window."),
    }

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["volatility"] = {
        "basis": "derived",
        "detail": (
            "Realized volatility, drawdown and trend per GPU, computed from the "
            "daily market average across the providers tracked today. "
            "Previously computed from the monthly series, which after being "
            "rebuilt on one comparable basis holds four points -- three "
            "returns. Annualized volatility is only reported where at least 20 "
            "daily returns stand behind it, and the observation count is "
            "published with every figure."
        ),
    }
    log_ok("Volatility", f"{len(out)} GPUs from daily observations")
    return data


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

    data = rebuild_historical(data, history, log_ok=log_ok)
    data = compute_volatility_daily(data, history, log_ok=log_ok)

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
