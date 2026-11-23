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

# A price the provider table carries but no derived figure may use. Same rule
# as the page's priceFreshness (index.html) and update_data._usable: flagged
# stale, a hardcoded fallback constant, or a reading more than
# STALE_AFTER_HOURS older than the run that published it.
#
# The record used to take every published price. GCP's T4 has been a $0.35
# fallback constant on every day of it, so volatility.T4 was built on that one
# frozen number alone ("stable, 0%"), the T4 monthly average carried it, and
# the anomaly feed reported it "unchanged for 200 days". A carried-forward
# number is not an observation of the market, so it is not history.
STALE_AFTER_HOURS = 48


def _usable(entry, ref_ts=None):
    price = entry.get("price_per_gpu_hr")
    if not price or price <= 0:
        return False
    if entry.get("stale") or entry.get("source") == "hardcoded_fallback":
        return False
    seen = entry.get("last_updated")
    if ref_ts and seen:
        try:
            age = (datetime.fromisoformat(ref_ts) - datetime.fromisoformat(seen)).total_seconds()
        except (TypeError, ValueError):
            return True  # unparseable or naive/aware mix: no evidence it is old
        if age > STALE_AFTER_HOURS * 3600:
            return False
    return True


def _snapshot(providers, ref_ts):
    """{provider: {gpu: price}} of the usable prices in one published state."""
    day = {}
    for prov_key, prov in (providers or {}).items():
        for gpu_id, info in (prov.get("gpus") or {}).items():
            if _usable(info, ref_ts):
                day.setdefault(prov_key, {})[gpu_id] = round(float(info["price_per_gpu_hr"]), 4)
    return day


def _run(args, cwd=None):
    return subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=120).stdout


def backfill_from_git(history, repo_dir, log_info=lambda m: None, max_commits=400, screened=None):
    """Fill `history` with per-provider prices from past commits of data.json,
    and screen dates recorded before the usability rule existed.

    Only dates absent from `history`, or not yet in `screened`, are read, so
    this is expensive once and almost free afterwards. Dates the git history
    does not reach (CI clones shallow) stay unscreened until a run that can
    see them -- `python scripts/price_history.py --screen` from a full clone.
    Returns the number of dates added.
    """
    screened = screened if screened is not None else set()
    log = _run(["git", "log", f"--max-count={max_commits}", "--format=%H %ad",
                "--date=short", "--", DATA_FILE], cwd=repo_dir).strip()
    if not log:
        return 0

    # Newest commit for a given day wins; walking newest-first means the first
    # time we see a date is the one to keep.
    seen_dates = set()
    added = purged = 0
    # Once the record is full, older dates would only be read to be trimmed
    # again at the end of the run -- dozens of `git show`s for nothing.
    oldest = min(history) if len(history) >= HISTORY_DAYS else None
    for line in log.split("\n"):
        parts = line.split()
        if len(parts) != 2:
            continue
        sha, date = parts
        if oldest and date < oldest:
            continue
        if date in seen_dates or (date in history and date in screened):
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

        clean = _snapshot(snap.get("providers"), snap.get("last_updated"))
        if date in history:
            # Recorded before stale and fallback prices were kept out: drop
            # the points that commit shows were not usable when published.
            # Only a point that matches the commit is touched -- anything else
            # was recorded by a later run of that day and is not ours to judge.
            pubd = {pk: {g: round(float(i.get("price_per_gpu_hr") or 0), 4)
                         for g, i in (p.get("gpus") or {}).items()}
                    for pk, p in (snap.get("providers") or {}).items()}
            for prov_key, gpus in list((history[date] or {}).items()):
                for gpu_id, price in list((gpus or {}).items()):
                    if gpu_id in (clean.get(prov_key) or {}):
                        continue
                    if abs((pubd.get(prov_key) or {}).get(gpu_id, -1) - price) < 1e-4:
                        gpus.pop(gpu_id)
                        purged += 1
                if not gpus:
                    history[date].pop(prov_key)
            screened.add(date)
        elif clean:
            history[date] = clean
            screened.add(date)
            added += 1

    if added:
        log_info(f"Price history: backfilled {added} days from git")
    if purged:
        log_info(f"Price history: dropped {purged} stale/fallback points recorded before screening")
    return added


# Scraper faults: spans in which a *live*, fresh-looking price was wrong
# because our reader was. Unlike a methodology change, where both sides are
# real prices measured differently, the points inside a fault are not prices
# at all, so they are kept out of the record rather than merely flagged.
#
# A fault closes by itself: on the first recorded day on which any of its GPUs
# is back within `restored_ratio` of its last pre-fault price, the reader has
# been fixed and every point from then on is kept. That avoids a hand edit on
# the day the fix lands, and an open-ended fault silently deleting good data
# for ever. Set `until` only if the fix restores a genuinely different price.
SCRAPER_FAULTS = [
    {
        # Together added a Preemptible column ahead of On-Demand and the
        # reader took the first dollar in each row: H100 3.99 -> 1.99, H200
        # 5.99 -> 2.99, B200 8.19 -> 4.09/4.65. Fixed in update_data
        # (read the column headed on-demand).
        "from": "2026-11-11",
        "until": None,
        "provider": "Together",
        "gpus": ["H100-SXM", "H200", "B200", "B300"],
        "restored_ratio": 0.8,
        "reason": "scraper read Together's Preemptible column as on-demand",
    },
]


def _fault_end(history, fault):
    """First date the fault is over, or None while it is still open."""
    if fault.get("until"):
        return fault["until"]
    dates = sorted(history)
    for date in dates:
        if date < fault["from"]:
            continue
        for gpu_id in fault["gpus"]:
            ref = None
            for d in dates:
                if d >= fault["from"]:
                    break
                ref = ((history[d] or {}).get(fault["provider"]) or {}).get(gpu_id) or ref
            now = ((history[date] or {}).get(fault["provider"]) or {}).get(gpu_id)
            if ref and now and now >= fault["restored_ratio"] * ref:
                return date
    return None


def drop_scraper_faults(history):
    """Remove points inside an open or closed SCRAPER_FAULTS span, in place.

    Runs on every read as well as at record time, so points recorded before
    a fault was registered stop leaking into history, volatility and signals.
    Returns {(provider, gpu): reason} for listings currently inside a fault.
    """
    active = {}
    for fault in SCRAPER_FAULTS:
        end = _fault_end(history, fault)
        for date in sorted(history):
            if date < fault["from"] or (end and date >= end):
                continue
            gpus = (history[date] or {}).get(fault["provider"]) or {}
            for gpu_id in fault["gpus"]:
                gpus.pop(gpu_id, None)
            if not gpus:
                (history[date] or {}).pop(fault["provider"], None)
        if not end:
            for gpu_id in fault["gpus"]:
                active[(fault["provider"], gpu_id)] = fault["reason"]
    return active


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
    {
        "date": "2026-11-09",
        "providers": ["Vast.ai"],
        "gpus": None,
        "reason": ("on-demand changed from the single cheapest offer per name to the "
                   "25th percentile of >=3 offers, pinned by memory size (40GB A100s "
                   "had been published as 80GB)"),
    },
    {
        "date": "2026-11-09",
        "providers": ["Azure"],
        "gpus": None,
        "reason": ("on-demand now read from one exact Linux VM size per GPU; it had "
                   "pooled Windows rows and non-InfiniBand sizes (H100 11.61 -> 12.29) "
                   "and published the 40GB A100 as the 80GB"),
    },
    {
        "date": "2026-11-09",
        "providers": ["Lambda"],
        "gpus": None,
        "reason": "on-demand now the 8x-cluster rate wherever Lambda sells 8x (H100 had been the 4x rate)",
    },
    {
        "date": "2026-11-09",
        "providers": ["AWS"],
        "gpus": ["T4"],
        "reason": ("now read live from the 4-GPU g4dn.12xlarge, like every other row; it was a "
                   "frozen fallback for the 1-GPU g4dn.xlarge"),
    },
    {
        "date": "2026-11-09",
        "kind": "joined",
        "providers": ["Oracle", "Nebius", "DigitalOcean", "Vultr", "Crusoe"],
        "gpus": None,
        "reason": "providers added; kept out of monthly averages until they have 90 days of history",
    },
    {
        "date": "2026-11-11",
        "kind": "scraper_fault",  # see SCRAPER_FAULTS
        "providers": ["Together"],
        "gpus": ["H100-SXM", "H200", "B200", "B300"],
        "reason": ("scraper correction: on-demand was read from the Preemptible column "
                   "(H100 3.99 -> 1.99); those points are dropped from history "
                   "(SCRAPER_FAULTS) and the restored on-demand rate is not a market move"),
    },
    {
        "date": "2026-11-12",
        "kind": "joined",
        "providers": ["Hyperbolic", "Paperspace", "Hot Aisle"],
        "gpus": None,
        "reason": "providers added; kept out of monthly averages until they have 90 days of history",
    },
]

# Joins and gaps are also detected from the record itself (rebuild_historical): a listing absent for longer than this, or appearing for
# the first time, is reported as joined/left for the month rather than read
# as a move. The hand list above stays for documented method changes; a
# listing vanishing for a quarter (Azure MI300X, 2026-07-28 to 11-09) should
# not need anyone to notice it.


def _recent_joins(days=90):
    """Providers added within `days`: excluded from averages so their arrival
    cannot read as a price move."""
    horizon = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return {p for c in METHODOLOGY_CHANGES
            if c.get("kind") == "joined" and c["date"] >= horizon for p in c["providers"]}


def _correction_in_window(provider, gpu, since, until=None):
    """Reason a provider/GPU comparison from `since` to `until` (default: now)
    is not like-for-like."""
    for change in METHODOLOGY_CHANGES:
        if change["date"] <= since or (until and change["date"] > until):
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
    # provider joining or leaving look like a price move. A tracked provider
    # can still drop one listing and return to it, which is why change_pct
    # below compares matched providers only.
    current = set(data.get("providers") or {}) - _recent_joins()

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

    # Per GPU, month and provider: that provider's prices that month.
    by_month = {}
    for date in sorted(history):
        month = date[:7]
        if first_month and month < first_month:
            continue
        for provider, gpus in (history[date] or {}).items():
            if provider not in current:
                continue
            for gpu_id, price in (gpus or {}).items():
                if price and price > 0:
                    (by_month.setdefault(gpu_id, {}).setdefault(month, {})
                     .setdefault(provider, []).append((date, price)))

    historical = {}
    for gpu_id, months in by_month.items():
        series = {}
        prev = None
        for month, provs in sorted(months.items()):
            prices = [p for pts in provs.values() for _, p in pts]
            entry = {
                "avg": round(sum(prices) / len(prices), 2),
                "min": round(min(prices), 2),
                "max": round(max(prices), 2),
                "observations": len(prices),
                # Providers pricing *this* GPU this month. It was the count
                # across every GPU, so all of them read 11 in November.
                "providers": len(provs),
            }
            # Month-on-month change over the providers that priced the GPU in
            # both months, each weighted equally, leaving out any whose method
            # changed in between. `avg` is a level and moves whenever the
            # provider mix does: Azure's MI300X vanished on 2026-07-28 and
            # returned on 11-09, so the November avg (3.70) against October
            # (2.39) read +54% while RunPod -- the one listing in both months
            # -- had not moved at all.
            if prev:
                pm, pprovs = prev
                matched = sorted(
                    p for p in set(provs) & set(pprovs)
                    if not _correction_in_window(p, gpu_id, pprovs[p][0][0], provs[p][-1][0]))
                if matched:
                    now_m = sum(sum(x for _, x in provs[p]) / len(provs[p]) for p in matched)
                    then_m = sum(sum(x for _, x in pprovs[p]) / len(pprovs[p]) for p in matched)
                    entry["change_pct"] = round((now_m / then_m - 1) * 100, 1) if then_m else None
                else:
                    entry["change_pct"] = None
                entry["change_providers"] = matched
                # Composition changes, detected from the record rather than
                # hand-listed: they move `avg` but are not price moves.
                joined, left = sorted(set(provs) - set(pprovs)), sorted(set(pprovs) - set(provs))
                if joined:
                    entry["joined"] = joined
                if left:
                    entry["left"] = left
            # A month in which we changed how a provider of this GPU is read
            # is not comparable with its neighbours; say so rather than let
            # the step be read as market movement.
            changed = [c["reason"] for c in METHODOLOGY_CHANGES if c["date"][:7] == month
                       and c.get("kind") != "joined"
                       and (not c["gpus"] or gpu_id in c["gpus"])
                       and (c.get("scope") == "all" or set(c["providers"]) & (set(provs) | set((prev or (None, {}))[1])))]
            if changed:
                entry["measurement_changed"] = changed[0]
            series[month] = entry
            prev = (month, provs)
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
            "per-provider price record over the provider set tracked today, "
            "excluding stale, carried-forward and fallback prices. The "
            "month-on-month change (change_pct) is computed only over "
            "providers that priced the GPU in both months and whose method did "
            "not change in between, so a provider joining, leaving or "
            "returning (listed per month) cannot read as a price move. "
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
    #
    # A scraper fault is the exception. Its bad points are not in the record,
    # and the return spanning it is screened out below, so once the fault has
    # closed the provider's series is comparable either side and it stays in.
    # While it is open the provider has no readings for those GPUs, which
    # would skip every day as an incomplete basket, so it is out until then.
    open_faults = {f["provider"] for f in SCRAPER_FAULTS if not _fault_end(history, f)}
    unstable = {p for c in METHODOLOGY_CHANGES if c["date"] >= horizon and c.get("scope") != "all"
                for p in c["providers"]
                if c.get("kind") != "scraper_fault" or p in open_faults}
    # Marketplaces stay out permanently: their figure is an order-book
    # statistic that moves with whichever hosts are listed that hour, and it
    # was most of the "53.6% annualized volatility" the H100 panel reported on
    # list prices that had barely moved.
    marketplaces = {k for k, p in (data.get("providers") or {}).items()
                    if p.get("type") == "marketplace"}
    basket = sorted(current - unstable - marketplaces - _recent_joins())
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
    #
    # "Every day" counts the days on which the GPU was read at all. Stale and
    # carried-forward prices are no longer recorded, and on those runs every
    # scraped provider is missing together -- a day nobody was read is no
    # observation, not a basket member dropping out.
    coverage, read_days = {}, {}
    for date in dates:
        for provider, gpus in (history[date] or {}).items():
            if provider not in basket:
                continue
            for gpu_id, price in (gpus or {}).items():
                if price and price > 0:
                    coverage.setdefault(gpu_id, {}).setdefault(provider, 0)
                    coverage[gpu_id][provider] += 1
                    read_days.setdefault(gpu_id, set()).add(date)
    consistent = {
        gpu_id: {p for p, n in provs.items() if n >= 0.9 * len(read_days[gpu_id])}
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

    # A GPU-level average has to be checked for methodology changes among its
    # constituents. The market average is dominated by whoever lists the part
    # -- MI300X is essentially RunPod and Vast.ai -- so RunPod's tier fix
    # showed up as MI300X "rising 378% in a week" with 1194% annualized
    # volatility. That is our change, not a market. Only the providers
    # actually in the average can move it, so only they are checked.
    def _changed_since(gpu_id, since, until=None):
        for provider in sorted(consistent.get(gpu_id, ())):
            why = _correction_in_window(provider, gpu_id, since, until)
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

        # A return spanning one of our own changes measures the change, not
        # the market, so it is left out. Every return used is screened: this
        # used to check only the last 90 points while computing over all ~199,
        # so Together's 2026-08-20 table switch sat inside H100 volatility.
        rets, vol_skipped = [], []
        for (d0, a), (d1, b) in zip(pts, pts[1:]):
            if not a:
                continue
            why = _changed_since(gpu_id, d0, d1)
            if why:
                vol_skipped.append(why)
                continue
            rets.append(b / a - 1)
        vol = median_move = None
        days_moved = None
        if len(rets) >= 20:
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
        if vol_skipped:
            rec["returns_excluded"] = len(vol_skipped)
            rec["returns_excluded_reason"] = vol_skipped[0]
        why = why30 or why7 or why90
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
        "excluded_marketplaces": sorted(marketplaces),
        "excluded_new": sorted(_recent_joins() & set(data.get("providers") or {})),
        "note": ("Averaged over dedicated-cloud providers whose basis held for the "
                 "whole window. Providers that changed how we read them, or joined, "
                 "within the last 90 days rejoin automatically once that falls out "
                 "of the window. Marketplaces (order-book prices) are always excluded."),
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
            "published with every figure. Stale and fallback prices are not "
            "in the record, and any daily return spanning a methodology change "
            "of a provider in the average is left out (returns_excluded)."
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
    screened = set(block.get("screened_dates") or [])

    # One-time (per missing or unscreened date) backfill of the exact
    # published record.
    backfill_from_git(history, repo_dir, log_info=log_info, screened=screened)

    # Only usable prices are recorded (see _usable). A stale or fallback
    # price stays in the provider table, flagged, but it is not an
    # observation and must not become one here.
    # The run's own date, so re-running on a saved data.json (the --screen
    # entry point below) re-records that day instead of inventing today.
    today = (data.get("last_updated") or datetime.now(timezone.utc).isoformat())[:10]
    snapshot = _snapshot(providers, data.get("last_updated"))
    if snapshot:
        history[today] = snapshot
        screened.add(today)

    # Read-time screens, so points recorded before a rule existed stop
    # leaking: scraper-fault spans, and any listing now on a fallback
    # constant -- every recorded point equal to that constant is the
    # constant, not a reading (GCP T4, $0.35, on every day of the record).
    in_fault = drop_scraper_faults(history)
    for prov_key, prov in providers.items():
        for gpu_id, info in (prov.get("gpus") or {}).items():
            if info.get("source") != "hardcoded_fallback" or not info.get("price_per_gpu_hr"):
                continue
            fb = round(float(info["price_per_gpu_hr"]), 4)
            for date in history:
                gpus = (history[date] or {}).get(prov_key) or {}
                if gpu_id in gpus and abs(gpus[gpu_id] - fb) < 1e-4:
                    gpus.pop(gpu_id)
                if prov_key in history[date] and not gpus:
                    history[date].pop(prov_key)

    for old in sorted(history)[:-HISTORY_DAYS]:
        history.pop(old, None)
    screened &= set(history)

    week_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    # Moves only for listings recorded today: a stale, fallback or in-fault
    # price has no "current" to compare, and comparing one produced
    # "unchanged for 200 days" for a constant that was never read.
    moves = {}
    for prov_key, prov in providers.items():
        for gpu_id, info in (prov.get("gpus") or {}).items():
            now_price = ((history.get(today) or {}).get(prov_key) or {}).get(gpu_id)
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
        # Dates checked against the published commit for stale/fallback
        # points; the rest were recorded before that rule and await a run
        # whose git history reaches them.
        "screened_dates": sorted(screened),
        "unscreened_days": len(set(dates) - screened),
        "scraper_faults_open": sorted(f"{p}|{g}" for p, g in in_fault),
        # Committed-rate record, maintained by term_structure.record_term_history.
        "term_days": block.get("term_days") or {},
    }

    meta = data.setdefault("_meta", {}).setdefault("sections", {})
    meta["price_history"] = {
        "basis": "measured",
        "detail": (
            "Each provider's published price per GPU, per day. Backfilled from "
            "the git history of data.json -- every commit is an exact record of "
            "what was published that day -- and appended once per run "
            "thereafter. The change feed was not used: it only logs moves of 3% "
            "or more, so a series rebuilt from it drifts. Only usable prices "
            "are recorded: stale, carried forward more than "
            f"{STALE_AFTER_HOURS}h, or hardcoded fallback prices are not "
            "observations and are left out (and purged from days recorded "
            "before that rule). Days on which our scraper read the wrong "
            "column are dropped. Windows containing one of this pipeline's "
            "own scraper corrections are flagged, because those moves are not "
            "the market."
        ),
    }
    log_ok("Price History", f"{len(dates)} days, {sum(len(v) for v in moves.values())} provider/GPU series")
    return data


if __name__ == "__main__":
    # One-off screen of the accumulated record from a full clone. CI clones
    # 60 commits deep, so days older than that can only be screened here:
    #   python scripts/price_history.py --screen [path/to/data.json]
    import os
    import sys
    if "--screen" not in sys.argv:
        sys.exit("usage: price_history.py --screen [data.json]")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = [a for a in sys.argv[1:] if a != "--screen"]
    path = args[0] if args else os.path.join(root, DATA_FILE)
    with open(path) as f:
        d = json.load(f)
    pr = lambda *a: print(*a)
    d = build_price_history(d, root, log_info=pr, log_ok=pr)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    print("unscreened days left:", d["price_history"]["unscreened_days"])
