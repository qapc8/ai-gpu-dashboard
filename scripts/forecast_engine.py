#!/usr/bin/env python3
"""
GPU Price Forecast Engine
=========================
Generates GPU price forecasts based on historical data and market signals.

Imported by update_data.py and called weekly to regenerate the forecasts
section of data.json.

Model: Non-linear monthly forecast with regime detection, momentum carry,
mean-reversion, seasonal adjustments, and event-driven shocks.
Pure Python (math stdlib only).
"""

import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_YEAR = 2026
HOURS_PER_YEAR = 8760
USEFUL_LIFE_YEARS = 3
FLOOR_MARGIN = 1.5
FORECAST_MONTHS = 12

# Seasonal demand multipliers (month index 1-12)
# Q4 (Oct-Dec): high demand from year-end procurement
# Q1 (Jan-Mar): post-holiday softness
# Q2 (Apr-Jun): moderate
# Q3 (Jul-Sep): pre-Q4 ramp
SEASONAL_FACTORS = {
    1: 1.02, 2: 1.01, 3: 1.00,    # Q1: slight upward (post-correction)
    4: 0.99, 5: 0.98, 6: 0.97,    # Q2: mild decline
    7: 0.98, 8: 0.99, 9: 1.00,    # Q3: stabilization
    10: 1.02, 11: 1.04, 12: 1.03, # Q4: demand spike
}

# Momentum half-life in months: how fast recent trend reverts to long-term
MOMENTUM_HALFLIFE = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_monthly_prices(gpu_history: dict) -> list[tuple[str, float]]:
    """Return [(period_str, avg_price), ...] sorted chronologically."""
    items = []
    for period, info in gpu_history.items():
        avg = info.get("avg")
        if avg is not None:
            items.append((period, float(avg)))
    items.sort(key=lambda x: x[0])
    return items


def _compute_mom_pct_changes(prices: list[float]) -> list[float]:
    """Compute (P[i+1]-P[i])/P[i] for consecutive prices."""
    changes = []
    for i in range(len(prices) - 1):
        if prices[i] > 0:
            changes.append((prices[i + 1] - prices[i]) / prices[i])
    return changes


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _advance_month(year: int, month: int, offset: int) -> tuple[int, int]:
    """Advance a year/month by offset months."""
    m = month + offset - 1
    return year + m // 12, (m % 12) + 1


def _fmt_month(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def _detect_regime(sorted_prices: list[tuple[str, float]]) -> dict:
    """
    Analyze recent price history to detect current regime.
    Returns dict with:
      - regime: "declining" | "stable" | "recovering" | "volatile"
      - recent_momentum: monthly rate of change (negative = declining)
      - long_term_rate: overall monthly rate from full history
      - volatility: stdev of monthly pct changes
      - acceleration: is the decline speeding up or slowing down?
    """
    prices = [p for _, p in sorted_prices]
    n = len(prices)

    if n < 3:
        return {
            "regime": "stable",
            "recent_momentum": 0.0,
            "long_term_rate": 0.0,
            "volatility": 0.15,
            "acceleration": 0.0,
        }

    all_changes = _compute_mom_pct_changes(prices)
    volatility = _stdev(all_changes) if all_changes else 0.0

    # Long-term rate: median of all changes
    long_term_rate = _median(all_changes)

    # Recent momentum: weighted average of last 3 months (more recent = higher weight)
    recent = all_changes[-3:] if len(all_changes) >= 3 else all_changes
    weights = list(range(1, len(recent) + 1))  # 1, 2, 3 (most recent gets highest)
    total_w = sum(weights)
    recent_momentum = sum(r * w for r, w in zip(recent, weights)) / total_w

    # Acceleration: compare last 3 months rate vs prior 3 months
    acceleration = 0.0
    if len(all_changes) >= 6:
        recent_avg = _mean(all_changes[-3:])
        prior_avg = _mean(all_changes[-6:-3])
        acceleration = recent_avg - prior_avg

    # Classify regime
    if recent_momentum > 0.02:
        regime = "recovering"
    elif recent_momentum > -0.01:
        regime = "stable"
    elif volatility > abs(recent_momentum) * 2 and volatility > 0.05:
        regime = "volatile"
    else:
        regime = "declining"

    return {
        "regime": regime,
        "recent_momentum": recent_momentum,
        "long_term_rate": long_term_rate,
        "volatility": volatility,
        "acceleration": acceleration,
    }


# ---------------------------------------------------------------------------
# Market Factor Analysis (same as before, kept modular)
# ---------------------------------------------------------------------------

def _supply_factor(gpu_id: str, data: dict) -> tuple[float, str]:
    indicators = data.get("indicators", {})
    lead_times = indicators.get("gpu_lead_times", {})
    gpu_lt = lead_times.get(gpu_id, {})
    weeks = gpu_lt.get("weeks", 4)
    status = gpu_lt.get("status", "available").lower()
    notes = []

    if status == "constrained" or weeks > 8:
        lt_factor = 0.7
        notes.append(f"constrained supply ({weeks}wk lead time) slows decline")
    elif status == "limited" or 4 <= weeks <= 8:
        lt_factor = 0.85
        notes.append(f"limited supply ({weeks}wk lead time)")
    else:
        lt_factor = 1.15
        notes.append(f"available supply ({weeks}wk lead time) accelerates decline")

    shipments = indicators.get("data_center_gpu_shipments_k", {})
    shipment_keys = sorted(shipments.keys())
    dc_adj = 1.0
    if len(shipment_keys) >= 5:
        latest_val = shipments[shipment_keys[-1]]
        yoy_val = shipments[shipment_keys[-5]]
        if yoy_val > 0:
            yoy_growth = (latest_val - yoy_val) / yoy_val
            if yoy_growth > 0.20:
                dc_adj = 1.05
                notes.append(f"DC shipments +{yoy_growth*100:.0f}% YoY accelerates decline")

    return lt_factor * dc_adj, "; ".join(notes)


def _demand_factor(data: dict) -> tuple[float, str]:
    inference = data.get("inference", {})
    notes = []
    total_tokens = sum(
        m.get("tokens_7d", 0) for m in inference.values()
        if isinstance(m.get("tokens_7d"), (int, float))
    )

    if total_tokens > 5000:
        token_factor = 0.85
        notes.append(f"high token demand ({total_tokens:.0f}B/wk) slows decline")
    elif total_tokens > 2000:
        token_factor = 1.0
        notes.append(f"moderate token demand ({total_tokens:.0f}B/wk)")
    else:
        token_factor = 1.1
        notes.append(f"low token demand ({total_tokens:.0f}B/wk) accelerates decline")

    indicators = data.get("indicators", {})
    capex = indicators.get("ai_capex_bn", {})
    est_2026 = capex.get("2026_est", capex.get(2026, 0))
    est_2025 = capex.get("2025_est", capex.get("2025", capex.get(2025, 0)))
    capex_adj = 1.0
    if est_2025 and est_2025 > 0:
        capex_growth = (est_2026 - est_2025) / est_2025
        if capex_growth > 0.30:
            capex_adj = 0.90
            notes.append(f"AI CapEx +{capex_growth*100:.0f}% strong demand")
        elif capex_growth < 0.15:
            capex_adj = 1.05
            notes.append(f"AI CapEx +{capex_growth*100:.0f}% weak demand")
        else:
            notes.append(f"AI CapEx +{capex_growth*100:.0f}% moderate")

    return token_factor * capex_adj, "; ".join(notes)


def _competitive_factor(gpu_id: str, data: dict) -> tuple[float, str]:
    specs = data.get("specs", {})
    gpu_spec = specs.get(gpu_id, {})
    vendor = gpu_spec.get("vendor", "NVIDIA")
    indicators = data.get("indicators", {})
    notes = []

    amd_share = indicators.get("amd_gpu_market_share_pct", {})
    amd_sorted = sorted(amd_share.items())
    amd_gaining = False
    amd_growth_pct = 0.0
    if len(amd_sorted) >= 2:
        oldest, newest = amd_sorted[0][1], amd_sorted[-1][1]
        if oldest > 0:
            amd_growth_pct = ((newest - oldest) / oldest) * 100
            amd_gaining = amd_growth_pct > 5

    factor = 1.0
    if vendor == "NVIDIA":
        if amd_gaining:
            acceleration = min(0.10, amd_growth_pct / 100 * 0.5)
            factor *= (1 + acceleration)
            notes.append(f"AMD gaining share (+{amd_growth_pct:.0f}%)")
        gpu_tier = gpu_spec.get("tier", "")
        gpu_tflops = gpu_spec.get("fp16_tflops", 0)
        gpu_msrp = gpu_spec.get("msrp_usd", 1)
        if gpu_msrp > 0 and gpu_tflops > 0:
            ratio = gpu_tflops / gpu_msrp
            for oid, ospec in specs.items():
                if oid != gpu_id and ospec.get("vendor") == "AMD" and ospec.get("tier") == gpu_tier:
                    ot, om = ospec.get("fp16_tflops", 0), ospec.get("msrp_usd", 1)
                    if om > 0 and ot > 0 and ot / om > ratio:
                        factor *= 1.05
                        notes.append(f"competitive pressure from {oid}")
                        break
    elif vendor == "AMD":
        if amd_gaining:
            factor *= 0.95
            notes.append("AMD gaining share — prices hold")
        else:
            notes.append("AMD share stable")
        gpu_tier = gpu_spec.get("tier", "")
        gpu_release = gpu_spec.get("release_year", 2020)
        for oid, ospec in specs.items():
            if ospec.get("vendor") == "NVIDIA" and ospec.get("tier") == gpu_tier:
                if ospec.get("release_year", 2020) > gpu_release:
                    factor *= 1.05
                    notes.append(f"NVIDIA {oid} gen pressure")
                    break

    if not notes:
        notes.append("no significant competitive pressure")
    return factor, "; ".join(notes)


def _regulatory_factor(gpu_id: str, data: dict) -> tuple[float, str]:
    supplychain = data.get("supplychain", {})
    export_controls = supplychain.get("export_controls", [])
    cutoff_str = "2025-09"
    restricting = liberalizing = 0
    notes = []

    for event in export_controls:
        if event.get("date", "") < cutoff_str:
            continue
        affected = event.get("affected_gpus", [])
        status = event.get("status", "")
        category = event.get("category", "").lower()

        gpu_affected = False
        for ag in affected:
            ag_n = ag.upper().replace(" ", "").replace("-", "")
            gpu_n = gpu_id.upper().replace(" ", "").replace("-", "")
            if gpu_n in ag_n or ag_n in gpu_n or gpu_n.startswith(ag_n):
                gpu_affected = True
                break
            if "all" in ag.lower() and "gpu" in ag.lower():
                gpu_affected = True
                break
        if not gpu_affected:
            continue

        if status == "rescinded" or "deregulation" in category:
            liberalizing += 1
        elif "export control" in category or "retaliation" in category or status in ("enacted", "in_effect", "proposed"):
            restricting += 1

    factor = (0.97 ** restricting) * (1.02 ** liberalizing)
    if restricting + liberalizing > 0:
        notes.append(f"{restricting} restricting, {liberalizing} liberalizing events")
    else:
        notes.append("no recent regulatory events")
    return factor, "; ".join(notes)


def _displacement_factor(gpu_id: str, data: dict) -> tuple[float, str]:
    specs = data.get("specs", {})
    gpu_spec = specs.get(gpu_id, {})
    vendor = gpu_spec.get("vendor", "")
    release_year = gpu_spec.get("release_year", CURRENT_YEAR)
    fp16 = gpu_spec.get("fp16_tflops", 0)
    msrp = gpu_spec.get("msrp_usd", 1)
    notes = []

    if msrp <= 0 or fp16 <= 0:
        return 1.0, "insufficient spec data"

    ratio = fp16 / msrp
    hits = 0
    for oid, ospec in specs.items():
        if oid == gpu_id or ospec.get("vendor") != vendor:
            continue
        if ospec.get("release_year", 2000) <= release_year:
            continue
        of, om = ospec.get("fp16_tflops", 0), ospec.get("msrp_usd", 1)
        if om <= 0 or of <= 0:
            continue
        improvement = (of / om - ratio) / ratio if ratio > 0 else 0
        if improvement > 0.05:
            hits += 1
            notes.append(f"{oid} +{improvement*100:.0f}% TFLOPS/$")

    factor = min(1.25, 1.0 + hits * 0.05) if hits > 0 else 1.0
    if not notes:
        notes.append("no significant generational displacement")
    return factor, "; ".join(notes)


# ---------------------------------------------------------------------------
# Price Floor
# ---------------------------------------------------------------------------

def _price_floor(gpu_id: str, data: dict) -> float:
    specs = data.get("specs", {})
    msrp = specs.get(gpu_id, {}).get("msrp_usd", 0)
    if msrp > 0:
        return round(msrp / (USEFUL_LIFE_YEARS * HOURS_PER_YEAR) * FLOOR_MARGIN, 4)
    return 0.0


def _elasticity(gpu_id: str, data: dict, volatility: float) -> float:
    specs = data.get("specs", {})
    release_year = specs.get(gpu_id, {}).get("release_year", CURRENT_YEAR)
    age_factor = min(5, CURRENT_YEAR - release_year)
    raw = -1 * (volatility * 2 + age_factor * 0.1)
    return round(_clamp(raw, -0.6, -0.05), 4)


# ---------------------------------------------------------------------------
# Monthly Curve Generation (core non-linear model)
# ---------------------------------------------------------------------------

def _generate_monthly_curve(
    current_avg: float,
    regime: dict,
    adjusted_lambda: float,
    floor: float,
    start_year: int,
    start_month: int,
) -> list[dict]:
    """
    Generate 12 monthly forecast points with non-linear dynamics.

    The model blends:
    1. Recent momentum (carries forward with exponential decay)
    2. Long-term structural rate (mean-reversion target)
    3. Seasonal demand patterns
    4. Volatility-scaled confidence bands

    This produces curves that can show plateaus, acceleration,
    recovery phases, and seasonal bumps — not just smooth decay.
    """
    recent_momentum = regime["recent_momentum"]
    long_term_rate = regime["long_term_rate"]
    volatility = regime["volatility"]
    acceleration = regime["acceleration"]

    # Structural rate from the multi-factor adjusted lambda
    # Convert lambda (decay constant) to monthly rate
    # lambda > 0 means decline, so structural_rate < 0
    structural_rate = -adjusted_lambda

    # If long-term trend is clearly negative, use the stronger signal
    if long_term_rate < structural_rate:
        structural_rate = (structural_rate + long_term_rate) / 2

    # For recovering regimes, allow positive structural rate near-term
    if regime["regime"] == "recovering":
        # Blend: momentum dominates initially, then structural takes over
        structural_rate = min(structural_rate * 0.5, -0.005)

    points = []
    price = current_avg

    for t in range(1, FORECAST_MONTHS + 1):
        y, m = _advance_month(start_year, start_month, t)

        # Momentum decays exponentially toward structural rate
        # Half-life determines how fast recent trend fades
        decay_weight = math.exp(-math.log(2) * t / MOMENTUM_HALFLIFE)

        # Blended monthly rate: momentum + structural
        monthly_rate = (recent_momentum * decay_weight +
                        structural_rate * (1 - decay_weight))

        # Acceleration effect (fading): if decline was accelerating,
        # continue that pattern but dampen it
        if abs(acceleration) > 0.001:
            accel_effect = acceleration * decay_weight * 0.5
            monthly_rate += accel_effect

        # Seasonal adjustment
        seasonal = SEASONAL_FACTORS.get(m, 1.0)
        monthly_rate *= seasonal

        # Cap monthly change to prevent extremes
        monthly_rate = _clamp(monthly_rate, -0.12, 0.08)

        # Apply monthly change
        price = price * (1 + monthly_rate)

        # Enforce floor
        price = max(price, floor)

        # Confidence bands widen with time and volatility
        # Use asymmetric bands: upside wider than downside for declining,
        # downside wider for recovering
        vol_t = volatility * math.sqrt(t)

        if regime["regime"] == "recovering":
            low = price * (1 - vol_t * 1.8)
            high = price * (1 + vol_t * 1.2)
        elif regime["regime"] == "volatile":
            low = price * (1 - vol_t * 2.0)
            high = price * (1 + vol_t * 2.0)
        else:
            low = price * (1 - vol_t * 1.5)
            high = price * (1 + vol_t * 1.0)

        low = max(low, floor * 0.9)
        high = max(high, price * 1.01)

        points.append({
            "month": _fmt_month(y, m),
            "mid": round(price, 4),
            "low": round(low, 4),
            "high": round(high, 4),
        })

    return points


# ---------------------------------------------------------------------------
# Main Forecast Function
# ---------------------------------------------------------------------------

def generate_forecasts(data: dict) -> dict:
    """Generate GPU price forecasts for all GPUs with historical data."""
    historical = data.get("historical", {})
    if not historical:
        return {}

    forecasts = {}
    for gpu_id, gpu_history in historical.items():
        forecast = _forecast_single_gpu(gpu_id, gpu_history, data)
        if forecast is not None:
            forecasts[gpu_id] = forecast

    return forecasts


def _forecast_single_gpu(gpu_id: str, gpu_history: dict, data: dict) -> dict | None:
    """Generate forecast for a single GPU."""

    sorted_prices = _sorted_monthly_prices(gpu_history)
    if not sorted_prices:
        return None

    current_avg = sorted_prices[-1][1]
    if current_avg <= 0:
        return None

    num_points = len(sorted_prices)

    # Detect regime from historical patterns
    regime = _detect_regime(sorted_prices)

    # Compute base lambda from recent history for factor adjustment
    if num_points >= 3:
        recent = [p for _, p in sorted_prices[-7:]]
        log_changes = []
        for i in range(len(recent) - 1):
            if recent[i] > 0 and recent[i + 1] > 0:
                log_changes.append(math.log(recent[i + 1] / recent[i]))
        base_lambda = -_median(log_changes) if log_changes else 0.0
        if base_lambda < 0:
            base_lambda = 0.005
    else:
        base_lambda = 0.0

    # Collect market factors
    factor_log = []
    supply_f, supply_note = _supply_factor(gpu_id, data)
    factor_log.append(("supply_pressure", supply_f, supply_note))
    demand_f, demand_note = _demand_factor(data)
    factor_log.append(("demand_signal", demand_f, demand_note))
    competitive_f, competitive_note = _competitive_factor(gpu_id, data)
    factor_log.append(("competitive_pressure", competitive_f, competitive_note))
    regulatory_f, regulatory_note = _regulatory_factor(gpu_id, data)
    factor_log.append(("regulatory_impact", regulatory_f, regulatory_note))
    displacement_f, displacement_note = _displacement_factor(gpu_id, data)
    factor_log.append(("generational_displacement", displacement_f, displacement_note))

    # Adjusted lambda
    adjusted_lambda = base_lambda * supply_f * demand_f * competitive_f * regulatory_f * displacement_f
    adjusted_lambda = min(adjusted_lambda, 0.08)

    # Price floor
    floor = _price_floor(gpu_id, data)

    # Determine start month from last historical period
    last_period = sorted_prices[-1][0]
    parts = last_period.split("-")
    start_year, start_month = int(parts[0]), int(parts[1])

    # Generate monthly curve
    is_static = num_points < 3
    if is_static:
        # Static forecast with flat curve and wide bands
        monthly = []
        for t in range(1, FORECAST_MONTHS + 1):
            y, m = _advance_month(start_year, start_month, t)
            monthly.append({
                "month": _fmt_month(y, m),
                "mid": round(current_avg, 4),
                "low": round(current_avg * 0.85, 4),
                "high": round(current_avg * 1.10, 4),
            })
    else:
        monthly = _generate_monthly_curve(
            current_avg, regime, adjusted_lambda, floor,
            start_year, start_month,
        )

    # Build result
    result = {
        "current_avg": round(current_avg, 4),
        "forecast_monthly": monthly,
        "regime": regime["regime"],
    }

    # Extract summary points (3mo, 6mo, 12mo) from the monthly curve
    indicators = data.get("indicators", {})
    lt_weeks = indicators.get("gpu_lead_times", {}).get(gpu_id, {}).get("weeks", 4)

    for idx, label in [(2, "3mo"), (5, "6mo"), (11, "12mo")]:
        if idx < len(monthly):
            pt = monthly[idx]
            result[f"forecast_{label}"] = {
                "low": pt["low"],
                "mid": pt["mid"],
                "high": pt["high"],
            }
            t = idx + 1
            if is_static:
                confidence = max(20, 40 - (t * 3))
            else:
                confidence = max(20, 100 - (t * 5) - (regime["volatility"] * 200) - (lt_weeks * 2))
            result[f"confidence_{label}"] = round(_clamp(confidence, 20, 95), 1)

    # Price floor
    result["price_floor"] = round(floor, 4) if floor > 0 else round(monthly[-1]["mid"] * 0.6, 4)

    # Elasticity
    result["elasticity_coefficient"] = _elasticity(gpu_id, data, regime["volatility"])

    # Factor weights
    result["factors"] = {
        "historical_trend": round(base_lambda, 6),
        "supply_pressure": round(supply_f, 4),
        "demand_signal": round(demand_f, 4),
        "competitive_pressure": round(competitive_f, 4),
        "regulatory_impact": round(regulatory_f, 4),
        "generational_displacement": round(displacement_f, 4),
    }

    # Methodology
    result["methodology"] = _build_methodology(
        gpu_id, base_lambda, adjusted_lambda, regime,
        num_points, is_static, factor_log
    )

    return result


def _build_methodology(
    gpu_id: str,
    base_lambda: float,
    adjusted_lambda: float,
    regime: dict,
    num_points: int,
    is_static: bool,
    factor_log: list,
) -> str:
    if is_static:
        return (
            f"Static forecast for {gpu_id} — insufficient historical data "
            f"({num_points} data points). Low confidence. "
            f"Prices held constant with wide uncertainty bands."
        )

    regime_desc = {
        "declining": "steady decline",
        "stable": "price stability",
        "recovering": "price recovery",
        "volatile": "high volatility",
    }

    parts = [
        f"Non-linear model with {regime_desc.get(regime['regime'], 'mixed')} regime "
        f"detected from {min(6, num_points)}-month history "
        f"(momentum={regime['recent_momentum']*100:+.1f}%/mo)."
    ]

    adjustments = []
    for name, factor, note in factor_log:
        if abs(factor - 1.0) > 0.005:
            pct = (factor - 1.0) * 100
            label = name.replace("_", " ")
            adjustments.append(f"{pct:+.0f}% {label}")
    if adjustments:
        parts.append("Factors: " + ", ".join(adjustments) + ".")

    for name, factor, note in factor_log:
        if name == "regulatory_impact" and "event" in note:
            parts.append(note.capitalize() + ".")
            break

    parts.append(
        f"Momentum decays toward structural rate over {MOMENTUM_HALFLIFE:.0f}-month half-life "
        f"with seasonal adjustments. "
        f"Volatility: {regime['volatility']*100:.1f}%/mo."
    )

    return " ".join(parts)
