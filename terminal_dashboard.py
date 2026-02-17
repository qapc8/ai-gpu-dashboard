#!/usr/bin/env python3
"""
Bloomberg-Style AI GPU Pricing Terminal Dashboard
Rich-based terminal UI with live data, charts, and AI analysis.
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich.live import Live
from rich.align import Align
from rich import box
from rich.progress_bar import ProgressBar
from rich.style import Style
from rich.markdown import Markdown

from gpu_data import (
    GPU_SPECS, CLOUD_PRICING, HISTORICAL_PRICING, MARKET_INDICATORS,
    REGIONAL_DATA, WORKLOAD_RECOMMENDATIONS,
    TCO_COMPONENTS, INFERENCE_BENCHMARKS, SPOT_MARKET, NEWS_FEED,
    get_cheapest_by_gpu, get_price_comparison_matrix,
    generate_market_summary, get_regional_summary, get_workload_recommendations,
    get_utilization_summary, get_reservation_analysis, get_price_forecasts,
    get_competitive_landscape, get_sustainability_summary, get_supply_chain_summary,
    get_model_hardware_fit, UTILIZATION_METRICS, RESERVATION_ANALYTICS,
    PRICE_FORECASTS, COMPETITIVE_MOAT, GPU_CARBON_FOOTPRINT,
    SUPPLY_CHAIN_RISK, EXPORT_CONTROL_TRACKER, MODEL_HARDWARE_FIT,
)

console = Console()

# ============================================================================
# SPARKLINE GENERATOR
# ============================================================================

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list) -> str:
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK_CHARS[min(int((v - mn) / rng * 7), 7)] for v in values)


def trend_arrow(change_pct: float) -> str:
    if change_pct < -5:
        return "[bold green]▼▼[/]"
    elif change_pct < -1:
        return "[green]▼[/]"
    elif change_pct < 1:
        return "[yellow]→[/]"
    elif change_pct < 5:
        return "[red]▲[/]"
    else:
        return "[bold red]▲▲[/]"


def avail_color(avail: str) -> str:
    colors = {"scarce": "bold red", "limited": "red", "moderate": "yellow", "good": "green", "abundant": "bold green"}
    return colors.get(avail, "white")


# ============================================================================
# DASHBOARD SECTIONS
# ============================================================================

def render_header() -> Panel:
    now = datetime.now()
    title = Text()
    title.append("  AI GPU MARKET TERMINAL  ", style="bold white on blue")
    title.append("  ", style="")
    title.append(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}  ", style="bold white on dark_green")
    title.append("  ", style="")
    title.append("  LIVE  ", style="bold white on red")

    subtitle = Text()
    subtitle.append(f"  NVDA ${MARKET_INDICATORS['nvidia_stock']['current']:.2f} ", style="bold green" if MARKET_INDICATORS['nvidia_stock']['ytd_change'] > 0 else "bold red")
    subtitle.append(f"({MARKET_INDICATORS['nvidia_stock']['ytd_change']:+.1f}% YTD)  ", style="green" if MARKET_INDICATORS['nvidia_stock']['ytd_change'] > 0 else "red")
    subtitle.append(" │ ", style="dim")
    subtitle.append(f"AMD ${MARKET_INDICATORS['amd_stock']['current']:.2f} ", style="bold green" if MARKET_INDICATORS['amd_stock']['ytd_change'] > 0 else "bold red")
    subtitle.append(f"({MARKET_INDICATORS['amd_stock']['ytd_change']:+.1f}% YTD)  ", style="green" if MARKET_INDICATORS['amd_stock']['ytd_change'] > 0 else "red")
    subtitle.append(" │ ", style="dim")
    subtitle.append(f"GPU Market: $95.8B (2025E)  ", style="cyan")
    subtitle.append(" │ ", style="dim")
    subtitle.append(f"AI CapEx: $150B (2025E)", style="cyan")

    grid = Table.grid(padding=0)
    grid.add_row(Align.center(title))
    grid.add_row(Align.center(subtitle))

    return Panel(grid, style="bold blue", box=box.DOUBLE)


def render_price_matrix() -> Panel:
    table = Table(
        title="GPU PRICE MATRIX — On-Demand $/hr Per GPU",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        row_styles=["", "dim"],
        padding=(0, 1),
    )

    table.add_column("GPU", style="bold white", min_width=18)
    table.add_column("VRAM", style="cyan", justify="right")
    table.add_column("Arch", style="yellow")
    table.add_column("Best $/hr", justify="right", style="bold green")
    table.add_column("Provider", style="green")
    table.add_column("Avg $/hr", justify="right")
    table.add_column("Max $/hr", justify="right", style="red")
    table.add_column("#Prov", justify="center")
    table.add_column("Spread", justify="right")
    table.add_column("MoM Δ", justify="right")
    table.add_column("Trend", justify="center")
    table.add_column("TFLOPS/$", justify="right", style="magenta")

    matrix = get_price_comparison_matrix()
    for row in matrix:
        qoq = row["monthly_change_pct"]
        qoq_style = "green" if qoq < 0 else "red" if qoq > 0 else "yellow"
        spread_style = "green" if row["price_spread_pct"] > 30 else "yellow"

        # Get sparkline from historical data
        hist = HISTORICAL_PRICING.get(row["gpu_id"], {})
        spark_vals = [hist[k]["avg"] for k in sorted(hist.keys())]
        spark = sparkline(spark_vals) if spark_vals else "—"

        table.add_row(
            row["name"],
            f"{row['vram_gb']}GB",
            row["arch"],
            f"${row['cheapest_price']:.2f}",
            row["cheapest_provider"],
            f"${row['avg_price']:.2f}",
            f"${row['most_expensive']:.2f}",
            str(row["num_providers"]),
            f"[{spread_style}]{row['price_spread_pct']:.0f}%[/]",
            f"[{qoq_style}]{qoq:+.1f}%[/]",
            spark if spark != "—" else trend_arrow(qoq),
            f"{row['flops_per_dollar']:.0f}",
        )

    return Panel(table, border_style="blue", box=box.ROUNDED)


def render_provider_comparison(gpu_id: str = "H100-SXM") -> Panel:
    providers = get_cheapest_by_gpu(gpu_id)
    spec = GPU_SPECS.get(gpu_id, {})

    table = Table(
        title=f"PROVIDER COMPARISON — {spec.get('name', gpu_id)}",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    table.add_column("Provider", style="bold white", min_width=12)
    table.add_column("Instance", style="dim")
    table.add_column("On-Demand", justify="right", style="bold yellow")
    table.add_column("Spot", justify="right", style="green")
    table.add_column("1yr Res.", justify="right")
    table.add_column("3yr Res.", justify="right", style="cyan")
    table.add_column("Monthly", justify="right", style="bold white")
    table.add_column("Annual", justify="right")
    table.add_column("vs Best", justify="right")

    if providers:
        best = providers[0]["price_per_gpu_hr"]
        for p in providers:
            premium = ((p["price_per_gpu_hr"] - best) / best * 100) if best > 0 else 0
            prem_style = "green" if premium == 0 else "yellow" if premium < 20 else "red"
            spot_str = f"${p['spot_price']:.2f}" if p["spot_price"] else "N/A"

            table.add_row(
                p["provider"],
                p["instance"][:20],
                f"${p['price_per_gpu_hr']:.2f}",
                spot_str,
                f"${p['reserved_1yr']:.2f}",
                f"${p['reserved_3yr']:.2f}",
                f"${p['price_monthly']:,.0f}",
                f"${p['price_monthly'] * 12:,.0f}",
                f"[{prem_style}]{premium:+.1f}%[/]",
            )

    return Panel(table, border_style="green", box=box.ROUNDED)


def render_historical_trends() -> Panel:
    table = Table(
        title="HISTORICAL PRICE TRENDS — Quarterly Average $/hr",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    table.add_column("GPU", style="bold white", min_width=14)

    # Collect all periods
    all_periods = set()
    for gpu_id in ["H100-SXM", "H200", "B200", "A100-80GB", "A100-40GB", "MI300X", "RTX-4090"]:
        if gpu_id in HISTORICAL_PRICING:
            all_periods.update(HISTORICAL_PRICING[gpu_id].keys())
    periods = sorted(all_periods)[-8:]  # Last 8 quarters

    for p in periods:
        table.add_column(p, justify="right", min_width=8)
    table.add_column("Spark", justify="center", min_width=10)
    table.add_column("Avail", justify="center")

    for gpu_id in ["H100-SXM", "H200", "B200", "A100-80GB", "A100-40GB", "MI300X", "RTX-4090"]:
        if gpu_id not in HISTORICAL_PRICING:
            continue
        hist = HISTORICAL_PRICING[gpu_id]
        row_vals = []
        cells = []
        for p in periods:
            if p in hist:
                val = hist[p]["avg"]
                row_vals.append(val)
                cells.append(f"${val:.2f}")
            else:
                cells.append("[dim]—[/]")

        spark = sparkline(row_vals) if row_vals else "—"

        # Get latest availability
        latest_period = sorted(hist.keys())[-1]
        avail = hist[latest_period]["availability"]
        avail_str = f"[{avail_color(avail)}]{avail.upper()}[/]"

        table.add_row(gpu_id, *cells, spark, avail_str)

    return Panel(table, border_style="magenta", box=box.ROUNDED)


def render_regional_dashboard() -> Panel:
    table = Table(
        title="GLOBAL GPU MARKET — Regional Analysis",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    table.add_column("Region", style="bold white", min_width=18)
    table.add_column("Share", justify="right", style="cyan")
    table.add_column("YoY Growth", justify="right")
    table.add_column("Demand Idx", justify="center")
    table.add_column("Premium", justify="right")
    table.add_column("Energy $/kWh", justify="right")
    table.add_column("DCs", justify="right")
    table.add_column("Reg. Score", justify="right")
    table.add_column("Top Providers", style="dim")

    for region, data in REGIONAL_DATA.items():
        growth_style = "bold green" if data["yoy_growth_pct"] > 40 else "green" if data["yoy_growth_pct"] > 25 else "yellow"

        # Demand bar
        demand = data["gpu_demand_index"]
        bar_filled = "█" * (demand // 10)
        bar_empty = "░" * (10 - demand // 10)
        demand_color = "green" if demand >= 80 else "yellow" if demand >= 50 else "red"
        demand_str = f"[{demand_color}]{bar_filled}{bar_empty}[/] {demand}"

        table.add_row(
            region,
            f"{data['market_share_pct']}%",
            f"[{growth_style}]{data['yoy_growth_pct']:+.1f}%[/]",
            demand_str,
            f"+{data['avg_price_premium_pct']}%",
            f"${data['energy_cost_kwh']:.3f}",
            str(data["data_centers_count"]),
            f"{data['regulatory_score']}/10",
            ", ".join(data["top_providers"][:3]),
        )

    return Panel(table, border_style="yellow", box=box.ROUNDED)


def render_market_indicators() -> Panel:
    mi = MARKET_INDICATORS

    # GPU Shipments sparkline
    shipments = mi["data_center_gpu_shipments_k"]
    ship_vals = [shipments[k] for k in sorted(shipments.keys())]
    ship_spark = sparkline(ship_vals)

    # Lead time sparkline (inverted - lower is better)
    lead = mi["h100_lead_time_weeks"]
    lead_vals = [lead[k] for k in sorted(lead.keys())]
    lead_spark = sparkline(lead_vals)

    text = Text()
    text.append("═══ EQUITY INDICATORS ═══\n", style="bold cyan")
    nvda = mi["nvidia_stock"]
    text.append(f"  NVDA  ${nvda['current']:>8.2f}  ", style="bold")
    text.append(f"1M: {nvda['1m_change']:+.1f}%  ", style="green" if nvda['1m_change'] > 0 else "red")
    text.append(f"3M: {nvda['3m_change']:+.1f}%  ", style="green" if nvda['3m_change'] > 0 else "red")
    text.append(f"YTD: {nvda['ytd_change']:+.1f}%  ", style="green" if nvda['ytd_change'] > 0 else "red")
    text.append(f"52W: ${nvda['52w_low']:.0f}-${nvda['52w_high']:.0f}\n", style="dim")

    amd = mi["amd_stock"]
    text.append(f"  AMD   ${amd['current']:>8.2f}  ", style="bold")
    text.append(f"1M: {amd['1m_change']:+.1f}%  ", style="green" if amd['1m_change'] > 0 else "red")
    text.append(f"3M: {amd['3m_change']:+.1f}%  ", style="green" if amd['3m_change'] > 0 else "red")
    text.append(f"YTD: {amd['ytd_change']:+.1f}%  ", style="green" if amd['ytd_change'] > 0 else "red")
    text.append(f"52W: ${amd['52w_low']:.0f}-${amd['52w_high']:.0f}\n", style="dim")

    text.append("\n═══ MARKET SIZE ($B) ═══\n", style="bold cyan")
    mkt = mi["gpu_market_size_bn"]
    text.append(f"  2023: ${mkt['2023']}B  2024: ${mkt['2024']}B  2025: ${mkt['2025']}B  ", style="bold")
    text.append(f"2026E: ${mkt['2026_est']}B  2027E: ${mkt['2027_est']}B\n", style="yellow")

    text.append("\n═══ AI CAPEX ($B) ═══\n", style="bold cyan")
    capex = mi["ai_capex_bn"]
    text.append(f"  2023: ${capex['2023']}B  2024: ${capex['2024']}B  ", style="bold")
    text.append(f"2025E: ${capex['2025_est']}B  2026E: ${capex['2026_est']}B  2027E: ${capex['2027_est']}B\n", style="yellow")

    text.append(f"\n═══ DC GPU SHIPMENTS (K units) ═══\n", style="bold cyan")
    text.append(f"  Trend: {ship_spark}  Latest: {ship_vals[-1]:,}K  ", style="bold green")
    text.append(f"(+{((ship_vals[-1] - ship_vals[-5]) / ship_vals[-5] * 100):.0f}% YoY)\n", style="green")

    text.append(f"\n═══ H100 LEAD TIME (weeks) ═══\n", style="bold cyan")
    text.append(f"  Trend: {lead_spark}  Latest: {lead_vals[-1]}wk  ", style="bold green")
    text.append(f"(from {lead_vals[0]}wk in 2023-Q1)\n", style="green")

    return Panel(text, title="[bold]MARKET INDICATORS[/]", border_style="cyan", box=box.ROUNDED)


def render_workload_guide() -> Panel:
    recs = get_workload_recommendations()

    table = Table(
        title="WORKLOAD-BASED GPU GUIDE",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    table.add_column("Workload", style="bold white", min_width=22)
    table.add_column("Recommended GPUs", style="yellow")
    table.add_column("Min GPUs", justify="center")
    table.add_column("Budget/mo", justify="right", style="green")
    table.add_column("Best Value", style="bold cyan")

    for workload, rec in recs.items():
        gpus = ", ".join(rec["recommended"][:3])
        budget = f"${rec['budget_monthly_low']:,}-${rec['budget_monthly_high']:,}"
        table.add_row(
            workload,
            gpus,
            str(rec["min_gpus"]),
            budget,
            rec["best_value"],
        )

    return Panel(table, border_style="green", box=box.ROUNDED)


def render_ai_analysis() -> Panel:
    try:
        from ai_analyzer import get_quick_summary
        summary = get_quick_summary(use_cache=True)
    except Exception as e:
        summary = f"AI analysis unavailable: {e}"

    return Panel(
        Markdown(summary),
        title="[bold]AI MARKET ANALYSIS[/]",
        subtitle="[dim]Powered by LLM — Auto-refreshes hourly[/]",
        border_style="bright_magenta",
        box=box.DOUBLE,
    )


def render_cost_calculator() -> Panel:
    text = Text()
    text.append("═══ QUICK COST ESTIMATES ═══\n\n", style="bold cyan")

    scenarios = [
        ("Train 7B LLM (1 week, 8xH100)", 8, 2.23, 168),
        ("Fine-tune 70B (3 days, 4xA100-80)", 4, 1.10, 72),
        ("Inference server (1xA10G, 30d)", 1, 1.006, 730),
        ("Batch processing (8xA100, spot, 48h)", 8, 0.70 * 0.35, 48),
        ("Dev/test (1xRTX-4090, 8hr/day, 22d)", 1, 0.22, 176),
    ]

    for label, gpus, price, hours in scenarios:
        cost = gpus * price * hours
        text.append(f"  {label}\n", style="bold white")
        text.append(f"    {gpus} GPU × ${price:.2f}/hr × {hours}h = ", style="dim")
        text.append(f"${cost:,.0f}\n", style="bold green")

    return Panel(text, title="[bold]COST CALCULATOR[/]", border_style="green", box=box.ROUNDED)


def render_news_feed() -> Panel:
    text = Text()
    for n in NEWS_FEED[:10]:
        sent_style = "green" if n["sentiment"] == "positive" else "red" if n["sentiment"] == "negative" else "yellow"
        impact_style = "bold red" if n["impact"] == "high" else "yellow" if n["impact"] == "medium" else "dim"
        text.append(f"  [{n['date']}] ", style="dim")
        text.append(f"{n['source']}: ", style="bold cyan")
        text.append(f"{n['headline']}\n", style=sent_style)
        text.append(f"    Category: {n['category']}  ", style="dim")
        text.append(f"Impact: ", style="dim")
        text.append(f"{n['impact'].upper()}\n", style=impact_style)
    return Panel(text, title="[bold]NEWS FEED[/]", border_style="red", box=box.ROUNDED)


def render_spot_market() -> Panel:
    table = Table(
        title="SPOT MARKET — Live Bid/Ask ($/hr)",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    table.add_column("GPU", style="bold white", min_width=12)
    table.add_column("Bid", justify="right", style="bold green")
    table.add_column("Ask", justify="right", style="bold red")
    table.add_column("Spread", justify="right", style="yellow")
    table.add_column("24h Vol", justify="right")
    table.add_column("Avail", justify="right", style="cyan")
    table.add_column("30d Vol%", justify="right")
    table.add_column("24h Price", justify="center", min_width=12)

    for gpu, data in SPOT_MARKET.items():
        prices = data.get("hourly_prices_24h", [])
        spark = sparkline(prices) if prices else "—"
        vol_style = "red" if data["volatility_30d"] > 15 else "yellow" if data["volatility_30d"] > 8 else "green"

        table.add_row(
            gpu,
            f"${data['bid']:.2f}",
            f"${data['ask']:.2f}",
            f"{data['spread_pct']:.1f}%",
            f"{data['24h_volume_gpu_hrs']:,}",
            f"{data['available_gpus']:,}",
            f"[{vol_style}]{data['volatility_30d']:.1f}%[/]",
            spark,
        )

    return Panel(table, border_style="yellow", box=box.ROUNDED)


def render_inference_economics() -> Panel:
    table = Table(
        title="INFERENCE ECONOMICS — $/Million Tokens (Self-Hosted)",
        box=box.SIMPLE_HEAVY,
        title_style="bold cyan",
        header_style="bold white on dark_blue",
        padding=(0, 1),
    )

    # Collect top providers
    models = sorted(INFERENCE_BENCHMARKS.keys(), key=lambda m: INFERENCE_BENCHMARKS[m].get("rank", 99))
    prov_set = set()
    for m in models:
        prov_set.update(INFERENCE_BENCHMARKS[m].get("providers", {}).keys())
    top_provs = ["OpenRouter", "OpenAI API", "Anthropic API", "Google Vertex", "DeepSeek API", "Together"]
    provs = [p for p in top_provs if p in prov_set][:6]

    table.add_column("#", style="dim", min_width=3)
    table.add_column("Model", style="bold white", min_width=18)
    table.add_column("Params", justify="right", min_width=6)
    for p in provs:
        table.add_column(p, justify="right", min_width=10)
    table.add_column("Best", justify="right", style="bold green", min_width=8)

    for m in models:
        d = INFERENCE_BENCHMARKS[m]
        rank = str(d.get("rank", ""))
        params = f"{d['params_b']}B"
        providers = d.get("providers", {})
        cells = []
        for p in provs:
            price = providers.get(p)
            if price is not None:
                cells.append(f"[green]${price:.2f}[/]" if price < 1 else f"[yellow]${price:.2f}[/]")
            else:
                cells.append("[dim]—[/]")
        best_prices = [v for v in providers.values() if v > 0]
        best = f"${min(best_prices):.2f}" if best_prices else "—"
        table.add_row(rank, m, params, *cells, best)

    return Panel(table, border_style="magenta", box=box.ROUNDED)


def render_tco_breakdown() -> Panel:
    text = Text()
    text.append("═══ TOTAL COST OF OWNERSHIP — Hidden Costs per GPU-hour ═══\n\n", style="bold cyan")

    components = list(TCO_COMPONENTS.keys())
    comp_labels = {
        "networking": "Network", "storage": "Storage", "egress": "Egress",
        "energy_overhead": "Energy", "ops_management": "Ops/Mgmt"
    }

    # Gather GPUs
    gpu_set = set()
    for c in components:
        gpu_set.update(TCO_COMPONENTS[c].get("cost_per_gpu_hr", {}).keys())
    gpus = sorted(gpu_set)

    for g in gpus[:6]:
        # Get base GPU cost
        matrix = get_price_comparison_matrix()
        base = next((r["cheapest_price"] for r in matrix if r["gpu_id"] == g), 0)

        overhead = sum(TCO_COMPONENTS[c].get("cost_per_gpu_hr", {}).get(g, 0) for c in components)
        total = base + overhead
        oh_pct = (overhead / total * 100) if total > 0 else 0

        text.append(f"  {g:<12}", style="bold white")
        text.append(f"GPU: ${base:.2f}", style="green")
        text.append("  +  ", style="dim")
        for c in components:
            cost = TCO_COMPONENTS[c].get("cost_per_gpu_hr", {}).get(g, 0)
            text.append(f"{comp_labels.get(c, c)}: ${cost:.2f}  ", style="dim")
        text.append(f"= ", style="dim")
        text.append(f"${total:.2f}/hr", style="bold yellow")
        text.append(f"  ({oh_pct:.0f}% overhead)", style="red")
        text.append(f"  ~${total * 730:,.0f}/mo\n", style="dim")

    return Panel(text, title="[bold]TCO ANALYSIS[/]", border_style="yellow", box=box.ROUNDED)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def render_utilization_metrics() -> Panel:
    """Render GPU utilization and efficiency metrics."""
    util = get_utilization_summary()
    table = Table(title="GPU Utilization & Efficiency", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("GPU", style="cyan bold", width=14)
    table.add_column("Avg Util %", justify="right", width=10)
    table.add_column("Avg Efficiency", justify="right", width=12)
    table.add_column("Best Provider", width=14)
    table.add_column("Best Score", justify="right", width=10)
    table.add_column("Trend (5mo)", width=14)

    for gpu_id, data in util.items():
        best = max(data["providers"].items(), key=lambda x: x[1]["efficiency_score"])
        trend = best[1].get("utilization_trend", [])
        trend_str = sparkline(trend) if trend else ""
        ut_color = "green" if data["avg_utilization"] >= 75 else "yellow" if data["avg_utilization"] >= 60 else "red"
        table.add_row(
            gpu_id, f"[{ut_color}]{data['avg_utilization']:.1f}%[/]",
            f"{data['avg_efficiency']:.1f}/100", best[0],
            f"[green]{best[1]['efficiency_score']}[/]", trend_str,
        )
    return Panel(table, border_style="bright_cyan", box=box.DOUBLE)


def render_reservation_analysis() -> Panel:
    """Render reservation and commitment analytics."""
    res = RESERVATION_ANALYTICS
    table = Table(title="Reservation & Commitment Analysis", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("GPU", style="cyan bold", width=12)
    table.add_column("On-Demand", justify="right", width=10)
    table.add_column("Spot", justify="right", width=10)
    table.add_column("1yr Res.", justify="right", width=10)
    table.add_column("3yr Res.", justify="right", width=10)
    table.add_column("Low Util", width=10)
    table.add_column("High Util", width=10)

    for gpu_id, r in res.items():
        table.add_row(
            gpu_id, f"${r['on_demand_rate']:.2f}",
            f"[green]${r['spot_avg_rate']:.2f}[/]",
            f"${r['reserved_1yr_rate']:.2f}",
            f"[green]${r['reserved_3yr_rate']:.2f}[/]",
            r["recommended_commitment"]["low_util"]["type"],
            r["recommended_commitment"]["high_util"]["type"],
        )
    return Panel(table, border_style="bright_yellow", box=box.DOUBLE)


def render_price_forecasts() -> Panel:
    """Render price elasticity and forecasting."""
    fc = PRICE_FORECASTS
    table = Table(title="Price Forecasts & Elasticity", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("GPU", style="cyan bold", width=12)
    table.add_column("Current", justify="right", width=8)
    table.add_column("3mo", justify="right", width=8)
    table.add_column("6mo", justify="right", width=8)
    table.add_column("12mo", justify="right", width=8)
    table.add_column("Floor", justify="right", width=8)
    table.add_column("Elasticity", justify="right", width=10)
    table.add_column("Supply", width=12)

    for gpu_id in sorted(fc.keys(), key=lambda g: fc[g]["current_avg"], reverse=True):
        f = fc[gpu_id]
        chg = (f["forecast_12mo"]["mid"] - f["current_avg"]) / f["current_avg"] * 100
        chg_color = "green" if chg < -10 else "yellow" if chg < 0 else "red"
        table.add_row(
            gpu_id, f"${f['current_avg']:.2f}",
            f"${f['forecast_3mo']['mid']:.2f}", f"${f['forecast_6mo']['mid']:.2f}",
            f"[{chg_color}]${f['forecast_12mo']['mid']:.2f}[/]",
            f"${f['price_floor']:.2f}",
            f"{f['elasticity_coefficient']:.2f}", f"{f['supply_signal']}",
        )
    return Panel(table, border_style="bright_blue", box=box.DOUBLE)


def render_competitive_moat() -> Panel:
    """Render competitive landscape tracker."""
    comp = COMPETITIVE_MOAT
    table = Table(title="Competitive Moat Tracker", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Vendor", style="cyan bold", width=14)
    table.add_column("Perf", justify="right", width=6)
    table.add_column("Ecosystem", justify="right", width=10)
    table.add_column("Software", justify="right", width=8)
    table.add_column("Price/Perf", justify="right", width=10)
    table.add_column("Moat", justify="right", width=6)
    table.add_column("Share", justify="right", width=8)
    table.add_column("Products", width=20)

    for vendor, d in comp.items():
        moat_color = "green" if d["moat_strength_score"] >= 70 else "yellow" if d["moat_strength_score"] >= 40 else "red"
        table.add_row(
            vendor.replace("_", " "), str(d["performance_score"]),
            str(d["ecosystem_maturity"]), str(d["software_compatibility"]),
            str(d["price_performance_ratio"]),
            f"[{moat_color}]{d['moat_strength_score']}[/]",
            f"{d['market_share_pct']}%",
            ", ".join(d["key_products"][:3]),
        )
    return Panel(table, border_style="bright_magenta", box=box.DOUBLE)


def render_sustainability_index() -> Panel:
    """Render energy and sustainability metrics."""
    sus = get_sustainability_summary()
    table = Table(title="Sustainability & Energy Index", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Provider", style="cyan bold", width=12)
    table.add_column("Avg Score", justify="right", width=10)
    table.add_column("Green %", justify="right", width=8)
    table.add_column("Avg PUE", justify="right", width=8)
    table.add_column("Best Region", width=14)
    table.add_column("Worst Region", width=14)

    for provider, data in sus["providers"].items():
        sc_color = "green" if data["avg_sustainability_score"] >= 80 else "yellow" if data["avg_sustainability_score"] >= 60 else "red"
        table.add_row(
            provider, f"[{sc_color}]{data['avg_sustainability_score']}/100[/]",
            f"{data['avg_green_energy_pct']}%", f"{data['avg_pue']}",
            data["best_region"], data["worst_region"],
        )
    return Panel(table, border_style="bright_green", box=box.DOUBLE)


def render_supply_chain_risk() -> Panel:
    """Render supply chain risk dashboard."""
    sc = SUPPLY_CHAIN_RISK
    table = Table(title="Supply Chain Risk Dashboard", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Vendor", style="cyan bold", width=14)
    table.add_column("Risk Score", justify="right", width=10)
    table.add_column("TSMC Dep.", justify="right", width=10)
    table.add_column("Lead Time", justify="right", width=10)
    table.add_column("Trend", width=12)
    table.add_column("Key Bottlenecks", width=30)

    for vendor, d in sc.items():
        risk_color = "red" if d["supply_risk_score"] > 50 else "yellow" if d["supply_risk_score"] > 35 else "green"
        table.add_row(
            vendor, f"[{risk_color}]{d['supply_risk_score']}/100[/]",
            f"{d['tsmc_dependency_pct']}%", f"{d['lead_time_weeks']} wks",
            d["risk_trend"], ", ".join(d["bottlenecks"][:2]),
        )
    return Panel(table, border_style="bright_red", box=box.DOUBLE)


def render_model_fit_matrix() -> Panel:
    """Render model-to-hardware fit matrix."""
    mf = MODEL_HARDWARE_FIT
    table = Table(title="Model-to-Hardware Fit Matrix", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("Size", style="magenta bold", width=6)
    table.add_column("GPU", style="cyan bold", width=12)
    table.add_column("Config", width=16)
    table.add_column("Throughput", justify="right", width=10)
    table.add_column("$/M Tok", justify="right", width=8)
    table.add_column("VRAM Head.", justify="right", width=10)
    table.add_column("Fit Score", justify="right", width=10)

    for size, data in mf.items():
        entries = sorted(data["gpus"].items(), key=lambda x: x[1]["fit_score"], reverse=True)
        for i, (gpu, d) in enumerate(entries[:4]):
            fit_color = "green" if d["fit_score"] >= 85 else "yellow" if d["fit_score"] >= 70 else "red"
            table.add_row(
                size if i == 0 else "", gpu, d["optimal_config"],
                f"{d['throughput_tok_s']} tok/s", f"${d['cost_per_1m_tokens']}",
                f"{d['vram_headroom_pct']}%",
                f"[{fit_color}]{d['fit_score']}/100[/]",
            )
        table.add_section()
    return Panel(table, border_style="bright_cyan", box=box.DOUBLE)


def print_full_dashboard():
    """Print the complete terminal dashboard."""
    console.clear()

    # Header
    console.print(render_header())
    console.print()

    # Price Matrix
    console.print(render_price_matrix())
    console.print()

    # Provider Comparison for top GPUs
    console.print(render_provider_comparison("H100-SXM"))
    console.print()

    # Historical Trends
    console.print(render_historical_trends())
    console.print()

    # Market Indicators + Regional side by side
    console.print(render_market_indicators())
    console.print()

    console.print(render_regional_dashboard())
    console.print()

    # News Feed
    console.print(render_news_feed())
    console.print()

    # Spot Market
    console.print(render_spot_market())
    console.print()

    # Inference Economics
    console.print(render_inference_economics())
    console.print()

    # TCO Breakdown
    console.print(render_tco_breakdown())
    console.print()

    # Workload Guide + Cost Calculator
    console.print(render_workload_guide())
    console.print()
    console.print(render_cost_calculator())
    console.print()

    # Utilization & Reservation
    console.print(render_utilization_metrics())
    console.print()
    console.print(render_reservation_analysis())
    console.print()

    # Forecasting & Competitive
    console.print(render_price_forecasts())
    console.print()
    console.print(render_competitive_moat())
    console.print()

    # Sustainability & Supply Chain
    console.print(render_sustainability_index())
    console.print()
    console.print(render_supply_chain_risk())
    console.print()

    # Model Fit
    console.print(render_model_fit_matrix())
    console.print()

    # AI Analysis
    console.print(render_ai_analysis())
    console.print()

    # Footer
    footer = Text()
    footer.append("  [Q]uit  ", style="bold white on dark_blue")
    footer.append("  [R]efresh  ", style="bold white on dark_green")
    footer.append("  [A]I Deep Analysis  ", style="bold white on dark_magenta")
    footer.append("  [1-9] GPU Drill-down  ", style="bold white on dark_red")
    footer.append(f"  Last updated: {datetime.now().strftime('%H:%M:%S')}  ", style="dim")
    console.print(Panel(footer, box=box.ROUNDED))


def run_interactive():
    """Run the interactive terminal dashboard."""
    print_full_dashboard()

    console.print("\n[bold cyan]Commands:[/] [R]efresh | [A]I Analysis | [S]pot | [I]nference | [N]ews | [H]elp | [Q]uit\n")

    while True:
        try:
            cmd = console.input("[bold blue]GPU>[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("r", "refresh"):
            print_full_dashboard()
        elif cmd in ("a", "ai", "analysis"):
            console.print("\n[bold magenta]Generating AI Analysis...[/]\n")
            try:
                from ai_analyzer import get_all_analyses
                analyses = get_all_analyses(use_cache=False)
                for key, text in analyses.items():
                    if key == "generated_at":
                        continue
                    console.print(Panel(
                        Markdown(str(text)),
                        title=f"[bold]{key.upper().replace('_', ' ')}[/]",
                        border_style="magenta",
                        box=box.DOUBLE,
                    ))
                    console.print()
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
        elif cmd in ("h100", "h", "1"):
            console.print(render_provider_comparison("H100-SXM"))
        elif cmd in ("a100", "2"):
            console.print(render_provider_comparison("A100-80GB"))
        elif cmd in ("h200", "3"):
            console.print(render_provider_comparison("H200"))
        elif cmd in ("mi300x", "mi300", "4"):
            console.print(render_provider_comparison("MI300X"))
        elif cmd in ("b200", "5"):
            console.print(render_provider_comparison("B200"))
        elif cmd in ("4090", "rtx4090", "6"):
            console.print(render_provider_comparison("RTX-4090"))
        elif cmd in ("trends", "t"):
            console.print(render_historical_trends())
        elif cmd in ("regional", "reg", "g"):
            console.print(render_regional_dashboard())
        elif cmd in ("market", "m"):
            console.print(render_market_indicators())
        elif cmd in ("workload", "w"):
            console.print(render_workload_guide())
        elif cmd in ("cost", "c"):
            console.print(render_cost_calculator())
        elif cmd in ("spot", "s"):
            console.print(render_spot_market())
        elif cmd in ("inference", "inf", "i"):
            console.print(render_inference_economics())
        elif cmd in ("tco",):
            console.print(render_tco_breakdown())
        elif cmd in ("news", "n"):
            console.print(render_news_feed())
        elif cmd in ("util", "utilization"):
            console.print(render_utilization_metrics())
        elif cmd in ("res", "reservations"):
            console.print(render_reservation_analysis())
        elif cmd in ("forecast", "fc"):
            console.print(render_price_forecasts())
        elif cmd in ("competitive", "comp"):
            console.print(render_competitive_moat())
        elif cmd in ("sus", "sustainability", "green"):
            console.print(render_sustainability_index())
        elif cmd in ("supply", "risk"):
            console.print(render_supply_chain_risk())
        elif cmd in ("fit", "model", "modelfit"):
            console.print(render_model_fit_matrix())
        elif cmd in ("notes",):
            console.print("\n[bold orange1]Generating AI Market Notes...[/]\n")
            try:
                from ai_analyzer import generate_market_notes
                result = generate_market_notes(use_cache=True)
                console.print(Panel(Markdown(result), title="[bold]AI ANALYST MARKET NOTES[/]", border_style="bright_red", box=box.DOUBLE))
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
        elif cmd.startswith("gpu "):
            gpu_name = cmd[4:].upper().replace(" ", "-")
            # Try to match
            matched = None
            for gid in GPU_SPECS:
                if gpu_name in gid.upper():
                    matched = gid
                    break
            if matched:
                console.print(render_provider_comparison(matched))
                console.print("\n[bold magenta]Generating AI deep-dive...[/]\n")
                try:
                    from ai_analyzer import analyze_specific_gpu
                    result = analyze_specific_gpu(matched)
                    console.print(Panel(Markdown(result), title=f"[bold]AI Analysis: {matched}[/]", border_style="magenta"))
                except Exception as e:
                    console.print(f"[red]Error: {e}[/]")
            else:
                console.print(f"[red]GPU not found: {gpu_name}[/]")
        elif cmd in ("web", "server"):
            console.print("[bold cyan]Starting web server...[/]")
            os.system(f"python3 {os.path.join(os.path.dirname(__file__), 'server.py')} &")
            console.print("[green]Web dashboard available at http://localhost:8050[/]")
        elif cmd in ("help", "?"):
            console.print(Panel(
                "[bold]Available Commands:[/]\n\n"
                "  [cyan]r, refresh[/]      — Refresh dashboard\n"
                "  [cyan]a, ai[/]           — Full AI analysis\n"
                "  [cyan]t, trends[/]       — Historical price trends\n"
                "  [cyan]g, regional[/]     — Regional market data\n"
                "  [cyan]m, market[/]       — Market indicators\n"
                "  [cyan]w, workload[/]     — Workload recommendations\n"
                "  [cyan]c, cost[/]         — Cost calculator\n"
                "  [cyan]s, spot[/]         — Spot market live data\n"
                "  [cyan]i, inference[/]    — Inference economics\n"
                "  [cyan]tco[/]             — TCO breakdown analysis\n"
                "  [cyan]n, news[/]         — Latest news feed\n"
                "  [cyan]notes[/]           — AI analyst market notes\n"
                "  [cyan]util[/]            — GPU utilization metrics\n"
                "  [cyan]res[/]             — Reservation analytics\n"
                "  [cyan]fc, forecast[/]    — Price forecasts\n"
                "  [cyan]comp[/]            — Competitive moat tracker\n"
                "  [cyan]sus, green[/]      — Sustainability index\n"
                "  [cyan]supply, risk[/]    — Supply chain risk\n"
                "  [cyan]fit, model[/]      — Model-to-hardware fit\n"
                "  [cyan]gpu <name>[/]      — GPU deep-dive (e.g. gpu h100)\n"
                "  [cyan]1-8[/]             — Quick GPU comparison\n"
                "  [cyan]web[/]             — Start web dashboard\n"
                "  [cyan]q, quit[/]         — Exit\n",
                title="[bold]HELP[/]",
                border_style="cyan",
            ))
        else:
            console.print("[dim]Unknown command. Type 'help' for options.[/]")


if __name__ == "__main__":
    if "--static" in sys.argv:
        print_full_dashboard()
    else:
        run_interactive()
