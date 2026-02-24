"""
AI-Powered GPU Market Analyzer
Uses LLM API to generate deep market insights, trend analysis, and recommendations.
"""

import json
import os
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime

from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from gpu_data import (
    generate_market_summary,
    get_price_comparison_matrix,
    get_regional_summary,
    get_workload_recommendations,
    get_utilization_summary,
    get_price_forecasts,
    get_competitive_landscape,
    get_sustainability_summary,
    get_supply_chain_summary,
    HISTORICAL_PRICING,
    MARKET_INDICATORS,
    GPU_SPECS,
    CLOUD_PRICING,
    UTILIZATION_METRICS,
    PRICE_FORECASTS,
    COMPETITIVE_MOAT,
    SUPPLY_CHAIN_RISK,
)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "ai_analysis_cache.json")
NEWS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "news_cache.json")


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
    """Call the LLM API for analysis."""
    import requests as _req
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    try:
        resp = _req.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # Support both standard content and reasoning models
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if content:
            return content
        elif reasoning:
            return reasoning
        else:
            return str(msg)
    except _req.exceptions.HTTPError as e:
        body = e.response.text[:300] if e.response else str(e)
        return f"[LLM API Error {e.response.status_code if e.response else '?'}]: {body}"
    except Exception as e:
        return f"[LLM Error]: {str(e)}"


def _load_cache() -> dict:
    """Load cached analysis results."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < 3600:
                return cache
        except Exception:
            pass
    return {}


def _save_cache(data: dict):
    """Save analysis results to cache."""
    data["timestamp"] = time.time()
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _build_market_context() -> str:
    """Build comprehensive market context for the LLM."""
    summary = generate_market_summary()
    regional = get_regional_summary()

    context = "## CURRENT GPU CLOUD PRICING ($/hr per GPU, on-demand)\n\n"
    for row in summary["comparison_matrix"][:12]:
        context += (
            f"- {row['name']}: ${row['cheapest_price']:.2f}/hr (cheapest: {row['cheapest_provider']}) "
            f"to ${row['most_expensive']:.2f}/hr | {row['num_providers']} providers | "
            f"MoM change: {row['monthly_change_pct']:+.1f}% | "
            f"TFLOPS/$: {row['flops_per_dollar']}\n"
        )

    context += "\n## HISTORICAL PRICE TRENDS\n\n"
    for gpu_id in ["H100-SXM", "H200", "B200", "A100-80GB", "MI300X", "RTX-4090"]:
        if gpu_id in HISTORICAL_PRICING:
            prices = HISTORICAL_PRICING[gpu_id]
            periods = sorted(prices.keys())
            context += f"- {gpu_id}: "
            for p in periods[-4:]:
                context += f"{p}=${prices[p]['avg']:.2f} "
            context += "\n"

    context += "\n## MARKET INDICATORS\n\n"
    mi = MARKET_INDICATORS
    context += f"- NVDA: ${mi['nvidia_stock']['current']} (YTD: {mi['nvidia_stock']['ytd_change']:+.1f}%)\n"
    context += f"- AMD: ${mi['amd_stock']['current']} (YTD: {mi['amd_stock']['ytd_change']:+.1f}%)\n"
    context += f"- GPU Market Size: $52.4B (2023) → $95.8B (2025) → $128.5B est (2026)\n"
    context += f"- AI CapEx: $55B (2023) → $150B est (2025) → $210B est (2026)\n"
    context += f"- H100 Lead Time: down from 52 weeks (2023-Q1) to ~1 week (2026-Q1)\n"

    context += "\n## REGIONAL MARKET\n\n"
    for region, data in regional.items():
        context += (
            f"- {region}: {data['market_share_pct']}% share, {data['yoy_growth_pct']:+.1f}% YoY, "
            f"demand index: {data['gpu_demand_index']}/100, premium: +{data['avg_price_premium_pct']}%\n"
        )

    context += f"\n## DATE: {datetime.now().strftime('%Y-%m-%d')}\n"
    return context


def analyze_market_trends(use_cache: bool = True) -> str:
    """Generate comprehensive market trend analysis."""
    if use_cache:
        cache = _load_cache()
        if "market_trends" in cache:
            return cache["market_trends"]

    system_prompt = (
        "You are a senior GPU market analyst at a Bloomberg-tier financial intelligence firm. "
        "You provide institutional-grade analysis of the AI GPU compute market. "
        "Be specific with numbers, cite data points, identify key inflection points, "
        "and provide actionable intelligence. Use bullet points for clarity. "
        "Format sections with headers using ##."
    )

    context = _build_market_context()
    user_prompt = f"""Based on this real-time GPU market data, provide a comprehensive market trend analysis:

{context}

Provide analysis covering:
1. **Price Trajectory Analysis**: Current pricing trends across GPU tiers, rate of price decline, inflection points
2. **Supply/Demand Dynamics**: Lead time improvements, availability shifts, supply chain signals
3. **Generational Transition Impact**: How Blackwell (B200/GB200) is affecting Hopper/Ampere pricing
4. **Competitive Landscape**: How AMD MI300X/MI325X is pressuring NVIDIA pricing
5. **Key Signals to Watch**: What data points should procurement teams monitor

Be concise but data-dense. This is for a financial terminal dashboard."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=3000)

    cache = _load_cache()
    cache["market_trends"] = result
    _save_cache(cache)
    return result


def analyze_regional_opportunities(use_cache: bool = True) -> str:
    """Generate regional opportunity analysis."""
    if use_cache:
        cache = _load_cache()
        if "regional_analysis" in cache:
            return cache["regional_analysis"]

    system_prompt = (
        "You are a GPU infrastructure strategist advising on global compute deployment. "
        "Provide data-driven regional analysis with specific recommendations."
    )

    context = _build_market_context()
    user_prompt = f"""Analyze regional GPU market opportunities:

{context}

Cover:
1. **Regional Price Arbitrage**: Where are the best deals by region?
2. **Emerging Markets**: Which regions are growing fastest and why?
3. **Regulatory Impact**: How do regulations (EU AI Act, China export controls) affect pricing?
4. **Energy Cost Impact**: How energy costs affect total cost of ownership by region
5. **Strategic Recommendations**: Where to deploy for cost optimization vs. latency vs. compliance

Be specific with numbers and recommendations."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=2500)

    cache = _load_cache()
    cache["regional_analysis"] = result
    _save_cache(cache)
    return result


def analyze_investment_outlook(use_cache: bool = True) -> str:
    """Generate investment and procurement outlook."""
    if use_cache:
        cache = _load_cache()
        if "investment_outlook" in cache:
            return cache["investment_outlook"]

    system_prompt = (
        "You are a senior technology investment analyst specializing in AI infrastructure. "
        "Provide actionable investment intelligence on the GPU compute market."
    )

    context = _build_market_context()
    user_prompt = f"""Based on this GPU market data, provide investment outlook:

{context}

Cover:
1. **Buy vs. Rent Decision Framework**: When to commit to reserved instances vs. spot vs. on-demand
2. **Price Floor Analysis**: Where are GPU prices likely to bottom out for each tier?
3. **Technology Transition Timing**: When to migrate from A100→H100→B200
4. **Provider Risk Assessment**: Which providers offer best long-term value/stability?
5. **Budget Planning**: Cost projections for next 4 quarters by workload type
6. **Alpha Opportunities**: Underpriced capacity, arbitrage opportunities, or upcoming disruptions

This is for institutional decision-makers. Be specific and actionable."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=3000)

    cache = _load_cache()
    cache["investment_outlook"] = result
    _save_cache(cache)
    return result


def get_quick_summary(use_cache: bool = True) -> str:
    """Generate a quick 1-paragraph executive summary."""
    if use_cache:
        cache = _load_cache()
        if "quick_summary" in cache:
            return cache["quick_summary"]

    system_prompt = "You are a GPU market analyst. Be extremely concise."
    context = _build_market_context()
    user_prompt = f"""In 3-4 sentences, summarize the current AI GPU market state:

{context}

Focus on: current pricing levels, direction of travel, key risks, and one actionable insight."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=500)

    cache = _load_cache()
    cache["quick_summary"] = result
    _save_cache(cache)
    return result


def analyze_specific_gpu(gpu_id: str) -> str:
    """Deep-dive analysis on a specific GPU."""
    if gpu_id not in GPU_SPECS:
        return f"Unknown GPU: {gpu_id}"

    spec = GPU_SPECS[gpu_id]
    from gpu_data import get_cheapest_by_gpu
    providers = get_cheapest_by_gpu(gpu_id)
    trends = HISTORICAL_PRICING.get(gpu_id, {})

    system_prompt = "You are a GPU market analyst. Provide a focused analysis."
    user_prompt = f"""Deep-dive on {spec['name']}:

Specs: {json.dumps(spec, indent=2)}
Providers (by price): {json.dumps(providers[:5], indent=2)}
Historical pricing: {json.dumps(trends, indent=2)}

Provide:
1. Current market positioning and value proposition
2. Price trend analysis and forecast
3. Best use cases at current pricing
4. Buy/wait/skip recommendation
5. Best provider recommendation with reasoning"""

    return _call_llm(system_prompt, user_prompt, max_tokens=2000)


def generate_market_notes(use_cache: bool = True) -> str:
    """Generate AI analyst market notes with predictions."""
    if use_cache:
        cache = _load_cache()
        if "market_notes" in cache:
            return cache["market_notes"]

    from gpu_data import NEWS_FEED, SPOT_MARKET, INFERENCE_BENCHMARKS

    system_prompt = (
        "You are a senior GPU market strategist writing analyst notes for an institutional trading desk. "
        "Write in the style of Bloomberg Intelligence or Goldman Sachs research notes. "
        "Be specific, data-driven, and include actionable predictions with timeframes. "
        "Use bullet points. Include BUY/SELL/HOLD signals where appropriate."
    )

    news_ctx = "\n".join(f"- [{n['date']}] {n['source']}: {n['headline']} (sentiment: {n['sentiment']}, impact: {n['impact']})" for n in NEWS_FEED[:10])

    spot_ctx = "\n".join(f"- {gpu}: bid=${s['bid']:.2f} ask=${s['ask']:.2f} spread={s['spread_pct']:.1f}% vol={s['24h_volume_gpu_hrs']:,} avail={s['available_gpus']:,} volatility={s['volatility_30d']:.1f}%"
                         for gpu, s in SPOT_MARKET.items())

    context = _build_market_context()

    user_prompt = f"""Write institutional analyst notes covering:

## MARKET DATA
{context}

## RECENT NEWS
{news_ctx}

## SPOT MARKET (LIVE)
{spot_ctx}

Generate:
1. **MARKET OUTLOOK (30/60/90 day)** — Specific price predictions for H100, B200, A100, MI300X with confidence levels
2. **KEY TRADE IDEAS** — What to buy/sell/hold right now. Include specific providers and commitment types (spot vs reserved)
3. **RISK SIGNALS** — What could disrupt the current trend (upside and downside risks)
4. **SECTOR ROTATION CALL** — Which GPU tiers are rotating (who's moving from A100→H100→B200)
5. **ALPHA OPPORTUNITIES** — Underpriced capacity, arbitrage between providers, timing opportunities

Write as 5-8 concise bullet-point notes, each with a clear actionable signal. Include [BUY], [SELL], [HOLD], or [WATCH] tags."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=4000)

    cache = _load_cache()
    cache["market_notes"] = result
    _save_cache(cache)
    return result


def analyze_efficiency_optimization(use_cache: bool = True) -> str:
    """Generate efficiency and utilization optimization analysis."""
    if use_cache:
        cache = _load_cache()
        if "efficiency_optimization" in cache:
            return cache["efficiency_optimization"]

    system_prompt = (
        "You are a GPU infrastructure efficiency consultant. "
        "Provide data-driven recommendations for optimizing GPU utilization and reducing waste. "
        "Be specific with numbers and actionable recommendations."
    )

    util_summary = get_utilization_summary()
    util_ctx = ""
    for gpu_id, data in util_summary.items():
        util_ctx += f"- {gpu_id}: avg utilization {data['avg_utilization']:.1f}%, avg efficiency {data['avg_efficiency']:.1f}/100\n"
        best = max(data["providers"].items(), key=lambda x: x[1]["efficiency_score"])
        worst = min(data["providers"].items(), key=lambda x: x[1]["efficiency_score"])
        util_ctx += f"  Best: {best[0]} ({best[1]['efficiency_score']}/100), Worst: {worst[0]} ({worst[1]['efficiency_score']}/100)\n"

    context = _build_market_context()
    user_prompt = f"""Analyze GPU utilization efficiency and provide optimization recommendations:

## MARKET CONTEXT
{context}

## UTILIZATION DATA
{util_ctx}

Provide:
1. **Utilization Gap Analysis**: Where is compute being wasted? Which GPU/provider combos have the worst utilization?
2. **Right-sizing Recommendations**: Which workloads should migrate to different GPU tiers?
3. **Scheduling Optimization**: How to improve off-peak utilization and reduce idle costs
4. **Provider Efficiency Ranking**: Which providers deliver best utilization rates and why?
5. **Cost Savings Estimate**: Quantify potential savings from optimization

Be specific and actionable. Include dollar amounts where possible."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=3000)

    cache = _load_cache()
    cache["efficiency_optimization"] = result
    _save_cache(cache)
    return result


def analyze_price_forecasts(use_cache: bool = True) -> str:
    """Generate price forecasting analysis."""
    if use_cache:
        cache = _load_cache()
        if "price_forecasts" in cache:
            return cache["price_forecasts"]

    system_prompt = (
        "You are a quantitative analyst specializing in GPU compute pricing models. "
        "Provide data-driven price forecasts with confidence intervals. "
        "Think like a commodity trader analyzing supply/demand dynamics."
    )

    forecasts = get_price_forecasts()
    forecast_ctx = ""
    for gpu_id, f in forecasts.items():
        forecast_ctx += (
            f"- {gpu_id}: current=${f['current_avg']:.2f}, elasticity={f['elasticity_coefficient']:.2f}, "
            f"3mo=${f['forecast_3mo']['mid']:.2f} (conf:{f['forecast_3mo']['confidence']:.0%}), "
            f"12mo=${f['forecast_12mo']['mid']:.2f} (conf:{f['forecast_12mo']['confidence']:.0%}), "
            f"floor=${f['price_floor']:.2f}, supply={f['supply_signal']}, demand={f['demand_signal']}, "
            f"pattern={f['pattern_match']}\n"
        )

    competitive = get_competitive_landscape()
    comp_ctx = ""
    for vendor, data in competitive.items():
        comp_ctx += (
            f"- {vendor}: moat={data['moat_strength_score']}/100, share={data['market_share_pct']}%, "
            f"perf={data['performance_score']}/100, price/perf={data['price_performance_ratio']}/100, "
            f"trend={data['market_share_trend']}\n"
        )

    context = _build_market_context()
    user_prompt = f"""Generate comprehensive GPU price forecasts and competitive analysis:

## MARKET CONTEXT
{context}

## PRICE FORECAST MODELS
{forecast_ctx}

## COMPETITIVE LANDSCAPE
{comp_ctx}

Provide:
1. **30/60/90-Day Price Outlook**: Specific price targets for each major GPU with confidence levels
2. **Elasticity Analysis**: How demand changes affect pricing — which GPUs are most/least elastic?
3. **Competitive Displacement**: How AMD/Google/AWS custom silicon affects NVIDIA pricing
4. **Generational Crossover Points**: When does it become cheaper to use B200 vs H100 per FLOP?
5. **Arbitrage Opportunities**: Price inefficiencies between providers or commitment types
6. **Risk Scenarios**: Bull/bear cases for GPU pricing over next 12 months

Include specific numbers and actionable trade recommendations."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=3500)

    cache = _load_cache()
    cache["price_forecasts"] = result
    _save_cache(cache)
    return result


def analyze_sustainability_risk(use_cache: bool = True) -> str:
    """Generate sustainability and supply chain risk analysis."""
    if use_cache:
        cache = _load_cache()
        if "sustainability_risk" in cache:
            return cache["sustainability_risk"]

    system_prompt = (
        "You are an ESG analyst and supply chain risk specialist for AI infrastructure. "
        "Provide institutional-grade analysis of sustainability metrics and geopolitical risks. "
        "Include specific data points and actionable recommendations."
    )

    sustainability = get_sustainability_summary()
    sus_ctx = ""
    for provider, data in sustainability["providers"].items():
        sus_ctx += (
            f"- {provider}: avg sustainability={data['avg_sustainability_score']}/100, "
            f"green energy={data['avg_green_energy_pct']}%, PUE={data['avg_pue']}, "
            f"best={data['best_region']}, worst={data['worst_region']}\n"
        )

    supply = get_supply_chain_summary()
    supply_ctx = ""
    for vendor, data in supply["vendors"].items():
        supply_ctx += (
            f"- {vendor}: risk_score={data['supply_risk_score']}/100, TSMC_dep={data['tsmc_dependency_pct']}%, "
            f"lead_time={data['lead_time_weeks']}wk, trend={data['risk_trend']}, "
            f"bottlenecks={', '.join(data['bottlenecks'][:2])}\n"
        )

    export_ctx = "\n".join(
        f"- [{e['date']}] {e['regulation']}: {e['description']} (impact: {e['impact']})"
        for e in supply["export_controls"][-5:]
    )

    user_prompt = f"""Analyze sustainability and supply chain risks for GPU compute:

## SUSTAINABILITY METRICS
{sus_ctx}

## SUPPLY CHAIN RISK
{supply_ctx}

## RECENT EXPORT CONTROLS
{export_ctx}

Provide:
1. **ESG Ranking**: Rank providers by sustainability — where should ESG-conscious buyers deploy?
2. **Carbon Cost Analysis**: Total carbon footprint per GPU-hour by region — best/worst combos
3. **Supply Chain Vulnerabilities**: Single points of failure, TSMC concentration risk
4. **Geopolitical Risk Map**: How export controls and trade tensions affect availability/pricing
5. **Regulatory Outlook**: What's coming next in AI regulation that affects GPU procurement?
6. **Resilience Recommendations**: How to build a supply-chain-resilient GPU strategy

Be specific with data and actionable recommendations."""

    result = _call_llm(system_prompt, user_prompt, max_tokens=3000)

    cache = _load_cache()
    cache["sustainability_risk"] = result
    _save_cache(cache)
    return result


def generate_daily_news() -> list:
    """Generate daily AI news headlines using LLM, with 24-hour (date-based) cache."""
    from gpu_data import NEWS_FEED

    today = datetime.now().strftime("%Y-%m-%d")

    # Check separate news cache (daily TTL based on date string)
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, "r") as f:
                cache = json.load(f)
            if cache.get("date") == today and isinstance(cache.get("news"), list) and len(cache["news"]) > 0:
                return cache["news"]
        except Exception:
            pass

    # Build LLM prompt
    context = _build_market_context()
    system_prompt = (
        "You are a financial news wire editor covering the AI GPU compute market. "
        "Generate realistic news headlines that could appear on Bloomberg, CNBC, Reuters, "
        "TechCrunch, The Register, TrendForce, or other major outlets. "
        "Headlines should be specific, data-driven, and reflect current market dynamics. "
        "Return ONLY valid JSON — no markdown fences, no commentary."
    )

    user_prompt = f"""Based on this GPU market data, generate exactly 15 realistic news headlines for today ({today}).

{context}

Return a JSON array of exactly 15 objects with this schema:
[
  {{
    "date": "{today}",
    "source": "<realistic news source name>",
    "headline": "<specific, data-driven headline>",
    "category": "<one of: pricing, demand, supply, policy, earnings, expansion>",
    "sentiment": "<one of: bullish, bearish, neutral, positive, negative>",
    "impact": "<one of: high, medium, low>"
  }}
]

Requirements:
- Use varied sources: Bloomberg, CNBC, Reuters, TechCrunch, The Register, TrendForce, Motley Fool, Wolf Street, MIT Tech Review, AI Business 2.0, Federal Register, TechSpot
- Cover all categories: at least 2 pricing, 2 demand, 2 supply, 1 policy, 1 earnings, 1 expansion
- Mix sentiments: not all bullish or bearish
- Spread dates: some today, some 1-5 days ago (use {today} as the most recent)
- Make headlines specific with real company names, dollar amounts, and percentages
- Return ONLY the JSON array, no other text"""

    try:
        raw = _call_llm(system_prompt, user_prompt, max_tokens=3000)

        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        news = json.loads(text)

        # Validate shape
        if not isinstance(news, list) or len(news) < 5:
            raise ValueError("Expected list of at least 5 news items")
        required_keys = {"date", "source", "headline", "category", "sentiment", "impact"}
        for item in news:
            if not isinstance(item, dict) or not required_keys.issubset(item.keys()):
                raise ValueError(f"News item missing required keys: {item}")

        # Save to cache
        os.makedirs(os.path.dirname(NEWS_CACHE_FILE), exist_ok=True)
        with open(NEWS_CACHE_FILE, "w") as f:
            json.dump({"date": today, "news": news, "timestamp": time.time()}, f, indent=2)

        return news

    except Exception:
        # Fallback: return static NEWS_FEED with dates shifted to today
        from datetime import timedelta
        if NEWS_FEED:
            most_recent = max(n["date"] for n in NEWS_FEED)
            most_recent_dt = datetime.strptime(most_recent, "%Y-%m-%d")
            today_dt = datetime.strptime(today, "%Y-%m-%d")
            gap = (today_dt - most_recent_dt).days
            shifted = []
            for n in NEWS_FEED:
                item = dict(n)
                orig_dt = datetime.strptime(item["date"], "%Y-%m-%d")
                item["date"] = (orig_dt + timedelta(days=gap)).strftime("%Y-%m-%d")
                shifted.append(item)
            return shifted
        return NEWS_FEED


def get_all_analyses(use_cache: bool = True) -> dict:
    """Get all analyses in one call."""
    return {
        "quick_summary": get_quick_summary(use_cache),
        "market_trends": analyze_market_trends(use_cache),
        "regional_opportunities": analyze_regional_opportunities(use_cache),
        "investment_outlook": analyze_investment_outlook(use_cache),
        "market_notes": generate_market_notes(use_cache),
        "efficiency_optimization": analyze_efficiency_optimization(use_cache),
        "price_forecasts": analyze_price_forecasts(use_cache),
        "sustainability_risk": analyze_sustainability_risk(use_cache),
        "generated_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("AI GPU MARKET ANALYZER - Generating Analysis...")
    print("=" * 80)

    print("\n📊 Quick Summary:")
    print("-" * 40)
    print(get_quick_summary(use_cache=False))

    print("\n📈 Market Trends:")
    print("-" * 40)
    print(analyze_market_trends(use_cache=False))

    print("\n🌍 Regional Opportunities:")
    print("-" * 40)
    print(analyze_regional_opportunities(use_cache=False))

    print("\n💰 Investment Outlook:")
    print("-" * 40)
    print(analyze_investment_outlook(use_cache=False))
