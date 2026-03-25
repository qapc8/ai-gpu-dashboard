#!/usr/bin/env python3
"""
Daily Lightweight Updater
==========================
Fetches only news RSS and stock prices, then merges into data.json.
Designed to run daily via GitHub Actions (fast, no heavy API calls).

Usage:
    python scripts/update_daily.py

Dependencies:
    pip install requests feedparser
"""

import json
import os
import sys
from datetime import datetime, timezone

# Re-use helpers from the main update script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from update_data import (
    DATA_JSON,
    EMBEDDED_DATA_JSON,
    AI_ANALYSIS_JSON,
    EMBEDDED_AI_JSON,
    load_existing,
    save_json,
    log_info,
    log_ok,
    log_fail,
    _results,
    fetch_stock_price,
    merge_stocks_into_data,
    fetch_news_rss,
    merge_news_into_data,
)


def main():
    print("=" * 60)
    print("  GPU Market Daily Update (News + Stocks)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Load existing data
    data = load_existing(DATA_JSON, EMBEDDED_DATA_JSON)
    if not data:
        log_info("No existing data.json found. Cannot run daily update without base data.")
        return 1

    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # ---- Stock Prices ----
    print("[1/2] Stock Prices")
    print("-" * 40)
    stocks = []
    for ticker in ["NVDA", "AMD"]:
        try:
            stock = fetch_stock_price(ticker)
            stocks.append(stock)
        except Exception as exc:
            log_fail(f"Stock/{ticker}", str(exc))
    data = merge_stocks_into_data(data, stocks)
    print()

    # ---- News ----
    print("[2/2] News Headlines")
    print("-" * 40)
    try:
        articles = fetch_news_rss()
        data = merge_news_into_data(data, articles)
    except Exception as exc:
        log_fail("Google News RSS", str(exc))
    print()

    # ---- Save ----
    print("Saving files...")
    print("-" * 40)
    save_json(DATA_JSON, data)

    # ---- Summary ----
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if _results["success"]:
        print(f"  Updated ({len(_results['success'])}):")
        for s in _results["success"]:
            print(f"    + {s}")
    if _results["failed"]:
        print(f"  Failed ({len(_results['failed'])}):")
        for f in _results["failed"]:
            print(f"    - {f}")
    if not _results["failed"]:
        print("  All data sources fetched successfully.")
    print()
    print(f"  data.json: {DATA_JSON}")
    print("=" * 60)

    return 0 if not _results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
