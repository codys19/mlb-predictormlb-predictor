"""
fetch_pitchers.py
-----------------
Downloads starting pitcher statistics from Baseball Reference
and saves them as a CSV.

HOW TO RUN:
    python data/fetch_pitchers.py

OUTPUT:
    raw/pitcher_stats.csv
"""

import os
import time
import pandas as pd
from pybaseball import pitching_stats_bref

# ── Settings ──────────────────────────────────────────────────────────────────

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Only keep pitchers who started at least this many games
MIN_GAMES_STARTED = 5

OUTPUT_PATH = "raw/pitcher_stats.csv"


# ── Download pitcher stats ─────────────────────────────────────────────────────

def fetch_all_pitchers():
    os.makedirs("raw", exist_ok=True)

    all_pitchers = []

    for season in SEASONS:
        print(f"\n📅 Downloading {season} pitcher stats...")

        try:
            df = pitching_stats_bref(season)

            df["season"] = season

            # Filter to starting pitchers only
            if "GS" in df.columns:
                df = df[df["GS"] >= MIN_GAMES_STARTED].copy()

            # Add recency weight — 2026 slightly lower due to partial season
            weight_map = {2021: 0.6, 2022: 0.7, 2023: 0.8, 2024: 1.0, 2025: 1.0, 2026: 0.75}
            df["recency_weight"] = weight_map.get(season, 0.6)

            all_pitchers.append(df)
            print(f"  ✅ Found {len(df)} starting pitchers")

        except Exception as e:
            print(f"  ⚠️  Could not download {season}: {e}")

        time.sleep(2)

    if not all_pitchers:
        print("❌ No pitcher data downloaded.")
        return

    final_df = pd.concat(all_pitchers, ignore_index=True)

    # ── Clean up column names ──────────────────────────────────────────────────
    final_df.columns = (
        final_df.columns
        .str.lower()
        .str.replace("/", "_per_", regex=False)
        .str.replace("%", "_pct", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("+", "_plus", regex=False)
    )

    sort_cols = [c for c in ["season", "era"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Done! Saved {len(final_df):,} pitcher records to: {OUTPUT_PATH}")
    print(f"\nPitchers per season:")
    print(final_df.groupby("season").size().to_string())
    print(f"\nFirst 5 rows preview:")
    print(final_df.head())


# ── Run it ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fetch_all_pitchers()