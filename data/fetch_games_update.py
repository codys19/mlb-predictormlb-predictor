"""
fetch_games_update.py
---------------------
Incremental updater — refreshes ONLY the current season in game_results.csv.
Runs in ~2 minutes vs ~15 minutes for the full fetch_games.py.

Designed to be run weekly (via GitHub Actions) to keep rolling stats
current mid-season. Historical seasons (2021-2025) are left untouched.

HOW TO RUN:
    python data/fetch_games_update.py

OUTPUT:
    raw/game_results.csv  (current season rows replaced, history preserved)
"""

import os
import pandas as pd
from datetime import datetime
from pybaseball import schedule_and_record

CURRENT_SEASON = datetime.now().year
OUTPUT_PATH    = "raw/game_results.csv"

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

WEIGHT_MAP = {2021: 0.6, 2022: 0.7, 2023: 0.8, 2024: 1.0, 2025: 1.0, 2026: 0.75}

KEEP_COLS = {
    "Date":          "date",
    "team":          "team",
    "Home_Away":     "home_away",
    "Opp":           "opponent",
    "R":             "runs_scored",
    "RA":            "runs_allowed",
    "W/L":           "result",
    "home_team_won": "home_team_won",
    "Attendance":    "attendance",
}


def label_winner(df, team):
    df = df.copy()
    df["team"] = team
    df["home_team_won"] = None
    for i, row in df.iterrows():
        result   = str(row.get("W/L", "")).strip().upper()
        location = str(row.get("Home_Away", "")).strip().upper()
        if result not in ("W", "L"):
            continue
        if location == "HOME":
            df.at[i, "home_team_won"] = 1 if result == "W" else 0
        else:
            df.at[i, "home_team_won"] = 1 if result == "L" else 0
    return df


def fetch_current_season():
    print(f"⬇️  Fetching {CURRENT_SEASON} season for all 30 teams...")
    frames = []; skipped = 0

    for team in TEAMS:
        try:
            df = schedule_and_record(CURRENT_SEASON, team)
            df = label_winner(df, team)
            frames.append(df)
            print(f"  ✅ {team}: {len(df)} games")
        except Exception as e:
            print(f"  ⚠️  {team}: skipped ({e})")
            skipped += 1

    if not frames:
        print("❌ No data fetched — pybaseball may be down or season hasn't started.")
        return None

    df = pd.concat(frames, ignore_index=True)

    # Clean columns
    existing = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["result"])
    df = df[df["result"].str.upper().str[:1].isin(["W", "L"])]
    df["season"] = CURRENT_SEASON
    df["recency_weight"] = WEIGHT_MAP.get(CURRENT_SEASON, 0.75)

    print(f"\n  → {len(df):,} completed games for {CURRENT_SEASON} ({skipped} teams skipped)")
    return df


def merge_into_existing(new_df):
    if not os.path.exists(OUTPUT_PATH):
        print(f"⚠️  {OUTPUT_PATH} not found — writing fresh file")
        new_df.sort_values("date").reset_index(drop=True).to_csv(OUTPUT_PATH, index=False)
        return new_df

    existing = pd.read_csv(OUTPUT_PATH)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")

    before = len(existing)
    # Drop old current-season rows, replace with fresh fetch
    historical = existing[existing["season"] != CURRENT_SEASON]
    combined   = pd.concat([historical, new_df], ignore_index=True)
    combined   = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    added = len(combined) - before
    print(f"\n📊 Merge summary:")
    print(f"  Historical rows kept:  {len(historical):,}  (seasons {CURRENT_SEASON-1} and earlier)")
    print(f"  Current season rows:   {len(new_df):,}  ({CURRENT_SEASON})")
    print(f"  Net change:            {added:+,} rows")
    print(f"  Total rows now:        {len(combined):,}")
    print(f"\nGames per season:")
    print(combined.groupby("season").size().to_string())
    return combined


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    print(f"🔄 Weekly game results refresh — updating {CURRENT_SEASON} data\n")

    new_df = fetch_current_season()
    if new_df is not None:
        merge_into_existing(new_df)
        print(f"\n✅ Done! {OUTPUT_PATH} is up to date.")
        print(f"   Next: model will use fresh rolling stats at tomorrow's 7 AM run.")
    else:
        print("\n❌ Update failed — game_results.csv unchanged.")
        exit(1)
