"""
fetch_games.py
--------------
Downloads historical MLB game results (who won, runs scored, etc.)
and saves them as a CSV file you can use to train your model.

HOW TO RUN:
    python data/fetch_games.py

OUTPUT:
    raw/game_results.csv
"""

import os
import pandas as pd
from pybaseball import schedule_and_record

# ── Settings ──────────────────────────────────────────────────────────────────

# Which seasons to download
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]

# Where to save the file
OUTPUT_PATH = "raw/game_results.csv"

# ── Team list (all 30 MLB teams) ───────────────────────────────────────────────
TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

# ── Helper: label each game with Home/Away win ─────────────────────────────────

def label_winner(df, team):
    df = df.copy()
    df["team"] = team
    df["home_team_won"] = None

    for i, row in df.iterrows():
        result = str(row.get("W/L", "")).strip().upper()
        location = str(row.get("Home_Away", "")).strip().upper()

        if result not in ("W", "L"):
            continue

        if location == "HOME":
            df.at[i, "home_team_won"] = 1 if result == "W" else 0
        else:
            df.at[i, "home_team_won"] = 1 if result == "L" else 0

    return df


# ── Main download loop ─────────────────────────────────────────────────────────

def fetch_all_games():
    os.makedirs("raw", exist_ok=True)

    all_games = []

    for season in SEASONS:
        print(f"\n📅 Downloading {season} season...")

        season_frames = []

        for team in TEAMS:
            try:
                df = schedule_and_record(season, team)
                df = label_winner(df, team)
                season_frames.append(df)
                print(f"  ✅ {team}: {len(df)} games")
            except Exception as e:
                print(f"  ⚠️  {team}: skipped ({e})")

        if season_frames:
            season_df = pd.concat(season_frames, ignore_index=True)
            all_games.append(season_df)

    if not all_games:
        print("❌ No data downloaded.")
        return

    final_df = pd.concat(all_games, ignore_index=True)

    # ── Clean up columns ───────────────────────────────────────────────────────
    keep_cols = {
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

    existing = {k: v for k, v in keep_cols.items() if k in final_df.columns}
    final_df = final_df[list(existing.keys())].rename(columns=existing)

    final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")

    # Drop postponed / future games
    final_df = final_df.dropna(subset=["result"])
    final_df = final_df[final_df["result"].str.upper().str[:1].isin(["W", "L"])]

    # Add season year
    final_df["season"] = final_df["date"].dt.year

    # Recency weights — 2026 slightly lower due to small sample size so far
    weight_map = {2021: 0.6, 2022: 0.7, 2023: 0.8, 2024: 1.0, 2025: 1.0, 2026: 0.75}
    final_df["recency_weight"] = final_df["season"].map(weight_map).fillna(0.6)

    final_df = final_df.sort_values("date").reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Done! Saved {len(final_df):,} game records to: {OUTPUT_PATH}")
    print(f"\nGames per season:")
    print(final_df.groupby("season").size().to_string())


# ── Run it ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fetch_all_games()