"""
build_features.py
-----------------
Builds the model training dataset. Now includes individual starter stats
instead of just team-average pitcher stats.

HOW TO RUN:
    python data/build_features.py

OUTPUT:
    raw/training_data.csv
"""

import os
import pandas as pd
import numpy as np

GAMES_PATH       = "raw/game_results.csv"
PITCHER_PATH     = "raw/pitcher_stats.csv"
STARTER_PATH     = "raw/starter_logs.csv"
TEAM_POOL_PATH   = "raw/team_starter_pool.csv"
OUTPUT_PATH      = "raw/training_data.csv"

ROLLING_WINDOW   = 15
SEASONS_IN_ORDER = [2021, 2022, 2023, 2024, 2025, 2026]
GAMES_PER_SEASON = 162

TEAM_NAME_MAP = {
    "Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Boston":"BOS",
    "Chicago":"CHC","Cincinnati":"CIN","Cleveland":"CLE","Colorado":"COL",
    "Detroit":"DET","Houston":"HOU","Kansas City":"KCR","Los Angeles":"LAD",
    "Miami":"MIA","Milwaukee":"MIL","Minnesota":"MIN","New York":"NYY",
    "Oakland":"OAK","Athletics":"ATH","Philadelphia":"PHI","Pittsburgh":"PIT",
    "San Diego":"SDP","Seattle":"SEA","San Francisco":"SFG","St. Louis":"STL",
    "Tampa Bay":"TBR","Texas":"TEX","Toronto":"TOR","Washington":"WSN",
}

def normalize_team(raw):
    parts = [p.strip() for p in str(raw).split(",")]
    return TEAM_NAME_MAP.get(parts[-1], parts[-1].upper()[:3])


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data():
    print("📂 Loading raw data...")

    games    = pd.read_csv(GAMES_PATH)
    pitchers = pd.read_csv(PITCHER_PATH)

    # Load starter logs if available
    if os.path.exists(STARTER_PATH):
        starters  = pd.read_csv(STARTER_PATH)
        team_pool = pd.read_csv(TEAM_POOL_PATH) if os.path.exists(TEAM_POOL_PATH) else None
        print(f"  ✅ Starter logs loaded: {len(starters):,} rows")
    else:
        starters  = None
        team_pool = None
        print("  ⚠️  No starter logs — using team averages (run fetch_starter_logs.py for better accuracy)")

    print(f"  Raw games: {len(games):,}")

    # Fix home_away and result
    games["home_away"] = (
        games["home_away"].astype(str).str.strip()
        .replace({"@": "AWAY", "Home": "HOME"})
    )
    games["result"] = (
        games["result"].astype(str).str.strip()
        .str.split("-").str[0].str.upper()
    )

    # Assign season by row position
    def assign_season(group):
        idx   = np.minimum(np.arange(len(group)) // GAMES_PER_SEASON, len(SEASONS_IN_ORDER) - 1)
        group = group.copy()
        group["season"] = [SEASONS_IN_ORDER[i] for i in idx]
        return group

    games = games.groupby("team", group_keys=False).apply(assign_season, include_groups=True)
    games["season"] = games["season"].astype(int)

    # Build synthetic date
    games = games.sort_values(["season","team"]).reset_index(drop=True)
    games["game_index"] = games.groupby(["team","season"]).cumcount()
    games["date"] = (
        pd.to_datetime(games["season"].astype(str) + "-01-01")
        + pd.to_timedelta(games["game_index"], unit="D")
    )

    # Recompute home_team_won
    def compute_winner(row):
        r = row["result"]; h = row["home_away"]
        if r not in ("W","L"): return np.nan
        return (1 if r=="W" else 0) if h=="HOME" else (1 if r=="L" else 0)

    games["home_team_won"] = games.apply(compute_winner, axis=1)
    games["runs_scored"]   = pd.to_numeric(games["runs_scored"],  errors="coerce")
    games["runs_allowed"]  = pd.to_numeric(games["runs_allowed"], errors="coerce")
    games["team"]          = games["team"].str.upper().str.strip().replace("OAK","ATH")
    games["opponent"]      = games["opponent"].str.upper().str.strip().replace("OAK","ATH")

    games = games.dropna(subset=["home_team_won","runs_scored","runs_allowed"])
    games["home_team_won"] = games["home_team_won"].astype(int)

    weight_map = {2021:0.6,2022:0.7,2023:0.8,2024:1.0,2025:1.0,2026:0.75}
    games["recency_weight"] = games["season"].map(weight_map).fillna(0.8)

    pitchers["team_abbr"] = pitchers["tm"].apply(normalize_team)
    pitchers["season"]    = pitchers["season"].astype(int)

    games = games.sort_values(["date","team"]).reset_index(drop=True)
    print(f"  ✅ Clean games: {len(games):,}")
    return games, pitchers, starters, team_pool


# ── Step 2: Rolling team stats ────────────────────────────────────────────────

def build_team_rolling_stats(games, window=ROLLING_WINDOW):
    print(f"\n📊 Building rolling stats (last {window} games)...")
    games = games.sort_values(["team","date"]).copy()

    games["rolling_runs_scored"] = (
        games.groupby("team")["runs_scored"]
        .transform(lambda x: x.shift(1).rolling(window,min_periods=3).mean())
    )
    games["rolling_runs_allowed"] = (
        games.groupby("team")["runs_allowed"]
        .transform(lambda x: x.shift(1).rolling(window,min_periods=3).mean())
    )
    games["result_binary"] = (games["result"]=="W").astype(int)
    games["rolling_win_rate"] = (
        games.groupby("team")["result_binary"]
        .transform(lambda x: x.shift(1).rolling(window,min_periods=3).mean())
    )
    games["rolling_run_diff"] = games["rolling_runs_scored"] - games["rolling_runs_allowed"]

    games = games.sort_values("date").reset_index(drop=True)
    print("  ✅ Done")
    return games


# ── Step 3: One row per game ──────────────────────────────────────────────────

def build_game_rows(games):
    print("\n🔗 Merging home/away into one row per game...")

    home = games[games["home_away"]=="HOME"].copy()
    away = games[games["home_away"]=="AWAY"].copy()

    home = home.rename(columns={
        "team":"home_team","opponent":"away_team",
        "rolling_runs_scored":"home_rolling_runs_scored",
        "rolling_runs_allowed":"home_rolling_runs_allowed",
        "rolling_win_rate":"home_rolling_win_rate",
        "rolling_run_diff":"home_rolling_run_diff",
    })
    away = away.rename(columns={
        "team":"away_team_check",
        "rolling_runs_scored":"away_rolling_runs_scored",
        "rolling_runs_allowed":"away_rolling_runs_allowed",
        "rolling_win_rate":"away_rolling_win_rate",
        "rolling_run_diff":"away_rolling_run_diff",
    })

    home_cols = [c for c in ["date","season","home_team","away_team","home_team_won",
        "home_rolling_runs_scored","home_rolling_runs_allowed",
        "home_rolling_win_rate","home_rolling_run_diff","recency_weight"] if c in home.columns]
    away_cols = [c for c in ["date","away_team_check","away_rolling_runs_scored",
        "away_rolling_runs_allowed","away_rolling_win_rate","away_rolling_run_diff"] if c in away.columns]

    merged = pd.merge(home[home_cols], away[away_cols],
        left_on=["date","away_team"], right_on=["date","away_team_check"],
        how="inner").drop(columns=["away_team_check"],errors="ignore")

    print(f"  ✅ {len(merged):,} game rows")
    return merged


# ── Step 4: Attach pitcher/starter stats ─────────────────────────────────────

def attach_pitcher_stats(games, pitchers, starters, team_pool):
    print("\n⚾ Attaching pitcher stats...")

    pitcher_cols = ["era","whip","so","bb","ip","hr","so9","so_per_w"]
    available    = [c for c in pitcher_cols if c in pitchers.columns]

    team_pitching = (
        pitchers.groupby(["team_abbr","season"])[available]
        .mean().reset_index()
    )

    home_p = team_pitching.rename(columns={"team_abbr":"home_team",**{c:f"home_avg_{c}" for c in available}})
    away_p = team_pitching.rename(columns={"team_abbr":"away_team",**{c:f"away_avg_{c}" for c in available}})

    games = pd.merge(games, home_p, on=["home_team","season"], how="left")
    games = pd.merge(games, away_p, on=["away_team","season"], how="left")

    matched = games["home_avg_era"].notna().sum()
    print(f"  Team ERA: {matched:,}/{len(games):,} matched")

    # Attach starter pool if available
    if team_pool is not None:
        team_pool["season"] = team_pool["season"].astype(int)
        starter_cols = [c for c in team_pool.columns if c.startswith("team_avg_starter_")]

        home_sp = team_pool.rename(columns={"team_abbr":"home_team",
            **{c:c.replace("team_avg_starter_","home_starter_") for c in starter_cols}})
        away_sp = team_pool.rename(columns={"team_abbr":"away_team",
            **{c:c.replace("team_avg_starter_","away_starter_") for c in starter_cols}})

        home_keep = ["home_team","season"] + [c.replace("team_avg_starter_","home_starter_") for c in starter_cols]
        away_keep = ["away_team","season"] + [c.replace("team_avg_starter_","away_starter_") for c in starter_cols]

        games = pd.merge(games, home_sp[[c for c in home_keep if c in home_sp.columns]], on=["home_team","season"], how="left")
        games = pd.merge(games, away_sp[[c for c in away_keep if c in away_sp.columns]], on=["away_team","season"], how="left")

        sp_matched = sum(1 for c in games.columns if "home_starter_" in c and games[c].notna().sum() > 0)
        print(f"  Starter pool: {sp_matched} feature columns attached")

    print("  ✅ Done")
    return games


# ── Step 5: Difference features ──────────────────────────────────────────────

def add_extra_features(games):
    print("\n➕ Adding difference features...")

    games["win_rate_diff"]    = games["home_rolling_win_rate"]    - games["away_rolling_win_rate"]
    games["run_diff_diff"]    = games["home_rolling_run_diff"]    - games["away_rolling_run_diff"]
    games["runs_scored_diff"] = games["home_rolling_runs_scored"] - games["away_rolling_runs_scored"]

    if "home_avg_era" in games.columns and "away_avg_era" in games.columns:
        games["era_diff"]  = games["away_avg_era"]  - games["home_avg_era"]
        games["whip_diff"] = games["away_avg_whip"] - games["home_avg_whip"]
        if "home_avg_so9" in games.columns:
            games["so9_diff"] = games["home_avg_so9"] - games["away_avg_so9"]

    # Starter difference features
    if "home_starter_era" in games.columns and "away_starter_era" in games.columns:
        games["starter_era_diff"]  = games["away_starter_era"]  - games["home_starter_era"]
        games["starter_whip_diff"] = games["away_starter_whip"] - games["home_starter_whip"]
        if "home_starter_k_per_9" in games.columns:
            games["starter_k_diff"]   = games["home_starter_k_per_9"]   - games["away_starter_k_per_9"]
        if "home_starter_fip_proxy" in games.columns:
            games["starter_fip_diff"] = games["away_starter_fip_proxy"] - games["home_starter_fip_proxy"]
        print("  Starter diff features added ✅")

    games["is_home"] = 1
    print("  ✅ Done")
    return games


# ── Step 6: Save ──────────────────────────────────────────────────────────────

def save_training_data(games):
    print("\n💾 Saving...")
    games = games.dropna(subset=["home_team_won","home_rolling_win_rate","away_rolling_win_rate"])
    games = games.sort_values("date").reset_index(drop=True)
    games.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved {len(games):,} training rows → {OUTPUT_PATH}")
    print(f"\nGames per season:")
    print(games.groupby("season").size().to_string())
    print(f"\nHome win rate: {games['home_team_won'].mean():.1%}")

    starter_cols = [c for c in games.columns if "starter_" in c]
    if starter_cols:
        matched = games[[starter_cols[0]]].notna().sum().iloc[0]
        print(f"\nStarter features: {len(starter_cols)} cols, {matched:,}/{len(games):,} games ({matched/len(games):.0%})")
    else:
        print("\n⚠️  No starter features — run fetch_starter_logs.py first")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    games, pitchers, starters, team_pool = load_data()
    games = build_team_rolling_stats(games)
    games = build_game_rows(games)
    games = attach_pitcher_stats(games, pitchers, starters, team_pool)
    games = add_extra_features(games)
    save_training_data(games)
