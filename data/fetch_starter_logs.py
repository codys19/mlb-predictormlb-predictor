"""
fetch_starter_logs.py
---------------------
Downloads individual starter pitcher stats using the MLB Stats API.
Builds ERA, WHIP, K/9, BB/9 per pitcher per season.

HOW TO RUN:
    python data/fetch_starter_logs.py

OUTPUT:
    raw/starter_logs.csv
    raw/team_starter_pool.csv
"""

import pandas as pd
import numpy as np
import statsapi
import time
import os

SEASONS    = [2021, 2022, 2023, 2024, 2025, 2026]
OUTPUT_CSV = "raw/starter_logs.csv"
POOL_CSV   = "raw/team_starter_pool.csv"

TEAM_ID_MAP = {
    109:"ARI", 144:"ATL", 110:"BAL", 111:"BOS", 112:"CHC", 145:"CHW",
    113:"CIN", 114:"CLE", 115:"COL", 116:"DET", 117:"HOU", 118:"KCR",
    108:"LAA", 119:"LAD", 146:"MIA", 158:"MIL", 142:"MIN", 121:"NYM",
    147:"NYY", 133:"ATH", 143:"PHI", 134:"PIT", 135:"SDP", 136:"SEA",
    137:"SFG", 138:"STL", 139:"TBR", 140:"TEX", 141:"TOR", 120:"WSN",
}


def fetch_season_pitchers(season):
    """Fetch all pitchers who started games in a season via MLB Stats API."""
    print(f"  📅 {season}...")
    records = []

    for team_id, team_abbr in TEAM_ID_MAP.items():
        try:
            # Get team roster for the season
            roster = statsapi.roster(team_id, season=season, rosterType="fullSeason")
            if not roster:
                continue

            # Parse roster lines — format: "#  Name  POS"
            for line in roster.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                pos = parts[-1]
                if pos != "P":
                    continue  # only pitchers

                # Name is everything between number and position
                name = " ".join(parts[1:-1])

                # Look up player stats
                try:
                    results = statsapi.player_stat_data(
                        statsapi.lookup_player(name)[0]["id"],
                        group="pitching",
                        type="season",
                        season=season
                    )
                    stats = results.get("stats", [{}])[0].get("stats", {})

                    gs   = int(stats.get("gamesStarted", 0))
                    if gs < 1:
                        continue  # skip relievers

                    ip   = float(stats.get("inningsPitched", 0) or 0)
                    er   = int(stats.get("earnedRuns", 0) or 0)
                    h    = int(stats.get("hits", 0) or 0)
                    bb   = int(stats.get("baseOnBalls", 0) or 0)
                    so   = int(stats.get("strikeOuts", 0) or 0)
                    hr   = int(stats.get("homeRuns", 0) or 0)

                    era  = float(stats.get("era", 99) or 99)
                    whip = float(stats.get("whip", 99) or 99)

                    k9   = (so / ip * 9) if ip > 0 else np.nan
                    bb9  = (bb / ip * 9) if ip > 0 else np.nan
                    hr9  = (hr / ip * 9) if ip > 0 else np.nan
                    fip  = ((13*hr + 3*bb - 2*so) / ip + 3.2) if ip > 0 else np.nan

                    records.append({
                        "name":       name,
                        "season":     season,
                        "team_abbr":  team_abbr,
                        "gs":         gs,
                        "ip":         ip,
                        "era":        era,
                        "whip":       whip,
                        "k_per_9":    round(k9,  2) if not np.isnan(k9)  else np.nan,
                        "bb_per_9":   round(bb9, 2) if not np.isnan(bb9) else np.nan,
                        "hr_per_9":   round(hr9, 2) if not np.isnan(hr9) else np.nan,
                        "fip_proxy":  round(fip, 2) if not np.isnan(fip) else np.nan,
                    })
                    time.sleep(0.05)  # gentle rate limit

                except (IndexError, KeyError, TypeError):
                    continue

        except Exception as e:
            continue

    print(f"     → {len(records)} starters found")
    return records


def build_team_pool(df):
    """Average stats of each team's rotation per season."""
    feature_cols = ["era","whip","k_per_9","bb_per_9","hr_per_9","fip_proxy"]
    available    = [c for c in feature_cols if c in df.columns]

    pool = (
        df[df["gs"] >= 5]
        .groupby(["team_abbr","season"])[available]
        .mean()
        .reset_index()
    )
    pool = pool.rename(columns={c: f"team_avg_starter_{c}" for c in available})
    return pool


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    print("🎯 Fetching starter stats via MLB Stats API...\n")

    all_records = []
    for season in SEASONS:
        records = fetch_season_pitchers(season)
        all_records.extend(records)
        time.sleep(1)

    if not all_records:
        print("❌ No data. Trying fallback method...")

        # ── Fallback: use existing pitcher_stats.csv ─────────────────────────
        # Build starter features directly from the pitcher_stats.csv we already have
        print("\n🔄 Building from existing pitcher_stats.csv...")
        pitchers = pd.read_csv("raw/pitcher_stats.csv")

        TEAM_NAME_MAP = {
            "Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Boston":"BOS",
            "Chicago":"CHC","Cincinnati":"CIN","Cleveland":"CLE","Colorado":"COL",
            "Detroit":"DET","Houston":"HOU","Kansas City":"KCR","Los Angeles":"LAD",
            "Miami":"MIA","Milwaukee":"MIL","Minnesota":"MIN","New York":"NYY",
            "Oakland":"OAK","Athletics":"ATH","Philadelphia":"PHI","Pittsburgh":"PIT",
            "San Diego":"SDP","Seattle":"SEA","San Francisco":"SFG","St. Louis":"STL",
            "Tampa Bay":"TBR","Texas":"TEX","Toronto":"TOR","Washington":"WSN",
        }
        def norm(raw):
            parts=[p.strip() for p in str(raw).split(",")]
            return TEAM_NAME_MAP.get(parts[-1], parts[-1].upper()[:3])

        pitchers["team_abbr"] = pitchers["tm"].apply(norm)
        pitchers["season"]    = pitchers["season"].astype(int)

        # Compute k_per_9, bb_per_9, fip_proxy from raw columns
        if "so" in pitchers.columns and "ip" in pitchers.columns:
            pitchers["k_per_9"]   = pitchers["so"]  / pitchers["ip"].replace(0,np.nan) * 9
            pitchers["bb_per_9"]  = pitchers["bb"]  / pitchers["ip"].replace(0,np.nan) * 9
            pitchers["hr_per_9"]  = pitchers["hr"]  / pitchers["ip"].replace(0,np.nan) * 9
            pitchers["fip_proxy"] = (
                (13*pitchers["hr"] + 3*pitchers["bb"] - 2*pitchers["so"])
                / pitchers["ip"].replace(0,np.nan) + 3.2
            )

        cols = ["name","season","team_abbr","gs","ip","era","whip",
                "k_per_9","bb_per_9","hr_per_9","fip_proxy"]
        rename = {"so9":"k_per_9","so_per_w":"so_per_w_ratio"}
        pitchers = pitchers.rename(columns=rename)
        available = [c for c in cols if c in pitchers.columns]
        df = pitchers[available].copy()
        df.columns = [c.lower() for c in df.columns]

        df.to_csv(OUTPUT_CSV, index=False)
        pool = build_team_pool(df)
        pool.to_csv(POOL_CSV, index=False)

        print(f"\n✅ Fallback complete:")
        print(f"   {OUTPUT_CSV}:  {len(df):,} rows")
        print(f"   {POOL_CSV}: {len(pool):,} team-seasons")
        print(f"\nSample:")
        print(df.head(5)[["name","season","team_abbr","gs","era","whip","k_per_9"]].to_string(index=False))

    else:
        df   = pd.DataFrame(all_records)
        pool = build_team_pool(df)

        df.to_csv(OUTPUT_CSV, index=False)
        pool.to_csv(POOL_CSV, index=False)

        print(f"\n✅ Done:")
        print(f"   {OUTPUT_CSV}:  {len(df):,} pitcher-season rows")
        print(f"   {POOL_CSV}: {len(pool):,} team-seasons")
        print(f"\nSample:")
        print(df.head(5)[["name","season","team_abbr","gs","era","whip","k_per_9"]].to_string(index=False))