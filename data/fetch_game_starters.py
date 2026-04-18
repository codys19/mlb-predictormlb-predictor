"""
fetch_game_starters.py
----------------------
Builds a historical game → starter mapping using the MLB Stats API (mlb-statsapi).
NO rate limits. NO Baseball Reference. Runs in ~2-3 minutes.

Uses statsapi.schedule() to pull all completed games per season, extracting
the home/away probable pitchers (which for completed games = actual starters).

HOW TO RUN:
    python data/fetch_game_starters.py

OUTPUT:
    raw/game_starters.csv   (date, season, team_abbr, starter_name)

RUNTIME: ~2-3 minutes (6 season-level API calls + boxscore fallback only for
         games where probable_pitcher field is empty)
"""

import pandas as pd
import numpy as np
import statsapi
import time
import os

SEASONS    = [2021, 2022, 2023, 2024, 2025, 2026]
OUTPUT_CSV = "raw/game_starters.csv"

SEASON_DATES = {
    2021: ("03/01/2021", "11/30/2021"),
    2022: ("03/01/2022", "11/30/2022"),
    2023: ("03/01/2023", "11/30/2023"),
    2024: ("03/01/2024", "11/30/2024"),
    2025: ("03/01/2025", "11/30/2025"),
    2026: ("03/01/2026", "11/30/2026"),
}

TEAM_MAP = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CHW",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Cleveland Indians":"CLE",
    "Colorado Rockies":"COL","Detroit Tigers":"DET","Houston Astros":"HOU",
    "Kansas City Royals":"KCR","Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD",
    "Miami Marlins":"MIA","Milwaukee Brewers":"MIL","Minnesota Twins":"MIN",
    "New York Mets":"NYM","New York Yankees":"NYY","Oakland Athletics":"ATH",
    "Athletics":"ATH","Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT",
    "San Diego Padres":"SDP","Seattle Mariners":"SEA","San Francisco Giants":"SFG",
    "St. Louis Cardinals":"STL","Tampa Bay Rays":"TBR","Texas Rangers":"TEX",
    "Toronto Blue Jays":"TOR","Washington Nationals":"WSN",
}


def clean_name(raw):
    if not raw or str(raw).strip() in ("", "None", "TBD"):
        return ""
    name = str(raw).strip()
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name


def get_starter_from_boxscore(game_id, home_abbr, away_abbr):
    """Fallback: read first pitcher from boxscore when schedule() has no pitcher name."""
    try:
        box = statsapi.boxscore_data(game_id)
        home_p = ""; away_p = ""
        for side, key in [("home", home_abbr), ("away", away_abbr)]:
            pitchers = box.get(side, {}).get("pitchers", [])
            if pitchers:
                pid   = pitchers[0]
                pinfo = box.get(side, {}).get("players", {}).get(f"ID{pid}", {})
                n     = pinfo.get("person", {}).get("fullName", "")
                if side == "home": home_p = clean_name(n)
                else: away_p = clean_name(n)
        return home_p, away_p
    except Exception:
        return "", ""


def fetch_season(season):
    start, end = SEASON_DATES[season]
    records = []; boxscore_calls = 0

    print(f"  Fetching schedule {start} → {end}...")
    try:
        sched = statsapi.schedule(start_date=start, end_date=end, sportId=1)
    except Exception as e:
        print(f"  ❌ Failed: {e}"); return records, 0

    completed = [g for g in sched if g.get("status") in
                 ("Final","Game Over","Completed Early")]
    print(f"  {len(completed):,} completed games found")

    missing = []
    for g in completed:
        hn = g.get("home_name",""); an = g.get("away_name","")
        ha = TEAM_MAP.get(hn, hn[:3].upper() if hn else "")
        aa = TEAM_MAP.get(an, an[:3].upper() if an else "")
        if not ha or not aa: continue

        game_date = g.get("game_date","")
        if not game_date: continue

        home_p = clean_name(g.get("home_probable_pitcher","") or "")
        away_p = clean_name(g.get("away_probable_pitcher","") or "")

        if not home_p or not away_p:
            missing.append((g.get("game_id"), game_date, ha, aa, home_p, away_p))
            continue

        records.append({"date":game_date,"season":season,"team_abbr":ha,"starter_name":home_p})
        records.append({"date":game_date,"season":season,"team_abbr":aa,"starter_name":away_p})

    if missing:
        print(f"  {len(missing)} games missing pitcher name — fetching boxscores...")
        for gid, game_date, ha, aa, home_p, away_p in missing:
            bs_home, bs_away = get_starter_from_boxscore(gid, ha, aa)
            home_p = home_p or bs_home
            away_p = away_p or bs_away
            boxscore_calls += 1
            time.sleep(0.1)
            if home_p: records.append({"date":game_date,"season":season,"team_abbr":ha,"starter_name":home_p})
            if away_p: records.append({"date":game_date,"season":season,"team_abbr":aa,"starter_name":away_p})

    print(f"  → {len(records):,} starter entries  ({boxscore_calls} boxscore fallback calls)")
    return records, boxscore_calls


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    print("🎯 Fetching game→starter assignments via MLB Stats API (no rate limits)\n")

    all_records = []; total_bc = 0
    for season in SEASONS:
        print(f"\n📅 Season {season}...")
        records, bc = fetch_season(season)
        all_records.extend(records)
        total_bc += bc
        time.sleep(0.5)

    if not all_records:
        print("\n❌ No records fetched. Check: pip install mlb-statsapi")
        exit(1)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["date","team_abbr"], keep="first")
    df = df[df["starter_name"].str.strip() != ""]
    df = df.sort_values(["season","date","team_abbr"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"✅  Saved {len(df):,} game-starter records → {OUTPUT_CSV}")
    print(f"    ({total_bc} boxscore fallback calls)")
    print(f"\nCoverage by season:")
    print(df.groupby("season").size().rename("starter_entries").to_string())
    print(f"\nSample:")
    print(df.head(10)[["date","season","team_abbr","starter_name"]].to_string(index=False))
    print()
    for season in SEASONS:
        n = len(df[df["season"]==season])
        ok = n >= 30 * 120
        print(f"  {'✅' if ok else '⚠️ '} {season}: {n} entries {'✓' if ok else '← may be partial'}")
    print(f"\nNext: python data/build_features.py")