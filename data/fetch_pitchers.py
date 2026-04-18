"""
fetch_game_starters.py
----------------------
Builds a historical game → starter mapping using the MLB Stats API directly
via requests (no mlb-statsapi wrapper needed — avoids version incompatibilities).

HOW TO RUN:
    python data/fetch_game_starters.py

OUTPUT:
    raw/game_starters.csv   (date, season, team_abbr, starter_name)

RUNTIME: ~2-3 minutes (6 season-level API calls + boxscore fallback only for
         games where probablePitcher field is empty)
"""

import pandas as pd
import requests
import time
import os

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
OUTPUT_CSV = "raw/game_starters.csv"

SEASON_DATES = {
    2021: ("2021-03-01", "2021-11-30"),
    2022: ("2022-03-01", "2022-11-30"),
    2023: ("2023-03-01", "2023-11-30"),
    2024: ("2024-03-01", "2024-11-30"),
    2025: ("2025-03-01", "2025-11-30"),
    2026: ("2026-03-01", "2026-11-30"),
}

# Direct MLB Stats API endpoints
SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&startDate={start}&endDate={end}"
    "&gameType=R&hydrate=probablePitcher"
    "&fields=dates,date,games,gamePk,status,detailedState,"
    "teams,home,away,team,name,probablePitcher,fullName"
)
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

TEAM_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "ATH", "Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}

FINAL_STATES = {"Final", "Game Over", "Completed Early"}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-predictor/1.0"})


def clean_name(raw):
    if not raw or str(raw).strip() in ("", "None", "TBD"):
        return ""
    name = str(raw).strip()
    # "Last, First" → "First Last"
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    return name


def get_starter_from_boxscore(game_pk):
    """Fallback: pull first pitcher listed per side from the boxscore."""
    try:
        r = SESSION.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=10)
        r.raise_for_status()
        data = r.json()
        home_p = ""; away_p = ""
        for side in ("home", "away"):
            pitchers = data.get("teams", {}).get(side, {}).get("pitchers", [])
            players  = data.get("teams", {}).get(side, {}).get("players", {})
            if pitchers:
                pid  = f"ID{pitchers[0]}"
                name = players.get(pid, {}).get("person", {}).get("fullName", "")
                if side == "home":
                    home_p = clean_name(name)
                else:
                    away_p = clean_name(name)
        return home_p, away_p
    except Exception:
        return "", ""


def fetch_season(season):
    start, end = SEASON_DATES[season]
    url = SCHEDULE_URL.format(start=start, end=end)

    print(f"  Fetching schedule {start} → {end}...")
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        sched_data = r.json()
    except Exception as e:
        print(f"  ❌ HTTP request failed: {e}")
        return [], 0

    records = []; missing = []; boxscore_calls = 0

    for day in sched_data.get("dates", []):
        game_date = day.get("date", "")
        for game in day.get("games", []):
            state = game.get("status", {}).get("detailedState", "")
            if state not in FINAL_STATES:
                continue

            game_pk  = game.get("gamePk")
            home_obj = game.get("teams", {}).get("home", {})
            away_obj = game.get("teams", {}).get("away", {})

            home_name = home_obj.get("team", {}).get("name", "")
            away_name = away_obj.get("team", {}).get("name", "")
            ha = TEAM_MAP.get(home_name, home_name[:3].upper() if home_name else "")
            aa = TEAM_MAP.get(away_name, away_name[:3].upper() if away_name else "")
            if not ha or not aa:
                continue

            home_p = clean_name(home_obj.get("probablePitcher", {}).get("fullName", "") or "")
            away_p = clean_name(away_obj.get("probablePitcher", {}).get("fullName", "") or "")

            if home_p and away_p:
                records.append({"date": game_date, "season": season, "team_abbr": ha, "starter_name": home_p})
                records.append({"date": game_date, "season": season, "team_abbr": aa, "starter_name": away_p})
            else:
                missing.append((game_pk, game_date, ha, aa, home_p, away_p))

    completed_total = len(records) // 2 + len(missing)
    print(f"  {completed_total:,} completed games found")

    if missing:
        print(f"  {len(missing)} games missing pitcher name — fetching boxscores...")
        for game_pk, game_date, ha, aa, home_p, away_p in missing:
            bs_home, bs_away = get_starter_from_boxscore(game_pk)
            home_p = home_p or bs_home
            away_p = away_p or bs_away
            boxscore_calls += 1
            time.sleep(0.1)
            if home_p:
                records.append({"date": game_date, "season": season, "team_abbr": ha, "starter_name": home_p})
            if away_p:
                records.append({"date": game_date, "season": season, "team_abbr": aa, "starter_name": away_p})

    print(f"  → {len(records):,} starter entries  ({boxscore_calls} boxscore fallback calls)")
    return records, boxscore_calls


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    print("🎯 Fetching game→starter assignments via MLB Stats API (direct HTTP)\n")

    all_records = []; total_bc = 0
    for season in SEASONS:
        print(f"\n📅 Season {season}...")
        records, bc = fetch_season(season)
        all_records.extend(records)
        total_bc += bc
        time.sleep(0.5)

    if not all_records:
        print("\n❌ No records fetched. Check your internet connection.")
        exit(1)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["date", "team_abbr"], keep="first")
    df = df[df["starter_name"].str.strip() != ""]
    df = df.sort_values(["season", "date", "team_abbr"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"✅  Saved {len(df):,} game-starter records → {OUTPUT_CSV}")
    print(f"    ({total_bc} boxscore fallback calls)")
    print(f"\nCoverage by season:")
    print(df.groupby("season").size().rename("starter_entries").to_string())
    print(f"\nSample:")
    print(df.head(10)[["date", "season", "team_abbr", "starter_name"]].to_string(index=False))
    print()
    for season in SEASONS:
        n  = len(df[df["season"] == season])
        ok = n >= 30 * 120
        print(f"  {'✅' if ok else '⚠️ '} {season}: {n} entries {'✓' if ok else '← may be partial'}")
    print(f"\nNext: python data/build_features.py")