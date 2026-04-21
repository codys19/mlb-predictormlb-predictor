"""
fetch_bullpen.py
----------------
Builds historical bullpen fatigue data by pulling boxscores from the
MLB Stats API (free, no auth). For each game, extracts all relief pitcher
appearances and innings pitched. Output is used by build_features.py to
add bullpen fatigue columns to the training data.

HOW TO RUN:
    python data/fetch_bullpen.py

RUNTIME: ~45-60 minutes (one boxscore API call per game, ~15,000 games)
         Saves progress every 500 games — safe to restart if interrupted.

OUTPUT:
    raw/bullpen_usage.csv   (date, team_abbr, relief_ip, relief_appearances)
"""

import pandas as pd
import requests
import time
import os
import json

SEASONS    = [2021, 2022, 2023, 2024, 2025, 2026]
OUTPUT_CSV = "raw/bullpen_usage.csv"
PROGRESS   = "raw/bullpen_progress.json"   # tracks which game_pks are done

SEASON_DATES = {
    2021: ("2021-03-01", "2021-11-30"),
    2022: ("2022-03-01", "2022-11-30"),
    2023: ("2023-03-01", "2023-11-30"),
    2024: ("2024-03-01", "2024-11-30"),
    2025: ("2025-03-01", "2025-11-30"),
    2026: ("2026-03-01", "2026-11-30"),
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

FINAL_STATES = {"Final", "Game Over", "Completed Early"}

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&startDate={start}&endDate={end}&gameType=R"
    "&fields=dates,date,games,gamePk,status,detailedState,teams,home,away,team,name"
)
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-predictor/1.0"})


def parse_ip(ip_str):
    """Convert '2.1' (2 innings + 1 out) to decimal innings."""
    try:
        ip = float(ip_str)
        full = int(ip)
        outs = round((ip - full) * 10)
        return round(full + outs / 3, 4)
    except:
        return 0.0


def fetch_game_pks(season):
    """Get all completed regular-season game PKs for a season."""
    start, end = SEASON_DATES[season]
    url = SCHEDULE_URL.format(start=start, end=end)
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ❌ Schedule fetch failed: {e}")
        return []

    games = []
    for day in data.get("dates", []):
        game_date = day.get("date", "")
        for game in day.get("games", []):
            state = game.get("status", {}).get("detailedState", "")
            if state not in FINAL_STATES:
                continue
            game_pk = game.get("gamePk")
            home_name = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_name = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            ha = TEAM_MAP.get(home_name, "")
            aa = TEAM_MAP.get(away_name, "")
            if game_pk and ha and aa:
                games.append({"game_pk": game_pk, "date": game_date, "home": ha, "away": aa})

    return games


def fetch_bullpen_from_boxscore(game_pk, game_date, home_abbr, away_abbr):
    """
    Pull reliever usage from a boxscore.
    Returns list of {date, team_abbr, relief_ip, relief_appearances} rows.
    Starter = first pitcher listed per side. Everyone else = reliever.
    """
    try:
        r = SESSION.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=10)
        r.raise_for_status()
        box = r.json()
    except Exception:
        return []

    rows = []
    for side, team_abbr in [("home", home_abbr), ("away", away_abbr)]:
        team_data = box.get("teams", {}).get(side, {})
        pitchers  = team_data.get("pitchers", [])
        players   = team_data.get("players", {})

        if len(pitchers) < 2:
            # Only starter pitched — no bullpen usage
            rows.append({
                "date":                game_date,
                "team_abbr":           team_abbr,
                "game_pk":             game_pk,
                "relief_appearances":  0,
                "relief_ip":           0.0,
            })
            continue

        relief_ip = 0.0
        relief_appearances = 0
        for pid in pitchers[1:]:   # skip starter (index 0)
            pdata  = players.get(f"ID{pid}", {})
            pstats = pdata.get("stats", {}).get("pitching", {})
            ip_str = pstats.get("inningsPitched", "0")
            ip     = parse_ip(ip_str)
            if ip > 0:
                relief_ip += ip
                relief_appearances += 1

        rows.append({
            "date":               game_date,
            "team_abbr":          team_abbr,
            "game_pk":            game_pk,
            "relief_appearances": relief_appearances,
            "relief_ip":          round(relief_ip, 4),
        })

    return rows


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return set(json.load(f))
    return set()


def save_progress(done_pks):
    with open(PROGRESS, "w") as f:
        json.dump(list(done_pks), f)


def load_existing():
    if os.path.exists(OUTPUT_CSV):
        return pd.read_csv(OUTPUT_CSV)
    return pd.DataFrame()


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    print("⚾ Fetching historical bullpen usage from MLB Stats API\n")
    print("   This takes ~45-60 min. Progress is saved every 500 games.")
    print("   Safe to Ctrl+C and restart — will resume where it left off.\n")

    done_pks   = load_progress()
    all_rows   = load_existing().to_dict("records") if os.path.exists(OUTPUT_CSV) else []
    done_set   = set(done_pks)

    print(f"  Resuming from {len(done_set):,} previously completed games\n")

    total_calls = 0
    new_rows    = []

    for season in SEASONS:
        print(f"📅 Season {season} — fetching schedule...")
        games = fetch_game_pks(season)
        todo  = [g for g in games if g["game_pk"] not in done_set]
        print(f"  {len(games):,} total games, {len(todo):,} need boxscores\n")

        for i, g in enumerate(todo):
            rows = fetch_bullpen_from_boxscore(
                g["game_pk"], g["date"], g["home"], g["away"]
            )
            new_rows.extend(rows)
            done_set.add(g["game_pk"])
            total_calls += 1
            time.sleep(0.05)   # ~20 req/sec — well within free limits

            if total_calls % 100 == 0:
                pct = i / max(len(todo), 1) * 100
                print(f"  {season}: {i+1}/{len(todo)} ({pct:.0f}%)  "
                      f"— {total_calls:,} total calls")

            if total_calls % 500 == 0:
                # Save progress checkpoint
                save_progress(done_set)
                combined = pd.DataFrame(all_rows + new_rows)
                combined.to_csv(OUTPUT_CSV, index=False)
                print(f"  💾 Checkpoint saved ({len(combined):,} rows)")

        print(f"  ✅ Season {season} done\n")

    # Final save
    save_progress(done_set)
    final = pd.DataFrame(all_rows + new_rows)
    final = final.drop_duplicates(subset=["date", "team_abbr"], keep="last")
    final = final.sort_values(["date", "team_abbr"]).reset_index(drop=True)
    final.to_csv(OUTPUT_CSV, index=False)

    print(f"{'='*55}")
    print(f"✅ Saved {len(final):,} team-game rows → {OUTPUT_CSV}")
    print(f"   {total_calls:,} boxscore API calls made\n")
    print(f"Coverage by season:")
    final["season"] = pd.to_datetime(final["date"]).dt.year
    print(final.groupby("season").size().rename("games").to_string())
    print(f"\nSample:")
    print(final.head(10).to_string(index=False))
    print(f"\nNext: python data/build_features.py")

    # Clean up progress file when done
    if os.path.exists(PROGRESS):
        os.remove(PROGRESS)
        print("   (Progress file cleaned up)")
