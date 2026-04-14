"""
grade_picks.py
--------------
Run this each evening (or next morning) to grade yesterday's picks
against actual results and update your running W/L record.

RUN:
    python data/grade_picks.py

It will:
  1. Load yesterday's saved picks from raw/picks_YYYY-MM-DD.json
  2. Fetch actual game results from MLB Stats API
  3. Compare picks vs results
  4. Update raw/record.json with W/L
  5. Save a history HTML page for yesterday
"""

import json
import os
import statsapi
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBClassifier

RECORD_PATH = "raw/record.json"

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CHW",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KCR",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"OAK","Athletics":"ATH",
    "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SDP",
    "Seattle Mariners":"SEA","San Francisco Giants":"SFG","St. Louis Cardinals":"STL",
    "Tampa Bay Rays":"TBR","Texas Rangers":"TEX","Toronto Blue Jays":"TOR",
    "Washington Nationals":"WSN",
}

def load_record():
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f:
            return json.load(f)
    return {
        "all_time":{"w":0,"l":0},
        "by_conf":{"50":{"w":0,"l":0},"55":{"w":0,"l":0},"60":{"w":0,"l":0},"70":{"w":0,"l":0}},
        "daily":{},
    }

def save_record(record):
    with open(RECORD_PATH,"w") as f:
        json.dump(record, f, indent=2)

def conf_bucket(conf):
    if conf >= 0.80: return "70"
    if conf >= 0.70: return "70"
    if conf >= 0.60: return "60"
    if conf >= 0.55: return "55"
    return "50"

def grade_date(date_str):
    picks_path = f"raw/picks_{date_str}.json"

    if not os.path.exists(picks_path):
        print(f"⚠️  No picks file found for {date_str}")
        print(f"   Expected: {picks_path}")
        return

    with open(picks_path) as f:
        picks = json.load(f)

    # Fetch actual results
    print(f"📊 Fetching results for {date_str}...")
    dt       = datetime.strptime(date_str, "%Y-%m-%d")
    date_fmt = dt.strftime("%m/%d/%Y")
    sched    = statsapi.schedule(date=date_fmt)

    # Build result lookup: home_abbr → winner_abbr
    results = {}
    for g in sched:
        if g.get("status") not in ("Final","Game Over","Completed Early"):
            continue
        hn = g.get("home_name",""); an = g.get("away_name","")
        ha = TEAM_NAME_TO_ABBR.get(hn, hn[:3].upper())
        aa = TEAM_NAME_TO_ABBR.get(an, an[:3].upper())
        hs = g.get("home_score",0); as_ = g.get("away_score",0)
        winner = ha if hs > as_ else aa
        results[f"{ha}_{aa}"] = {"winner": winner, "home_score": hs, "away_score": as_,
                                  "home_abbr": ha, "away_abbr": aa,
                                  "home_name": hn, "away_name": an}

    # Grade picks
    record   = load_record()
    day_w    = 0
    day_l    = 0
    graded   = []

    print(f"\n📋 Results for {date_str}:")
    print("─" * 60)

    for pick in picks:
        key    = f"{pick['home_abbr']}_{pick['away_abbr']}"
        result = results.get(key)

        if not result:
            print(f"  ⚠️  No result found for {pick['home_abbr']} vs {pick['away_abbr']}")
            pick["result"] = "no_result"
            graded.append(pick)
            continue

        correct = result["winner"] == pick["pick_abbr"]
        pick["result"]       = "W" if correct else "L"
        pick["actual_winner"]= result["winner"]
        pick["home_score"]   = result["home_score"]
        pick["away_score"]   = result["away_score"]

        # Update record
        bucket = conf_bucket(pick["confidence"])
        if correct:
            record["all_time"]["w"]       += 1
            record["by_conf"][bucket]["w"] += 1
            day_w += 1
        else:
            record["all_time"]["l"]       += 1
            record["by_conf"][bucket]["l"] += 1
            day_l += 1

        icon = "✅" if correct else "❌"
        score_str = f"{result['home_score']}–{result['away_score']}"
        print(f"  {icon} Picked {pick['pick_abbr']:<4}  |  "
              f"{pick['home_abbr']} {score_str} {pick['away_abbr']}  |  "
              f"Actual: {result['winner']}  ({int(pick['confidence']*100)}%)")

        graded.append(pick)

    # Save updated picks
    with open(picks_path,"w") as f:
        json.dump(graded, f, indent=2)

    # Update daily record
    record["daily"][date_str] = {"w": day_w, "l": day_l}
    save_record(record)

    at = record["all_time"]
    pct = f"{at['w']/(at['w']+at['l']):.1%}" if (at['w']+at['l']) > 0 else "—"
    print(f"\n📊 Today: {day_w}–{day_l}")
    print(f"   All time: {at['w']}–{at['l']} ({pct})")

    # Generate history page for this date
    generate_history_html(graded, date_str, day_w, day_l)


def generate_history_html(picks, date_str, day_w, day_l):
    """Save a history page so the date nav can link back to it."""
    dt    = datetime.strptime(date_str, "%Y-%m-%d")
    title = dt.strftime("%A, %B %d %Y")
    path  = f"history_{date_str}.html"

    rows = ""
    for p in picks:
        if p.get("result") == "no_result": continue
        correct   = p.get("result") == "W"
        icon      = "✅" if correct else "❌"
        score_str = f"{p.get('home_score','')}–{p.get('away_score','')}"
        conf_pct  = int(p["confidence"]*100)
        clr       = "#16a34a" if correct else "#dc2626"

        rows += f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:14px;font-weight:600;color:{clr};">{icon} {p['pick']}</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:2px;">{p['home_name']} vs {p['away_name']}</div>
      <div style="font-size:12px;color:#374151;margin-top:2px;">Final: {score_str} · Winner: {p.get('actual_winner','?')}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;font-weight:500;color:#374151;">{conf_pct}% conf.</div>
    </div>
  </div>
</div>"""

    wl_clr = "#16a34a" if day_w > day_l else "#dc2626"
    html   = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB Picks · {date_str}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
  .wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
</style>
</head>
<body>
<div class="wrap">
  <div style="margin-bottom:20px;">
    <a href="mlb_predictor.html" style="font-size:12px;color:#9ca3af;text-decoration:none;">← Back to today</a>
    <h1 style="font-size:22px;font-weight:500;margin-top:8px;">⚾ {title}</h1>
    <div style="font-size:18px;font-weight:500;color:{wl_clr};margin-top:6px;">{day_w}–{day_l} · {f"{day_w/(day_w+day_l):.0%}" if (day_w+day_l)>0 else "—"}</div>
  </div>
  {rows}
</div>
</body>
</html>"""

    with open(path,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ History page saved → {path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Grade a specific date: python data/grade_picks.py 2026-04-09
        date_str = sys.argv[1]
    else:
        # Default: grade yesterday
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Grading picks for {date_str}...")
    grade_date(date_str)
