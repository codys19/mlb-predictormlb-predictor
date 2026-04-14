"""
generate_predictions.py
-----------------------
Runs your MLB model for today's games and generates mlb_predictor.html.
Tracks your actual pick record in raw/record.json.

RUN EVERY MORNING:
    python data/generate_predictions.py

Then run this in the evening to grade yesterday's picks:
    python data/grade_picks.py

OUTPUT:
    mlb_predictor.html   ← open in Chrome
    raw/picks_YYYY-MM-DD.json  ← today's picks saved for grading later
    raw/record.json      ← your running W/L record
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import statsapi
import requests
import json
import os

# ── Paste your Odds API key here ──────────────────────────────────────────────
ODDS_API_KEY = "YOUR_KEY_HERE"
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "raw/model.json"
GAMES_PATH   = "raw/game_results.csv"
PITCHER_PATH = "raw/pitcher_stats.csv"
OUTPUT_HTML  = "mlb_predictor.html"
RECORD_PATH  = "raw/record.json"
ROLLING_WINDOW = 15

FEATURE_COLS = [
    "home_rolling_runs_scored", "home_rolling_runs_allowed",
    "home_rolling_win_rate",    "home_rolling_run_diff",
    "away_rolling_runs_scored", "away_rolling_runs_allowed",
    "away_rolling_win_rate",    "away_rolling_run_diff",
    "win_rate_diff", "run_diff_diff", "runs_scored_diff", "is_home",
    "home_avg_era", "home_avg_whip", "home_avg_so9", "home_avg_so_per_w",
    "away_avg_era", "away_avg_whip", "away_avg_so9", "away_avg_so_per_w",
    "era_diff", "whip_diff", "so9_diff",
]

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

ODDS_NAME_TO_ABBR = {**TEAM_NAME_TO_ABBR, "Oakland Athletics":"ATH"}


# ── Record helpers ────────────────────────────────────────────────────────────

def load_record():
    """Load running W/L record. Creates a fresh one if it doesn't exist."""
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f:
            return json.load(f)
    return {
        "all_time":    {"w": 0, "l": 0},
        "by_conf":     {"50":{"w":0,"l":0},"55":{"w":0,"l":0},"60":{"w":0,"l":0},"70":{"w":0,"l":0}},
        "daily":       {},   # "2026-04-09": {"w": 3, "l": 1}
    }

def save_record(record):
    with open(RECORD_PATH, "w") as f:
        json.dump(record, f, indent=2)

def record_display(record):
    """Build the W/L strings shown in the header card."""
    at = record["all_time"]
    at_str = f"{at['w']}–{at['l']}"
    at_pct = f"{at['w']/(at['w']+at['l']):.1%}" if (at['w']+at['l']) > 0 else "—"

    # This month
    now = datetime.now()
    month_w = month_l = 0
    for date_str, d in record["daily"].items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt.year == now.year and dt.month == now.month:
                month_w += d["w"]; month_l += d["l"]
        except: pass
    month_str = f"{month_w}–{month_l}"

    # This week (last 7 days)
    week_w = week_l = 0
    for date_str, d in record["daily"].items():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if (now - dt).days <= 7:
                week_w += d["w"]; week_l += d["l"]
        except: pass
    week_str = f"{week_w}–{week_l}"

    # Yesterday
    yest = (now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yd   = record["daily"].get(yest, {})
    yest_str = f"{yd.get('w',0)}–{yd.get('l',0)}" if yd else "—"

    return at_str, at_pct, month_str, week_str, yest_str


# ── Model + data loading ──────────────────────────────────────────────────────

def load_model():
    print("🤖 Loading model...")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    print("  ✅ Done")
    return model

def build_current_team_stats():
    print("📊 Building team rolling stats...")
    games = pd.read_csv(GAMES_PATH)
    games["home_away"] = games["home_away"].astype(str).str.strip().replace({"@":"AWAY","Home":"HOME"})
    games["result"]    = games["result"].astype(str).str.strip().str.split("-").str[0].str.upper()

    SEASONS = [2021,2022,2023,2024,2025,2026]
    def assign_season(group):
        idx   = np.minimum(np.arange(len(group)) // 162, len(SEASONS)-1)
        group = group.copy()
        group["season"] = [SEASONS[i] for i in idx]
        return group

    games = games.groupby("team", group_keys=False).apply(assign_season, include_groups=True)
    games = games.sort_values(["season","team"]).reset_index(drop=True)
    games["game_index"] = games.groupby(["team","season"]).cumcount()
    games["date"] = (pd.to_datetime(games["season"].astype(str)+"-01-01")
                     + pd.to_timedelta(games["game_index"], unit="D"))
    games["runs_scored"]   = pd.to_numeric(games["runs_scored"],  errors="coerce")
    games["runs_allowed"]  = pd.to_numeric(games["runs_allowed"], errors="coerce")
    games["result_binary"] = (games["result"] == "W").astype(int)
    games["team"]          = games["team"].str.upper().str.strip()

    team_stats = {}
    for team, grp in games.groupby("team"):
        grp    = grp.sort_values("date")
        recent = grp.tail(ROLLING_WINDOW)
        last5  = grp.tail(5)
        rs = recent["runs_scored"].mean(); ra = recent["runs_allowed"].mean()
        wr = recent["result_binary"].mean()
        team_stats[team] = {
            "rolling_runs_scored": rs, "rolling_runs_allowed": ra,
            "rolling_win_rate": wr,    "rolling_run_diff": rs - ra,
            "last5": "".join(["W" if r==1 else "L" for r in last5["result_binary"]]),
            "last5_w": int(last5["result_binary"].sum()),
            "last5_l": int(5 - last5["result_binary"].sum()),
        }
    print(f"  ✅ {len(team_stats)} teams")
    return team_stats

def build_pitcher_stats():
    print("⚾ Loading pitcher stats...")
    pitchers = pd.read_csv(PITCHER_PATH)
    MAP = {"Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Boston":"BOS",
           "Chicago":"CHC","Cincinnati":"CIN","Cleveland":"CLE","Colorado":"COL",
           "Detroit":"DET","Houston":"HOU","Kansas City":"KCR","Los Angeles":"LAD",
           "Miami":"MIA","Milwaukee":"MIL","Minnesota":"MIN","New York":"NYY",
           "Oakland":"OAK","Athletics":"ATH","Philadelphia":"PHI","Pittsburgh":"PIT",
           "San Diego":"SDP","Seattle":"SEA","San Francisco":"SFG","St. Louis":"STL",
           "Tampa Bay":"TBR","Texas":"TEX","Toronto":"TOR","Washington":"WSN"}
    def norm(raw):
        parts = [p.strip() for p in str(raw).split(",")]
        return MAP.get(parts[-1], parts[-1].upper()[:3])
    pitchers["team_abbr"] = pitchers["tm"].apply(norm)
    cols = [c for c in ["era","whip","so9","so_per_w"] if c in pitchers.columns]
    pitchers["season"] = pitchers["season"].astype(int)
    latest = pitchers.sort_values("season").groupby("team_abbr").last().reset_index()
    return {row["team_abbr"]: {f"avg_{c}": row.get(c, np.nan) for c in cols}
            for _, row in latest.iterrows()}


# ── Odds ──────────────────────────────────────────────────────────────────────

def fetch_odds():
    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("⚠️  No odds key — skipping odds")
        return {}
    print("💰 Fetching odds...")
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={"apiKey":ODDS_API_KEY,"regions":"us","markets":"h2h",
                    "oddsFormat":"american","bookmakers":"draftkings,fanduel,betmgm"},
            timeout=10
        )
        out = {}
        for game in r.json():
            h = ODDS_NAME_TO_ABBR.get(game.get("home_team",""), "")
            a = ODDS_NAME_TO_ABBR.get(game.get("away_team",""), "")
            if not h or not a: continue
            hl, al = [], []
            for bk in game.get("bookmakers",[]):
                for mkt in bk.get("markets",[]):
                    if mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            ab = ODDS_NAME_TO_ABBR.get(o["name"],"")
                            if ab == h: hl.append(o["price"])
                            elif ab == a: al.append(o["price"])
            if hl and al:
                out[f"{h}_{a}"] = {"home_odds":int(np.mean(hl)),"away_odds":int(np.mean(al))}
        print(f"  ✅ Odds for {len(out)} games")
        return out
    except Exception as e:
        print(f"  ⚠️  Failed: {e}"); return {}


# ── Schedule ──────────────────────────────────────────────────────────────────

def get_todays_games():
    print("📅 Fetching schedule...")
    today = datetime.now().strftime("%m/%d/%Y")
    sched = statsapi.schedule(date=today)
    games = []
    for g in sched:
        if g.get("status") in ("Final","Game Over","Completed Early"): continue
        hn = g.get("home_name",""); an = g.get("away_name","")
        ha = TEAM_NAME_TO_ABBR.get(hn, hn[:3].upper())
        aa = TEAM_NAME_TO_ABBR.get(an, an[:3].upper())
        try:
            dt   = datetime.strptime(g["game_datetime"],"%Y-%m-%dT%H:%M:%SZ")
            hour = dt.hour - 4
            ampm = "AM" if hour < 12 else "PM"
            hour = hour if hour <= 12 else hour - 12
            hour = 12 if hour == 0 else hour
            tstr = f"{hour}:{dt.strftime('%M')} {ampm} ET"
        except:
            tstr = g.get("game_time","TBD")
        games.append({"home_name":hn,"away_name":an,"home_abbr":ha,"away_abbr":aa,"time":tstr,"game_id":g.get("game_id")})
    print(f"  ✅ {len(games)} games")
    return games


# ── Predict ───────────────────────────────────────────────────────────────────

def american_to_prob(o):
    if o is None: return None
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def calc_ev(prob, odds):
    if odds is None or prob is None: return None
    profit = odds if odds > 0 else 10000/abs(odds)
    return round((prob * profit) - ((1-prob)*100), 1)

def predict_games(games, model, team_stats, pitcher_stats, odds):
    print("🔮 Running predictions...")
    results = []
    for g in games:
        ha = g["home_abbr"]; aa = g["away_abbr"]
        hs = team_stats.get(ha,{}); as_ = team_stats.get(aa,{})
        hp = pitcher_stats.get(ha,{}); ap = pitcher_stats.get(aa,{})
        if not hs or not as_:
            print(f"  ⚠️  No stats for {ha}/{aa}"); continue

        row = {
            "home_rolling_runs_scored": hs.get("rolling_runs_scored",4.5),
            "home_rolling_runs_allowed":hs.get("rolling_runs_allowed",4.5),
            "home_rolling_win_rate":    hs.get("rolling_win_rate",0.5),
            "home_rolling_run_diff":    hs.get("rolling_run_diff",0.0),
            "away_rolling_runs_scored": as_.get("rolling_runs_scored",4.5),
            "away_rolling_runs_allowed":as_.get("rolling_runs_allowed",4.5),
            "away_rolling_win_rate":    as_.get("rolling_win_rate",0.5),
            "away_rolling_run_diff":    as_.get("rolling_run_diff",0.0),
            "win_rate_diff":  hs.get("rolling_win_rate",0.5) - as_.get("rolling_win_rate",0.5),
            "run_diff_diff":  hs.get("rolling_run_diff",0.0) - as_.get("rolling_run_diff",0.0),
            "runs_scored_diff": hs.get("rolling_runs_scored",4.5) - as_.get("rolling_runs_scored",4.5),
            "is_home": 1,
            "home_avg_era":   hp.get("avg_era",4.5),   "home_avg_whip":   hp.get("avg_whip",1.3),
            "home_avg_so9":   hp.get("avg_so9",8.0),   "home_avg_so_per_w":hp.get("avg_so_per_w",2.5),
            "away_avg_era":   ap.get("avg_era",4.5),   "away_avg_whip":   ap.get("avg_whip",1.3),
            "away_avg_so9":   ap.get("avg_so9",8.0),   "away_avg_so_per_w":ap.get("avg_so_per_w",2.5),
            "era_diff":  ap.get("avg_era",4.5)  - hp.get("avg_era",4.5),
            "whip_diff": ap.get("avg_whip",1.3) - hp.get("avg_whip",1.3),
            "so9_diff":  hp.get("avg_so9",8.0)  - ap.get("avg_so9",8.0),
        }

        X         = pd.DataFrame([row])[FEATURE_COLS]
        prob_home = float(model.predict_proba(X)[0][1])
        prob_away = 1 - prob_home

        go        = odds.get(f"{ha}_{aa}", {})
        home_odds = go.get("home_odds"); away_odds = go.get("away_odds")

        if prob_home >= prob_away:
            pick=g["home_name"]; pick_abbr=ha; conf=prob_home; pick_odds=home_odds
        else:
            pick=g["away_name"]; pick_abbr=aa; conf=prob_away; pick_odds=away_odds

        ev = calc_ev(conf, pick_odds)
        results.append({**g,
            "prob_home":prob_home,"prob_away":prob_away,
            "pick":pick,"pick_abbr":pick_abbr,"confidence":conf,
            "home_odds":home_odds,"away_odds":away_odds,"pick_odds":pick_odds,"ev":ev,
            "h_wr":hs.get("rolling_win_rate",0.5),"a_wr":as_.get("rolling_win_rate",0.5),
            "h_rd":hs.get("rolling_run_diff",0.0),"a_rd":as_.get("rolling_run_diff",0.0),
            "h_last5":hs.get("last5",""),"a_last5":as_.get("last5",""),
            "h_last5_w":hs.get("last5_w",0),"h_last5_l":hs.get("last5_l",0),
            "a_last5_w":as_.get("last5_w",0),"a_last5_l":as_.get("last5_l",0),
        })
        print(f"  {ha} vs {aa}  →  {pick_abbr} ({int(conf*100)}%)  EV:{ev}")

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ── Save picks for grading later ──────────────────────────────────────────────

def save_picks(predictions, date_str):
    path = f"raw/picks_{date_str}.json"
    picks = [{
        "home_abbr": g["home_abbr"], "away_abbr": g["away_abbr"],
        "home_name": g["home_name"], "away_name": g["away_name"],
        "pick_abbr": g["pick_abbr"], "pick":      g["pick"],
        "confidence":g["confidence"],"pick_odds": g["pick_odds"],
        "game_id":   g.get("game_id"),
        "result":    None,   # filled in by grade_picks.py
    } for g in predictions]
    with open(path, "w") as f:
        json.dump(picks, f, indent=2)
    print(f"  ✅ Picks saved → {path}")


# ── HTML generation ───────────────────────────────────────────────────────────

def ev_color(ev):
    if ev is None: return "#9ca3af"
    if ev >= 8:  return "#16a34a"
    if ev >= 3:  return "#65a30d"
    if ev >= 0:  return "#d97706"
    return "#dc2626"

def ev_label(ev):
    if ev is None: return "no odds"
    if ev >= 8:  return f"+{ev} excellent"
    if ev >= 3:  return f"+{ev} good"
    if ev >= 0:  return f"+{ev} marginal"
    return f"{ev} skip"

def conf_color(c):
    if c >= 0.70: return "#16a34a"
    if c >= 0.60: return "#65a30d"
    if c >= 0.55: return "#d97706"
    return "#9ca3af"

def odds_str(o):
    if o is None: return "—"
    return f"+{o}" if o > 0 else str(o)

def form_dots(s):
    out = ""
    for c in s:
        clr = "#16a34a" if c == "W" else "#dc2626"
        out += f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{clr};margin-right:2px;"></span>'
    return out

def generate_html(predictions, record, today_str, date_str):
    print("🖥️  Generating HTML...")

    # Record card
    at_str, at_pct, month_str, week_str, yest_str = record_display(record)
    at   = record["all_time"]
    total = at["w"] + at["l"]
    pct_w = at["w"] / total * 100 if total > 0 else 0

    # Confidence breakdown
    bc = record["by_conf"]
    def conf_cell(key, label):
        w = bc.get(key,{}).get("w",0); l = bc.get(key,{}).get("l",0)
        t = w + l
        pct = f"{w/t:.0%}" if t > 0 else "—"
        clr = "#16a34a" if t > 0 and w/t >= 0.60 else ("#65a30d" if t > 0 and w/t >= 0.55 else "#374151")
        return f"""<div style="background:#fff;border-radius:8px;padding:10px 6px;text-align:center;border:0.5px solid #e5e7eb;">
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">{label}</div>
          <div style="font-size:13px;font-weight:500;">{w}–{l}</div>
          <div style="font-size:10px;color:{clr};margin-top:1px;">{pct}</div>
        </div>"""

    # Date nav — only show days we have picks files for
    nav = f'<a href="mlb_predictor.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#111;color:#fff;">Today · {datetime.now().strftime("%b %d")}</a>'
    for date_key, day in sorted(record["daily"].items(), reverse=True)[:5]:
        try:
            dt    = datetime.strptime(date_key, "%Y-%m-%d")
            label = dt.strftime("%b %d")
            w     = day["w"]; l = day["l"]
            wl_clr = "#16a34a" if w > l else ("#dc2626" if l > w else "#9ca3af")
            nav  += f'<a href="history_{date_key}.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#f3f4f6;color:#374151;">{label}<span style="font-size:11px;color:{wl_clr};"> {w}–{l}</span></a>'
        except: pass

    # Game cards
    cards = ""
    for i, g in enumerate(predictions):
        conf     = g["confidence"]
        ph       = int(g["prob_home"]*100)
        pa       = int(g["prob_away"]*100)
        clr      = conf_color(conf)
        ev       = g["ev"]
        ev_clr   = ev_color(ev)
        ev_lbl   = ev_label(ev)
        conf_pct = int(conf*100)
        pick_is_home = g["pick_abbr"] == g["home_abbr"]

        if pick_is_home:
            away_sty="color:#9ca3af;"; home_sty="font-weight:600;"
            bar_l=pa; bar_r=ph; lclr="#93c5fd"; rclr=clr
        else:
            away_sty="font-weight:600;"; home_sty="color:#9ca3af;"
            bar_l=pa; bar_r=ph; lclr=clr; rclr="#93c5fd"

        opp_abbr = g["away_abbr"] if pick_is_home else g["home_abbr"]

        cards += f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle({i})" style="padding:16px 18px;cursor:pointer;user-select:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:11px;color:#9ca3af;">{g['time']}</span>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:13px;font-weight:500;color:{clr};">{conf_pct}% confidence</span>
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;{away_sty}">{g['away_name']}</div>
        <div style="font-size:11px;color:#9ca3af;">Away · {g['a_last5_w']}–{g['a_last5_l']} last 5</div>
        <div style="margin-top:3px;">{form_dots(g['a_last5'])}</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{odds_str(g['away_odds'])}</div>
      </div>
      <div style="text-align:center;padding:0 10px;flex-shrink:0;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Win prob</div>
        <div style="display:flex;align-items:center;gap:4px;">
          <span style="font-size:26px;font-weight:500;color:{'#bbb' if pick_is_home else '#111'};">{pa}%</span>
          <span style="font-size:14px;color:#d1d5db;">–</span>
          <span style="font-size:26px;font-weight:500;color:{'#111' if pick_is_home else '#bbb'};">{ph}%</span>
        </div>
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;{home_sty}">{g['home_name']}</div>
        <div style="font-size:11px;color:#9ca3af;">Home · {g['h_last5_w']}–{g['h_last5_l']} last 5</div>
        <div style="margin-top:3px;text-align:right;">{form_dots(g['h_last5'])}</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{odds_str(g['home_odds'])}</div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{bar_l}%;background:{lclr};border-radius:99px 0 0 99px;"></div>
        <div style="width:{bar_r}%;background:{rclr};border-radius:0 99px 99px 0;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{g['away_abbr']} {pa}%</span>
        <span style="color:#374151;font-weight:500;">Proj: {g['pick']} wins</span>
        <span>{g['home_abbr']} {ph}%</span>
      </div>
    </div>
    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:15px;font-weight:600;">{g['pick']}</span>
            <span style="font-size:13px;font-weight:500;padding:2px 10px;border-radius:20px;background:#f9fafb;color:#374151;">{odds_str(g['pick_odds'])}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</div>
          <div style="font-size:11px;color:#d1d5db;margin-top:4px;" id="hint{i}">tap for details ▾</div>
        </div>
      </div>
    </div>
  </div>
  <div id="body{i}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:6px;">Moneyline pick</div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
      <span style="font-size:19px;font-weight:500;">{g['pick']}</span>
      <span style="font-size:14px;font-weight:500;color:#374151;">{odds_str(g['pick_odds'])}</span>
      <span style="font-size:12px;color:#9ca3af;">book {int(american_to_prob(g['pick_odds'])*100) if g['pick_odds'] else '—'}% → model {conf_pct}%</span>
      <span style="font-size:13px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</span>
    </div>
    <div style="font-size:12px;color:#9ca3af;margin-bottom:10px;">EV = expected value per $100 using model probability vs book price</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:2px;">{g['away_abbr']} ML</div>
        <div style="font-size:14px;font-weight:500;">{odds_str(g['away_odds'])}</div>
        <div style="font-size:11px;color:#9ca3af;">Win rate: {g['a_wr']:.0%} · RD: {g['a_rd']:+.1f}</div>
      </div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:2px;">{g['home_abbr']} ML</div>
        <div style="font-size:14px;font-weight:500;">{odds_str(g['home_odds'])}</div>
        <div style="font-size:11px;color:#9ca3af;">Win rate: {g['h_wr']:.0%} · RD: {g['h_rd']:+.1f}</div>
      </div>
    </div>
    <div style="font-size:11px;color:#9ca3af;background:#f9fafb;border-radius:8px;padding:8px 10px;">Based on last {ROLLING_WINDOW}-game rolling stats + season pitcher averages. Starter not yet included.</div>
    <div style="text-align:center;margin-top:12px;"><span onclick="toggle({i})" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse ▴</span></div>
  </div>
</div>"""

    n = len(predictions)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB AI Predictor</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
  .wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
  @media(max-width:480px){{.wrap{{padding:14px 10px 40px}}}}
</style>
</head>
<body>
<div class="wrap">
  <div style="margin-bottom:20px;">
    <h1 style="font-size:22px;font-weight:500;">⚾ MLB AI Predictor</h1>
    <div style="font-size:13px;color:#9ca3af;margin-top:3px;">{today_str} · {n} games</div>
  </div>

  <div style="background:#f9fafb;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px 18px;margin-bottom:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:12px;">Model record</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;">
      <div style="text-align:center;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">All time</div>
        <div style="font-size:22px;font-weight:500;">{at_str}</div>
        <div style="font-size:11px;color:#16a34a;">{at_pct}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">This month</div>
        <div style="font-size:22px;font-weight:500;">{month_str}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">This week</div>
        <div style="font-size:22px;font-weight:500;">{week_str}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Yesterday</div>
        <div style="font-size:22px;font-weight:500;">{yest_str}</div>
      </div>
    </div>
    <div style="height:5px;background:#e5e7eb;border-radius:99px;overflow:hidden;margin-bottom:4px;">
      <div style="width:{pct_w:.1f}%;height:100%;background:#16a34a;border-radius:99px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-bottom:12px;">
      <span>Hit rate</span><span>Target 65% · Break-even 52.4%</span>
    </div>
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:8px;">Record by confidence</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
      {conf_cell("50","50–59%")}
      {conf_cell("55","60–69%")}
      {conf_cell("60","70–79%")}
      {conf_cell("70","80%+")}
    </div>
  </div>

  <div style="overflow-x:auto;white-space:nowrap;margin-bottom:16px;padding-bottom:4px;">{nav}</div>

  <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;gap:5px;">
    EV: <span style="color:#16a34a;">+8 excellent</span> ·
    <span style="color:#65a30d;">+3 good</span> ·
    <span style="color:#d97706;">marginal</span> ·
    <span style="color:#dc2626;">neg = skip</span>
  </div>

  {cards}

  <div style="margin-top:20px;padding:12px 14px;background:#fffbeb;border-radius:8px;border:0.5px solid #fde68a;">
    <p style="font-size:11px;color:#92400e;line-height:1.6;">AI-generated picks for informational purposes only. Always gamble responsibly.</p>
  </div>
</div>
<script>
function toggle(i){{
  var b=document.getElementById('body'+i);
  var h=document.getElementById('hint'+i);
  var o=b.style.display!=='none';
  b.style.display=o?'none':'block';
  h.innerHTML=o?'tap for details &#9662;':'collapse &#9652;';
}}
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Saved → {OUTPUT_HTML}")

def american_to_prob(o):
    if o is None: return None
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today_str = datetime.now().strftime("%A, %B %d %Y")
    date_str  = datetime.now().strftime("%Y-%m-%d")

    model         = load_model()
    team_stats    = build_current_team_stats()
    pitcher_stats = build_pitcher_stats()
    odds          = fetch_odds()
    games         = get_todays_games()
    record        = load_record()

    if not games:
        print("⚠️  No games found today.")
    else:
        predictions = predict_games(games, model, team_stats, pitcher_stats, odds)
        save_picks(predictions, date_str)
        generate_html(predictions, record, today_str, date_str)
        print(f"\n✅ Done! Open {OUTPUT_HTML} in Chrome.")
        print(f"\nRecord: {record['all_time']['w']}–{record['all_time']['l']} all time")