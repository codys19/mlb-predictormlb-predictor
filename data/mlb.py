"""
mlb.py
------
Run every morning. Grades yesterday, predicts today.
No emoji in print statements so Windows Task Scheduler doesn't crash.

RUN:
  python data/mlb.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime, timedelta, timezone
import statsapi
import requests
import json
import os
import glob

ODDS_API_KEY = "YOUR_KEY_HERE"  # paste your key here

MODEL_PATH      = "raw/model.json"
GAMES_PATH      = "raw/game_results.csv"
PITCHER_PATH    = "raw/pitcher_stats.csv"
STARTER_PATH    = "raw/starter_logs.csv"
TEAM_POOL_PATH  = "raw/team_starter_pool.csv"
OUTPUT_HTML     = "mlb_predictor.html"
RECORD_PATH     = "raw/record.json"
ROLLING_WINDOW  = 15

# Base features always used
BASE_FEATURES = [
    "home_rolling_runs_scored","home_rolling_runs_allowed",
    "home_rolling_win_rate","home_rolling_run_diff",
    "away_rolling_runs_scored","away_rolling_runs_allowed",
    "away_rolling_win_rate","away_rolling_run_diff",
    "win_rate_diff","run_diff_diff","runs_scored_diff","is_home",
    "home_avg_era","home_avg_whip","home_avg_so9","home_avg_so_per_w",
    "away_avg_era","away_avg_whip","away_avg_so9","away_avg_so_per_w",
    "era_diff","whip_diff","so9_diff",
]

# Starter features — only used if team_starter_pool.csv exists
STARTER_FEATURES = [
    "home_starter_era","home_starter_whip","home_starter_k_per_9",
    "home_starter_bb_per_9","home_starter_fip_proxy",
    "away_starter_era","away_starter_whip","away_starter_k_per_9",
    "away_starter_bb_per_9","away_starter_fip_proxy",
    "starter_era_diff","starter_whip_diff","starter_k_diff","starter_fip_diff",
]

TEAM_MAP = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CHW",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KCR",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"ATH","Athletics":"ATH",
    "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SDP",
    "Seattle Mariners":"SEA","San Francisco Giants":"SFG","St. Louis Cardinals":"STL",
    "Tampa Bay Rays":"TBR","Texas Rangers":"TEX","Toronto Blue Jays":"TOR",
    "Washington Nationals":"WSN",
}

PITCHER_NAME_MAP = {
    "Arizona":"ARI","Atlanta":"ATL","Baltimore":"BAL","Boston":"BOS",
    "Chicago":"CHC","Cincinnati":"CIN","Cleveland":"CLE","Colorado":"COL",
    "Detroit":"DET","Houston":"HOU","Kansas City":"KCR","Los Angeles":"LAD",
    "Miami":"MIA","Milwaukee":"MIL","Minnesota":"MIN","New York":"NYY",
    "Oakland":"ATH","Athletics":"ATH","Philadelphia":"PHI","Pittsburgh":"PIT",
    "San Diego":"SDP","Seattle":"SEA","San Francisco":"SFG","St. Louis":"STL",
    "Tampa Bay":"TBR","Texas":"TEX","Toronto":"TOR","Washington":"WSN",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def utc_to_et(utc_str):
    try:
        dt  = datetime.strptime(utc_str,"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        et  = dt - timedelta(hours=4)
        h   = et.hour; m=et.strftime("%M")
        ap  = "AM" if h<12 else "PM"
        h12 = h%12; h12=12 if h12==0 else h12
        return f"{h12}:{m} {ap} ET"
    except:
        return "TBD"

def american_to_prob(o):
    if o is None: return None
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def calc_ev(prob,odds):
    if odds is None or prob is None: return None
    profit = odds if odds>0 else 10000/abs(odds)
    return round((prob*profit)-((1-prob)*100),1)

def os_(o):
    if o is None: return "--"
    return f"+{o}" if o>0 else str(o)

def dots(s):
    return "".join([
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{"#16a34a" if c=="W" else "#dc2626"};margin-right:2px;"></span>'
        for c in s
    ])

def cc(c):
    if c>=0.70: return "#16a34a"
    if c>=0.60: return "#65a30d"
    if c>=0.55: return "#d97706"
    return "#9ca3af"

def ec(ev):
    if ev is None: return "#9ca3af"
    if ev>=8:  return "#16a34a"
    if ev>=3:  return "#65a30d"
    if ev>=0:  return "#d97706"
    return "#dc2626"

def el(ev):
    if ev is None: return "no odds"
    if ev>=8:  return f"+{ev} excellent"
    if ev>=3:  return f"+{ev} good"
    if ev>=0:  return f"+{ev} marginal"
    return f"{ev} skip"


# ═══════════════════════════════════════════════════════════════════
# PART 1 — GRADE ALL UNGRADED PAST PICKS
# ═══════════════════════════════════════════════════════════════════

def load_record():
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f: return json.load(f)
    return {"all_time":{"w":0,"l":0},
            "by_conf":{"50":{"w":0,"l":0},"55":{"w":0,"l":0},"60":{"w":0,"l":0},"70":{"w":0,"l":0}},
            "daily":{}}

def save_record(r):
    with open(RECORD_PATH,"w") as f: json.dump(r,f,indent=2)

def conf_bucket(c):
    if c>=0.70: return "70"
    if c>=0.60: return "60"
    if c>=0.55: return "55"
    return "50"

def grade_all_ungraded(record):
    today_str  = datetime.now().strftime("%Y-%m-%d")
    pick_files = sorted(glob.glob("raw/picks_*.json"))
    ungraded   = []
    for path in pick_files:
        basename = os.path.basename(path)
        date_str = basename.replace("picks_","").replace(".json","")
        if date_str == today_str: continue
        if date_str not in record["daily"]:
            ungraded.append((date_str, path))

    if not ungraded:
        print("All past picks already graded")
        return record

    for date_str, path in ungraded:
        record = grade_one_day(record, date_str, path)
    return record

def grade_one_day(record, date_str, path):
    with open(path) as f: picks = json.load(f)

    if all(p.get("result") not in (None,"no_result") for p in picks):
        if date_str not in record["daily"]:
            dw = sum(1 for p in picks if p.get("result")=="W")
            dl = sum(1 for p in picks if p.get("result")=="L")
            record["daily"][date_str]={"w":dw,"l":dl}
            save_record(record)
        return record

    print(f"Grading {date_str}...")
    dt = datetime.strptime(date_str,"%Y-%m-%d")
    try:
        sched = statsapi.schedule(date=dt.strftime("%m/%d/%Y"))
    except Exception as e:
        print(f"  Could not fetch results: {e}"); return record

    results={}
    for g in sched:
        if g.get("status") not in ("Final","Game Over","Completed Early"): continue
        hn=g.get("home_name",""); an=g.get("away_name","")
        ha=TEAM_MAP.get(hn,hn[:3].upper()); aa=TEAM_MAP.get(an,an[:3].upper())
        hs=g.get("home_score",0); as_=g.get("away_score",0)
        results[f"{ha}_{aa}"]={"winner":ha if hs>as_ else aa,
            "home_score":hs,"away_score":as_,"home_name":hn,"away_name":an,
            "home_abbr":ha,"away_abbr":aa}

    day_w=day_l=0; graded=[]
    print("-"*55)
    for pick in picks:
        if pick.get("result") not in (None,"no_result"):
            graded.append(pick)
            if pick["result"]=="W": day_w+=1
            elif pick["result"]=="L": day_l+=1
            continue
        key=f"{pick['home_abbr']}_{pick['away_abbr']}"
        result=results.get(key)
        if not result:
            pick["result"]="no_result"; graded.append(pick); continue
        correct=result["winner"]==pick["pick_abbr"]
        pick.update({"result":"W" if correct else "L","actual_winner":result["winner"],
                     "home_score":result["home_score"],"away_score":result["away_score"]})
        bkt=conf_bucket(pick["confidence"])
        if correct: record["all_time"]["w"]+=1; record["by_conf"][bkt]["w"]+=1; day_w+=1
        else:       record["all_time"]["l"]+=1; record["by_conf"][bkt]["l"]+=1; day_l+=1
        icon="OK" if correct else "XX"
        print(f"  {icon}  {pick['pick_abbr']:<4} | {pick['home_abbr']} {result['home_score']}-{result['away_score']} {pick['away_abbr']} | {result['winner']} ({int(pick['confidence']*100)}%)")
        graded.append(pick)

    record["daily"][date_str]={"w":day_w,"l":day_l}
    save_record(record)
    with open(path,"w") as f: json.dump(graded,f,indent=2)
    at=record["all_time"]; pct=f"{at['w']/(at['w']+at['l']):.1%}" if (at['w']+at['l'])>0 else "--"
    print(f"  {date_str}: {day_w}-{day_l}  |  All time: {at['w']}-{at['l']} ({pct})")
    _save_history_html(graded,date_str,day_w,day_l)
    return record

def _save_history_html(picks,date_str,day_w,day_l):
    dt=datetime.strptime(date_str,"%Y-%m-%d"); title=dt.strftime("%A, %B %d %Y")
    rows=""
    for p in picks:
        if p.get("result") in ("no_result",None): continue
        ok=p["result"]=="W"; clr="#16a34a" if ok else "#dc2626"; icon="WIN" if ok else "LOSS"
        hp=p.get("home_pitcher","TBD"); ap=p.get("away_pitcher","TBD")
        rows+=f"""<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:14px;font-weight:600;color:{clr};">[{icon}] {p['pick']}</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:2px;">{p['home_name']} vs {p['away_name']}</div>
      <div style="font-size:12px;color:#374151;margin-top:2px;">Final: {p.get('home_score','')}–{p.get('away_score','')} · Winner: {p.get('actual_winner','?')}</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:2px;">SP: {ap} vs {hp}</div>
    </div>
    <div style="font-size:13px;font-weight:500;color:#374151;">{int(p['confidence']*100)}%</div>
  </div>
</div>"""
    wlc="#16a34a" if day_w>=day_l else "#dc2626"
    total=day_w+day_l; pct=f"{day_w/total:.0%}" if total>0 else "--"
    html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB Picks - {date_str}</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}.wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}</style>
</head><body><div class="wrap">
  <div style="margin-bottom:20px;">
    <a href="mlb_predictor.html" style="font-size:12px;color:#9ca3af;text-decoration:none;">Back to today</a>
    <h1 style="font-size:22px;font-weight:500;margin-top:8px;">MLB {title}</h1>
    <div style="font-size:18px;font-weight:600;color:{wlc};margin-top:4px;">{day_w}–{day_l} · {pct}</div>
  </div>{rows}
</div></body></html>"""
    out=f"history_{date_str}.html"
    with open(out,"w",encoding="utf-8") as f: f.write(html)
    print(f"  History saved -> {out}")


# ═══════════════════════════════════════════════════════════════════
# PART 2 — BUILD DATA
# ═══════════════════════════════════════════════════════════════════

def load_model():
    print("\nLoading model...")
    m=XGBClassifier(); m.load_model(MODEL_PATH)
    # Detect which features the model was trained with
    feature_names = m.get_booster().feature_names
    print(f"  Model uses {len(feature_names)} features")
    return m, feature_names

def build_team_stats():
    print("Building team rolling stats...")
    games=pd.read_csv(GAMES_PATH)
    games["home_away"]=games["home_away"].astype(str).str.strip().replace({"@":"AWAY","Home":"HOME"})
    games["result"]=games["result"].astype(str).str.strip().str.split("-").str[0].str.upper()
    SEASONS=[2021,2022,2023,2024,2025,2026]
    def assign_season(grp):
        idx=np.minimum(np.arange(len(grp))//162,len(SEASONS)-1)
        grp=grp.copy(); grp["season"]=[SEASONS[i] for i in idx]; return grp
    games=games.groupby("team",group_keys=False).apply(assign_season,include_groups=True)
    games=games.sort_values(["season","team"]).reset_index(drop=True)
    games["game_index"]=games.groupby(["team","season"]).cumcount()
    games["date"]=(pd.to_datetime(games["season"].astype(str)+"-01-01")
                   +pd.to_timedelta(games["game_index"],unit="D"))
    games["runs_scored"]=pd.to_numeric(games["runs_scored"],errors="coerce")
    games["runs_allowed"]=pd.to_numeric(games["runs_allowed"],errors="coerce")
    games["result_binary"]=(games["result"]=="W").astype(int)
    games["team"]=games["team"].str.upper().str.strip().replace("OAK","ATH")
    games["opponent"]=games["opponent"].str.upper().str.strip().replace("OAK","ATH")
    out={}
    for team,grp in games.groupby("team"):
        grp=grp.sort_values("date"); recent=grp.tail(ROLLING_WINDOW); last5=grp.tail(5)
        rs=recent["runs_scored"].mean(); ra=recent["runs_allowed"].mean()
        wr=recent["result_binary"].mean()
        out[team]={"rolling_runs_scored":rs,"rolling_runs_allowed":ra,
                   "rolling_win_rate":wr,"rolling_run_diff":rs-ra,
                   "last5":"".join(["W" if r==1 else "L" for r in last5["result_binary"]]),
                   "last5_w":int(last5["result_binary"].sum()),
                   "last5_l":int(5-last5["result_binary"].sum())}
    print(f"  {len(out)} teams")
    return out

def build_pitcher_stats():
    print("Loading pitcher stats...")
    pitchers=pd.read_csv(PITCHER_PATH)
    def norm(raw):
        parts=[p.strip() for p in str(raw).split(",")]
        return PITCHER_NAME_MAP.get(parts[-1],parts[-1].upper()[:3])
    pitchers["team_abbr"]=pitchers["tm"].apply(norm)
    cols=[c for c in ["era","whip","so9","so_per_w"] if c in pitchers.columns]
    pitchers["season"]=pitchers["season"].astype(int)
    latest=pitchers.sort_values("season").groupby("team_abbr").last().reset_index()
    return {row["team_abbr"]:{f"avg_{c}":row.get(c,np.nan) for c in cols}
            for _,row in latest.iterrows()}

def build_starter_pool():
    """Load team starter pool for use in predictions."""
    if not os.path.exists(TEAM_POOL_PATH):
        return None
    pool=pd.read_csv(TEAM_POOL_PATH)
    pool["season"]=pool["season"].astype(int)
    # Build lookup: team_abbr+season -> starter stats dict
    starter_stats={}
    starter_cols=[c for c in pool.columns if c.startswith("team_avg_starter_")]
    for _,row in pool.iterrows():
        key=f"{row['team_abbr']}_{int(row['season'])}"
        starter_stats[key]={
            c.replace("team_avg_starter_","starter_"): row[c]
            for c in starter_cols
        }
    print(f"  Starter pool loaded: {len(starter_stats)} team-seasons")
    return starter_stats

def fetch_probable_starters():
    print("Fetching probable starters...")
    today=datetime.now().strftime("%m/%d/%Y")
    sched=statsapi.schedule(date=today)
    starters={}
    for g in sched:
        hn=g.get("home_name",""); an=g.get("away_name","")
        ha=TEAM_MAP.get(hn,hn[:3].upper()); aa=TEAM_MAP.get(an,an[:3].upper())
        hp=g.get("home_probable_pitcher","TBD") or "TBD"
        ap=g.get("away_probable_pitcher","TBD") or "TBD"
        def clean(n):
            if not n or n=="TBD": return "TBD"
            if "," in n:
                p=n.split(","); return f"{p[1].strip()} {p[0].strip()}"
            return n
        starters[f"{ha}_{aa}"]={"home_pitcher":clean(hp),"away_pitcher":clean(ap)}
    filled=len([k for k,v in starters.items() if v["home_pitcher"]!="TBD"])
    print(f"  Starters for {filled}/{len(starters)} games")
    return starters

def fetch_odds():
    if ODDS_API_KEY=="YOUR_KEY_HERE":
        print("No odds key set -- skipping"); return {}
    print("Fetching odds...")
    try:
        r=requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={"apiKey":ODDS_API_KEY,"regions":"us","markets":"h2h",
                    "oddsFormat":"american","bookmakers":"draftkings,fanduel,betmgm"},timeout=10)
        out={}
        for game in r.json():
            h=TEAM_MAP.get(game.get("home_team",""),""); a=TEAM_MAP.get(game.get("away_team",""),"")
            if not h or not a: continue
            hl,al=[],[]
            for bk in game.get("bookmakers",[]):
                for mkt in bk.get("markets",[]):
                    if mkt["key"]=="h2h":
                        for o in mkt["outcomes"]:
                            ab=TEAM_MAP.get(o["name"],"")
                            if ab==h: hl.append(o["price"])
                            elif ab==a: al.append(o["price"])
            if hl and al: out[f"{h}_{a}"]={"home_odds":int(np.mean(hl)),"away_odds":int(np.mean(al))}
        print(f"  Odds for {len(out)} games"); return out
    except Exception as e:
        print(f"  Failed: {e}"); return {}

def get_schedule():
    print("Fetching today's schedule...")
    today=datetime.now().strftime("%m/%d/%Y")
    sched=statsapi.schedule(date=today)
    games=[]
    for g in sched:
        if g.get("status") in ("Final","Game Over","Completed Early"): continue
        hn=g.get("home_name",""); an=g.get("away_name","")
        ha=TEAM_MAP.get(hn,hn[:3].upper()); aa=TEAM_MAP.get(an,an[:3].upper())
        tstr=utc_to_et(g.get("game_datetime",""))
        games.append({"home_name":hn,"away_name":an,"home_abbr":ha,"away_abbr":aa,
                      "time":tstr,"game_id":g.get("game_id")})
    print(f"  {len(games)} games today")
    return games


# ═══════════════════════════════════════════════════════════════════
# PART 3 — PREDICT
# ═══════════════════════════════════════════════════════════════════

def run_predictions(games,model,feature_names,ts,ps,odds,starters,starter_pool):
    print("Running predictions...")
    current_season=datetime.now().year
    results=[]

    for g in games:
        ha=g["home_abbr"]; aa=g["away_abbr"]
        hs=ts.get(ha,{}); as_=ts.get(aa,{})
        hp=ps.get(ha,{}); ap=ps.get(aa,{})
        if not hs or not as_: print(f"  No stats for {ha}/{aa}"); continue

        # Base row
        row={
            "home_rolling_runs_scored": hs.get("rolling_runs_scored",4.5),
            "home_rolling_runs_allowed":hs.get("rolling_runs_allowed",4.5),
            "home_rolling_win_rate":    hs.get("rolling_win_rate",0.5),
            "home_rolling_run_diff":    hs.get("rolling_run_diff",0.0),
            "away_rolling_runs_scored": as_.get("rolling_runs_scored",4.5),
            "away_rolling_runs_allowed":as_.get("rolling_runs_allowed",4.5),
            "away_rolling_win_rate":    as_.get("rolling_win_rate",0.5),
            "away_rolling_run_diff":    as_.get("rolling_run_diff",0.0),
            "win_rate_diff":  hs.get("rolling_win_rate",0.5)-as_.get("rolling_win_rate",0.5),
            "run_diff_diff":  hs.get("rolling_run_diff",0.0)-as_.get("rolling_run_diff",0.0),
            "runs_scored_diff":hs.get("rolling_runs_scored",4.5)-as_.get("rolling_runs_scored",4.5),
            "is_home":1,
            "home_avg_era":  hp.get("avg_era",4.5),  "home_avg_whip": hp.get("avg_whip",1.3),
            "home_avg_so9":  hp.get("avg_so9",8.0),  "home_avg_so_per_w":hp.get("avg_so_per_w",2.5),
            "away_avg_era":  ap.get("avg_era",4.5),  "away_avg_whip": ap.get("avg_whip",1.3),
            "away_avg_so9":  ap.get("avg_so9",8.0),  "away_avg_so_per_w":ap.get("avg_so_per_w",2.5),
            "era_diff":  ap.get("avg_era",4.5) -hp.get("avg_era",4.5),
            "whip_diff": ap.get("avg_whip",1.3)-hp.get("avg_whip",1.3),
            "so9_diff":  hp.get("avg_so9",8.0) -ap.get("avg_so9",8.0),
        }

        # Add starter pool features if model expects them
        if starter_pool is not None and any("starter_" in f for f in feature_names):
            h_key=f"{ha}_{current_season}"; a_key=f"{aa}_{current_season}"
            # Fall back to previous season if current not available
            h_sp=starter_pool.get(h_key, starter_pool.get(f"{ha}_{current_season-1}",{}))
            a_sp=starter_pool.get(a_key, starter_pool.get(f"{aa}_{current_season-1}",{}))

            for k,v in h_sp.items():
                row[f"home_{k}"] = v
            for k,v in a_sp.items():
                row[f"away_{k}"] = v

            # Compute diff features
            h_era = h_sp.get("starter_era", row["home_avg_era"])
            a_era = a_sp.get("starter_era", row["away_avg_era"])
            h_whip= h_sp.get("starter_whip", row["home_avg_whip"])
            a_whip= a_sp.get("starter_whip", row["away_avg_whip"])
            h_k9  = h_sp.get("starter_k_per_9", row["home_avg_so9"])
            a_k9  = a_sp.get("starter_k_per_9", row["away_avg_so9"])
            h_fip = h_sp.get("starter_fip_proxy", row["home_avg_era"])
            a_fip = a_sp.get("starter_fip_proxy", row["away_avg_era"])

            row["starter_era_diff"]  = a_era  - h_era
            row["starter_whip_diff"] = a_whip - h_whip
            row["starter_k_diff"]    = h_k9   - a_k9
            row["starter_fip_diff"]  = a_fip  - h_fip

        # Build feature vector using exactly the features the model expects
        X = pd.DataFrame([{f: row.get(f, 0.0) for f in feature_names}])

        prob_home=float(model.predict_proba(X)[0][1]); prob_away=1-prob_home
        go=odds.get(f"{ha}_{aa}",{}); home_odds=go.get("home_odds"); away_odds=go.get("away_odds")
        st=starters.get(f"{ha}_{aa}",{}); home_p=st.get("home_pitcher","TBD"); away_p=st.get("away_pitcher","TBD")

        if prob_home>=prob_away:
            pick=g["home_name"]; pick_abbr=ha; conf=prob_home; pick_odds=home_odds
        else:
            pick=g["away_name"]; pick_abbr=aa; conf=prob_away; pick_odds=away_odds

        ev=calc_ev(conf,pick_odds)
        results.append({**g,
            "prob_home":prob_home,"prob_away":prob_away,
            "pick":pick,"pick_abbr":pick_abbr,"confidence":conf,
            "home_odds":home_odds,"away_odds":away_odds,"pick_odds":pick_odds,"ev":ev,
            "h_wr":hs.get("rolling_win_rate",0.5),"a_wr":as_.get("rolling_win_rate",0.5),
            "h_rd":hs.get("rolling_run_diff",0.0),"a_rd":as_.get("rolling_run_diff",0.0),
            "h_last5":hs.get("last5",""),"a_last5":as_.get("last5",""),
            "h_last5_w":hs.get("last5_w",0),"h_last5_l":hs.get("last5_l",0),
            "a_last5_w":as_.get("last5_w",0),"a_last5_l":as_.get("last5_l",0),
            "home_pitcher":home_p,"away_pitcher":away_p,
        })
        print(f"  {ha} vs {aa}  ->  {pick_abbr} ({int(conf*100)}%)  {away_p} vs {home_p}")

    results.sort(key=lambda x: x["confidence"],reverse=True)
    return results

def save_todays_picks(predictions,date_str):
    path=f"raw/picks_{date_str}.json"
    if os.path.exists(path):
        print(f"  Picks already saved for {date_str}"); return
    with open(path,"w") as f:
        json.dump([{"home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],
            "home_name":g["home_name"],"away_name":g["away_name"],
            "pick_abbr":g["pick_abbr"],"pick":g["pick"],
            "confidence":g["confidence"],"pick_odds":g.get("pick_odds"),
            "home_pitcher":g.get("home_pitcher","TBD"),
            "away_pitcher":g.get("away_pitcher","TBD"),
            "game_id":g.get("game_id"),"result":None} for g in predictions],f,indent=2)
    print(f"  Picks saved -> {path}")


# ═══════════════════════════════════════════════════════════════════
# PART 4 — HTML
# ═══════════════════════════════════════════════════════════════════

def record_strings(record):
    at=record["all_time"]; total=at["w"]+at["l"]
    at_str=f"{at['w']}-{at['l']}"; at_pct=f"{at['w']/total:.1%}" if total>0 else "--"
    pct_w=at["w"]/total*100 if total>0 else 0
    now=datetime.now(); mw=ml=ww=wl=0
    for ds,d in record["daily"].items():
        try:
            dt=datetime.strptime(ds,"%Y-%m-%d")
            if dt.year==now.year and dt.month==now.month: mw+=d["w"]; ml+=d["l"]
            if (now-dt).days<=7: ww+=d["w"]; wl+=d["l"]
        except: pass
    yest=(now-timedelta(days=1)).strftime("%Y-%m-%d")
    yd=record["daily"].get(yest,{})
    yest_str=f"{yd.get('w',0)}-{yd.get('l',0)}" if yd else "--"
    return at_str,at_pct,f"{mw}-{ml}",f"{ww}-{wl}",yest_str,pct_w

def conf_cell_html(record,key,label):
    bc=record["by_conf"]; w=bc.get(key,{}).get("w",0); l=bc.get(key,{}).get("l",0); t=w+l
    pct=f"{w/t:.0%}" if t>0 else "--"
    clr="#16a34a" if t>0 and w/t>=0.60 else ("#65a30d" if t>0 and w/t>=0.55 else "#374151")
    return f"""<div style="background:#fff;border-radius:8px;padding:10px 6px;text-align:center;border:0.5px solid #e5e7eb;">
      <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">{label}</div>
      <div style="font-size:13px;font-weight:500;">{w}-{l}</div>
      <div style="font-size:10px;color:{clr};margin-top:1px;">{pct}</div></div>"""

def generate_html(predictions,record,today_str,date_str):
    print("Generating HTML...")
    at_str,at_pct,mo_str,wk_str,ye_str,pct_w=record_strings(record)

    nav=f'<a href="mlb_predictor.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#111;color:#fff;">Today - {datetime.now().strftime("%b %d")}</a>'
    for ds,d in sorted(record["daily"].items(),reverse=True)[:5]:
        try:
            lbl=datetime.strptime(ds,"%Y-%m-%d").strftime("%b %d")
            wlc="#16a34a" if d["w"]>d["l"] else ("#dc2626" if d["l"]>d["w"] else "#9ca3af")
            nav+=f'<a href="history_{ds}.html" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;text-decoration:none;margin-right:6px;background:#f3f4f6;color:#374151;">{lbl}<span style="font-size:11px;color:{wlc};"> {d["w"]}-{d["l"]}</span></a>'
        except: pass

    cards=""
    for i,g in enumerate(predictions):
        conf=g["confidence"]; pih=g["pick_abbr"]==g["home_abbr"]
        pick_pct=int(conf*100); clr=cc(conf)
        ev=g["ev"]; ev_clr=ec(ev); ev_lbl=el(ev)
        asty="color:#9ca3af;" if pih else "font-weight:600;"
        hsty="font-weight:600;" if pih else "color:#9ca3af;"
        away_pct=int(g["prob_away"]*100); home_pct=int(g["prob_home"]*100)
        away_bar_clr=clr if not pih else "#93c5fd"
        home_bar_clr=clr if pih else "#93c5fd"
        hp=g.get("home_pitcher","TBD"); ap=g.get("away_pitcher","TBD")
        bk_prob=int((american_to_prob(g["pick_odds"]) or 0)*100)
        if pih:
            prob_display=f'<span style="font-size:13px;color:#9ca3af;">{away_pct}%</span><span style="font-size:11px;color:#d1d5db;margin:0 4px;">-</span><span style="font-size:26px;font-weight:600;color:#111;">{home_pct}%</span>'
        else:
            prob_display=f'<span style="font-size:26px;font-weight:600;color:#111;">{away_pct}%</span><span style="font-size:11px;color:#d1d5db;margin:0 4px;">-</span><span style="font-size:13px;color:#9ca3af;">{home_pct}%</span>'

        cards+=f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle({i})" style="padding:16px 18px;cursor:pointer;user-select:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:11px;color:#9ca3af;">{g['time']}</span>
      <span style="font-size:13px;font-weight:500;color:{clr};">{pick_pct}% confidence</span>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;{asty}">{g['away_name']}</div>
        <div style="font-size:11px;color:#9ca3af;">Away · {g['a_last5_w']}-{g['a_last5_l']} last 5</div>
        <div style="margin-top:3px;">{dots(g['a_last5'])}</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">SP: {ap}</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(g['away_odds'])}</div>
      </div>
      <div style="text-align:center;padding:0 10px;flex-shrink:0;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:6px;">Win prob</div>
        {prob_display}
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;{hsty}">{g['home_name']}</div>
        <div style="font-size:11px;color:#9ca3af;">Home · {g['h_last5_w']}-{g['h_last5_l']} last 5</div>
        <div style="margin-top:3px;text-align:right;">{dots(g['h_last5'])}</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">{hp} SP</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(g['home_odds'])}</div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{away_pct}%;background:{away_bar_clr};border-radius:99px 0 0 99px;"></div>
        <div style="width:{home_pct}%;background:{home_bar_clr};border-radius:0 99px 99px 0;"></div>
      </div>
    </div>
    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:15px;font-weight:600;">{g['pick']}</span>
            <span style="font-size:13px;padding:2px 10px;border-radius:20px;background:#f9fafb;color:#374151;">{os_(g['pick_odds'])}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</div>
          <div style="font-size:11px;color:#d1d5db;margin-top:4px;" id="hint{i}">tap for details</div>
        </div>
      </div>
    </div>
  </div>
  <div id="body{i}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:6px;">Moneyline pick</div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
      <span style="font-size:19px;font-weight:500;">{g['pick']}</span>
      <span style="font-size:14px;font-weight:500;color:#374151;">{os_(g['pick_odds'])}</span>
      <span style="font-size:12px;color:#9ca3af;">book {bk_prob}% vs model {pick_pct}%</span>
      <span style="font-size:13px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</span>
    </div>
    <div style="font-size:12px;color:#9ca3af;margin-bottom:10px;">EV = expected value per $100 using model probability vs book price</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">{g['away_abbr']} (Away)</div>
        <div style="font-size:13px;font-weight:500;">{g['away_name']}</div>
        <div style="font-size:12px;color:#9ca3af;margin-top:2px;">ML: {os_(g['away_odds'])}</div>
        <div style="font-size:12px;color:#9ca3af;">Win rate: {g['a_wr']:.0%} · RD: {g['a_rd']:+.1f}</div>
        <div style="font-size:11px;color:#374151;margin-top:4px;">SP: {ap}</div>
      </div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">{g['home_abbr']} (Home)</div>
        <div style="font-size:13px;font-weight:500;">{g['home_name']}</div>
        <div style="font-size:12px;color:#9ca3af;margin-top:2px;">ML: {os_(g['home_odds'])}</div>
        <div style="font-size:12px;color:#9ca3af;">Win rate: {g['h_wr']:.0%} · RD: {g['h_rd']:+.1f}</div>
        <div style="font-size:11px;color:#374151;margin-top:4px;">SP: {hp}</div>
      </div>
    </div>
    <div style="font-size:11px;color:#9ca3af;background:#f9fafb;border-radius:8px;padding:8px 10px;">
      Based on last {ROLLING_WINDOW}-game rolling stats + starter pool ERA/WHIP/FIP averages.
    </div>
    <div style="text-align:center;margin-top:12px;"><span onclick="toggle({i})" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse</span></div>
  </div>
</div>"""

    n=len(predictions)
    html=f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB AI Predictor</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
.wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
@media(max-width:480px){{.wrap{{padding:14px 10px 40px}}}}</style>
</head><body><div class="wrap">
  <div style="margin-bottom:20px;">
    <h1 style="font-size:22px;font-weight:500;">MLB AI Predictor</h1>
    <div style="font-size:13px;color:#9ca3af;margin-top:3px;">{today_str} · {n} games</div>
  </div>
  <div style="background:#f9fafb;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px 18px;margin-bottom:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:12px;">Model record</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;">
      <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">All time</div><div style="font-size:22px;font-weight:500;">{at_str}</div><div style="font-size:11px;color:#16a34a;">{at_pct}</div></div>
      <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">This month</div><div style="font-size:22px;font-weight:500;">{mo_str}</div></div>
      <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">This week</div><div style="font-size:22px;font-weight:500;">{wk_str}</div></div>
      <div style="text-align:center;"><div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Yesterday</div><div style="font-size:22px;font-weight:500;">{ye_str}</div></div>
    </div>
    <div style="height:5px;background:#e5e7eb;border-radius:99px;overflow:hidden;margin-bottom:4px;">
      <div style="width:{pct_w:.1f}%;height:100%;background:#16a34a;border-radius:99px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-bottom:12px;">
      <span>Hit rate</span><span>Target 65% · Break-even 52.4%</span>
    </div>
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:8px;">Record by confidence</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
      {conf_cell_html(record,"50","50-59%")}{conf_cell_html(record,"55","60-69%")}{conf_cell_html(record,"60","70-79%")}{conf_cell_html(record,"70","80%+")}
    </div>
  </div>
  <div style="overflow-x:auto;white-space:nowrap;margin-bottom:16px;padding-bottom:4px;">{nav}</div>
  <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;gap:5px;">
    EV: <span style="color:#16a34a;">+8 excellent</span> · <span style="color:#65a30d;">+3 good</span> · <span style="color:#d97706;">marginal</span> · <span style="color:#dc2626;">neg = skip</span>
  </div>
  {cards}
  <div style="margin-top:20px;padding:12px 14px;background:#fffbeb;border-radius:8px;border:0.5px solid #fde68a;">
    <p style="font-size:11px;color:#92400e;line-height:1.6;">AI-generated picks for informational purposes only. Always gamble responsibly.</p>
  </div>
</div>
<script>
function toggle(i){{var b=document.getElementById('body'+i);var h=document.getElementById('hint'+i);var o=b.style.display!=='none';b.style.display=o?'none':'block';h.innerHTML=o?'tap for details':'collapse';}}
</script></body></html>"""

    with open(OUTPUT_HTML,"w",encoding="utf-8") as f: f.write(html)
    print(f"  Saved -> {OUTPUT_HTML}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__=="__main__":
    today_str=datetime.now().strftime("%A, %B %d %Y")
    date_str =datetime.now().strftime("%Y-%m-%d")

    record        = load_record()
    record        = grade_all_ungraded(record)

    print()
    model, feature_names = load_model()
    ts            = build_team_stats()
    ps            = build_pitcher_stats()
    starter_pool  = build_starter_pool()
    starters      = fetch_probable_starters()
    odds          = fetch_odds()
    games         = get_schedule()

    if not games:
        print("No games today.")
    else:
        preds = run_predictions(games,model,feature_names,ts,ps,odds,starters,starter_pool)
        save_todays_picks(preds,date_str)
        generate_html(preds,record,today_str,date_str)
        at=record["all_time"]
        print(f"\nDone! Open mlb_predictor.html in Chrome.")
        print(f"Record: {at['w']}-{at['l']} all time")