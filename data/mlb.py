"""
mlb.py
------
Run every morning. Grades yesterday, predicts today.
Tabs: ML / O/U / Props. $10 flat P&L tracking. Reasoning blurbs.

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

ODDS_API_KEY   = os.environ.get("ODDS_API_KEY", "")

MODEL_PATH     = "raw/model.json"
GAMES_PATH     = "raw/game_results.csv"
PITCHER_PATH   = "raw/pitcher_stats.csv"
STARTER_PATH   = "raw/starter_logs.csv"
TEAM_POOL_PATH = "raw/team_starter_pool.csv"
OUTPUT_HTML    = "mlb_predictor.html"
RECORD_PATH    = "raw/record.json"
ROLLING_WINDOW = 15

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


# ── Helpers ───────────────────────────────────────────────────────

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

def calc_pnl(result, odds):
    """$10 flat bet P&L."""
    if odds is None: return 0.0
    if result == "W":
        if odds > 0: return round(10 * odds / 100, 2)
        else: return round(10 * 100 / abs(odds), 2)
    else:
        return -10.0

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

def conf_bucket(c):
    if c>=0.70: return "70"
    if c>=0.60: return "60"
    if c>=0.55: return "55"
    return "50"


# ═══════════════════════════════════════════════════════════════════
# RECORD — load / save / migrate
# ═══════════════════════════════════════════════════════════════════

def _empty_bet_record():
    return {
        "all_time": {"w":0,"l":0},
        "by_conf": {
            "50":{"w":0,"l":0,"pnl":0.0},
            "55":{"w":0,"l":0,"pnl":0.0},
            "60":{"w":0,"l":0,"pnl":0.0},
            "70":{"w":0,"l":0,"pnl":0.0}
        },
        "daily": {}
    }

def load_record():
    if os.path.exists(RECORD_PATH):
        with open(RECORD_PATH) as f: r = json.load(f)
        # Migrate old flat structure → new nested ml/ou/props structure
        if "ml" not in r:
            old = r
            r = {"ml":_empty_bet_record(),"ou":_empty_bet_record(),"props":_empty_bet_record()}
            r["ml"]["all_time"] = old.get("all_time",{"w":0,"l":0})
            r["ml"]["daily"]    = old.get("daily",{})
            for k,v in old.get("by_conf",{}).items():
                r["ml"]["by_conf"][k] = {"w":v.get("w",0),"l":v.get("l",0),"pnl":0.0}
            # Backfill P&L from existing ML picks files
            r = _backfill_ml_pnl(r)
            save_record(r)
        # Ensure all three sections exist
        for bt in ["ml","ou","props"]:
            if bt not in r: r[bt] = _empty_bet_record()
            # Ensure pnl field exists in each bucket
            for bk in r[bt]["by_conf"].values():
                if "pnl" not in bk: bk["pnl"] = 0.0
        return r
    return {"ml":_empty_bet_record(),"ou":_empty_bet_record(),"props":_empty_bet_record()}

def _backfill_ml_pnl(record):
    for path in sorted(glob.glob("raw/picks_*.json")):
        try:
            with open(path) as f: picks = json.load(f)
            for p in picks:
                if p.get("result") not in ("W","L"): continue
                bkt = conf_bucket(p.get("confidence",0.5))
                pnl = calc_pnl(p["result"], p.get("pick_odds"))
                record["ml"]["by_conf"][bkt]["pnl"] = round(
                    record["ml"]["by_conf"][bkt].get("pnl",0) + pnl, 2)
        except: pass
    return record

def save_record(r):
    with open(RECORD_PATH,"w") as f: json.dump(r,f,indent=2)


# ═══════════════════════════════════════════════════════════════════
# PART 1 — GRADE ALL UNGRADED PAST PICKS
# ═══════════════════════════════════════════════════════════════════

def grade_all_ungraded(record):
    today_str  = datetime.now().strftime("%Y-%m-%d")

    # Grade ML picks
    for path in sorted(glob.glob("raw/picks_*.json")):
        date_str = os.path.basename(path).replace("picks_","").replace(".json","")
        if date_str == today_str: continue
        if date_str not in record["ml"]["daily"]:
            record = grade_ml_day(record, date_str, path)

    # Grade O/U picks
    for path in sorted(glob.glob("raw/ou_*.json")):
        date_str = os.path.basename(path).replace("ou_","").replace(".json","")
        if date_str == today_str: continue
        if date_str not in record["ou"]["daily"]:
            record = grade_ou_day(record, date_str, path)

    # Grade Props picks
    for path in sorted(glob.glob("raw/props_*.json")):
        date_str = os.path.basename(path).replace("props_","").replace(".json","")
        if date_str == today_str: continue
        if date_str not in record["props"]["daily"]:
            record = grade_props_day(record, date_str, path)

    return record

def _fetch_day_results(date_str):
    """Returns dict keyed HA_AA with scores, and also {game_id: {home_score, away_score, ha, aa}}."""
    dt = datetime.strptime(date_str,"%Y-%m-%d")
    try:
        sched = statsapi.schedule(date=dt.strftime("%m/%d/%Y"))
    except Exception as e:
        print(f"  Could not fetch results: {e}"); return {}, {}
    by_teams = {}; by_gameid = {}
    for g in sched:
        if g.get("status") not in ("Final","Game Over","Completed Early"): continue
        hn=g.get("home_name",""); an=g.get("away_name","")
        ha=TEAM_MAP.get(hn,hn[:3].upper()); aa=TEAM_MAP.get(an,an[:3].upper())
        hs=g.get("home_score",0); as_=g.get("away_score",0)
        gid=g.get("game_id")
        entry={"winner":ha if hs>as_ else aa,"home_score":hs,"away_score":as_,
               "home_name":hn,"away_name":an,"home_abbr":ha,"away_abbr":aa,
               "total":hs+as_,"game_id":gid}
        by_teams[f"{ha}_{aa}"] = entry
        if gid: by_gameid[str(gid)] = entry
    return by_teams, by_gameid

def grade_ml_day(record, date_str, path):
    with open(path) as f: picks = json.load(f)
    if all(p.get("result") not in (None,"no_result") for p in picks):
        if date_str not in record["ml"]["daily"]:
            dw=sum(1 for p in picks if p.get("result")=="W")
            dl=sum(1 for p in picks if p.get("result")=="L")
            record["ml"]["daily"][date_str]={"w":dw,"l":dl}
            save_record(record)
        return record

    print(f"Grading ML {date_str}...")
    by_teams, _ = _fetch_day_results(date_str)
    day_w=day_l=0; graded=[]
    print("-"*55)
    for pick in picks:
        if pick.get("result") not in (None,"no_result"):
            graded.append(pick)
            if pick["result"]=="W": day_w+=1
            elif pick["result"]=="L": day_l+=1
            continue
        key=f"{pick['home_abbr']}_{pick['away_abbr']}"
        result=by_teams.get(key)
        if not result:
            pick["result"]="no_result"; graded.append(pick); continue
        correct=result["winner"]==pick["pick_abbr"]
        pick.update({"result":"W" if correct else "L","actual_winner":result["winner"],
                     "home_score":result["home_score"],"away_score":result["away_score"]})
        bkt=conf_bucket(pick["confidence"])
        pnl=calc_pnl(pick["result"],pick.get("pick_odds"))
        if correct:
            record["ml"]["all_time"]["w"]+=1
            record["ml"]["by_conf"][bkt]["w"]+=1
            record["ml"]["by_conf"][bkt]["pnl"]=round(record["ml"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_w+=1
        else:
            record["ml"]["all_time"]["l"]+=1
            record["ml"]["by_conf"][bkt]["l"]+=1
            record["ml"]["by_conf"][bkt]["pnl"]=round(record["ml"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_l+=1
        icon="OK" if correct else "XX"
        print(f"  {icon}  {pick['pick_abbr']:<4} | {pick['home_abbr']} {result['home_score']}-{result['away_score']} {pick['away_abbr']} | {result['winner']} ({int(pick['confidence']*100)}%)")
        graded.append(pick)

    record["ml"]["daily"][date_str]={"w":day_w,"l":day_l}
    save_record(record)
    with open(path,"w") as f: json.dump(graded,f,indent=2)
    at=record["ml"]["all_time"]; tot=at["w"]+at["l"]
    pct=f"{at['w']/tot:.1%}" if tot>0 else "--"
    print(f"  ML {date_str}: {day_w}-{day_l}  |  All time: {at['w']}-{at['l']} ({pct})")
    _save_history_html(graded, date_str, day_w, day_l)
    return record

def grade_ou_day(record, date_str, path):
    with open(path) as f: picks = json.load(f)
    if all(p.get("result") not in (None,"no_result") for p in picks):
        if date_str not in record["ou"]["daily"]:
            dw=sum(1 for p in picks if p.get("result")=="W")
            dl=sum(1 for p in picks if p.get("result")=="L")
            record["ou"]["daily"][date_str]={"w":dw,"l":dl}
            save_record(record)
        return record

    print(f"Grading O/U {date_str}...")
    by_teams, _ = _fetch_day_results(date_str)
    day_w=day_l=0; graded=[]
    for pick in picks:
        if pick.get("result") not in (None,"no_result"):
            graded.append(pick)
            if pick["result"]=="W": day_w+=1
            elif pick["result"]=="L": day_l+=1
            continue
        key=f"{pick['home_abbr']}_{pick['away_abbr']}"
        result=by_teams.get(key)
        if not result:
            pick["result"]="no_result"; graded.append(pick); continue
        actual_total=result["total"]; line=pick["line"]
        if actual_total==line:
            pick["result"]="push"; graded.append(pick); continue
        actual_ou="over" if actual_total>line else "under"
        correct=(actual_ou==pick["pick"].lower())
        pick.update({"result":"W" if correct else "L","actual_total":actual_total})
        bkt=conf_bucket(pick["confidence"])
        pnl=calc_pnl(pick["result"],pick.get("pick_odds"))
        if correct:
            record["ou"]["all_time"]["w"]+=1
            record["ou"]["by_conf"][bkt]["w"]+=1
            record["ou"]["by_conf"][bkt]["pnl"]=round(record["ou"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_w+=1
        else:
            record["ou"]["all_time"]["l"]+=1
            record["ou"]["by_conf"][bkt]["l"]+=1
            record["ou"]["by_conf"][bkt]["pnl"]=round(record["ou"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_l+=1
        icon="OK" if correct else "XX"
        print(f"  {icon}  {pick['pick']} {line} | Actual: {actual_total} | {pick['home_abbr']} vs {pick['away_abbr']}")
        graded.append(pick)

    record["ou"]["daily"][date_str]={"w":day_w,"l":day_l}
    save_record(record)
    with open(path,"w") as f: json.dump(graded,f,indent=2)
    print(f"  O/U {date_str}: {day_w}-{day_l}")
    return record

def grade_props_day(record, date_str, path):
    with open(path) as f: picks = json.load(f)
    if all(p.get("result") not in (None,"no_result") for p in picks):
        if date_str not in record["props"]["daily"]:
            dw=sum(1 for p in picks if p.get("result")=="W")
            dl=sum(1 for p in picks if p.get("result")=="L")
            record["props"]["daily"][date_str]={"w":dw,"l":dl}
            save_record(record)
        return record

    print(f"Grading Props {date_str}...")
    _, by_gameid = _fetch_day_results(date_str)
    day_w=day_l=0; graded=[]
    for pick in picks:
        if pick.get("result") not in (None,"no_result"):
            graded.append(pick)
            if pick["result"]=="W": day_w+=1
            elif pick["result"]=="L": day_l+=1
            continue
        # Try to get pitcher Ks from box score
        gid=str(pick.get("game_id",""))
        actual_k=_fetch_pitcher_ks(gid, pick.get("pitcher",""), pick.get("team_abbr",""))
        if actual_k is None:
            pick["result"]="no_result"; graded.append(pick); continue
        line=pick["line"]
        if actual_k==line:
            pick["result"]="push"; graded.append(pick); continue
        actual_ou="over" if actual_k>line else "under"
        correct=(actual_ou==pick["pick"].lower())
        pick.update({"result":"W" if correct else "L","actual_k":actual_k})
        bkt=conf_bucket(pick["confidence"])
        pnl=calc_pnl(pick["result"],pick.get("pick_odds"))
        if correct:
            record["props"]["all_time"]["w"]+=1
            record["props"]["by_conf"][bkt]["w"]+=1
            record["props"]["by_conf"][bkt]["pnl"]=round(record["props"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_w+=1
        else:
            record["props"]["all_time"]["l"]+=1
            record["props"]["by_conf"][bkt]["l"]+=1
            record["props"]["by_conf"][bkt]["pnl"]=round(record["props"]["by_conf"][bkt]["pnl"]+pnl,2)
            day_l+=1
        icon="OK" if correct else "XX"
        print(f"  {icon}  {pick['pitcher']} K {pick['pick']} {line} | Actual: {actual_k}")
        graded.append(pick)

    record["props"]["daily"][date_str]={"w":day_w,"l":day_l}
    save_record(record)
    with open(path,"w") as f: json.dump(graded,f,indent=2)
    print(f"  Props {date_str}: {day_w}-{day_l}")
    return record

def _fetch_pitcher_ks(game_id, pitcher_name, team_abbr):
    """Fetch actual pitcher Ks from MLB Stats API box score."""
    if not game_id or game_id=="None": return None
    try:
        box = statsapi.boxscore_data(game_id)
        for side in ["home","away"]:
            pitchers = box.get(side,{}).get("pitchers",[])
            for pid in pitchers:
                pdata = box.get(side,{}).get("players",{}).get(f"ID{pid}",{})
                pname = pdata.get("person",{}).get("fullName","")
                stats = pdata.get("stats",{}).get("pitching",{})
                ks = stats.get("strikeOuts")
                if ks is not None:
                    last = pname.split()[-1].lower() if pname else ""
                    search = pitcher_name.split()[-1].lower() if pitcher_name else ""
                    if search and last and (search in last or last in search):
                        return int(ks)
    except: pass
    return None

def _save_history_html(picks, date_str, day_w, day_l):
    dt    = datetime.strptime(date_str, "%Y-%m-%d")
    title = dt.strftime("%A, %B %d %Y")
    cards = ""

    for i, p in enumerate(picks):
        result = p.get("result")
        if result in ("no_result", None): continue

        ok       = result == "W"
        conf     = p["confidence"]
        pick_pct = int(conf * 100)
        clr      = cc(conf)
        pick_tm  = p["pick"]
        pick_a   = p.get("pick_abbr", "")
        home_n   = p.get("home_name", ""); home_a = p.get("home_abbr", "")
        away_n   = p.get("away_name", ""); away_a = p.get("away_abbr", "")
        hp       = p.get("home_pitcher", "TBD")
        ap       = p.get("away_pitcher", "TBD")
        hs       = p.get("home_score", ""); as_ = p.get("away_score", "")
        home_odds = p.get("home_odds"); away_odds = p.get("away_odds")
        pick_odds = p.get("pick_odds")

        pih      = pick_a == home_a
        asty     = "color:#9ca3af;" if pih else "font-weight:600;"
        hsty     = "font-weight:600;" if pih else "color:#9ca3af;"
        away_pct = int((1 - conf) * 100) if pih else int(conf * 100)
        home_pct = 100 - away_pct
        away_bar = clr if not pih else "#93c5fd"
        home_bar = clr if pih else "#93c5fd"

        # WIN/LOSS badge
        if ok:
            badge = f'<span style="font-size:11px;font-weight:600;color:#16a34a;background:#dcfce7;padding:3px 9px;border-radius:20px;">✓ WIN</span>'
        else:
            badge = f'<span style="font-size:11px;font-weight:600;color:#dc2626;background:#fee2e2;padding:3px 9px;border-radius:20px;">✗ LOSS</span>'

        # Final score display
        if hs != "" and as_ != "":
            score_line = f'<span style="font-size:11px;color:#374151;font-weight:500;">Final: {away_a} {as_}–{hs} {home_a}</span>'
        else:
            score_line = ""

        odds_str = os_(pick_odds)

        cards += f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div style="padding:16px 18px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px;">
      <div style="display:flex;align-items:center;gap:8px;">{score_line}</div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:13px;font-weight:500;color:{clr};">{pick_pct}% conf</span>
        {badge}
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;{asty}">{away_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Away</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">SP: {ap}</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(away_odds)}</div>
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;{hsty}">{home_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Home</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">{hp} SP</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(home_odds)}</div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{away_pct}%;background:{away_bar};border-radius:99px 0 0 99px;"></div>
        <div style="width:{home_pct}%;background:{home_bar};border-radius:0 99px 99px 0;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{away_a} {away_pct}%</span>
        <span style="color:#374151;font-weight:500;">Proj: {pick_tm} wins</span>
        <span>{home_a} {home_pct}%</span>
      </div>
    </div>
    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:15px;font-weight:600;">{pick_tm}</span>
            <span style="font-size:13px;padding:2px 10px;border-radius:20px;background:#f9fafb;color:#374151;">{odds_str}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>"""

    wlc   = "#16a34a" if day_w >= day_l else "#dc2626"
    total = day_w + day_l
    pct   = f"{day_w/total:.0%}" if total > 0 else "--"
    if not cards:
        cards = '<div style="padding:32px;text-align:center;color:#9ca3af;font-size:13px;">No graded picks for this day.</div>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB Picks - {date_str}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
  .wrap{{max-width:680px;margin:0 auto;padding:20px 14px 60px}}
  a{{text-decoration:none}}
</style>
</head><body><div class="wrap">
  <div style="margin-bottom:20px;">
    <a href="mlb_predictor.html" style="font-size:12px;color:#9ca3af;">← Back to today</a>
    <h1 style="font-size:22px;font-weight:500;margin-top:10px;">⚾ MLB · {title}</h1>
    <div style="font-size:24px;font-weight:600;color:{wlc};margin-top:6px;">{day_w}–{day_l} <span style="font-size:15px;font-weight:400;color:#9ca3af;">· {pct}</span></div>
  </div>
  {cards}
  <script>
  function toggle(id){{
    var b=document.getElementById('body_'+id);
    var h=document.getElementById('hint_'+id);
    if(!b)return;
    var o=b.style.display!=='none';
    b.style.display=o?'none':'block';
    if(h) h.innerHTML=o?'tap for details':'collapse';
  }}
  </script>
</div></body></html>"""

    out = f"history_{date_str}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  History saved -> {out}")


# ═══════════════════════════════════════════════════════════════════
# PART 2 — BUILD DATA
# ═══════════════════════════════════════════════════════════════════

def load_model():
    print("\nLoading model...")
    m=XGBClassifier(); m.load_model(MODEL_PATH)
    feature_names = m.get_booster().feature_names
    print(f"  Model uses {len(feature_names)} features")
    return m, feature_names

def build_team_stats():
    print("Building team rolling stats...")
    games=pd.read_csv(GAMES_PATH)
    games["home_away"]=games["home_away"].astype(str).str.strip().replace({"@":"AWAY","Home":"HOME"})
    games["result"]=games["result"].astype(str).str.strip().str.split("-").str[0].str.upper()
    SEASONS=[2021,2022,2023,2024,2025,2026]
    result_frames = []
    for team, grp in games.groupby("team"):
        grp = grp.copy()
        idx = np.minimum(np.arange(len(grp))//162, len(SEASONS)-1)
        grp["season"] = [SEASONS[i] for i in idx]
        result_frames.append(grp)
    games = pd.concat(result_frames).reset_index(drop=True)
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
    if not os.path.exists(TEAM_POOL_PATH): return None
    pool=pd.read_csv(TEAM_POOL_PATH)
    pool["season"]=pool["season"].astype(int)
    starter_stats={}
    starter_cols=[c for c in pool.columns if c.startswith("team_avg_starter_")]
    for _,row in pool.iterrows():
        key=f"{row['team_abbr']}_{int(row['season'])}"
        starter_stats[key]={c.replace("team_avg_starter_","starter_"):row[c] for c in starter_cols}
    print(f"  Starter pool loaded: {len(starter_stats)} team-seasons")
    return starter_stats

def build_starter_stats_by_name():
    """
    Build a name-keyed lookup for individual pitcher season stats.
    Used by run_predictions() and predict_props() to replace team rotation
    averages with the actual starting pitcher's ERA/WHIP/K9.

    Returns: {pitcher_name: {era, whip, k_per_9, bb_per_9, fip_proxy}}
    Also stores {"__lastname_index__": {last_name: [full_name, ...]}} for fuzzy matching.
    """
    if not os.path.exists(STARTER_PATH): return {}
    print("Loading individual starter stats by name...")
    df=pd.read_csv(STARTER_PATH)
    df["season"]=df["season"].astype(int)
    if "gs" in df.columns:
        df=df[df["gs"]>=3].copy()
    df=df.sort_values("season")
    latest=df.groupby("name").last().reset_index()
    def sf(val,default):
        try: v=float(val or default); return v if not np.isnan(v) and v>0 else default
        except: return default
    stats_by_name={}
    lastname_index={}
    for _,row in latest.iterrows():
        name=str(row.get("name","")).strip()
        if not name or name=="nan": continue
        stats_by_name[name]={
            "era":      sf(row.get("era"),      4.50),
            "whip":     sf(row.get("whip"),     1.30),
            "k_per_9":  sf(row.get("k_per_9"),  8.00),
            "bb_per_9": sf(row.get("bb_per_9"), 3.00),
            "fip_proxy":sf(row.get("fip_proxy"),4.50),
        }
        last=name.split()[-1].lower()
        lastname_index.setdefault(last,[]).append(name)
    stats_by_name["__lastname_index__"]=lastname_index
    print(f"  Individual starter stats: {len(stats_by_name)-1} pitchers")
    return stats_by_name

def _lookup_pitcher_stats(pitcher_name, stats_by_name, pool_fallback=None):
    """
    Look up an individual pitcher's stats. Priority:
      1. Exact name match
      2. Last-name fuzzy match (unique last names only; uses first initial for ambiguous)
      3. pool_fallback dict
      4. League averages
    """
    LEAGUE_AVG={"era":4.50,"whip":1.30,"k_per_9":8.00,"bb_per_9":3.00,"fip_proxy":4.50}
    if not pitcher_name or pitcher_name=="TBD": return pool_fallback or LEAGUE_AVG
    if pitcher_name in stats_by_name: return stats_by_name[pitcher_name]
    lastname_index=stats_by_name.get("__lastname_index__",{})
    last=pitcher_name.split()[-1].lower()
    candidates=lastname_index.get(last,[])
    if len(candidates)==1: return stats_by_name[candidates[0]]
    if len(candidates)>1:
        first=pitcher_name[0].lower() if pitcher_name else ""
        for cname in candidates:
            if cname[0].lower()==first: return stats_by_name[cname]
        return stats_by_name[candidates[0]]
    return pool_fallback or LEAGUE_AVG

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
    if not ODDS_API_KEY:
        print("No odds key -- skipping"); return {}
    print("Fetching ML odds...")
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
        print(f"  ML odds for {len(out)} games"); return out
    except Exception as e:
        print(f"  Failed: {e}"); return {}

def fetch_totals_odds():
    """Fetch run totals (O/U lines) from Odds API."""
    if not ODDS_API_KEY: return {}
    print("Fetching totals odds...")
    try:
        r=requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/",
            params={"apiKey":ODDS_API_KEY,"regions":"us","markets":"totals",
                    "oddsFormat":"american","bookmakers":"draftkings,fanduel,betmgm"},timeout=10)
        out={}
        for game in r.json():
            h=TEAM_MAP.get(game.get("home_team",""),""); a=TEAM_MAP.get(game.get("away_team",""),"")
            if not h or not a: continue
            over_odds_list=[]; under_odds_list=[]; lines=[]
            for bk in game.get("bookmakers",[]):
                for mkt in bk.get("markets",[]):
                    if mkt["key"]=="totals":
                        for o in mkt["outcomes"]:
                            if o["name"]=="Over":
                                over_odds_list.append(o["price"]); lines.append(o.get("point",8.5))
                            elif o["name"]=="Under":
                                under_odds_list.append(o["price"])
            if lines:
                out[f"{h}_{a}"]={
                    "line": round(np.mean(lines),1),
                    "over_odds": int(np.mean(over_odds_list)) if over_odds_list else -110,
                    "under_odds": int(np.mean(under_odds_list)) if under_odds_list else -110,
                }
        print(f"  Totals for {len(out)} games"); return out
    except Exception as e:
        print(f"  Totals failed: {e}"); return {}

def fetch_props_odds():
    """Fetch pitcher strikeout props from Odds API."""
    if not ODDS_API_KEY: return []
    print("Fetching pitcher K props...")
    try:
        # First get event IDs
        r=requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/events/",
            params={"apiKey":ODDS_API_KEY},timeout=10)
        events=r.json()
        props=[]
        count=0
        for ev in events[:16]:  # limit to save API calls
            h=TEAM_MAP.get(ev.get("home_team",""),""); a=TEAM_MAP.get(ev.get("away_team",""),"")
            if not h or not a: continue
            eid=ev.get("id","")
            try:
                pr=requests.get(f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{eid}/odds",
                    params={"apiKey":ODDS_API_KEY,"regions":"us","markets":"pitcher_strikeouts",
                            "oddsFormat":"american","bookmakers":"draftkings,fanduel"},timeout=10)
                count+=1
                for bk in pr.json().get("bookmakers",[]):
                    for mkt in bk.get("markets",[]):
                        if mkt["key"]=="pitcher_strikeouts":
                            pitcher_lines={}
                            for o in mkt["outcomes"]:
                                pname=o.get("description","")
                                side=o.get("name","")
                                pt=o.get("point",5.5)
                                price=o.get("price",-115)
                                if pname not in pitcher_lines:
                                    pitcher_lines[pname]={"line":pt,"over_odds":None,"under_odds":None,"home_abbr":h,"away_abbr":a,"game_id":eid}
                                if side=="Over": pitcher_lines[pname]["over_odds"]=price
                                elif side=="Under": pitcher_lines[pname]["under_odds"]=price
                            for pname,pd_ in pitcher_lines.items():
                                if pd_["over_odds"] and pd_["under_odds"]:
                                    props.append({**pd_,"pitcher":pname})
                    break  # one bookmaker enough
            except: pass
        print(f"  Props: {len(props)} pitcher lines from {count} games")
        return props
    except Exception as e:
        print(f"  Props fetch failed: {e}"); return []

ODDS_HISTORY_PATH = "raw/odds_history.csv"

def archive_odds(odds, date_str):
    """
    Append today's moneyline odds to raw/odds_history.csv.
    Builds the historical odds dataset needed to eventually train
    implied-probability as a model feature (Priority #3).
    Zero extra API calls — reuses odds already fetched by fetch_odds().
    """
    if not odds:
        return
    rows = []
    for key, o in odds.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        ha, aa = parts
        home_odds = o.get("home_odds")
        away_odds = o.get("away_odds")
        if home_odds is None or away_odds is None:
            continue
        home_impl = round(american_to_prob(home_odds), 4)
        away_impl = round(american_to_prob(away_odds), 4)
        rows.append({
            "date":           date_str,
            "home_abbr":      ha,
            "away_abbr":      aa,
            "home_odds":      home_odds,
            "away_odds":      away_odds,
            "home_impl_prob": home_impl,
            "away_impl_prob": away_impl,
        })
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(ODDS_HISTORY_PATH):
        df_existing = pd.read_csv(ODDS_HISTORY_PATH)
        df_existing = df_existing[df_existing["date"] != date_str]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(ODDS_HISTORY_PATH, index=False)
    print(f"  Archived {len(rows)} odds entries -> {ODDS_HISTORY_PATH}")


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


def fetch_live_bullpen():
    """
    Fetch last 3 days of completed games and compute per-team bullpen load.
    Returns: {team_abbr: {"ip_3d": float, "appearances_3d": int}}

    Uses MLB Stats API boxscores — free, no auth, ~10-20 calls.
    Only runs on game days (skips gracefully if no recent games).
    """
    print("Fetching live bullpen fatigue (last 3 days)...")
    today   = datetime.now()
    results = {}  # {team_abbr: {"ip": 0.0, "apps": 0}}

    def parse_ip(ip_str):
        try:
            ip   = float(ip_str)
            full = int(ip)
            outs = round((ip - full) * 10)
            return round(full + outs / 3, 4)
        except:
            return 0.0

    calls = 0
    for days_back in range(1, 4):   # yesterday, 2 days ago, 3 days ago
        check_date = (today - timedelta(days=days_back)).strftime("%m/%d/%Y")
        try:
            sched = statsapi.schedule(date=check_date)
        except Exception:
            continue

        for g in sched:
            if g.get("status") not in ("Final", "Game Over", "Completed Early"):
                continue
            hn = g.get("home_name",""); an = g.get("away_name","")
            ha = TEAM_MAP.get(hn, hn[:3].upper())
            aa = TEAM_MAP.get(an, an[:3].upper())
            gid = g.get("game_id")
            if not gid:
                continue

            try:
                box = statsapi.boxscore_data(gid)
                calls += 1
            except Exception:
                continue

            for side, team in [("home", ha), ("away", aa)]:
                pitchers = box.get(side, {}).get("pitchers", [])
                players  = box.get(side, {}).get("players", {})
                if len(pitchers) < 2:
                    continue   # no relievers used
                if team not in results:
                    results[team] = {"ip": 0.0, "apps": 0}
                for pid in pitchers[1:]:   # skip starter
                    pdata  = players.get(f"ID{pid}", {})
                    pstats = pdata.get("stats", {}).get("pitching", {})
                    ip     = parse_ip(pstats.get("inningsPitched", "0"))
                    if ip > 0:
                        results[team]["ip"]   = round(results[team]["ip"] + ip, 4)
                        results[team]["apps"] += 1

    fatigue = {team: {"ip_3d": v["ip"], "appearances_3d": v["apps"]}
               for team, v in results.items()}
    print(f"  Bullpen fatigue computed for {len(fatigue)} teams ({calls} boxscore calls)")
    return fatigue


# ═══════════════════════════════════════════════════════════════════
# PART 3 — PREDICT
# ═══════════════════════════════════════════════════════════════════

def run_predictions(games,model,feature_names,ts,ps,odds,starters,starter_pool,starter_stats_by_name=None,bullpen_fatigue=None):
    if starter_stats_by_name is None: starter_stats_by_name={}
    if bullpen_fatigue is None: bullpen_fatigue={}
    print("Running ML predictions...")
    current_season=datetime.now().year
    results=[]
    for g in games:
        ha=g["home_abbr"]; aa=g["away_abbr"]
        hs=ts.get(ha,{}); as_=ts.get(aa,{})
        hp=ps.get(ha,{}); ap=ps.get(aa,{})
        if not hs or not as_: print(f"  No stats for {ha}/{aa}"); continue

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

        if any("starter_" in f for f in feature_names):
            # ── Individual starter lookup (falls back to pool → league avg) ──
            st_game = starters.get(f"{ha}_{aa}",{})
            home_p_n = st_game.get("home_pitcher","TBD")
            away_p_n = st_game.get("away_pitcher","TBD")

            # Build pool fallbacks for when individual stats not found
            h_pool={}; a_pool={}
            if starter_pool is not None:
                h_raw=starter_pool.get(f"{ha}_{current_season}",starter_pool.get(f"{ha}_{current_season-1}",{}))
                a_raw=starter_pool.get(f"{aa}_{current_season}",starter_pool.get(f"{aa}_{current_season-1}",{}))
                if h_raw: h_pool={"era":h_raw.get("starter_era",row["home_avg_era"]),"whip":h_raw.get("starter_whip",row["home_avg_whip"]),"k_per_9":h_raw.get("starter_k_per_9",row["home_avg_so9"]),"bb_per_9":h_raw.get("starter_bb_per_9",3.0),"fip_proxy":h_raw.get("starter_fip_proxy",row["home_avg_era"])}
                if a_raw: a_pool={"era":a_raw.get("starter_era",row["away_avg_era"]),"whip":a_raw.get("starter_whip",row["away_avg_whip"]),"k_per_9":a_raw.get("starter_k_per_9",row["away_avg_so9"]),"bb_per_9":a_raw.get("starter_bb_per_9",3.0),"fip_proxy":a_raw.get("starter_fip_proxy",row["away_avg_era"])}

            h_sp = _lookup_pitcher_stats(home_p_n, starter_stats_by_name, h_pool or None)
            a_sp = _lookup_pitcher_stats(away_p_n, starter_stats_by_name, a_pool or None)

            row["home_starter_era"]=h_sp["era"]; row["home_starter_whip"]=h_sp["whip"]
            row["home_starter_k_per_9"]=h_sp["k_per_9"]; row["home_starter_bb_per_9"]=h_sp["bb_per_9"]
            row["home_starter_fip_proxy"]=h_sp["fip_proxy"]
            row["away_starter_era"]=a_sp["era"]; row["away_starter_whip"]=a_sp["whip"]
            row["away_starter_k_per_9"]=a_sp["k_per_9"]; row["away_starter_bb_per_9"]=a_sp["bb_per_9"]
            row["away_starter_fip_proxy"]=a_sp["fip_proxy"]
            row["starter_era_diff"]=a_sp["era"]-h_sp["era"]
            row["starter_whip_diff"]=a_sp["whip"]-h_sp["whip"]
            row["starter_k_diff"]=h_sp["k_per_9"]-a_sp["k_per_9"]
            row["starter_fip_diff"]=a_sp["fip_proxy"]-h_sp["fip_proxy"]

        # ── Bullpen fatigue features (if model was trained with them) ───
        if any("bullpen" in f for f in feature_names):
            h_bp = bullpen_fatigue.get(ha, {})
            a_bp = bullpen_fatigue.get(aa, {})
            row["home_bullpen_ip_3d"]          = h_bp.get("ip_3d", 0.0)
            row["away_bullpen_ip_3d"]          = a_bp.get("ip_3d", 0.0)
            row["home_bullpen_appearances_3d"] = h_bp.get("appearances_3d", 0)
            row["away_bullpen_appearances_3d"] = a_bp.get("appearances_3d", 0)
            row["bullpen_fatigue_diff"]        = (row["home_bullpen_ip_3d"]
                                                  - row["away_bullpen_ip_3d"])

        X=pd.DataFrame([{f: row.get(f,0.0) for f in feature_names}])
        prob_home=float(model.predict_proba(X)[0][1]); prob_away=1-prob_home
        go=odds.get(f"{ha}_{aa}",{}); home_odds_=go.get("home_odds"); away_odds_=go.get("away_odds")

        # ── Blend model probability with market implied probability ──────
        # Market captures injuries, weather, lineup changes the model can't see.
        # Weight: 70% model + 30% market. Falls back to 100% model if no odds.
        # This also archives signal for future training (see archive_odds).
        if home_odds_ is not None and away_odds_ is not None:
            mkt_home = american_to_prob(home_odds_)
            mkt_away = american_to_prob(away_odds_)
            # Normalise market probs (remove vig)
            mkt_total = mkt_home + mkt_away
            mkt_home_norm = mkt_home / mkt_total
            # Blend
            MARKET_WEIGHT = 0.30
            prob_home = (1 - MARKET_WEIGHT) * prob_home + MARKET_WEIGHT * mkt_home_norm
            prob_away = 1 - prob_home

        st=starters.get(f"{ha}_{aa}",{}); home_p=st.get("home_pitcher","TBD"); away_p=st.get("away_pitcher","TBD")

        if prob_home>=prob_away:
            pick=g["home_name"]; pick_abbr=ha; conf=prob_home; pick_odds=home_odds_
        else:
            pick=g["away_name"]; pick_abbr=aa; conf=prob_away; pick_odds=away_odds_

        ev=calc_ev(conf,pick_odds)
        results.append({**g,
            "prob_home":prob_home,"prob_away":prob_away,
            "pick":pick,"pick_abbr":pick_abbr,"confidence":conf,
            "home_odds":home_odds_,"away_odds":away_odds_,"pick_odds":pick_odds,"ev":ev,
            "h_wr":hs.get("rolling_win_rate",0.5),"a_wr":as_.get("rolling_win_rate",0.5),
            "h_rd":hs.get("rolling_run_diff",0.0),"a_rd":as_.get("rolling_run_diff",0.0),
            "h_rs":hs.get("rolling_runs_scored",4.5),"a_rs":as_.get("rolling_runs_scored",4.5),
            "h_ra":hs.get("rolling_runs_allowed",4.5),"a_ra":as_.get("rolling_runs_allowed",4.5),
            "h_last5":hs.get("last5",""),"a_last5":as_.get("last5",""),
            "h_last5_w":hs.get("last5_w",0),"h_last5_l":hs.get("last5_l",0),
            "a_last5_w":as_.get("last5_w",0),"a_last5_l":as_.get("last5_l",0),
            "home_pitcher":home_p,"away_pitcher":away_p,
            "h_era":hp.get("avg_era",4.5),"a_era":ap.get("avg_era",4.5),
            "h_so9":hp.get("avg_so9",8.0),"a_so9":ap.get("avg_so9",8.0),
        })
        print(f"  {ha} vs {aa}  ->  {pick_abbr} ({int(conf*100)}%)  {away_p} vs {home_p}")

    results.sort(key=lambda x: x["confidence"],reverse=True)
    return results

def predict_totals(games, ts, totals_odds):
    """Predict over/under for each game using rolling run averages."""
    print("Predicting totals...")
    results=[]
    for g in games:
        ha=g["home_abbr"]; aa=g["away_abbr"]
        key=f"{ha}_{aa}"
        tod=totals_odds.get(key)
        if not tod: continue
        hs=ts.get(ha,{}); as_=ts.get(aa,{})
        if not hs or not as_: continue

        line=tod["line"]
        h_rs=hs.get("rolling_runs_scored",4.5); h_ra=hs.get("rolling_runs_allowed",4.5)
        a_rs=as_.get("rolling_runs_scored",4.5); a_ra=as_.get("rolling_runs_allowed",4.5)
        # Expected total: blend of offensive and defensive estimates
        proj_total=round((h_rs + a_rs + h_ra + a_ra) / 2, 1)
        margin=proj_total - line
        # Confidence: further from line = more confident, cap at 80%
        raw_conf=0.50 + min(abs(margin)/3.0, 0.30)
        conf=round(raw_conf, 3)

        if margin > 0:
            pick="Over"; pick_odds=tod["over_odds"]
        else:
            pick="Under"; pick_odds=tod["under_odds"]

        ev=calc_ev(conf, pick_odds)
        results.append({
            "home_name":g["home_name"],"away_name":g["away_name"],
            "home_abbr":ha,"away_abbr":aa,"time":g["time"],"game_id":g["game_id"],
            "line":line,"pick":pick,"confidence":conf,
            "pick_odds":pick_odds,"over_odds":tod["over_odds"],"under_odds":tod["under_odds"],
            "ev":ev,"proj_total":proj_total,
            "h_rs":h_rs,"a_rs":a_rs,"h_ra":h_ra,"a_ra":a_ra,
        })
        print(f"  {ha} vs {aa}  ->  {pick} {line} ({int(conf*100)}%)  proj: {proj_total}")
    results.sort(key=lambda x: x["confidence"],reverse=True)
    return results

def build_pitcher_gamelogs():
    """
    Build a name-keyed lookup of each pitcher's recent K/9 from their
    last 5 starts using pybaseball game logs.

    Returns: {pitcher_name: {"recent_k_per_9": float, "recent_starts": int,
                              "recent_k_per_start": float, "recent_ip_per_start": float}}

    Falls back gracefully if pybaseball is unavailable or rate-limited.
    Uses a 30-day window to capture ~5 recent starts.
    """
    print("Loading recent pitcher game logs (last 30 days)...")
    try:
        from pybaseball import pitching_stats_range
        end   = datetime.now()
        start = end - timedelta(days=30)
        df = pitching_stats_range(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )
        if df is None or len(df) == 0:
            print("  No recent game log data available")
            return {}

        # Filter to starters (GS > 0) with at least 1 start
        if "GS" in df.columns:
            df = df[df["GS"] >= 1]

        out = {}
        name_col = "Name" if "Name" in df.columns else (
                   "PlayerName" if "PlayerName" in df.columns else None)
        if name_col is None:
            print("  Could not find pitcher name column in game logs")
            return {}

        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name or name == "nan":
                continue
            gs  = float(row.get("GS", 1)) or 1
            so  = float(row.get("SO", 0) or 0)
            ip  = float(row.get("IP", 0) or 0)
            if ip > 0:
                k_per_9        = round(so / ip * 9, 2)
                k_per_start    = round(so / gs, 2)
                ip_per_start   = round(ip / gs, 2)
            else:
                k_per_9 = 8.0; k_per_start = 5.0; ip_per_start = 5.5
            out[name] = {
                "recent_k_per_9":      k_per_9,
                "recent_k_per_start":  k_per_start,
                "recent_ip_per_start": ip_per_start,
                "recent_starts":       int(gs),
            }

        print(f"  Recent game logs loaded: {len(out)} pitchers (last 30 days)")
        return out

    except ImportError:
        print("  pybaseball not available — props will use season K/9")
        return {}
    except Exception as e:
        print(f"  Game log fetch failed ({e}) — props will use season K/9")
        return {}


def predict_props(games, starters, ps, starter_pool, props_odds, starter_stats_by_name=None, pitcher_gamelogs=None):
    """Predict pitcher strikeout props using recent game log K rates."""
    if starter_stats_by_name is None: starter_stats_by_name={}
    if pitcher_gamelogs is None: pitcher_gamelogs={}
    print("Predicting pitcher props...")
    current_season=datetime.now().year
    results=[]
    seen=set()
    for prop in props_odds:
        ha=prop.get("home_abbr",""); aa=prop.get("away_abbr","")
        pitcher=prop.get("pitcher",""); line=prop.get("line",5.5)
        if not pitcher or pitcher in seen: continue

        # Find which team's starter this is
        st=starters.get(f"{ha}_{aa}",{})
        if pitcher==st.get("home_pitcher",""):
            team_abbr=ha; opp_abbr=aa; is_home=True
        elif pitcher==st.get("away_pitcher",""):
            team_abbr=aa; opp_abbr=ha; is_home=False
        else:
            hp=st.get("home_pitcher",""); ap=st.get("away_pitcher","")
            last=pitcher.split()[-1].lower() if pitcher else ""
            if last and hp and last in hp.lower(): team_abbr=ha; opp_abbr=aa; is_home=True
            elif last and ap and last in ap.lower(): team_abbr=aa; opp_abbr=ha; is_home=False
            else: continue

        # ── K/9 lookup priority: recent logs → season individual → pool → team avg ──
        k_per_9     = None
        innings     = 5.5   # default projected IP
        data_source = "team_avg"

        # 1. Recent game logs (last 30 days) — best signal
        gl = pitcher_gamelogs.get(pitcher)
        if gl is None:
            # Try last-name fuzzy match
            last = pitcher.split()[-1].lower() if pitcher else ""
            for pname, pgl in pitcher_gamelogs.items():
                if pname.split()[-1].lower() == last:
                    gl = pgl; break
        if gl and gl.get("recent_starts", 0) >= 2:
            k_per_9     = gl["recent_k_per_9"]
            innings     = gl.get("recent_ip_per_start", 5.5)
            data_source = f"recent_{gl['recent_starts']}gs"

        # 2. Season individual stats
        if k_per_9 is None:
            individual = _lookup_pitcher_stats(pitcher, starter_stats_by_name)
            if individual.get("k_per_9") and individual["k_per_9"] != 8.0:
                k_per_9     = individual["k_per_9"]
                data_source = "season_individual"

        # 3. Team rotation pool
        if k_per_9 is None and starter_pool:
            sp_key = f"{team_abbr}_{current_season}"
            sp = starter_pool.get(sp_key, starter_pool.get(f"{team_abbr}_{current_season-1}", {}))
            if sp.get("starter_k_per_9"):
                k_per_9     = sp["starter_k_per_9"]
                data_source = "pool"

        # 4. Team average fallback
        if k_per_9 is None:
            k_per_9     = ps.get(team_abbr, {}).get("avg_so9", 8.0)
            data_source = "team_avg"

        # Blend with opponent K-susceptibility (how much the opposing lineup Ks)
        opp_so9    = ps.get(opp_abbr, {}).get("avg_so9", 8.0)
        blended_k9 = round(k_per_9 * 0.65 + opp_so9 * 0.35, 2)  # more weight on pitcher
        proj_k     = round(blended_k9 * innings / 9, 1)

        margin   = proj_k - line
        raw_conf = 0.50 + min(abs(margin) / 2.0, 0.28)
        conf     = round(raw_conf, 3)

        if margin > 0:
            pick = "Over";  pick_odds = prop["over_odds"]
        else:
            pick = "Under"; pick_odds = prop["under_odds"]

        ev        = calc_ev(conf, pick_odds)
        game_name = f"{aa} @ {ha}"
        game_time = "TBD"
        for g in games:
            if g["home_abbr"] == ha and g["away_abbr"] == aa:
                game_time = g["time"]; break

        seen.add(pitcher)
        results.append({
            "pitcher":pitcher,"team_abbr":team_abbr,"opp_abbr":opp_abbr,
            "home_abbr":ha,"away_abbr":aa,"game_name":game_name,"time":game_time,
            "game_id":prop.get("game_id"),
            "line":line,"pick":pick,"confidence":conf,
            "pick_odds":pick_odds,"over_odds":prop["over_odds"],"under_odds":prop["under_odds"],
            "ev":ev,"proj_k":proj_k,"k_per_9":k_per_9,"is_home":is_home,
            "data_source":data_source,
        })
        print(f"  {pitcher} K {pick} {line} ({int(conf*100)}%)  proj:{proj_k}  [{data_source}]")
    results.sort(key=lambda x: x["confidence"],reverse=True)
    return results


# ── Reasoning blurbs ──────────────────────────────────────────────

def ml_reasoning(g):
    pick_abbr=g["pick_abbr"]; opp_abbr=g["away_abbr"] if g["pick_abbr"]==g["home_abbr"] else g["home_abbr"]
    pick_rs=g["h_rs"] if pick_abbr==g["home_abbr"] else g["a_rs"]
    opp_ra=g["a_ra"] if pick_abbr==g["home_abbr"] else g["h_ra"]
    pick_era=g["h_era"] if pick_abbr==g["home_abbr"] else g["a_era"]
    opp_era=g["a_era"] if pick_abbr==g["home_abbr"] else g["h_era"]
    wr=g["h_wr"] if pick_abbr==g["home_abbr"] else g["a_wr"]
    hp=g.get("home_pitcher","TBD"); ap=g.get("away_pitcher","TBD")
    pick_p=hp if pick_abbr==g["home_abbr"] else ap
    opp_p=ap if pick_abbr==g["home_abbr"] else hp
    conf_pct=int(g["confidence"]*100)
    blurb=(f"{pick_abbr} averaging {pick_rs:.1f} runs/game over last {ROLLING_WINDOW} and "
           f"allowing {opp_ra:.1f} runs against — "
           f"{'a strong offensive edge' if pick_rs>opp_ra else 'backed by pitching'}. "
           f"Staff ERA of {pick_era:.2f} vs {opp_abbr}'s {opp_era:.2f}. "
           f"Model gives {pick_abbr} {conf_pct}% win probability with "
           f"{'solid starter advantage' if pick_era < opp_era else 'lineup edge'}.")
    return blurb

def ou_reasoning(g):
    proj=g["proj_total"]; line=g["line"]; pick=g["pick"]
    h=g["home_abbr"]; a=g["away_abbr"]
    h_rs=g["h_rs"]; a_rs=g["a_rs"]; h_ra=g["h_ra"]; a_ra=g["a_ra"]
    direction="above" if proj>line else "below"
    run_desc="both offenses running hot" if (h_rs+a_rs)/2>5 else "pitching-heavy matchup expected"
    blurb=(f"Rolling combined run average: {h} {h_rs:.1f} RS/{h_ra:.1f} RA, "
           f"{a} {a_rs:.1f} RS/{a_ra:.1f} RA. "
           f"Model projects {proj} total runs — {direction} the {line} line ({run_desc}). "
           f"Lean {pick}.")
    return blurb

def prop_reasoning(p):
    pitcher=p["pitcher"]; line=p["line"]; proj=p["proj_k"]; pick=p["pick"]
    k9=p["k_per_9"]; opp=p["opp_abbr"]
    direction="above" if proj>line else "below"
    edge=abs(proj-line)
    blurb=(f"{pitcher} posting {k9:.1f} K/9 this season. "
           f"Projected ~{proj} strikeouts in ~5.5 IP vs {opp} — {direction} the {line} line. "
           f"{'Comfortable edge' if edge>0.8 else 'Slim edge'}, lean {pick}.")
    return blurb


# ── Save picks ────────────────────────────────────────────────────

def save_todays_picks(predictions, date_str):
    path=f"raw/picks_{date_str}.json"
    if os.path.exists(path):
        print(f"  ML picks already saved for {date_str}"); return
    with open(path,"w") as f:
        json.dump([{"home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],
            "home_name":g["home_name"],"away_name":g["away_name"],
            "pick_abbr":g["pick_abbr"],"pick":g["pick"],
            "confidence":g["confidence"],"pick_odds":g.get("pick_odds"),
            "home_pitcher":g.get("home_pitcher","TBD"),
            "away_pitcher":g.get("away_pitcher","TBD"),
            "game_id":g.get("game_id"),"result":None} for g in predictions],f,indent=2)
    print(f"  ML picks saved -> {path}")

def save_ou_picks(predictions, date_str):
    path=f"raw/ou_{date_str}.json"
    if os.path.exists(path):
        print(f"  O/U picks already saved for {date_str}"); return
    with open(path,"w") as f:
        json.dump([{"home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],
            "home_name":g["home_name"],"away_name":g["away_name"],
            "pick":g["pick"],"line":g["line"],"confidence":g["confidence"],
            "pick_odds":g.get("pick_odds"),"game_id":g.get("game_id"),"result":None}
            for g in predictions],f,indent=2)
    print(f"  O/U picks saved -> {path}")

def save_props_picks(predictions, date_str):
    path=f"raw/props_{date_str}.json"
    if os.path.exists(path):
        print(f"  Props picks already saved for {date_str}"); return
    with open(path,"w") as f:
        json.dump([{"pitcher":g["pitcher"],"team_abbr":g["team_abbr"],
            "home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],
            "pick":g["pick"],"line":g["line"],"confidence":g["confidence"],
            "pick_odds":g.get("pick_odds"),"game_id":g.get("game_id"),"result":None}
            for g in predictions],f,indent=2)
    print(f"  Props picks saved -> {path}")


# ═══════════════════════════════════════════════════════════════════
# PART 4 — HTML
# ═══════════════════════════════════════════════════════════════════

def record_strings(rec):
    at=rec["all_time"]; total=at["w"]+at["l"]
    at_str=f"{at['w']}-{at['l']}"; at_pct=f"{at['w']/total:.1%}" if total>0 else "--"
    pct_w=at["w"]/total*100 if total>0 else 0
    now=datetime.now(); mw=ml_=ww=wl=0
    for ds,d in rec["daily"].items():
        try:
            dt=datetime.strptime(ds,"%Y-%m-%d")
            if dt.year==now.year and dt.month==now.month: mw+=d["w"]; ml_+=d["l"]
            if (now-dt).days<=7: ww+=d["w"]; wl+=d["l"]
        except: pass
    yest=(now-timedelta(days=1)).strftime("%Y-%m-%d")
    yd=rec["daily"].get(yest,{})
    yest_str=f"{yd.get('w',0)}-{yd.get('l',0)}" if yd else "--"
    return at_str,at_pct,f"{mw}-{ml_}",f"{ww}-{wl}",yest_str,pct_w

def conf_cell_html(rec, key, label):
    bc=rec["by_conf"]; w=bc.get(key,{}).get("w",0); l=bc.get(key,{}).get("l",0)
    pnl=bc.get(key,{}).get("pnl",0.0); t=w+l
    pct=f"{w/t:.0%}" if t>0 else "--"
    clr="#16a34a" if t>0 and w/t>=0.60 else ("#65a30d" if t>0 and w/t>=0.55 else "#374151")
    pnl_clr="#16a34a" if pnl>=0 else "#dc2626"
    pnl_str=f"+${pnl:.2f}" if pnl>=0 else f"-${abs(pnl):.2f}"
    return f"""<div style="background:#fff;border-radius:8px;padding:10px 6px;text-align:center;border:0.5px solid #e5e7eb;">
      <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">{label}</div>
      <div style="font-size:13px;font-weight:500;">{w}-{l}</div>
      <div style="font-size:10px;color:{clr};margin-top:1px;">{pct}</div>
      <div style="font-size:10px;font-weight:600;color:{pnl_clr};margin-top:4px;border-top:0.5px solid #f3f4f6;padding-top:4px;">$10 flat: {pnl_str}</div></div>"""

def build_history_section(date_str):
    """Build HTML cards for a past day's picks from the saved JSON file."""
    path = f"raw/picks_{date_str}.json"
    if not os.path.exists(path):
        return "", 0, 0
    try:
        picks = json.load(open(path))
    except:
        return "", 0, 0

    graded = [p for p in picks if p.get("result") not in (None, "no_result")]
    if not graded:
        return "", 0, 0

    day_w = sum(1 for p in graded if p.get("result") == "W")
    day_l = sum(1 for p in graded if p.get("result") == "L")
    cards = ""

    for i, p in enumerate(graded):
        result   = p.get("result")
        ok       = result == "W"
        conf     = p["confidence"]
        pick_pct = int(conf * 100)
        clr      = cc(conf)
        pick_tm  = p["pick"]
        pick_a   = p.get("pick_abbr", "")
        home_n   = p.get("home_name", ""); home_a = p.get("home_abbr", "")
        away_n   = p.get("away_name", ""); away_a = p.get("away_abbr", "")
        hp       = p.get("home_pitcher", "TBD")
        ap       = p.get("away_pitcher", "TBD")
        hs       = p.get("home_score", ""); as_ = p.get("away_score", "")
        home_odds = p.get("home_odds"); away_odds = p.get("away_odds")
        pick_odds = p.get("pick_odds")

        pih      = pick_a == home_a
        asty     = "color:#9ca3af;" if pih else "font-weight:600;"
        hsty     = "font-weight:600;" if pih else "color:#9ca3af;"
        away_pct = int((1-conf)*100) if pih else int(conf*100)
        home_pct = 100 - away_pct
        away_bar = clr if not pih else "#93c5fd"
        home_bar = clr if pih else "#93c5fd"

        badge = (f'<span style="font-size:11px;font-weight:600;color:#16a34a;background:#dcfce7;padding:3px 9px;border-radius:20px;">✓ WIN</span>'
                 if ok else
                 f'<span style="font-size:11px;font-weight:600;color:#dc2626;background:#fee2e2;padding:3px 9px;border-radius:20px;">✗ LOSS</span>')

        score_txt = f"Final: {away_a} {as_}–{hs} {home_a}" if hs != "" and as_ != "" else ""

        uid = f"h{date_str.replace('-','')}_{i}"
        cards += f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle('{uid}')" style="padding:16px 18px;cursor:pointer;user-select:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px;">
      <span style="font-size:11px;color:#9ca3af;">{score_txt}</span>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:13px;font-weight:500;color:{clr};">{pick_pct}% conf</span>
        {badge}
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;{asty}">{away_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Away</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">SP: {ap}</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(away_odds)}</div>
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;{hsty}">{home_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Home</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:3px;">{hp} SP</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{os_(home_odds)}</div>
      </div>
    </div>
    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{away_pct}%;background:{away_bar};border-radius:99px 0 0 99px;"></div>
        <div style="width:{home_pct}%;background:{home_bar};border-radius:0 99px 99px 0;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{away_a} {away_pct}%</span>
        <span style="color:#374151;font-weight:500;">Pick: {pick_tm}</span>
        <span>{home_a} {home_pct}%</span>
      </div>
    </div>
    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span style="font-size:15px;font-weight:600;">{pick_tm}</span>
            <span style="font-size:13px;padding:2px 10px;border-radius:20px;background:#f9fafb;color:#374151;">{os_(pick_odds)}</span>
          </div>
        </div>
        <div style="font-size:11px;color:#d1d5db;margin-top:4px;" id="hint_{uid}">tap for details</div>
      </div>
    </div>
  </div>
  <div id="body_{uid}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">{away_a} (Away)</div>
        <div style="font-size:13px;font-weight:500;">{away_n}</div>
        <div style="font-size:12px;color:#9ca3af;margin-top:2px;">ML: {os_(away_odds)}</div>
        <div style="font-size:11px;color:#374151;margin-top:4px;">SP: {ap}</div>
      </div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">{home_a} (Home)</div>
        <div style="font-size:13px;font-weight:500;">{home_n}</div>
        <div style="font-size:12px;color:#9ca3af;margin-top:2px;">ML: {os_(home_odds)}</div>
        <div style="font-size:11px;color:#374151;margin-top:4px;">SP: {hp}</div>
      </div>
    </div>
    <div style="text-align:center;"><span onclick="toggle('{uid}')" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse</span></div>
  </div>
</div>"""

    return cards, day_w, day_l


def generate_html(ml_preds, ou_preds, prop_preds, record, today_str, date_str):
    print("Generating HTML...")
    ml_rec=record["ml"]; ou_rec=record["ou"]; props_rec=record["props"]
    at_str,at_pct,mo_str,wk_str,ye_str,pct_w = record_strings(ml_rec)

    # Build history sections (last 8 days with picks)
    history_days = sorted(ml_rec["daily"].items(), reverse=True)[:8]
    history_sections = ""
    nav_history = ""

    for ds, d in history_days:
        try:
            lbl = datetime.strptime(ds, "%Y-%m-%d").strftime("%b %d")
            wlc = "#16a34a" if d["w"]>d["l"] else ("#dc2626" if d["l"]>d["w"] else "#9ca3af")
            hist_cards, hw, hl = build_history_section(ds)
            if not hist_cards:
                continue
            ht = hw + hl
            hpct = f"{hw/ht:.0%}" if ht > 0 else "--"
            nav_history += f'<button onclick="showDay(\'{ds}\')" id="nav_{ds}" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;border:none;cursor:pointer;margin-right:6px;background:#f3f4f6;color:#374151;">{lbl}<span style="font-size:11px;color:{wlc};"> {d["w"]}-{d["l"]}</span></button>'
            history_sections += f"""
<div id="section-{ds}" style="display:none;">
  <div style="margin-bottom:16px;">
    <div style="font-size:18px;font-weight:600;color:{wlc};">{hw}–{hl} <span style="font-size:13px;font-weight:400;color:#9ca3af;">· {hpct} · {datetime.strptime(ds,'%Y-%m-%d').strftime('%B %d %Y')}</span></div>
  </div>
  {hist_cards}
</div>"""
        except:
            pass

    # Nav — today button + history buttons
    nav = f'<button onclick="showDay(\'today\')" id="nav_today" style="display:inline-block;padding:7px 14px;border-radius:20px;font-size:12px;font-weight:500;border:none;cursor:pointer;margin-right:6px;background:#111;color:#fff;">Today · {datetime.now().strftime("%b %d")}</button>'
    nav += nav_history

    # ML cards
    ml_cards=""
    for i,g in enumerate(ml_preds):
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
        pick_name=g["pick"]
        reason=ml_reasoning(g)

        ml_cards+=f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle('ml{i}')" style="padding:16px 18px;cursor:pointer;user-select:none;">
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
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{g['away_abbr']} {away_pct}%</span>
        <span style="color:#374151;font-weight:500;">Proj: {pick_name} wins</span>
        <span>{g['home_abbr']} {home_pct}%</span>
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
          <div style="font-size:11px;color:#d1d5db;margin-top:4px;" id="hint_ml{i}">tap for details</div>
        </div>
      </div>
    </div>
  </div>
  <div id="body_ml{i}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:6px;">Moneyline pick</div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
      <span style="font-size:19px;font-weight:500;">{g['pick']}</span>
      <span style="font-size:14px;font-weight:500;color:#374151;">{os_(g['pick_odds'])}</span>
      <span style="font-size:12px;color:#9ca3af;">book {bk_prob}% vs model {pick_pct}%</span>
      <span style="font-size:13px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</span>
    </div>
    <div style="font-size:12px;color:#374151;line-height:1.65;margin-bottom:10px;padding:10px 12px;background:#f9fafb;border-radius:8px;">{reason}</div>
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
    <div style="text-align:center;margin-top:12px;"><span onclick="toggle('ml{i}')" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse</span></div>
  </div>
</div>"""

    # O/U cards
    ou_cards=""
    if not ou_preds:
        ou_cards='<div style="padding:20px;text-align:center;color:#9ca3af;font-size:13px;">No totals available today.</div>'
    for i,g in enumerate(ou_preds):
        conf=g["confidence"]; pick_pct=int(conf*100); clr=cc(conf)
        ev=g["ev"]; ev_clr=ec(ev); ev_lbl=el(ev)
        pick=g["pick"]; line=g["line"]; proj=g["proj_total"]
        over_clr="#16a34a" if pick=="Over" else "#9ca3af"
        under_clr="#16a34a" if pick=="Under" else "#9ca3af"
        reason=ou_reasoning(g)
        ou_cards+=f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div onclick="toggle('ou{i}')" style="padding:16px 18px;cursor:pointer;user-select:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:11px;color:#9ca3af;">{g['time']} · {g['away_abbr']} @ {g['home_abbr']}</span>
      <span style="font-size:13px;font-weight:500;color:{clr};">{pick_pct}% confidence</span>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
      <div style="flex:1;">
        <div style="font-size:13px;color:#9ca3af;">{g['away_name']} @ {g['home_name']}</div>
        <div style="font-size:11px;color:#9ca3af;margin-top:4px;">Proj: {proj} runs · Line: {line}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">Total</div>
        <div style="display:flex;gap:8px;align-items:center;">
          <span style="font-size:18px;font-weight:600;color:{over_clr};">O {line}</span>
          <span style="font-size:14px;color:#d1d5db;">/</span>
          <span style="font-size:18px;font-weight:600;color:{under_clr};">U {line}</span>
        </div>
      </div>
    </div>
    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
          <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:16px;font-weight:600;color:{clr};">{'OVER' if pick=='Over' else 'UNDER'} {line}</span>
            <span style="font-size:13px;padding:2px 10px;border-radius:20px;background:#f9fafb;color:#374151;">{os_(g['pick_odds'])}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</div>
          <div style="font-size:11px;color:#d1d5db;margin-top:4px;" id="hint_ou{i}">tap for details</div>
        </div>
      </div>
    </div>
  </div>
  <div id="body_ou{i}" style="display:none;border-top:0.5px solid #f3f4f6;padding:16px 18px;">
    <div style="font-size:12px;color:#374151;line-height:1.65;margin-bottom:10px;padding:10px 12px;background:#f9fafb;border-radius:8px;">{reason}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;">
        <div style="font-size:10px;color:#9ca3af;">Over {line}</div>
        <div style="font-size:16px;font-weight:600;margin-top:4px;">{os_(g['over_odds'])}</div>
      </div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;">
        <div style="font-size:10px;color:#9ca3af;">Under {line}</div>
        <div style="font-size:16px;font-weight:600;margin-top:4px;">{os_(g['under_odds'])}</div>
      </div>
    </div>
    <div style="text-align:center;margin-top:12px;"><span onclick="toggle('ou{i}')" style="font-size:11px;color:#d1d5db;cursor:pointer;">collapse</span></div>
  </div>
</div>"""

    # Props cards
    prop_cards=""
    if not prop_preds:
        prop_cards='<div style="padding:20px;text-align:center;color:#9ca3af;font-size:13px;">No pitcher props available today.</div>'
    for i,p in enumerate(prop_preds):
        conf=p["confidence"]; pick_pct=int(conf*100); clr=cc(conf)
        ev=p["ev"]; ev_clr=ec(ev); ev_lbl=el(ev)
        pick=p["pick"]; line=p["line"]
        bg_clr="#eff6ff" if pick=="Over" else "#f5f3ff"
        bd_clr="#bfdbfe" if pick=="Over" else "#ddd6fe"
        arrow_clr="#2563eb" if pick=="Over" else "#7c3aed"
        arrow="OVER" if pick=="Over" else "UNDER"
        reason=prop_reasoning(p)
        prop_cards+=f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div style="padding:16px 18px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <span style="font-size:11px;color:#9ca3af;">{p['game_name']} · {p['time']}</span>
      <span style="font-size:12px;font-weight:500;color:{clr};">{pick_pct}% conf</span>
    </div>
    <div style="font-size:16px;font-weight:600;color:#111;margin-bottom:8px;">{p['pitcher']}</div>
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;border-radius:10px;background:{bg_clr};border:0.5px solid {bd_clr};">
      <span style="font-size:18px;font-weight:600;color:{arrow_clr};">{arrow}</span>
      <span style="font-size:16px;font-weight:500;color:#111;">{line} Strikeouts</span>
      <span style="font-size:13px;color:#374151;">{os_(p['pick_odds'])}</span>
      <span style="font-size:12px;font-weight:500;color:{ev_clr};">EV {ev_lbl}</span>
    </div>
    <div style="font-size:12px;color:#374151;line-height:1.65;margin-top:10px;padding:10px 12px;background:#f9fafb;border-radius:8px;">{reason}</div>
  </div>
</div>"""

    # Record section — 3 rows
    def rec_row(label, rec_):
        at=rec_["all_time"]; tot=at["w"]+at["l"]
        pct=f"{at['w']/tot:.1%}" if tot>0 else "--"
        return f'<div style="background:#fff;border-radius:8px;padding:8px 10px;text-align:center;border:0.5px solid #e5e7eb;"><div style="font-size:10px;color:#9ca3af;">{label}</div><div style="font-size:14px;font-weight:500;">{at["w"]}-{at["l"]}</div><div style="font-size:10px;color:#16a34a;">{pct}</div></div>'

    n=len(ml_preds)
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
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:12px;">Model record · Moneyline</div>
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
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px;">
      {rec_row("ML",ml_rec)}{rec_row("O/U",ou_rec)}{rec_row("Props",props_rec)}
    </div>
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:8px;">ML record by confidence</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
      {conf_cell_html(ml_rec,"50","50-59%")}{conf_cell_html(ml_rec,"55","60-69%")}{conf_cell_html(ml_rec,"60","70-79%")}{conf_cell_html(ml_rec,"70","80%+")}
    </div>
  </div>

  <div style="overflow-x:auto;white-space:nowrap;margin-bottom:16px;padding-bottom:4px;">{nav}</div>

  <div style="display:flex;gap:6px;margin-bottom:16px;">
    <button onclick="showTab('ml')" id="tab-ml" style="flex:1;padding:9px 0;border-radius:10px;border:none;cursor:pointer;font-size:13px;font-weight:600;background:#111;color:#fff;">ML</button>
    <button onclick="showTab('ou')" id="tab-ou" style="flex:1;padding:9px 0;border-radius:10px;border:none;cursor:pointer;font-size:13px;font-weight:600;background:#f3f4f6;color:#374151;">O/U</button>
    <button onclick="showTab('props')" id="tab-props" style="flex:1;padding:9px 0;border-radius:10px;border:none;cursor:pointer;font-size:13px;font-weight:600;background:#f3f4f6;color:#374151;">Props</button>
  </div>

  <div id="section-ml">
    <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;gap:5px;">
      EV: <span style="color:#16a34a;">+8 excellent</span> · <span style="color:#65a30d;">+3 good</span> · <span style="color:#d97706;">marginal</span> · <span style="color:#dc2626;">neg = skip</span>
    </div>
    {ml_cards}
  </div>

  <div id="section-ou" style="display:none;">
    <div style="display:flex;justify-content:flex-end;margin-bottom:12px;font-size:11px;color:#9ca3af;gap:5px;">
      EV: <span style="color:#16a34a;">+8 excellent</span> · <span style="color:#65a30d;">+3 good</span> · <span style="color:#d97706;">marginal</span> · <span style="color:#dc2626;">neg = skip</span>
    </div>
    {ou_cards}
  </div>

  <div id="section-props" style="display:none;">
    {prop_cards}
  </div>

  <div style="margin-top:20px;padding:12px 14px;background:#fffbeb;border-radius:8px;border:0.5px solid #fde68a;">
    <p style="font-size:11px;color:#92400e;line-height:1.6;">AI-generated picks for informational purposes only. Always gamble responsibly.</p>
  </div>
</div>

<!-- History sections (hidden by default, shown via showDay()) -->
<div id="history-container" style="display:none;max-width:680px;margin:0 auto;padding:0 14px 60px;">
  {history_sections}
</div>

<script>
function toggle(id){{
  var b=document.getElementById('body_'+id);
  var h=document.getElementById('hint_'+id);
  if(!b)return;
  var o=b.style.display!=='none';
  b.style.display=o?'none':'block';
  if(h) h.innerHTML=o?'tap for details':'collapse';
}}
function showTab(t){{
  ['ml','ou','props'].forEach(function(id){{
    var active=t===id;
    document.getElementById('section-'+id).style.display=active?'block':'none';
    document.getElementById('tab-'+id).style.background=active?'#111':'#f3f4f6';
    document.getElementById('tab-'+id).style.color=active?'#fff':'#374151';
  }});
}}
function showDay(day){{
  // Hide today's content or any open history section
  document.querySelector('.wrap').style.display = day==='today' ? 'block' : 'none';
  document.getElementById('history-container').style.display = day==='today' ? 'none' : 'block';

  // Hide all history sections
  document.querySelectorAll('[id^="section-202"]').forEach(function(el){{
    el.style.display='none';
  }});

  // Show the selected history section
  if(day!=='today'){{
    var sec=document.getElementById('section-'+day);
    if(sec) sec.style.display='block';
  }}

  // Update nav button styles
  document.querySelectorAll('[id^="nav_"]').forEach(function(btn){{
    btn.style.background='#f3f4f6';
    btn.style.color='#374151';
  }});
  var active=document.getElementById('nav_'+day);
  if(active){{ active.style.background='#111'; active.style.color='#fff'; }}
}}
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
    starter_stats_by_name = build_starter_stats_by_name()
    pitcher_gamelogs      = build_pitcher_gamelogs()
    starters      = fetch_probable_starters()
    odds          = fetch_odds()
    archive_odds(odds, date_str)
    totals_odds   = fetch_totals_odds()
    props_odds_raw= fetch_props_odds()
    games         = get_schedule()
    bullpen_fatigue = fetch_live_bullpen()

    if not games:
        print("No games today.")
        generate_html([],[],[],record,today_str,date_str)
    else:
        ml_preds   = run_predictions(games,model,feature_names,ts,ps,odds,starters,starter_pool,starter_stats_by_name,bullpen_fatigue)
        ou_preds   = predict_totals(games,ts,totals_odds)
        prop_preds = predict_props(games,starters,ps,starter_pool,props_odds_raw,starter_stats_by_name,pitcher_gamelogs)

        save_todays_picks(ml_preds,date_str)
        save_ou_picks(ou_preds,date_str)
        save_props_picks(prop_preds,date_str)

        generate_html(ml_preds,ou_preds,prop_preds,record,today_str,date_str)

        at=record["ml"]["all_time"]
        print(f"\nDone! Open mlb_predictor.html in Chrome.")
        print(f"ML Record: {at['w']}-{at['l']} all time")