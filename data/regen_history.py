"""
Run from C:\\MLB:
    python data/regen_history.py

Regenerates all history_*.html files using the same card style
as the NBA app — team names left/right, score center, probability
bar, WIN/LOSS badge with actual score in header.
"""
import json, os, glob
from datetime import datetime

def cc(c):
    if c >= 0.70: return "#16a34a"
    if c >= 0.60: return "#65a30d"
    if c >= 0.55: return "#d97706"
    return "#9ca3af"

def build_card(p, i):
    result   = p.get("result")
    ok       = result == "W"
    conf     = p["confidence"]
    conf_pct = int(conf * 100)
    conf_clr = cc(conf)

    home_n   = p.get("home_name", "Home")
    away_n   = p.get("away_name", "Away")
    home_a   = p.get("home_abbr", "HME")
    away_a   = p.get("away_abbr", "AWY")
    pick_tm  = p["pick"]
    pick_a   = p.get("pick_abbr", "")
    odds_raw = p.get("pick_odds")
    odds_str = f"{odds_raw:+d}" if odds_raw is not None else "--"
    hs       = p.get("home_score", "")
    as_      = p.get("away_score", "")
    winner_a = p.get("actual_winner", "")
    hp       = p.get("home_pitcher", "TBD")
    ap       = p.get("away_pitcher", "TBD")

    # Probability bar
    prob_home = round(conf * 100) if pick_a == home_a else round((1 - conf) * 100)
    prob_away = 100 - prob_home
    pick_home = pick_a == home_a

    # WIN/LOSS badge with actual score
    if hs != "" and as_ != "":
        score_badge = f"· {away_a} {as_}–{hs} {home_a}"
    else:
        score_badge = ""

    if result == "W":
        badge_bg = "#dcfce7"; badge_clr = "#16a34a"
        badge_txt = f"✓ WIN {score_badge}"
    elif result == "L":
        badge_bg = "#fee2e2"; badge_clr = "#dc2626"
        badge_txt = f"✗ LOSS {score_badge}"
    else:
        badge_bg = "#f3f4f6"; badge_clr = "#9ca3af"
        badge_txt = "—"

    # Odds display
    home_odds_raw = p.get("home_odds")
    away_odds_raw = p.get("away_odds")
    home_odds_str = f"{home_odds_raw:+d}" if home_odds_raw else "--"
    away_odds_str = f"{away_odds_raw:+d}" if away_odds_raw else "--"

    # Pick side highlight
    h_bold = "font-weight:600;color:#111;" if pick_home else "color:#9ca3af;"
    a_bold = "font-weight:600;color:#111;" if not pick_home else "color:#9ca3af;"

    return f"""
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;margin-bottom:12px;overflow:hidden;">
  <div style="padding:16px 18px;">

    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:11px;color:#9ca3af;">{away_n} @ {home_n}</span>
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:flex-end;">
        <span style="font-size:13px;font-weight:500;color:{conf_clr};">{conf_pct}% conf</span>
        <span style="font-size:11px;font-weight:600;color:{badge_clr};background:{badge_bg};padding:3px 9px;border-radius:20px;white-space:nowrap;">{badge_txt}</span>
      </div>
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
      <div style="flex:1;min-width:0;">
        <div style="font-size:15px;font-weight:500;{a_bold}overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{away_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Away</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{away_odds_str}</div>
      </div>
      <div style="text-align:center;padding:0 10px;flex-shrink:0;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">Final</div>
        <div style="display:flex;align-items:center;gap:5px;">
          <span style="font-size:26px;font-weight:500;color:{'#111' if not pick_home else '#bbb'};">{as_ if as_ != '' else '–'}</span>
          <span style="font-size:14px;color:#d1d5db;">–</span>
          <span style="font-size:26px;font-weight:500;color:{'#111' if pick_home else '#bbb'};">{hs if hs != '' else '–'}</span>
        </div>
      </div>
      <div style="flex:1;min-width:0;text-align:right;">
        <div style="font-size:15px;font-weight:500;{h_bold}overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{home_n}</div>
        <div style="font-size:11px;color:#9ca3af;">Home</div>
        <div style="font-size:12px;font-weight:500;color:#374151;margin-top:2px;">{home_odds_str}</div>
      </div>
    </div>

    <div style="margin-top:10px;">
      <div style="height:5px;background:#f3f4f6;border-radius:99px;overflow:hidden;display:flex;">
        <div style="width:{prob_away}%;background:#93c5fd;border-radius:99px 0 0 99px;"></div>
        <div style="width:{prob_home}%;background:{conf_clr};border-radius:0 99px 99px 0;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#9ca3af;margin-top:3px;">
        <span>{away_a} {prob_away}%</span>
        <span style="color:#374151;font-weight:500;">Pick: {pick_tm}</span>
        <span>{home_a} {prob_home}%</span>
      </div>
    </div>

    <div style="margin-top:10px;border-top:0.5px solid #f3f4f6;padding-top:10px;display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:10px;color:#9ca3af;margin-bottom:3px;">MODEL PICK</div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span style="font-size:15px;font-weight:600;color:#111;">{pick_tm}</span>
          <span style="font-size:13px;font-weight:500;padding:2px 8px;border-radius:20px;background:#f0fdf4;color:#166534;">{odds_str}</span>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:10px;color:#9ca3af;margin-bottom:2px;">Starters</div>
        <div style="font-size:11px;color:#374151;">{ap} vs {hp}</div>
      </div>
    </div>

  </div>
</div>"""


def regen_history(picks, date_str, day_w, day_l):
    dt    = datetime.strptime(date_str, "%Y-%m-%d")
    title = dt.strftime("%A, %B %d %Y")
    cards = ""

    for i, p in enumerate(picks):
        result = p.get("result")
        if result in ("no_result", None, "push"):
            continue
        cards += build_card(p, i)

    wlc   = "#16a34a" if day_w >= day_l else "#dc2626"
    total = day_w + day_l
    pct   = f"{day_w/total:.0%}" if total > 0 else "--"

    if not cards:
        cards = '<div style="padding:32px;text-align:center;color:#9ca3af;font-size:13px;">No graded picks for this day.</div>'

    return f"""<!DOCTYPE html>
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
</div></body></html>"""


if __name__ == "__main__":
    picks_files = sorted(glob.glob("raw/picks_*.json"))
    print(f"Found {len(picks_files)} picks files\n")
    regenerated = 0

    for path in picks_files:
        date_str = os.path.basename(path).replace("picks_", "").replace(".json", "")
        try:
            picks = json.load(open(path))
        except Exception as e:
            print(f"  ❌ {date_str}: could not read ({e})")
            continue

        graded = [p for p in picks if p.get("result") not in (None, "no_result")]
        if not graded:
            print(f"  ⏭  {date_str}: no graded picks — skipping")
            continue

        day_w = sum(1 for p in graded if p.get("result") == "W")
        day_l = sum(1 for p in graded if p.get("result") == "L")

        html = regen_history(graded, date_str, day_w, day_l)
        out  = f"history_{date_str}.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"  ✅ {date_str}: {day_w}-{day_l} ({len(graded)} picks) → {out}")
        regenerated += 1

    print(f"\n✅ Regenerated {regenerated} history pages")
    print("Next: git add history_*.html && git commit -m 'Regen history NBA style' && git push")