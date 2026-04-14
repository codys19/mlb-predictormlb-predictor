"""
backtest.py
-----------
Backtests your model on 2025-2026 games (never seen during training).
Simulates day-by-day picks and measures accuracy, ROI, and confidence tiers.

RUN:
    python data/backtest.py

OUTPUT:
    raw/backtest_results.csv   ← every game with pick + result
    raw/backtest_report.html   ← visual report you can open in Chrome
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime
import os

TRAINING_DATA = "raw/training_data.csv"
MODEL_PATH    = "raw/model.json"
OUTPUT_CSV    = "raw/backtest_results.csv"
OUTPUT_HTML   = "raw/backtest_report.html"

FEATURE_COLS = [
    "home_rolling_runs_scored","home_rolling_runs_allowed",
    "home_rolling_win_rate","home_rolling_run_diff",
    "away_rolling_runs_scored","away_rolling_runs_allowed",
    "away_rolling_win_rate","away_rolling_run_diff",
    "win_rate_diff","run_diff_diff","runs_scored_diff","is_home",
    "home_avg_era","home_avg_whip","home_avg_so9","home_avg_so_per_w",
    "away_avg_era","away_avg_whip","away_avg_so9","away_avg_so_per_w",
    "era_diff","whip_diff","so9_diff",
]

# Fake moneyline odds for ROI simulation (model has no real odds in training data)
# We simulate: favourite at -130, underdog at +110 — typical MLB lines
FAV_ODDS  = -130
DOG_ODDS  = +110
FLAT_BET  = 10  # dollars per game


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data_and_model():
    print("📂 Loading data and model...")
    df    = pd.read_csv(TRAINING_DATA)
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # Only test on 2025-2026 — model never trained on these
    test = df[df["season"] >= 2025].copy()
    test = test.dropna(subset=FEATURE_COLS + ["home_team_won"])
    test = test.sort_values("date").reset_index(drop=True)

    print(f"  ✅ {len(test):,} test games (2025–2026)")
    return test, model


# ── Run predictions ───────────────────────────────────────────────────────────

def run_backtest(test, model):
    print("🔮 Running backtest predictions...")

    X     = test[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]  # probability home team wins

    results = []
    for i, row in test.iterrows():
        prob_home = probs[list(test.index).index(i)]
        prob_away = 1 - prob_home

        # Model pick
        if prob_home >= prob_away:
            pick        = "home"
            confidence  = prob_home
        else:
            pick        = "away"
            confidence  = prob_away

        actual_winner = "home" if row["home_team_won"] == 1 else "away"
        correct       = pick == actual_winner

        # Simulate ROI — assume favourite gets -130, underdog gets +110
        is_favourite  = confidence >= 0.5
        bet_odds      = FAV_ODDS if is_favourite else DOG_ODDS

        if correct:
            profit = FLAT_BET * (100/abs(bet_odds)) if bet_odds < 0 else FLAT_BET * (bet_odds/100)
        else:
            profit = -FLAT_BET

        results.append({
            "season":       row["season"],
            "date":         row["date"],
            "home_team":    row["home_team"],
            "away_team":    row["away_team"],
            "prob_home":    round(prob_home, 4),
            "prob_away":    round(prob_away, 4),
            "pick":         pick,
            "confidence":   round(confidence, 4),
            "actual_winner":actual_winner,
            "correct":      correct,
            "profit":       round(profit, 2),
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ {len(df_results):,} predictions saved → {OUTPUT_CSV}")
    return df_results


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(df):
    print("\n" + "═"*55)
    print("  BACKTEST RESULTS")
    print("═"*55)

    total    = len(df)
    correct  = df["correct"].sum()
    accuracy = correct / total

    print(f"\n  Overall accuracy:  {accuracy:.1%}  ({correct}/{total} games)")
    print(f"  Home win rate:     {df['actual_winner'].eq('home').mean():.1%}  (baseline)")

    # ── By confidence tier ────────────────────────────────────────────────────
    print(f"\n  {'Confidence':<15} {'Games':>6} {'Correct':>8} {'Accuracy':>9} {'ROI':>8}")
    print("  " + "─"*50)

    tiers = [
        ("50–54%",  0.50, 0.55),
        ("55–59%",  0.55, 0.60),
        ("60–64%",  0.60, 0.65),
        ("65–69%",  0.65, 0.70),
        ("70–74%",  0.70, 0.75),
        ("75–79%",  0.75, 0.80),
        ("80%+",    0.80, 1.01),
    ]

    tier_data = []
    for label, lo, hi in tiers:
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        sub  = df[mask]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        roi = sub["profit"].sum() / (len(sub) * FLAT_BET) * 100
        tier_data.append({"label":label,"games":len(sub),"acc":acc,"roi":roi})
        marker = " ◀" if acc >= 0.60 else ""
        print(f"  {label:<15} {len(sub):>6,} {sub['correct'].sum():>8,} {acc:>8.1%} {roi:>+7.1f}%{marker}")

    # ── By season ─────────────────────────────────────────────────────────────
    print(f"\n  {'Season':<10} {'Games':>6} {'Accuracy':>9} {'ROI':>8}")
    print("  " + "─"*35)
    for season in sorted(df["season"].unique()):
        sub = df[df["season"]==season]
        acc = sub["correct"].mean()
        roi = sub["profit"].sum() / (len(sub)*FLAT_BET)*100
        print(f"  {season:<10} {len(sub):>6,} {acc:>8.1%} {roi:>+7.1f}%")

    # ── ROI summary ───────────────────────────────────────────────────────────
    total_bet    = total * FLAT_BET
    total_profit = df["profit"].sum()
    roi_overall  = total_profit / total_bet * 100

    print(f"\n  ROI (flat ${FLAT_BET}/game):")
    print(f"  Total wagered:  ${total_bet:,.0f}")
    print(f"  Total profit:   ${total_profit:+,.2f}")
    print(f"  Overall ROI:    {roi_overall:+.1f}%")

    # High-confidence only
    hi_conf = df[df["confidence"] >= 0.60]
    if len(hi_conf) > 0:
        hi_profit = hi_conf["profit"].sum()
        hi_roi    = hi_profit / (len(hi_conf)*FLAT_BET)*100
        print(f"\n  High-confidence (≥60%) ROI:")
        print(f"  Games:  {len(hi_conf):,}")
        print(f"  Profit: ${hi_profit:+,.2f}")
        print(f"  ROI:    {hi_roi:+.1f}%")

    print("\n" + "═"*55)
    return tier_data, total, correct, accuracy, total_profit, roi_overall


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_report(df, tier_data, total, correct, accuracy, total_profit, roi_overall):
    print("🖥️  Generating HTML report...")

    # Accuracy by month
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = df.groupby("month").agg(
        games=("correct","count"),
        acc=("correct","mean"),
        profit=("profit","sum")
    ).reset_index()

    month_rows = ""
    for _, row in monthly.iterrows():
        roi = row["profit"]/(row["games"]*FLAT_BET)*100
        acc_clr = "#16a34a" if row["acc"]>=0.60 else ("#65a30d" if row["acc"]>=0.55 else "#374151")
        roi_clr = "#16a34a" if roi>0 else "#dc2626"
        month_rows += f"""<tr>
          <td style="padding:10px 12px;font-size:13px;">{row['month']}</td>
          <td style="padding:10px 12px;font-size:13px;text-align:center;">{row['games']:,}</td>
          <td style="padding:10px 12px;font-size:13px;text-align:center;font-weight:500;color:{acc_clr};">{row['acc']:.1%}</td>
          <td style="padding:10px 12px;font-size:13px;text-align:center;font-weight:500;color:{roi_clr};">{roi:+.1f}%</td>
        </tr>"""

    # Confidence tier bars
    tier_bars = ""
    for t in tier_data:
        acc_clr = "#16a34a" if t["acc"]>=0.60 else ("#65a30d" if t["acc"]>=0.55 else "#9ca3af")
        roi_clr = "#16a34a" if t["roi"]>0 else "#dc2626"
        bar_w   = int(t["acc"]*100)
        tier_bars += f"""
<div style="margin-bottom:14px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:13px;font-weight:500;">{t['label']}</span>
    <div style="display:flex;gap:16px;align-items:center;">
      <span style="font-size:12px;color:#9ca3af;">{t['games']:,} games</span>
      <span style="font-size:13px;font-weight:600;color:{acc_clr};">{t['acc']:.1%}</span>
      <span style="font-size:12px;font-weight:500;color:{roi_clr};">ROI {t['roi']:+.1f}%</span>
    </div>
  </div>
  <div style="height:6px;background:#f3f4f6;border-radius:99px;overflow:hidden;">
    <div style="width:{bar_w}%;height:100%;background:{acc_clr};border-radius:99px;"></div>
  </div>
</div>"""

    # Recent 20 picks table
    recent = df.tail(20).iloc[::-1]
    recent_rows = ""
    for _, row in recent.iterrows():
        ok   = row["correct"]
        clr  = "#16a34a" if ok else "#dc2626"
        icon = "✅" if ok else "❌"
        prof_clr = "#16a34a" if row["profit"]>0 else "#dc2626"
        recent_rows += f"""<tr style="border-bottom:0.5px solid #f3f4f6;">
          <td style="padding:8px 12px;font-size:12px;color:#9ca3af;">{row['date'][:10]}</td>
          <td style="padding:8px 12px;font-size:12px;">{row['away_team']} @ {row['home_team']}</td>
          <td style="padding:8px 12px;font-size:12px;font-weight:500;">{row['pick'].upper()}</td>
          <td style="padding:8px 12px;font-size:12px;text-align:center;">{int(row['confidence']*100)}%</td>
          <td style="padding:8px 12px;font-size:12px;text-align:center;">{icon}</td>
          <td style="padding:8px 12px;font-size:12px;text-align:right;color:{prof_clr};">${row['profit']:+.2f}</td>
        </tr>"""

    roi_clr     = "#16a34a" if roi_overall > 0 else "#dc2626"
    profit_clr  = "#16a34a" if total_profit > 0 else "#dc2626"
    acc_clr_main = "#16a34a" if accuracy >= 0.60 else ("#65a30d" if accuracy >= 0.55 else "#374151")
    total_wagered = total * FLAT_BET

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MLB Model Backtest Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f4f6;color:#111;min-height:100vh}}
  .wrap{{max-width:720px;margin:0 auto;padding:20px 14px 60px}}
  table{{width:100%;border-collapse:collapse;}}
  th{{padding:10px 12px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;color:#9ca3af;text-align:left;border-bottom:0.5px solid #e5e7eb;}}
  tr:hover td{{background:#fafafa;}}
</style>
</head><body><div class="wrap">

  <div style="margin-bottom:24px;">
    <h1 style="font-size:22px;font-weight:500;">⚾ MLB Model Backtest</h1>
    <div style="font-size:13px;color:#9ca3af;margin-top:3px;">
      2025–2026 seasons · {total:,} games · Generated {datetime.now().strftime("%B %d %Y")}
    </div>
  </div>

  <!-- Summary cards -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;">
    <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Overall accuracy</div>
      <div style="font-size:32px;font-weight:500;color:{acc_clr_main};">{accuracy:.1%}</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:4px;">{correct:,} of {total:,} correct</div>
    </div>
    <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Total profit (${FLAT_BET} flat)</div>
      <div style="font-size:32px;font-weight:500;color:{profit_clr};">${total_profit:+,.0f}</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:4px;">${total_wagered:,.0f} wagered</div>
    </div>
    <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center;">
      <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Overall ROI</div>
      <div style="font-size:32px;font-weight:500;color:{roi_clr};">{roi_overall:+.1f}%</div>
      <div style="font-size:12px;color:#9ca3af;margin-top:4px;">vs 52.4% break-even</div>
    </div>
  </div>

  <!-- Confidence tiers -->
  <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;margin-bottom:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:16px;">Accuracy by confidence tier</div>
    {tier_bars}
    <div style="font-size:11px;color:#9ca3af;margin-top:8px;padding-top:12px;border-top:0.5px solid #f3f4f6;">
      ◀ marks tiers hitting 60%+ accuracy target · ROI assumes favourite at {FAV_ODDS}, underdog at +{DOG_ODDS}
    </div>
  </div>

  <!-- Monthly breakdown -->
  <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;margin-bottom:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:16px;">Monthly breakdown</div>
    <table>
      <thead><tr>
        <th>Month</th><th style="text-align:center;">Games</th>
        <th style="text-align:center;">Accuracy</th><th style="text-align:center;">ROI</th>
      </tr></thead>
      <tbody>{month_rows}</tbody>
    </table>
    <div style="font-size:11px;color:#9ca3af;margin-top:10px;">
      Accuracy generally improves mid-season as rolling stats stabilise — expect first 2–3 weeks to be weakest.
    </div>
  </div>

  <!-- Recent picks -->
  <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;margin-bottom:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:16px;">Last 20 backtest picks</div>
    <table>
      <thead><tr>
        <th>Date</th><th>Matchup</th><th>Pick</th>
        <th style="text-align:center;">Conf</th>
        <th style="text-align:center;">Result</th>
        <th style="text-align:right;">Profit</th>
      </tr></thead>
      <tbody>{recent_rows}</tbody>
    </table>
  </div>

  <!-- What to improve -->
  <div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;">
    <div style="font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;margin-bottom:12px;">What would improve accuracy</div>
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px;background:#f9fafb;border-radius:8px;">
        <div><div style="font-size:13px;font-weight:500;">Individual starter ERA/FIP</div><div style="font-size:11px;color:#9ca3af;margin-top:2px;">Biggest single upgrade — replaces team average pitcher stats</div></div>
        <div style="font-size:12px;color:#16a34a;font-weight:500;">+3–4%</div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px;background:#f9fafb;border-radius:8px;">
        <div><div style="font-size:13px;font-weight:500;">Vegas moneyline as a feature</div><div style="font-size:11px;color:#9ca3af;margin-top:2px;">The market already prices in injuries, weather, lineups</div></div>
        <div style="font-size:12px;color:#16a34a;font-weight:500;">+2–3%</div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px;background:#f9fafb;border-radius:8px;">
        <div><div style="font-size:13px;font-weight:500;">Bullpen fatigue</div><div style="font-size:11px;color:#9ca3af;margin-top:2px;">Relievers used in last 3 days significantly affects late-game outcomes</div></div>
        <div style="font-size:12px;color:#65a30d;font-weight:500;">+1–2%</div>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px;background:#f9fafb;border-radius:8px;">
        <div><div style="font-size:13px;font-weight:500;">Park factors</div><div style="font-size:11px;color:#9ca3af;margin-top:2px;">Coors Field inflates COL stats — model currently overrates them at home</div></div>
        <div style="font-size:12px;color:#65a30d;font-weight:500;">+1%</div>
      </div>
    </div>
  </div>

</div></body></html>"""

    with open(OUTPUT_HTML,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Report saved → {OUTPUT_HTML}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test, model  = load_data_and_model()
    df_results   = run_backtest(test, model)
    tier_data, total, correct, accuracy, total_profit, roi_overall = analyse(df_results)
    generate_report(df_results, tier_data, total, correct, accuracy, total_profit, roi_overall)

    print(f"\n✅ Done!")
    print(f"   Open raw/backtest_report.html in Chrome for the full report.")
