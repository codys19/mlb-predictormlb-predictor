"""
backtest.py
-----------
Backtests the model on 2025-2026 games. Simulates picks, measures accuracy/ROI.

RUN:  python data/backtest.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime

TRAINING_DATA = "raw/training_data.csv"
MODEL_PATH    = "raw/model.json"
OUTPUT_CSV    = "raw/backtest_results.csv"
OUTPUT_HTML   = "raw/backtest_report.html"

FEATURE_COLS = []   # populated from model at load time
FAV_ODDS  = -130
DOG_ODDS  = +110
FLAT_BET  = 10


def load_data_and_model():
    global FEATURE_COLS
    print("📂 Loading data and model...")
    df    = pd.read_csv(TRAINING_DATA)
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    # Always in sync with whatever model.json was last trained
    FEATURE_COLS = model.get_booster().feature_names
    print(f"  ✅ {len(FEATURE_COLS)} features loaded from model")

    # Starter columns are intentionally sparse (NaN when no individual match).
    # XGBoost handles NaN natively — do NOT dropna on those rows.
    required = [c for c in FEATURE_COLS if not any(
        c.startswith(p) for p in ("home_starter_", "away_starter_", "starter_", "has_individual_")
    )]

    test = df[df["season"] >= 2025].copy()
    test = test.dropna(subset=required + ["home_team_won"])
    test = test.sort_values("date").reset_index(drop=True)
    print(f"  ✅ {len(test):,} test games (2025-2026)")
    return test, model


def run_backtest(test, model):
    print("🔮 Running backtest predictions...")
    X     = test[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]

    results = []
    idx_list = list(test.index)
    for i, row in test.iterrows():
        ph = probs[idx_list.index(i)]
        pa = 1 - ph
        if ph >= pa:
            pick, conf = "home", ph
        else:
            pick, conf = "away", pa
        actual  = "home" if row["home_team_won"] == 1 else "away"
        correct = pick == actual
        odds    = FAV_ODDS if conf >= 0.5 else DOG_ODDS
        profit  = (FLAT_BET * (100/abs(odds)) if odds < 0 else FLAT_BET * (odds/100)) if correct else -FLAT_BET
        results.append({
            "season": row["season"], "date": row["date"],
            "home_team": row["home_team"], "away_team": row["away_team"],
            "prob_home": round(ph, 4), "prob_away": round(pa, 4),
            "pick": pick, "confidence": round(conf, 4),
            "actual_winner": actual, "correct": correct, "profit": round(profit, 2),
        })

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ {len(df_r):,} predictions saved → {OUTPUT_CSV}")
    return df_r


def analyse(df):
    print("\n" + "="*55)
    print("  BACKTEST RESULTS")
    print("="*55)

    total    = len(df)
    correct  = df["correct"].sum()
    accuracy = correct / total

    print(f"\n  Overall accuracy:  {accuracy:.1%}  ({correct}/{total} games)")
    print(f"  Home win rate:     {df['actual_winner'].eq('home').mean():.1%}  (baseline)")
    print(f"\n  {'Confidence':<15} {'Games':>6} {'Correct':>8} {'Accuracy':>9} {'ROI':>8}")
    print("  " + "-"*50)

    tiers = [
        ("50-54%", 0.50, 0.55), ("55-59%", 0.55, 0.60),
        ("60-64%", 0.60, 0.65), ("65-69%", 0.65, 0.70),
        ("70-74%", 0.70, 0.75), ("75-79%", 0.75, 0.80),
        ("80%+",   0.80, 1.01),
    ]
    tier_data = []
    for label, lo, hi in tiers:
        sub = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        roi = sub["profit"].sum() / (len(sub) * FLAT_BET) * 100
        tier_data.append({"label": label, "games": len(sub), "acc": acc, "roi": roi})
        mark = " <" if acc >= 0.60 else ""
        print(f"  {label:<15} {len(sub):>6,} {sub['correct'].sum():>8,} {acc:>8.1%} {roi:>+7.1f}%{mark}")

    print(f"\n  {'Season':<10} {'Games':>6} {'Accuracy':>9} {'ROI':>8}")
    print("  " + "-"*35)
    for s in sorted(df["season"].unique()):
        sub = df[df["season"]==s]
        print(f"  {s:<10} {len(sub):>6,} {sub['correct'].mean():>8.1%} {sub['profit'].sum()/(len(sub)*FLAT_BET)*100:>+7.1f}%")

    tp  = df["profit"].sum()
    roi = tp / (total * FLAT_BET) * 100
    print(f"\n  ROI (flat ${FLAT_BET}/game):")
    print(f"  Total wagered:  ${total*FLAT_BET:,.0f}")
    print(f"  Total profit:   ${tp:+,.2f}")
    print(f"  Overall ROI:    {roi:+.1f}%")

    hi = df[df["confidence"] >= 0.60]
    if len(hi):
        hp  = hi["profit"].sum()
        hr  = hp / (len(hi)*FLAT_BET)*100
        print(f"\n  High-confidence (>=60%) ROI:")
        print(f"  Games:  {len(hi):,}")
        print(f"  Profit: ${hp:+,.2f}")
        print(f"  ROI:    {hr:+.1f}%")

    print("\n" + "="*55)
    return tier_data, total, correct, accuracy, tp, roi


def generate_report(df, tier_data, total, correct, accuracy, total_profit, roi_overall):
    print("Generating HTML report...")
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = df.groupby("month").agg(
        games=("correct","count"), acc=("correct","mean"), profit=("profit","sum")
    ).reset_index()

    rows = ""
    for _, r in monthly.iterrows():
        roi = r["profit"]/(r["games"]*FLAT_BET)*100
        ac  = "#16a34a" if r["acc"]>=0.60 else ("#65a30d" if r["acc"]>=0.55 else "#374151")
        rc  = "#16a34a" if roi>0 else "#dc2626"
        rows += (f'<tr><td style="padding:10px 12px">{r["month"]}</td>'
                 f'<td style="padding:10px 12px;text-align:center">{r["games"]:,}</td>'
                 f'<td style="padding:10px 12px;text-align:center;color:{ac}">{r["acc"]:.1%}</td>'
                 f'<td style="padding:10px 12px;text-align:center;color:{rc}">{roi:+.1f}%</td></tr>')

    bars = ""
    for t in tier_data:
        ac = "#16a34a" if t["acc"]>=0.60 else ("#65a30d" if t["acc"]>=0.55 else "#9ca3af")
        rc = "#16a34a" if t["roi"]>0 else "#dc2626"
        bw = int(t["acc"]*100)
        bars += (f'<div style="margin-bottom:14px">'
                 f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                 f'<span style="font-size:13px;font-weight:500">{t["label"]}</span>'
                 f'<div style="display:flex;gap:12px">'
                 f'<span style="font-size:12px;color:#9ca3af">{t["games"]:,} games</span>'
                 f'<span style="font-size:13px;font-weight:600;color:{ac}">{t["acc"]:.1%}</span>'
                 f'<span style="font-size:12px;color:{rc}">ROI {t["roi"]:+.1f}%</span>'
                 f'</div></div>'
                 f'<div style="height:6px;background:#f3f4f6;border-radius:99px">'
                 f'<div style="width:{bw}%;height:100%;background:{ac};border-radius:99px"></div>'
                 f'</div></div>')

    recent = df.tail(20).iloc[::-1]
    rrows = ""
    for _, r in recent.iterrows():
        icon = "✅" if r["correct"] else "❌"
        pc   = "#16a34a" if r["profit"]>0 else "#dc2626"
        rrows += (f'<tr><td style="padding:8px 12px;font-size:12px;color:#9ca3af">{str(r["date"])[:10]}</td>'
                  f'<td style="padding:8px 12px;font-size:12px">{r["away_team"]} @ {r["home_team"]}</td>'
                  f'<td style="padding:8px 12px;font-size:12px;font-weight:500">{r["pick"].upper()}</td>'
                  f'<td style="padding:8px 12px;font-size:12px;text-align:center">{int(r["confidence"]*100)}%</td>'
                  f'<td style="padding:8px 12px;font-size:12px;text-align:center">{icon}</td>'
                  f'<td style="padding:8px 12px;font-size:12px;text-align:right;color:{pc}">${r["profit"]:+.2f}</td></tr>')

    ac_m = "#16a34a" if accuracy>=0.60 else ("#65a30d" if accuracy>=0.55 else "#374151")
    pc_m = "#16a34a" if total_profit>0 else "#dc2626"
    rc_m = "#16a34a" if roi_overall>0 else "#dc2626"

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>MLB Backtest</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#f3f4f6;color:#111}}
.w{{max-width:720px;margin:0 auto;padding:20px 14px 60px}}
table{{width:100%;border-collapse:collapse}}
th{{padding:10px 12px;font-size:11px;font-weight:500;text-transform:uppercase;color:#9ca3af;text-align:left;border-bottom:0.5px solid #e5e7eb}}
tr:hover td{{background:#fafafa}}</style></head><body><div class="w">
<div style="margin-bottom:24px"><h1 style="font-size:22px;font-weight:500">⚾ MLB Model Backtest</h1>
<div style="font-size:13px;color:#9ca3af;margin-top:3px">2025-2026 · {total:,} games · {datetime.now().strftime("%B %d %Y")}</div></div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px">
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center">
<div style="font-size:10px;color:#9ca3af;text-transform:uppercase;margin-bottom:6px">Overall accuracy</div>
<div style="font-size:32px;font-weight:500;color:{ac_m}">{accuracy:.1%}</div>
<div style="font-size:12px;color:#9ca3af;margin-top:4px">{correct:,} of {total:,}</div></div>
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center">
<div style="font-size:10px;color:#9ca3af;text-transform:uppercase;margin-bottom:6px">Total profit</div>
<div style="font-size:32px;font-weight:500;color:{pc_m}">${total_profit:+,.0f}</div>
<div style="font-size:12px;color:#9ca3af;margin-top:4px">${total*FLAT_BET:,.0f} wagered</div></div>
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:16px;text-align:center">
<div style="font-size:10px;color:#9ca3af;text-transform:uppercase;margin-bottom:6px">Overall ROI</div>
<div style="font-size:32px;font-weight:500;color:{rc_m}">{roi_overall:+.1f}%</div>
<div style="font-size:12px;color:#9ca3af;margin-top:4px">vs 52.4% break-even</div></div></div>
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;margin-bottom:20px">
<div style="font-size:10px;font-weight:500;text-transform:uppercase;color:#9ca3af;margin-bottom:16px">Accuracy by confidence tier</div>
{bars}</div>
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px;margin-bottom:20px">
<div style="font-size:10px;font-weight:500;text-transform:uppercase;color:#9ca3af;margin-bottom:16px">Monthly breakdown</div>
<table><thead><tr><th>Month</th><th style="text-align:center">Games</th><th style="text-align:center">Accuracy</th><th style="text-align:center">ROI</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div style="background:#fff;border-radius:12px;border:0.5px solid #e5e7eb;padding:20px">
<div style="font-size:10px;font-weight:500;text-transform:uppercase;color:#9ca3af;margin-bottom:16px">Last 20 picks</div>
<table><thead><tr><th>Date</th><th>Matchup</th><th>Pick</th><th style="text-align:center">Conf</th><th style="text-align:center">Result</th><th style="text-align:right">Profit</th></tr></thead>
<tbody>{rrows}</tbody></table></div>
</div></body></html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Report saved -> {OUTPUT_HTML}")


if __name__ == "__main__":
    test, model = load_data_and_model()
    df_r = run_backtest(test, model)
    td, total, correct, acc, tp, roi = analyse(df_r)
    generate_report(df_r, td, total, correct, acc, tp, roi)
    print("\n✅ Done! Open raw/backtest_report.html in Chrome.")
