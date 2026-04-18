# MLB AI Predictor — Project Handoff V3

## What This Is
A machine learning model that predicts MLB game winners, with a daily HTML dashboard showing picks, confidence levels, moneyline odds, EV, O/U picks, pitcher K props, $10 flat P&L tracking, and a running W/L record. Runs automatically every morning on GitHub Actions and is served via GitHub Pages.

---

## Live URL
```
https://codys19.github.io/mlb-predictormlb-predictor/mlb_predictor.html
```

## GitHub Repo
```
https://github.com/codys19/mlb-predictormlb-predictor
```
- **Branch**: main
- **Actions**: runs daily at 2 PM UTC (7 AM PT) via `.github/workflows/daily.yml`
- **Pages**: deployed from main branch root
- **Personal access token expires**: May 14, 2026 — regenerate before then

---

## Current Status
- **Model accuracy**: 55.5% overall, 60.6% at >70% confidence (tested on 2025–2026 games)
- **ML Record**: 17-17 as of April 15 2026 (live, tracked in record.json)
- **Daily script**: Working — grades yesterday, fetches odds, predicts today, generates HTML
- **GitHub Actions**: Running successfully every morning at ~7:59 AM PT
- **GitHub Pages**: Live and updating automatically
- **Tabs**: ML / O/U / Props — all three implemented and grading separately
- **$10 flat P&L**: Tracking per confidence tier
- **Python**: 3.12 | **Packages**: pandas, numpy, xgboost, scikit-learn, pybaseball, mlb-statsapi, requests

---

## File Structure
```
C:\MLB\
├── data\
│   └── mlb.py                    ← main daily script (MODIFIED — see below)
│   └── build_features.py         ← training data builder (MODIFIED — see below)
│   └── fetch_game_starters.py    ← NEW script — builds game→starter mapping
│   └── fetch_starter_logs.py     ← builds pitcher season stats
│   └── fetch_games.py
│   └── fetch_pitchers.py
│   └── train_model.py
│   └── backtest.py
├── raw\
│   ├── game_results.csv          ← 24,309 games (2021–2026)
│   ├── pitcher_stats.csv
│   ├── starter_logs.csv          ← pitcher season stats (ERA/WHIP/K9 per pitcher)
│   ├── team_starter_pool.csv     ← team rotation averages (fallback)
│   ├── game_starters.csv         ← NEW: (date, team) → actual starter name
│   ├── model.json                ← trained XGBoost model (37 features)
│   ├── record.json               ← running W/L record (ML + O/U + Props)
│   ├── picks_YYYY-MM-DD.json
│   ├── ou_YYYY-MM-DD.json
│   └── props_YYYY-MM-DD.json
├── mlb_predictor.html
├── history_YYYY-MM-DD.html
├── run_mlb.bat
├── .github/workflows/daily.yml
└── mlb_log.txt
```

---

## What Was Done This Session (Individual Starter Stats — Priority #1)

### The Problem Being Solved
The model was using **team rotation averages** for every game — meaning every game for a team used the same blended ERA/WHIP/K9 regardless of who was actually pitching. Individual starter stats are the highest-leverage accuracy improvement (+3-4% expected).

### Files Changed / Created

#### 1. `data/fetch_game_starters.py` — NEW FILE ✅ (fixed, ready to run)
Pulls historical game→starter mappings from MLB Stats API (free, no rate limits).
- Uses `statsapi.schedule(start_date=..., end_date=...)` — one call per season
- For completed games, `home_probable_pitcher` / `away_probable_pitcher` = actual starters
- Falls back to `statsapi.boxscore_data()` for games missing pitcher names
- Output: `raw/game_starters.csv` columns: `date, season, team_abbr, starter_name`
- Runtime: ~2-3 minutes for all 6 seasons

**IMPORTANT**: The script uses `start_date` / `end_date` (snake_case). This was a bug that was already fixed.

#### 2. `data/build_features.py` — MODIFIED ✅ (ready to use after step 1)
Now attaches **individual starter stats** per game instead of team rotation averages.

Key new logic in `attach_pitcher_stats()`:
- Loads `game_starters.csv` to get the actual starter for each (date, team)
- Looks up that starter in `starter_logs.csv` by name+season
- Fallback chain: `exact name match → prior season → last-name fuzzy → team pool average → league average`
- Tracks match rate and warns if < 30% individual matches
- All individual starter columns: `home_starter_era`, `home_starter_whip`, `home_starter_k_per_9`, `home_starter_bb_per_9`, `home_starter_fip_proxy` (+ away_ versions)
- Diff features: `starter_era_diff`, `starter_whip_diff`, `starter_k_diff`, `starter_fip_diff`

#### 3. `data/mlb.py` — MODIFIED ✅ (ready to deploy)
Two new functions added + `run_predictions()` and `predict_props()` updated.

**New functions:**
- `build_starter_stats_by_name()` — loads `starter_logs.csv` keyed by pitcher name (most recent season with 3+ GS). Builds a last-name index for fuzzy matching. Returns dict with `__lastname_index__` key.
- `_lookup_pitcher_stats(pitcher_name, stats_by_name, pool_fallback)` — looks up pitcher stats with fallback chain: exact → last-name fuzzy (uses first initial for ambiguous) → pool → league avg.

**Changes to `run_predictions()`:**
- Added `starter_stats_by_name=None` parameter
- Now looks up the actual probable starter's individual stats instead of the team pool average
- Falls back to pool → league average when starter not found
- Attaches all 10 individual starter feature columns + 4 diff features to the feature row

**Changes to `predict_props()`:**
- Added `starter_stats_by_name=None` parameter  
- Now uses individual pitcher K/9 for strikeout projections instead of team average

**Main block change:** Added `starter_stats_by_name = build_starter_stats_by_name()` and passes it into both `run_predictions()` and `predict_props()`.

---

## Steps Still Needed to Complete Priority #1

### Step 1 — Run fetch_game_starters.py (NOT DONE YET)
```bash
python data/fetch_game_starters.py
```
Expected output: `raw/game_starters.csv` with ~14,000+ rows. Check the season coverage summary at the end — each season should show 3,000+ entries (30 teams × ~100+ starts each).

If any season shows low coverage, it usually means that season's games didn't have `probable_pitcher` populated in the schedule API and the boxscore fallback didn't fully recover them. Not a blocker — pool averages fill in.

### Step 2 — Rebuild training data
```bash
python data/build_features.py
```
Watch for the "Individual starter match" line in the output. Aim for 60%+ individual match rate. Lower is okay — the fallback to pool averages means the model still trains, just with less individual signal.

### Step 3 — Retrain the model
```bash
python data/train_model.py
```

### Step 4 — Run backtest
```bash
python data/backtest.py
```
Compare to baseline: 55.5% overall, 60.6% at >70% confidence. Looking for improvement in the >70% bucket especially.

### Step 5 — Deploy
```bash
git add raw/game_starters.csv raw/model.json raw/training_data.csv
git add data/mlb.py data/build_features.py data/fetch_game_starters.py
git commit -m "Individual starter stats — priority #1 accuracy improvement"
git push
```
If push rejected (Actions committed first): `git stash → git pull --rebase → git stash pop → git push`

---

## Model Details

### Features (37 total — trained model)
- Rolling team stats (last 15 games): runs scored/allowed, win rate, run diff
- Difference features: win_rate_diff, run_diff_diff, runs_scored_diff
- Team pitcher averages: ERA, WHIP, SO9, SO/W
- Starter pool features: ERA, WHIP, K/9, BB/9, FIP proxy
- `is_home`

### New Features (after retrain — 14 additional)
- `home_starter_era`, `home_starter_whip`, `home_starter_k_per_9`, `home_starter_bb_per_9`, `home_starter_fip_proxy`
- `away_starter_era`, `away_starter_whip`, `away_starter_k_per_9`, `away_starter_bb_per_9`, `away_starter_fip_proxy`
- `starter_era_diff`, `starter_whip_diff`, `starter_k_diff`, `starter_fip_diff`

### Training
- Train: 2021–2024 | Test: 2025–2026
- Algorithm: XGBoost (n_estimators=500, max_depth=4, learning_rate=0.05)

### Backtest Results (2025–2026, current model)
| Confidence | Games | Accuracy | ROI |
|---|---|---|---|
| 70–74% | 112 | 63.4% | +12.1% |
| 80%+ | 43 | 74.4% | +31.6% |

**Only bet 70%+ picks.**

---

## record.json Structure
```json
{
  "ml": {
    "all_time": {"w": 17, "l": 17},
    "by_conf": {
      "50": {"w": 2, "l": 5, "pnl": -22.50},
      "55": {"w": 6, "l": 3, "pnl": 18.40},
      "60": {"w": 6, "l": 7, "pnl": -8.20},
      "70": {"w": 3, "l": 2, "pnl": 12.10}
    },
    "daily": {"2026-04-13": {"w": 3, "l": 3}}
  },
  "ou": { "...same structure..." },
  "props": { "...same structure..." }
}
```

---

## API Keys
- **The Odds API**: GitHub Secret `ODDS_API_KEY`. Free tier = 500 requests/month. Props fetch uses ~16 calls/day.
- **mlb-statsapi**: Free, no key needed.

---

## GitHub Workflow (`daily.yml`)
```yaml
on:
  schedule:
    - cron: '0 14 * * *'   # 2 PM UTC = ~7 AM PT
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-predictor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install pandas numpy xgboost scikit-learn requests mlb-statsapi pybaseball
      - run: python data/mlb.py
        env: { ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }} }
      - run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add raw/record.json mlb_predictor.html
          git add raw/picks_*.json raw/ou_*.json raw/props_*.json history_*.html || true
          git diff --staged --quiet || git commit -m "Daily picks $(date +'%Y-%m-%d')"
          git push
```

---

## Remaining Roadmap (Priority Order)

### ✅ DONE: Individual starter stats (+3-4% ML accuracy)
Files modified, just needs fetch_game_starters.py to run and then retrain.

### 2. Refresh game_results.csv weekly
Add weekly cron job to keep rolling stats current mid-season. Prevents stale team stats after a few weeks.

### 3. Vegas odds as ML feature (+2-3% accuracy)
Feed historical closing lines into training data. Requires Odds API paid historical endpoint.

### 4. Bullpen fatigue feature (+1-2% accuracy)
Track reliever usage over last 3 days per team.

### 5. Better props model
Individual pitcher game logs from pybaseball instead of season K/9 averages.

---

## Retraining Pipeline
```bash
python data/fetch_games.py
python data/fetch_pitchers.py
python data/fetch_starter_logs.py
python data/fetch_game_starters.py   ← NEW, run this too
python data/build_features.py
python data/train_model.py
python data/backtest.py
```

## When Token Expires (May 14, 2026)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` + `workflow` scopes
3. `git remote set-url origin https://codys19:NEW_TOKEN@github.com/codys19/mlb-predictormlb-predictor.git`

## Known Issues / Quirks
1. **DeprecationWarning** about `DataFrameGroupBy.apply` — harmless
2. **OAK → ATH**: handled in TEAM_MAP
3. **Season assignment**: by row position in game_results.csv (162 rows per team per season)
4. **Props grading**: uses pitcher last-name matching against box score — may miss some games
5. **O/U confidence cap**: capped at 80% max
