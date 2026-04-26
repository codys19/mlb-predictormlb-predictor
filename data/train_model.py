"""
train_model.py
--------------
Trains XGBoost model. Automatically uses starter features if available.

Train: 2021-2024  |  Test: 2025-2026

RUN:
    python data/train_model.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

TRAINING_DATA = "raw/training_data.csv"

BASE_FEATURES = [
    "home_rolling_runs_scored","home_rolling_runs_allowed",
    "home_rolling_win_rate","home_rolling_run_diff",
    "away_rolling_runs_scored","away_rolling_runs_allowed",
    "away_rolling_win_rate","away_rolling_run_diff",
    "win_rate_diff","run_diff_diff","runs_scored_diff","is_home",
    # 5-game streak features
    "home_rolling_win_rate_5","home_rolling_run_diff_5",
    "away_rolling_win_rate_5","away_rolling_run_diff_5",
    "win_rate_diff_5","run_diff_diff_5",
]

TEAM_PITCHER_FEATURES = [
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

BULLPEN_FEATURES = [
    "home_bullpen_ip_3d","away_bullpen_ip_3d",
    "home_bullpen_appearances_3d","away_bullpen_appearances_3d",
    "bullpen_fatigue_diff",
]

PARK_FEATURES = [
    "home_park_factor",
    "park_factor_is_known",
]

TARGET = "home_team_won"


def load_data():
    print("📂 Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    print(f"  ✅ {len(df):,} rows, seasons: {sorted(df['season'].unique())}")
    return df


def build_feature_list(df):
    """Auto-detect which features are available and populated."""
    feature_cols = BASE_FEATURES.copy()

    # Add team pitcher features if populated
    tp_available = [c for c in TEAM_PITCHER_FEATURES if c in df.columns and df[c].notna().mean() > 0.5]
    if tp_available:
        feature_cols += tp_available
        print(f"  ✅ Team pitcher features: {len(tp_available)}")

    # Add bullpen fatigue features if available (requires fetch_bullpen.py to have run)
    bp_available = [c for c in BULLPEN_FEATURES if c in df.columns and df[c].notna().mean() > 0.5]
    if bp_available:
        feature_cols += bp_available
        print(f"  ✅ Bullpen fatigue features: {len(bp_available)}")
    else:
        print("  ℹ️  No bullpen features — run fetch_bullpen.py to add fatigue signal")

    # Add park factor features if available
    pk_available = [c for c in PARK_FEATURES if c in df.columns and df[c].notna().mean() > 0.5]
    if pk_available:
        feature_cols += pk_available
        print(f"  ✅ Park factor features: {len(pk_available)}")
    else:
        print("  ℹ️  No park features — add raw/park_factors.csv and rebuild")

    # Add starter features if populated.
    # Threshold is 0.25 (not 0.5) because starter cols are intentionally NaN
    # for unmatched games — XGBoost handles missing values natively, so 47%
    # population is perfectly fine and better than filling with pool averages.
    st_available = [c for c in STARTER_FEATURES if c in df.columns and df[c].notna().mean() > 0.25]
    if st_available:
        feature_cols += st_available
        print(f"  ✅ Starter features: {len(st_available)} — model will use individual starter stats!")
    else:
        print("  ℹ️  No starter features — using team averages (run fetch_starter_logs.py to upgrade)")

    print(f"  Total features: {len(feature_cols)}")
    return feature_cols


def split_data(df, feature_cols):
    print(f"\n✂️  Splitting (train=2021-2024, test=2025-2026)...")

    # Only require non-NaN on base/team features.
    # Starter columns are intentionally sparse (NaN when no individual match) —
    # XGBoost handles missing values natively so we must NOT drop those rows.
    required = [c for c in feature_cols if not any(
        c.startswith(p) for p in ("home_starter_","away_starter_","starter_","has_individual_","home_bullpen_","away_bullpen_","bullpen_")
    )]

    train = df[df["season"] <= 2024].dropna(subset=required + [TARGET])
    test  = df[df["season"] >= 2025].dropna(subset=required + [TARGET])

    X_train = train[feature_cols]; y_train = train[TARGET].astype(int)
    X_test  = test[feature_cols];  y_test  = test[TARGET].astype(int)

    print(f"  Train: {len(X_train):,} games  ({y_train.mean():.1%} home wins)")
    print(f"  Test:  {len(X_test):,} games   ({y_test.mean():.1%} home wins)")
    return X_train, y_train, X_test, y_test, test


def train_model(X_train, y_train):
    print("\n🤖 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,           # shallower trees generalise better
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,  # fewer features per tree reduces multicollinearity
        min_child_weight=15,   # require more samples per leaf
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=2.0,        # L2 regularization
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    print("  ✅ Done")
    return model


def evaluate(model, X_train, y_train, X_test, y_test):
    print("\n📊 Results:")
    print("─" * 50)

    train_acc  = accuracy_score(y_train, model.predict(X_train))
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    test_acc   = accuracy_score(y_test, test_preds)

    print(f"  Training accuracy : {train_acc:.1%}")
    print(f"  Test accuracy     : {test_acc:.1%}  ← real score")
    print()

    for threshold in [0.55, 0.60, 0.65, 0.70]:
        mask = (test_probs >= threshold) | (test_probs <= (1-threshold))
        if mask.sum() >= 10:
            acc = accuracy_score(y_test[mask], test_preds[mask])
            print(f"  >{threshold:.0%} confidence: {mask.sum():>5} games  →  {acc:.1%} accuracy")

    print()
    print(classification_report(y_test, test_preds, target_names=["Away wins","Home wins"]))
    return test_acc, test_probs, test_preds


def show_feature_importance(model, feature_cols):
    print("🔍 Top 15 features:")
    print("─" * 50)

    imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in imp.head(15).iterrows():
        if row["importance"] > 0:
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<40} {row['importance']:.4f}  {bar}")

    imp.to_csv("raw/feature_importance.csv", index=False)
    return imp


if __name__ == "__main__":
    df           = load_data()
    feature_cols = build_feature_list(df)
    X_train, y_train, X_test, y_test, test_df = split_data(df, feature_cols)
    model        = train_model(X_train, y_train)
    test_acc, test_probs, test_preds = evaluate(model, X_train, y_train, X_test, y_test)
    show_feature_importance(model, feature_cols)
    model.save_model("raw/model.json")
    print("\n💾 Model saved → raw/model.json")
    print(f"\n{'═'*50}")
    print(f"  FINAL TEST ACCURACY: {test_acc:.1%}")
    print(f"{'═'*50}")