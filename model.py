"""
model.py — World Cup predictor core
────────────────────────────────────
Data sources:
  1. Match results  → https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
                      Save as: results.csv
  2. FIFA rankings  → https://www.kaggle.com/datasets/cashncarry/fifaworldranking
                      Save as: fifa_rankings.csv

Both files go in the same directory as this file.
If either is missing the app still runs — results fall back to synthetic
data and rankings fall back to a neutral default of 50.

Deployment:
  Run once locally (with results.csv present) → model_artifacts/ is created.
  Commit model_artifacts/ to your repo → Streamlit Cloud loads from disk,
  never needs results.csv or fifa_rankings.csv.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from collections import Counter
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

DATA_PATH     = "results.csv"
RANKINGS_PATH = "fifa_rankings.csv"
WINDOW        = 15   # rolling form window

# ── Saved artifact paths — relative to this file so they work on any machine ──
_HERE            = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR    = os.path.join(_HERE, "model_artifacts")
MODEL_FILE       = os.path.join(ARTIFACTS_DIR, "xgb_model.json")
ENCODER_FILE     = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
FEATURE_COL_FILE = os.path.join(ARTIFACTS_DIR, "feature_cols.pkl")
DF_FILE          = os.path.join(ARTIFACTS_DIR, "match_data.parquet")
RANKINGS_FILE    = os.path.join(ARTIFACTS_DIR, "rankings_data.parquet")

# ── 2026 World Cup nations ────────────────────────────────────────────────────
TEAM_LIST = sorted([
    "Algeria", "Argentina", "Australia", "Austria", "Belgium",
    "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde", "Colombia",
    "Croatia", "Curacao", "Czechia", "DR Congo", "Ecuador", "Egypt",
    "England", "France", "Germany", "Ghana", "Haiti", "Iran", "Iraq",
    "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea",
    "Spain", "Sweden", "Switzerland", "Tunisia", "Turkey",
    "United States", "Uruguay", "Uzbekistan",
])

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 3.0,
    "FIFA World Cup qualification": 2.0,
    "Copa América": 2.0,
    "UEFA Euro": 2.0,
    "Africa Cup of Nations": 2.0,
    "Friendly": 0.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        st.warning(
            "⚠️ **results.csv not found.** Running on synthetic data. "
            "Download from Kaggle and place `results.csv` in the app directory.",
            icon="⚠️",
        )
        df = _generate_synthetic_data()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    def _outcome(row):
        if row["home_score"] > row["away_score"]:   return "home_win"
        elif row["home_score"] < row["away_score"]: return "away_win"
        else:                                        return "draw"

    df["outcome"] = df.apply(_outcome, axis=1)
    return df


def _generate_synthetic_data() -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", "2024-06-30", periods=8000)
    rows  = []
    for d in dates:
        home, away = rng.choice(TEAM_LIST, 2, replace=False)
        rows.append({
            "date":       d,
            "home_team":  home,
            "away_team":  away,
            "home_score": int(rng.poisson(1.5)),
            "away_score": int(rng.poisson(1.2)),
            "tournament": rng.choice(
                ["FIFA World Cup", "FIFA World Cup qualification",
                 "Friendly", "Copa América", "UEFA Euro"],
                p=[0.05, 0.20, 0.40, 0.10, 0.25],
            ),
            "neutral": bool(rng.integers(0, 2)),
        })
    return pd.DataFrame(rows)


# ── FIFA rankings ─────────────────────────────────────────────────────────────

def load_rankings() -> pd.DataFrame | None:
    """Return a sorted rankings DataFrame, or None if the file is missing."""
    if not os.path.exists(RANKINGS_PATH):
        return None
    rk = pd.read_csv(RANKINGS_PATH, parse_dates=["rank_date"])
    rk = rk[["rank_date", "country_full", "rank"]].dropna()
    rk = rk.sort_values("rank_date").reset_index(drop=True)
    return rk


def _get_ranking(team: str, date, rankings: pd.DataFrame | None) -> float:
    """Most recent FIFA rank for `team` before `date`. Defaults to 50 if unknown."""
    if rankings is None:
        return 50.0
    past = rankings[
        (rankings["country_full"] == team) &
        (rankings["rank_date"] <= date)
    ]
    if len(past) == 0:
        return 100.0
    return float(past.iloc[-1]["rank"])


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _team_rolling_stats(df: pd.DataFrame, team: str, before_date) -> dict:
    mask   = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    recent = df[mask].sort_values("date").tail(WINDOW)

    if len(recent) == 0:
        return dict(win_rate=0.33, draw_rate=0.33, loss_rate=0.34,
                    avg_gf=1.2, avg_ga=1.2, goal_diff=0.0, games=0)

    wins = draws = losses = gf = ga = 0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            gf += row["home_score"]; ga += row["away_score"]
            if   row["outcome"] == "home_win": wins   += 1
            elif row["outcome"] == "draw":     draws  += 1
            else:                              losses += 1
        else:
            gf += row["away_score"]; ga += row["home_score"]
            if   row["outcome"] == "away_win": wins   += 1
            elif row["outcome"] == "draw":     draws  += 1
            else:                              losses += 1

    n = len(recent)
    return dict(
        win_rate=wins / n, draw_rate=draws / n, loss_rate=losses / n,
        avg_gf=gf / n, avg_ga=ga / n, goal_diff=(gf - ga) / n, games=n,
    )


def _h2h_stats(df: pd.DataFrame, home: str, away: str, before_date) -> dict:
    mask = (
        (
            ((df["home_team"] == home) & (df["away_team"] == away)) |
            ((df["home_team"] == away) & (df["away_team"] == home))
        ) &
        (df["date"] < before_date)
    )
    h2h = df[mask]

    home_wins = draws = away_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home:
            if   row["outcome"] == "home_win": home_wins += 1
            elif row["outcome"] == "draw":     draws     += 1
            else:                              away_wins += 1
        else:
            if   row["outcome"] == "away_win": home_wins += 1
            elif row["outcome"] == "draw":     draws     += 1
            else:                              away_wins += 1

    total = len(h2h)
    if total == 0:
        return dict(h2h_home_win_rate=0.33, h2h_draw_rate=0.33,
                    h2h_away_win_rate=0.34, h2h_total=0)
    return dict(
        h2h_home_win_rate=home_wins / total,
        h2h_draw_rate=draws / total,
        h2h_away_win_rate=away_wins / total,
        h2h_total=total,
    )


def build_features(
    df: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Row-wise feature engineering. Only uses data before each match (no leakage)."""
    records = []
    for _, row in df.iterrows():
        hs  = _team_rolling_stats(df, row["home_team"], row["date"])
        as_ = _team_rolling_stats(df, row["away_team"], row["date"])
        h2h = _h2h_stats(df, row["home_team"], row["away_team"], row["date"])
        tw  = TOURNAMENT_WEIGHT.get(row.get("tournament", "Friendly"), 1.0)
        nv  = int(row.get("neutral", False))

        home_rank  = _get_ranking(row["home_team"], row["date"], rankings)
        away_rank  = _get_ranking(row["away_team"], row["date"], rankings)
        rank_delta = away_rank - home_rank

        records.append({
            # Home form
            "home_win_rate":  hs["win_rate"],
            "home_draw_rate": hs["draw_rate"],
            "home_loss_rate": hs["loss_rate"],
            "home_avg_gf":    hs["avg_gf"],
            "home_avg_ga":    hs["avg_ga"],
            "home_goal_diff": hs["goal_diff"],
            "home_games":     hs["games"],
            # Away form
            "away_win_rate":  as_["win_rate"],
            "away_draw_rate": as_["draw_rate"],
            "away_loss_rate": as_["loss_rate"],
            "away_avg_gf":    as_["avg_gf"],
            "away_avg_ga":    as_["avg_ga"],
            "away_goal_diff": as_["goal_diff"],
            "away_games":     as_["games"],
            # Differentials
            "win_rate_diff":  hs["win_rate"]  - as_["win_rate"],
            "goal_diff_diff": hs["goal_diff"] - as_["goal_diff"],
            "avg_gf_diff":    hs["avg_gf"]    - as_["avg_gf"],
            "avg_ga_diff":    hs["avg_ga"]    - as_["avg_ga"],
            # Head-to-head
            "h2h_home_win_rate": h2h["h2h_home_win_rate"],
            "h2h_draw_rate":     h2h["h2h_draw_rate"],
            "h2h_away_win_rate": h2h["h2h_away_win_rate"],
            "h2h_total":         h2h["h2h_total"],
            # Context
            "tournament_weight": tw,
            "neutral_venue":     nv,
            # Draw-signal
            "form_closeness": abs(hs["win_rate"] - as_["win_rate"]),
            # FIFA rankings
            "home_rank":  home_rank,
            "away_rank":  away_rank,
            "rank_delta": rank_delta,
            # Target
            "outcome": row["outcome"],
        })

    feat_df      = pd.DataFrame(records)
    feature_cols = [c for c in feat_df.columns if c != "outcome"]
    return feat_df, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def _artifacts_exist() -> bool:
    """True only when every artifact file is present on disk."""
    return all(os.path.exists(p) for p in [
        MODEL_FILE, ENCODER_FILE, FEATURE_COL_FILE, DF_FILE,
    ])


def _save_artifacts(clf, df, rankings, feature_cols, le):
    """Persist everything to disk so Streamlit Cloud never needs to retrain."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    clf.save_model(MODEL_FILE)
    joblib.dump(le,           ENCODER_FILE)
    joblib.dump(feature_cols, FEATURE_COL_FILE)
    df.to_parquet(DF_FILE, index=False)
    if rankings is not None:
        rankings.to_parquet(RANKINGS_FILE, index=False)
    print(f"✅ Artifacts saved to {ARTIFACTS_DIR}/")


def _load_artifacts():
    """Load pre-trained artifacts from disk — no CSV files required."""
    clf = XGBClassifier()
    clf.load_model(MODEL_FILE)
    le           = joblib.load(ENCODER_FILE)
    feature_cols = joblib.load(FEATURE_COL_FILE)
    df           = pd.read_parquet(DF_FILE)
    rankings     = (
        pd.read_parquet(RANKINGS_FILE)
        if os.path.exists(RANKINGS_FILE) else None
    )
    print("✅ Loaded pre-trained artifacts from disk")
    return clf, df, rankings, feature_cols, le


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _train_and_save() -> tuple:
    """Full training pipeline. Saves artifacts to disk when done."""
    df       = load_data()
    rankings = load_rankings()

    if rankings is None:
        st.info(
            "ℹ️ **fifa_rankings.csv not found.** Rankings defaulting to 50. "
            "Download from Kaggle for better accuracy.",
        )

    df = df[df["year"] >= 2000].copy()

    with st.spinner("Engineering features — runs once, then saved to disk..."):
        df["recency_weight"] = (df["year"] - 2000 + 1) ** 1.5
        w      = df["recency_weight"] / df["recency_weight"].sum()
        sample = df.sample(min(len(df), 8000), weights=w, random_state=42)
        feat_df, feature_cols = build_features(sample, rankings)

    feat_df = feat_df.dropna()
    X       = feat_df[feature_cols].values
    y_raw   = feat_df["outcome"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print("Label mapping    :", dict(enumerate(le.classes_)))
    print("Train distribution:", Counter(y_train))

    classes, counts = np.unique(y_train, return_counts=True)
    weight_map      = {c: len(y_train) / (len(classes) * cnt)
                       for c, cnt in zip(classes, counts)}
    sample_weight   = np.array([weight_map[yi] for yi in y_train])

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = clf.predict(X_test)
    print("\n── Model evaluation ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    feat_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(feat_imp.head(12))

    _save_artifacts(clf, df, rankings, feature_cols, le)
    return clf, df, rankings, feature_cols, le


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    Load from saved artifacts if present (fast, no CSVs needed).
    Train from scratch only when artifacts folder is missing.
    """
    if _artifacts_exist():
        return _load_artifacts()

    # No artifacts — need results.csv to train
    if not os.path.exists(DATA_PATH):
        st.error(
            "❌ **Cannot start:** `model_artifacts/` not found and `results.csv` is missing.\n\n"
            "**To fix:**\n"
            "- Add `results.csv` locally and run `streamlit run app.py` once to generate artifacts\n"
            "- Then commit the `model_artifacts/` folder to your repo before deploying"
        )
        st.stop()

    return _train_and_save()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_match(
    model,
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    neutral: bool,
    label_encoder: LabelEncoder,
    rankings: pd.DataFrame | None = None,
    tournament: str = "FIFA World Cup",
) -> dict | None:

    today   = pd.Timestamp.now()
    hs      = _team_rolling_stats(df, home_team, today)
    as_     = _team_rolling_stats(df, away_team, today)
    h2h_raw = _h2h_stats(df, home_team, away_team, today)
    tw      = TOURNAMENT_WEIGHT.get(tournament, 1.0)

    if hs["games"] == 0 and as_["games"] == 0:
        return None

    home_rank  = _get_ranking(home_team, today, rankings)
    away_rank  = _get_ranking(away_team, today, rankings)
    rank_delta = away_rank - home_rank

    def _build_X(h, a, hr, ar, rd, nv):
        """Build feature vector — order must exactly match build_features."""
        return np.array([[
            h["win_rate"],  h["draw_rate"],  h["loss_rate"],
            h["avg_gf"],    h["avg_ga"],     h["goal_diff"],  h["games"],
            a["win_rate"],  a["draw_rate"],  a["loss_rate"],
            a["avg_gf"],    a["avg_ga"],     a["goal_diff"],  a["games"],
            h["win_rate"]  - a["win_rate"],
            h["goal_diff"] - a["goal_diff"],
            h["avg_gf"]    - a["avg_gf"],
            h["avg_ga"]    - a["avg_ga"],
            h2h_raw["h2h_home_win_rate"],
            h2h_raw["h2h_draw_rate"],
            h2h_raw["h2h_away_win_rate"],
            h2h_raw["h2h_total"],
            tw,
            int(nv),
            abs(h["win_rate"] - a["win_rate"]),
            hr, ar, rd,
        ]])

    if neutral:
        # ── Neutral venue: run both perspectives and average to remove home bias
        # Forward:  home_team as "home", away_team as "away"
        # Reversed: away_team as "home", home_team as "away"
        probs_fwd = model.predict_proba(_build_X(hs, as_,  home_rank, away_rank,  rank_delta, True))[0]
        probs_rev = model.predict_proba(_build_X(as_, hs,  away_rank, home_rank, -rank_delta, True))[0]

        classes   = label_encoder.classes_   # ['away_win', 'draw', 'home_win']
        map_fwd   = dict(zip(classes, probs_fwd))
        map_rev   = dict(zip(classes, probs_rev))

        # In the reversed prediction home_win = away_team won, away_win = home_team won
        home_win_prob = (map_fwd.get("home_win", 0) + map_rev.get("away_win", 0)) / 2
        draw_prob     = (map_fwd.get("draw",     0) + map_rev.get("draw",     0)) / 2
        away_win_prob = (map_fwd.get("away_win", 0) + map_rev.get("home_win", 0)) / 2
    else:
        # ── Non-neutral: home advantage is real, single prediction is correct
        probs     = model.predict_proba(_build_X(hs, as_, home_rank, away_rank, rank_delta, False))[0]
        prob_map  = dict(zip(label_encoder.classes_, probs))
        home_win_prob = prob_map.get("home_win", 0.33)
        draw_prob     = prob_map.get("draw",     0.33)
        away_win_prob = prob_map.get("away_win", 0.34)

    # H2H counts for UI
    mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    h2h_df = df[mask]

    home_wins = draws = away_wins = 0
    for _, row in h2h_df.iterrows():
        if row["home_team"] == home_team:
            if   row["outcome"] == "home_win": home_wins += 1
            elif row["outcome"] == "draw":     draws     += 1
            else:                              away_wins += 1
        else:
            if   row["outcome"] == "away_win": home_wins += 1
            elif row["outcome"] == "draw":     draws     += 1
            else:                              away_wins += 1

    return {
        "home_win": home_win_prob,
        "draw":     draw_prob,
        "away_win": away_win_prob,
        "h2h": {"home_wins": home_wins, "draws": draws, "away_wins": away_wins},
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_team_stats(df: pd.DataFrame, team: str) -> dict:
    """Last-WINDOW match summary for the form display in app.py."""
    mask   = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < pd.Timestamp.now())
    )
    recent = df[mask].sort_values("date").tail(WINDOW)

    if len(recent) == 0:
        return dict(wins=0, draws=0, losses=0, gf=0.0, ga=0.0,
                    games=0, avg_gf=1.2, avg_ga=1.2)

    wins = draws = losses = gf = ga = 0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            gf += row["home_score"]; ga += row["away_score"]
            if   row["outcome"] == "home_win": wins   += 1
            elif row["outcome"] == "draw":     draws  += 1
            else:                              losses += 1
        else:
            gf += row["away_score"]; ga += row["home_score"]
            if   row["outcome"] == "away_win": wins   += 1
            elif row["outcome"] == "draw":     draws  += 1
            else:                              losses += 1

    n = len(recent)
    return dict(
        wins=wins, draws=draws, losses=losses,
        gf=gf / n, ga=ga / n, games=n,
        avg_gf=gf / n, avg_ga=ga / n,
    )