# ⚽ 2026 World Cup Match Predictor

A Streamlit app that predicts match outcomes (win / draw / loss) using a
Random Forest trained on historical international football results.

## Quick start

```bash
# 1. Clone / copy this folder, then install deps
pip install -r requirements.txt

# 2. (Recommended) Add real data
#    Download from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
#    Save the CSV as results.csv in this directory.
#    Without it, the app runs on synthetic data for demonstration.

# 3. Launch
streamlit run app.py
```

The app opens at http://localhost:8501

## Project structure

```
worldcup_predictor/
├── app.py           # Streamlit UI
├── model.py         # Feature engineering, training, prediction
├── requirements.txt
├── README.md
└── results.csv      # ← add this (not included)
```

## Features engineered

| Feature | Description |
|---|---|
| `home_win_rate` | Win % across last 10 matches |
| `home_avg_gf/ga` | Goals scored / conceded per game |
| `home_goal_diff` | Net goal difference per game |
| `win_rate_diff` | Home minus away win rate |
| `goal_diff_diff` | Home minus away goal differential |
| `h2h_home_win_rate` | Historical head-to-head win rate |
| `tournament_weight` | World Cup weighted 3×, friendlies 0.5× |
| `neutral_venue` | Boolean flag |

## Model

- **Algorithm:** Random Forest (300 trees, max depth 12)
- **Training data:** Post-1990 international results (up to 5,000 sampled rows)
- **Target:** `home_win` / `draw` / `away_win`
- **Typical accuracy:** ~52–56% (3-class; baseline is 33%)

## Improving accuracy

1. **Add ELO ratings** — download from eloratings.net and merge by team/date
2. **Squad quality features** — FIFA ranking delta at match date
3. **Tune hyperparameters** — use `GridSearchCV` on `n_estimators`, `max_depth`, `min_samples_leaf`
4. **Try XGBoost** — often 2–3% better than RF on tabular football data
5. **Use full dataset** — remove the 5,000-row sample cap in `load_or_train_model()`
