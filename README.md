# ⚽ AI Football Match Score Predictor

An end-to-end, production-ready machine learning system that predicts **exact football match scores**, **expected goals ($xG$)**, and **match outcome probabilities (Home / Draw / Away)** using **historical performance data** and **starting lineup player ratings**.

---

## 📌 Architecture & Methodology

Predicting football scores using direct multi-class classification is statistically suboptimal due to the sparsity of uncommon scorelines (e.g., 4-3, 5-2). This project implements the industry standard **Goal Expectancy ($xG$) framework with Dixon-Coles Bivariate Poisson adjustment**:

```
                       ┌──────────────────────────────┐
                       │  Historical Match Records    │
                       │  (Rolling Form & H2H Data)   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Feature Engineering        │
                       │   (34 Tactical & Form Feats) │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Dual HistGradientBoosting Regressors             │
             │ (Poisson Deviance Loss: Ensures Positive lambda) │
             └──────────────┬────────────────────┬──────────────┘
                            │                    │
              Home Goals    ▼                    ▼   Away Goals
              Expectancy (λ_H)                   Expectancy (λ_A)
                            │                    │
                            └─────────┬──────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Dixon-Coles Adjustment Matrix│
                       │ P(Home=i, Away=j) (8x8 grid) │
                       └──────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  Exact Scoreline              Win / Draw / Loss            Over/Under 2.5
 (e.g., 2 - 1, 14.8%)        (1X2 Probabilities)          & BTTS Likelihood
```

### 1. Feature Engineering (34 Dimensions)
- **Player Lineup Ratings**:
  - Overall Starting XI average rating
  - Positional Breakdown: Attack, Midfield, Defense, Goalkeeper ratings
  - Tactical Matchup Differentials: Home Attack vs Away Defense, Away Attack vs Home Defense, Midfield battle, Attack vs Goalkeeper
- **Historical Previous Match Data**:
  - 5-match rolling form: Points Per Game (PPG), Goals Scored, Goals Conceded, Goal Difference, Win Rate
  - Head-to-Head (H2H): Past meetings count, H2H win rates, H2H average goals scored
- **Fatigue & Rest**:
  - Rest days elapsed since each team's last fixture and rest disparity

### 2. Modeling & Mathematics
1. **Poisson Goal Expectancy**:
   $P(\text{Goals} = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
   Where $\lambda_H$ and $\lambda_A$ are predicted by gradient boosted regressors with Poisson loss.
2. **Dixon-Coles Low-Score Adjustment**:
   Adjusts joint probability $P(i, j) = \tau_{\rho}(i, j) \cdot P(i) P(j)$ for low scores $(0,0), (1,0), (0,1), (1,1)$ using correlation parameter $\rho \approx -0.06$ to capture defensive tactical conservatism.
3. **Strict Featurization Ordering**:
   Data is chronologically partitioned (Train: 70%, Validation: 15%, Test: 15%) to prevent future data leakage.

---

## 📁 Project Directory Layout

```
football_score_predictor/
├── full_project_app.py                       # 🌟 ALL-IN-ONE Full Stack (Frontend + Backend + DB + ML)
├── database.py                               # SQLite + SQLAlchemy Database Models & Seeding
├── app.py                                    # Streamlit Web Dashboard alternative
├── requirements.txt                          # Dependencies
├── README.md                                 # Documentation
├── data/
│   ├── raw/
│   │   └── matches_with_player_ratings.csv   # 1,520 historical league matches
│   └── processed/
│       ├── features.csv                      # 34 engineered feature columns
│       └── metrics_summary.json              # Validation & test metrics
├── models/
│   └── football_predictor.joblib             # Serialized trained model artifact
├── src/
│   ├── __init__.py
│   ├── data_generator.py                     # Realistic dataset generator
│   ├── feature_engineering.py                # Chronological rolling & H2H featurizer
│   ├── model.py                              # Poisson & Dixon-Coles predictor class
│   ├── train.py                              # Chronological training & evaluation pipeline
│   └── predict.py                            # CLI inference engine
└── tests/
    └── test_pipeline.py                      # Automated unit & integration tests
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train Model
```bash
# Generate multi-season match history with player ratings
python src/data_generator.py

# Extract chronological rolling form & tactical features
python src/feature_engineering.py

# Train model and generate evaluation metrics
python src/train.py
```

### 3. Run Command-Line Prediction
Predict any match with default team ratings:
```bash
python src/predict.py --home "Arsenal" --away "Chelsea"
```

Predict with custom lineup tactical ratings (e.g. customized forward attack or defense):
```bash
python src/predict.py --home "Manchester City" --away "Liverpool" --home-att 90 --away-att 88
```

Sample CLI Output:
```text
==============================================================
       MATCH PREDICTION: MANCHESTER CITY vs LIVERPOOL
==============================================================
 Expected Goals (xG)  :  Manchester City 2.10  -  0.25 Liverpool
 Most Probable Score  :  Manchester City 2 - 0 Liverpool  (21.12% confidence)
--------------------------------------------------------------
 Match Probabilities  :  Home Win: 81.1% | Draw: 15.8% | Away Win: 3.1%
 Over / Under 2.5     :  Over 2.5: 41.6% | Under 2.5: 58.4%
 Both Teams To Score  :  Yes: 19.5% | No: 80.5%
--------------------------------------------------------------
 Top 5 Likely Scorelines:
    - Score 2 - 0    : 21.12%
    - Score 1 - 0    : 19.78%
    - Score 3 - 0    : 14.81%
    - Score 0 - 0    : 9.84%
    - Score 4 - 0    : 7.79%
==============================================================
```

### 4. Launch the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The dashboard provides:
- **Interactive Team & Lineup Sliders**: Adjust Attack, Midfield, Defense, and Goalkeeper ratings in real-time.
- **Dynamic Scoreboard**: Displays expected goals ($xG$) and most probable scoreline.
- **Match Outcome Bars**: Win / Draw / Loss percentages.
- **Exact Score Heatmap**: 8x8 color-coded probability matrix.
- **Recent Form & H2H Comparison**: Previous performance metrics.

---

## 🧪 Running Unit Tests

Run the test suite to verify pipeline integrity:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 Evaluation Metrics

Evaluated on unseen chronological holdout test fixtures:
- **1X2 Match Outcome Accuracy**: **~65.8%** (outperforming standard benchmark of ~55%)
- **Home Goals MAE**: **~0.93 - 1.01**
- **Away Goals MAE**: **~0.78 - 0.92**
- **Exact Scoreline Accuracy**: **~9.2% - 17.5%** (random baseline is < 3%)
- **Both Teams to Score (BTTS) Accuracy**: **~59.2%**

---

## 🔌 Connecting to Live Data APIs

To feed real-world live data into the pipeline:
1. Replace `data/raw/matches_with_player_ratings.csv` with data from [Football-Data.org](https://www.football-data.org/) or [API-Football](https://www.api-football.com/).
2. Fetch live player ratings from WhoScored, Sofascore, or EA FC database.
3. Re-run `python src/feature_engineering.py` and `python src/train.py`.
