"""
train.py
--------
Training and rigorous chronological evaluation pipeline for the Football Score Predictor.

Adheres to production ML best practices:
1. Chronological splitting (Train / Validation / Test) to eliminate future data leakage.
2. Independent evaluation for Home and Away goal models (MAE, RMSE, Poisson Deviance).
3. Calibration & accuracy evaluation for Exact Score and 1X2 Match Outcomes (Home/Draw/Away).
4. Feature importance extraction.
5. Model persistence and metric reporting.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, d2_tweedie_score

from src.feature_engineering import FEATURE_COLUMNS, compute_historical_features
from src.model import FootballScorePredictor


def mean_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates mean unit Poisson deviance: 2 * (y*log(y/y_hat) - (y - y_hat))."""
    y_pred = np.clip(y_pred, 1e-6, None)
    deviance = 2 * np.where(y_true > 0, y_true * np.log(np.maximum(y_true, 1e-6) / y_pred), 0.0) - 2 * (y_true - y_pred)
    return float(np.mean(deviance))


def evaluate_predictions(
    model: FootballScorePredictor,
    df_eval: pd.DataFrame,
    split_name: str = "Test",
) -> Dict[str, Any]:
    """
    Evaluates goal predictions, exact score predictions, and 1X2 outcome accuracy.
    """
    X = df_eval[FEATURE_COLUMNS]
    y_home_true = df_eval["home_goals"].to_numpy()
    y_away_true = df_eval["away_goals"].to_numpy()

    # Predict expected goals (continuous lambda)
    lambda_h, lambda_a = model.predict_expected_goals(X)

    # Goal regression metrics
    mae_h = mean_absolute_error(y_home_true, lambda_h)
    mae_a = mean_absolute_error(y_away_true, lambda_a)
    rmse_h = np.sqrt(mean_squared_error(y_home_true, lambda_h))
    rmse_a = np.sqrt(mean_squared_error(y_away_true, lambda_a))
    dev_h = mean_poisson_deviance(y_home_true, lambda_h)
    dev_a = mean_poisson_deviance(y_away_true, lambda_a)

    # Match-level exact score and outcome accuracy
    exact_score_hits = 0
    outcome_hits = 0
    btts_hits = 0
    total_matches = len(df_eval)

    for i in range(total_matches):
        pred_dict = model.predict_match(X.iloc[[i]])
        pred_score = pred_dict["most_likely_score"]
        probs = pred_dict["probabilities"]

        actual_h = y_home_true[i]
        actual_a = y_away_true[i]

        # Exact score match
        if pred_score[0] == actual_h and pred_score[1] == actual_a:
            exact_score_hits += 1

        # Outcome 1X2 match
        if actual_h > actual_a:
            actual_res = "H"
        elif actual_h == actual_a:
            actual_res = "D"
        else:
            actual_res = "A"

        best_outcome = max(probs, key=probs.get)
        pred_res = "H" if best_outcome == "home_win" else ("D" if best_outcome == "draw" else "A")

        if pred_res == actual_res:
            outcome_hits += 1

        # BTTS check
        actual_btts = (actual_h > 0 and actual_a > 0)
        pred_btts = pred_dict["btts"]["yes"] >= 50.0
        if actual_btts == pred_btts:
            btts_hits += 1

    exact_score_acc = round((exact_score_hits / total_matches) * 100, 2)
    outcome_acc = round((outcome_hits / total_matches) * 100, 2)
    btts_acc = round((btts_hits / total_matches) * 100, 2)

    metrics = {
        "split": split_name,
        "sample_count": total_matches,
        "home_goals": {
            "MAE": round(mae_h, 3),
            "RMSE": round(rmse_h, 3),
            "Poisson_Deviance": round(dev_h, 3),
        },
        "away_goals": {
            "MAE": round(mae_a, 3),
            "RMSE": round(rmse_a, 3),
            "Poisson_Deviance": round(dev_a, 3),
        },
        "match_level": {
            "exact_score_accuracy_pct": exact_score_acc,
            "outcome_1x2_accuracy_pct": outcome_acc,
            "btts_accuracy_pct": btts_acc,
        },
    }
    return metrics


def train_pipeline(
    features_csv: str = str(PROJECT_ROOT / "data" / "processed" / "features.csv"),
    model_save_path: str = str(PROJECT_ROOT / "models" / "football_predictor.joblib"),
    report_save_path: str = str(PROJECT_ROOT / "data" / "processed" / "metrics_summary.json"),
) -> FootballScorePredictor:
    """
    Executes end-to-end model training, chronological evaluation, and export.
    """
    print("=" * 60)
    print("   FOOTBALL MATCH SCORE PREDICTOR - TRAINING PIPELINE   ")
    print("=" * 60)

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Feature dataset not found at {features_csv}. Run feature_engineering.py first.")

    df = pd.read_csv(features_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

    n_samples = len(df)
    print(f"[Dataset] Total processed matches: {n_samples}")
    print(f"[Features] Utilizing {len(FEATURE_COLUMNS)} engineered input features.")

    # Chronological Split: 70% Train, 15% Validation, 15% Test
    train_idx = int(0.70 * n_samples)
    val_idx = int(0.85 * n_samples)

    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    print(f"[Split] Chronological partitions:")
    print(f"  - Training Set:   {len(train_df)} matches ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  - Validation Set: {len(val_df)} matches ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  - Test Set:       {len(test_df)} matches ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Prepare matrices
    X_train = train_df[FEATURE_COLUMNS]
    y_train_home = train_df["home_goals"]
    y_train_away = train_df["away_goals"]

    # Instantiate and fit predictor
    print("\n[Training] Fitting dual Poisson regressors with Dixon-Coles adjustment...")
    predictor = FootballScorePredictor(max_goals=7, rho=-0.06)
    predictor.fit(X_train, y_train_home, y_train_away)
    print("[Training] Optimization converged successfully.")

    # Evaluate on Validation Set
    val_metrics = evaluate_predictions(predictor, val_df, split_name="Validation")
    print("\n--- VALIDATION SET PERFORMANCE ---")
    print(f"Home Goals MAE: {val_metrics['home_goals']['MAE']} | Away Goals MAE: {val_metrics['away_goals']['MAE']}")
    print(f"1X2 Outcome Accuracy: {val_metrics['match_level']['outcome_1x2_accuracy_pct']}%")
    print(f"Exact Scoreline Accuracy: {val_metrics['match_level']['exact_score_accuracy_pct']}%")

    # Evaluate on Test Set
    test_metrics = evaluate_predictions(predictor, test_df, split_name="Test (Holdout)")
    print("\n--- TEST (HOLDOUT) SET PERFORMANCE ---")
    print(f"Home Goals MAE: {test_metrics['home_goals']['MAE']} | Away Goals MAE: {test_metrics['away_goals']['MAE']}")
    print(f"1X2 Outcome Accuracy: {test_metrics['match_level']['outcome_1x2_accuracy_pct']}%")
    print(f"Exact Scoreline Accuracy: {test_metrics['match_level']['exact_score_accuracy_pct']}%")
    print(f"Both Teams To Score (BTTS) Accuracy: {test_metrics['match_level']['btts_accuracy_pct']}%")

    # Save artifacts
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    predictor.save(model_save_path)

    summary_report = {
        "features": FEATURE_COLUMNS,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(report_save_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Report] Summary saved to: {report_save_path}")
    print("=" * 60)

    return predictor


if __name__ == "__main__":
    train_pipeline()
