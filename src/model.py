"""
model.py
--------
Core prediction engine for football match scores using Poisson Goal Expectancy
and the Dixon-Coles bivariate score distribution.

Components:
- FootballScorePredictor: Trains dual Poisson/Gradient Boosting models for
  home and away expected goals (xG / lambda), then generates:
    1. Most probable exact scoreline (e.g., 2 - 1)
    2. Continuous expected score (e.g., 2.14 - 1.08)
    3. 8x8 score probability matrix
    4. Match outcome probabilities: Home Win %, Draw %, Away Win %
    5. Over / Under 2.5 goals probability
    6. Both Teams To Score (BTTS) probability
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


class FootballScorePredictor:
    """
    Predicts football match scorelines, expected goals, and probability distributions.
    """

    def __init__(self, max_goals: int = 7, rho: float = -0.06):
        """
        Args:
            max_goals: Maximum goals evaluated per team for the score matrix (0 to max_goals).
            rho: Dixon-Coles low-score correlation parameter (typically between -0.04 and -0.10).
        """
        self.max_goals = max_goals
        self.rho = rho

        # Scikit-learn HistGradientBoostingRegressor with Poisson deviance loss
        # ensures non-negative goal expectations and models non-linear interactions
        self.home_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.04,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=42,
            )),
        ])

        self.away_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.04,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=42,
            )),
        ])
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series) -> "FootballScorePredictor":
        """
        Fit home and away expected goals models.
        """
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        self.is_fitted = True
        return self

    def predict_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict expected goals (lambda) for Home and Away sides.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load from disk first.")
        lambda_h = self.home_model.predict(X)
        lambda_a = self.away_model.predict(X)
        # Numerical guard: ensure lambda is strictly positive and realistic
        lambda_h = np.clip(lambda_h, 0.1, 7.0)
        lambda_a = np.clip(lambda_a, 0.1, 7.0)
        return lambda_h, lambda_a

    def calculate_score_matrix(self, lambda_h: float, lambda_a: float) -> np.ndarray:
        """
        Computes the (max_goals+1) x (max_goals+1) probability matrix of exact scores
        using bivariate Poisson with Dixon-Coles adjustment.
        """
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        # Independent Poisson baseline
        for i in range(self.max_goals + 1):
            prob_i = poisson.pmf(i, lambda_h)
            for j in range(self.max_goals + 1):
                prob_j = poisson.pmf(j, lambda_a)
                matrix[i, j] = prob_i * prob_j

        # Dixon-Coles adjustment for low scores (0-0, 0-1, 1-0, 1-1)
        # to properly reflect match state dependence and defensive conservatism
        if self.rho != 0.0:
            tau_00 = max(0.0, 1.0 - lambda_h * lambda_a * self.rho)
            tau_01 = max(0.0, 1.0 + lambda_h * self.rho)
            tau_10 = max(0.0, 1.0 + lambda_a * self.rho)
            tau_11 = max(0.0, 1.0 - self.rho)

            matrix[0, 0] *= tau_00
            matrix[0, 1] *= tau_01
            matrix[1, 0] *= tau_10
            matrix[1, 1] *= tau_11

        # Renormalize to ensure sum of probabilities == 1.0
        total_prob = np.sum(matrix)
        if total_prob > 0:
            matrix /= total_prob

        return matrix

    def predict_match(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete match prediction for a single fixture feature row.

        Returns:
            Dict containing:
            - expected_goals: {"home": float, "away": float}
            - most_likely_score: (home_goals, away_goals)
            - most_likely_score_prob: float
            - probabilities: {"home_win": float, "draw": float, "away_win": float}
            - top_scorelines: list of {"score": "2-1", "probability": 0.12}
            - over_under_2_5: {"over": float, "under": float}
            - btts: {"yes": float, "no": float}
            - score_matrix: 2D numpy array
        """
        lambda_h_arr, lambda_a_arr = self.predict_expected_goals(X_single)
        lh = float(lambda_h_arr[0])
        la = float(lambda_a_arr[0])

        score_matrix = self.calculate_score_matrix(lh, la)

        # Calculate outcomes
        home_win_prob = float(np.sum(np.tril(score_matrix, -1)))
        draw_prob = float(np.sum(np.diag(score_matrix)))
        away_win_prob = float(np.sum(np.triu(score_matrix, 1)))

        # Find most likely exact score
        max_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        most_likely_score = (int(max_idx[0]), int(max_idx[1]))
        most_likely_prob = float(score_matrix[max_idx])

        # Top 5 most probable scorelines
        flattened_indices = np.argsort(score_matrix, axis=None)[::-1]
        top_scores = []
        for idx in flattened_indices[:6]:
            h, a = np.unravel_index(idx, score_matrix.shape)
            top_scores.append({
                "score": f"{h} - {a}",
                "home": int(h),
                "away": int(a),
                "probability": round(float(score_matrix[h, a]) * 100, 2),
            })

        # Over / Under 2.5 goals
        under_2_5_prob = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i + j < 2.5:
                    under_2_5_prob += score_matrix[i, j]
        over_2_5_prob = 1.0 - under_2_5_prob

        # Both Teams to Score (BTTS)
        btts_yes = float(np.sum(score_matrix[1:, 1:]))
        btts_no = 1.0 - btts_yes

        return {
            "expected_goals": {"home": round(lh, 2), "away": round(la, 2)},
            "most_likely_score": most_likely_score,
            "most_likely_score_prob": round(most_likely_prob * 100, 2),
            "probabilities": {
                "home_win": round(home_win_prob * 100, 1),
                "draw": round(draw_prob * 100, 1),
                "away_win": round(away_win_prob * 100, 1),
            },
            "top_scorelines": top_scores,
            "over_under_2_5": {
                "over": round(over_2_5_prob * 100, 1),
                "under": round(under_2_5_prob * 100, 1),
            },
            "btts": {
                "yes": round(btts_yes * 100, 1),
                "no": round(btts_no * 100, 1),
            },
            "score_matrix": score_matrix,
        }

    def save(self, filepath: str = "models/football_predictor.joblib") -> None:
        """Serialize trained model to disk."""
        joblib.dump(self, filepath)
        print(f"[Model] Saved model artifact to: {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/football_predictor.joblib") -> "FootballScorePredictor":
        """Load trained model artifact from disk."""
        model = joblib.load(filepath)
        return model
