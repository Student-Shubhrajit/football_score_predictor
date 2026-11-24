"""
test_pipeline.py
----------------
Automated unit & integration tests for football match score predictor.
Verifies data integrity, feature alignment, mathematical consistency, and inference output.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.feature_engineering import FEATURE_COLUMNS, build_single_match_features
from src.model import FootballScorePredictor
from src.predict import predict_matchup


class TestFootballPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = PROJECT_ROOT / "models" / "football_predictor.joblib"
        cls.assertTrue(cls.model_path.exists(), "Trained model artifact must exist.")
        cls.predictor = FootballScorePredictor.load(str(cls.model_path))

    def test_feature_columns_count(self):
        """Ensure feature vector contains expected 34 features."""
        self.assertEqual(len(FEATURE_COLUMNS), 34)

    def test_single_match_feature_builder(self):
        """Test single match feature builder alignment."""
        h_ratings = {"attack": 85, "midfield": 84, "defense": 82, "gk": 86, "overall": 84.3}
        a_ratings = {"attack": 80, "midfield": 80, "defense": 78, "gk": 80, "overall": 79.5}
        form = {"ppg": 2.0, "gf_avg": 2.2, "ga_avg": 0.8, "gd_avg": 1.4, "win_rate": 0.6}
        h2h = {"matches": 3, "home_win_rate": 0.66, "away_win_rate": 0.33, "home_avg_goals": 2.0, "away_avg_goals": 1.0}

        df_feat = build_single_match_features(
            home_team="Arsenal",
            away_team="Chelsea",
            home_ratings=h_ratings,
            away_ratings=a_ratings,
            home_form=form,
            away_form=form,
            h2h_stats=h2h,
        )
        self.assertEqual(list(df_feat.columns), FEATURE_COLUMNS)
        self.assertEqual(df_feat.shape, (1, 34))

    def test_probability_distribution_normalization(self):
        """Verify score matrix sums to approximately 100% and probabilities are valid."""
        res = predict_matchup(home_team="Arsenal", away_team="Chelsea")

        matrix = res["score_matrix"]
        total_matrix_prob = np.sum(matrix)
        self.assertAlmostEqual(total_matrix_prob, 1.0, places=4)

        probs = res["probabilities"]
        total_outcome_prob = probs["home_win"] + probs["draw"] + probs["away_win"]
        self.assertAlmostEqual(total_outcome_prob, 100.0, delta=0.5)

    def test_expected_goals_are_positive(self):
        """Expected goals must always be strictly positive."""
        res = predict_matchup(home_team="Liverpool", away_team="Everton")
        xg = res["expected_goals"]
        self.assertGreater(xg["home"], 0.0)
        self.assertGreater(xg["away"], 0.0)

    def test_most_likely_score_format(self):
        """Ensure predicted score consists of valid non-negative integers."""
        res = predict_matchup(home_team="Manchester City", away_team="Arsenal")
        mls = res["most_likely_score"]
        self.assertIsInstance(mls[0], int)
        self.assertIsInstance(mls[1], int)
        self.assertGreaterEqual(mls[0], 0)
        self.assertGreaterEqual(mls[1], 0)


if __name__ == "__main__":
    unittest.main()
