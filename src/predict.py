"""
predict.py
----------
Command-line inference interface for predicting football match scores.

Usage:
  python src/predict.py --home "Arsenal" --away "Chelsea"
  python src/predict.py --home "Liverpool" --away "Manchester City" --home-att 89 --away-att 90
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.model import FootballScorePredictor
from src.feature_engineering import build_single_match_features
from src.data_generator import TEAMS_CONFIG


def get_latest_team_form(team_name: str, features_df: pd.DataFrame) -> Dict[str, float]:
    """Extracts the most recent 5-match rolling form for a team from history."""
    # Find all rows where team was home or away
    home_matches = features_df[features_df["home_team"] == team_name]
    away_matches = features_df[features_df["away_team"] == team_name]

    latest_h = home_matches.iloc[-1] if len(home_matches) > 0 else None
    latest_a = away_matches.iloc[-1] if len(away_matches) > 0 else None

    # Pick the most recent one
    if latest_h is not None and latest_a is not None:
        if pd.to_datetime(latest_h["date"]) >= pd.to_datetime(latest_a["date"]):
            return {
                "ppg": latest_h["home_ppg_last5"],
                "gf_avg": latest_h["home_gf_last5"],
                "ga_avg": latest_h["home_ga_last5"],
                "gd_avg": latest_h["home_gd_last5"],
                "win_rate": latest_h["home_win_rate_last5"],
            }
        else:
            return {
                "ppg": latest_a["away_ppg_last5"],
                "gf_avg": latest_a["away_gf_last5"],
                "ga_avg": latest_a["away_ga_last5"],
                "gd_avg": latest_a["away_gd_last5"],
                "win_rate": latest_a["away_win_rate_last5"],
            }
    elif latest_h is not None:
        return {
            "ppg": latest_h["home_ppg_last5"],
            "gf_avg": latest_h["home_gf_last5"],
            "ga_avg": latest_h["home_ga_last5"],
            "gd_avg": latest_h["home_gd_last5"],
            "win_rate": latest_h["home_win_rate_last5"],
        }
    elif latest_a is not None:
        return {
            "ppg": latest_a["away_ppg_last5"],
            "gf_avg": latest_a["away_gf_last5"],
            "ga_avg": latest_a["away_ga_last5"],
            "gd_avg": latest_a["away_gd_last5"],
            "win_rate": latest_a["away_win_rate_last5"],
        }
    return {"ppg": 1.35, "gf_avg": 1.40, "ga_avg": 1.40, "gd_avg": 0.0, "win_rate": 0.35}


def get_h2h_stats(home_team: str, away_team: str, features_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates historical Head-to-Head metrics between two teams."""
    h2h = features_df[
        ((features_df["home_team"] == home_team) & (features_df["away_team"] == away_team))
        | ((features_df["home_team"] == away_team) & (features_df["away_team"] == home_team))
    ]
    if len(h2h) == 0:
        return {"matches": 0, "home_win_rate": 0.45, "away_win_rate": 0.30, "home_avg_goals": 1.5, "away_avg_goals": 1.15}

    latest_h2h = h2h.iloc[-1]
    if latest_h2h["home_team"] == home_team:
        return {
            "matches": latest_h2h["h2h_matches_prior"] + 1,
            "home_win_rate": latest_h2h["h2h_home_win_rate"],
            "away_win_rate": latest_h2h["h2h_away_win_rate"],
            "home_avg_goals": latest_h2h["h2h_home_avg_goals"],
            "away_avg_goals": latest_h2h["h2h_away_avg_goals"],
        }
    else:
        return {
            "matches": latest_h2h["h2h_matches_prior"] + 1,
            "home_win_rate": latest_h2h["h2h_away_win_rate"],
            "away_win_rate": latest_h2h["h2h_home_win_rate"],
            "home_avg_goals": latest_h2h["h2h_away_avg_goals"],
            "away_avg_goals": latest_h2h["h2h_home_avg_goals"],
        }


def predict_matchup(
    home_team: str,
    away_team: str,
    home_ratings: Optional[Dict[str, float]] = None,
    away_ratings: Optional[Dict[str, float]] = None,
    home_rest_days: int = 7,
    away_rest_days: int = 7,
    model_path: str = str(PROJECT_ROOT / "models" / "football_predictor.joblib"),
    features_csv: str = str(PROJECT_ROOT / "data" / "processed" / "features.csv"),
) -> Dict[str, Any]:
    """
    Executes end-to-end score prediction for a specific fixture.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    predictor = FootballScorePredictor.load(model_path)

    features_df = pd.read_csv(features_csv) if Path(features_csv).exists() else pd.DataFrame()

    # Load default team baseline ratings if not explicitly customized
    default_h = TEAMS_CONFIG.get(home_team, {"att": 78, "mid": 78, "def": 78, "gk": 78})
    default_a = TEAMS_CONFIG.get(away_team, {"att": 78, "mid": 78, "def": 78, "gk": 78})

    h_att = home_ratings.get("attack", default_h["att"]) if home_ratings else default_h["att"]
    h_mid = home_ratings.get("midfield", default_h["mid"]) if home_ratings else default_h["mid"]
    h_def = home_ratings.get("defense", default_h["def"]) if home_ratings else default_h["def"]
    h_gk = home_ratings.get("gk", default_h["gk"]) if home_ratings else default_h["gk"]
    h_ovr = round(0.27 * h_att + 0.33 * h_mid + 0.30 * h_def + 0.10 * h_gk, 1)

    a_att = away_ratings.get("attack", default_a["att"]) if away_ratings else default_a["att"]
    a_mid = away_ratings.get("midfield", default_a["mid"]) if away_ratings else default_a["mid"]
    a_def = away_ratings.get("defense", default_a["def"]) if away_ratings else default_a["def"]
    a_gk = away_ratings.get("gk", default_a["gk"]) if away_ratings else default_a["gk"]
    a_ovr = round(0.27 * a_att + 0.33 * a_mid + 0.30 * a_def + 0.10 * a_gk, 1)

    h_ratings_full = {"attack": h_att, "midfield": h_mid, "defense": h_def, "gk": h_gk, "overall": h_ovr}
    a_ratings_full = {"attack": a_att, "midfield": a_mid, "defense": a_def, "gk": a_gk, "overall": a_ovr}

    # Fetch rolling historical stats
    h_form = get_latest_team_form(home_team, features_df) if not features_df.empty else {}
    a_form = get_latest_team_form(away_team, features_df) if not features_df.empty else {}
    h2h = get_h2h_stats(home_team, away_team, features_df) if not features_df.empty else {}

    # Assemble feature vector
    X_single = build_single_match_features(
        home_team=home_team,
        away_team=away_team,
        home_ratings=h_ratings_full,
        away_ratings=a_ratings_full,
        home_form=h_form,
        away_form=a_form,
        h2h_stats=h2h,
        home_rest_days=home_rest_days,
        away_rest_days=away_rest_days,
    )

    pred = predictor.predict_match(X_single)
    pred["home_team"] = home_team
    pred["away_team"] = away_team
    pred["home_ratings"] = h_ratings_full
    pred["away_ratings"] = a_ratings_full
    pred["home_form"] = h_form
    pred["away_form"] = a_form
    pred["h2h"] = h2h
    return pred


def print_prediction_report(pred: Dict[str, Any]) -> None:
    """Format and print prediction results to console."""
    ht = pred["home_team"]
    at = pred["away_team"]
    mls = pred["most_likely_score"]
    prob_mls = pred["most_likely_score_prob"]
    xg = pred["expected_goals"]
    probs = pred["probabilities"]

    print("\n" + "=" * 62)
    print(f"       MATCH PREDICTION: {ht.upper()} vs {at.upper()}")
    print("=" * 62)
    print(f" Expected Goals (xG)  :  {ht} {xg['home']}  -  {xg['away']} {at}")
    print(f" Most Probable Score  :  {ht} {mls[0]} - {mls[1]} {at}  ({prob_mls}% confidence)")
    print("-" * 62)
    print(f" Match Probabilities  :  Home Win: {probs['home_win']}% | Draw: {probs['draw']}% | Away Win: {probs['away_win']}%")
    print(f" Over / Under 2.5     :  Over 2.5: {pred['over_under_2_5']['over']}% | Under 2.5: {pred['over_under_2_5']['under']}%")
    print(f" Both Teams To Score  :  Yes: {pred['btts']['yes']}% | No: {pred['btts']['no']}%")
    print("-" * 62)
    print(" Top 5 Likely Scorelines:")
    for item in pred["top_scorelines"][:5]:
        print(f"    - Score {item['score']:<8} : {item['probability']}%")
    print("-" * 62)
    print(f" Lineup Ratings:")
    print(f"    {ht:<18} Overall: {pred['home_ratings']['overall']} | Att: {pred['home_ratings']['attack']} | Mid: {pred['home_ratings']['midfield']} | Def: {pred['home_ratings']['defense']} | GK: {pred['home_ratings']['gk']}")
    print(f"    {at:<18} Overall: {pred['away_ratings']['overall']} | Att: {pred['away_ratings']['attack']} | Mid: {pred['away_ratings']['midfield']} | Def: {pred['away_ratings']['defense']} | GK: {pred['away_ratings']['gk']}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict football match scores from player ratings and previous data.")
    parser.add_argument("--home", type=str, default="Arsenal", help="Home team name")
    parser.add_argument("--away", type=str, default="Chelsea", help="Away team name")
    parser.add_argument("--home-att", type=float, default=None, help="Custom Home Attack rating")
    parser.add_argument("--away-att", type=float, default=None, help="Custom Away Attack rating")
    parser.add_argument("--home-def", type=float, default=None, help="Custom Home Defense rating")
    parser.add_argument("--away-def", type=float, default=None, help="Custom Away Defense rating")
    args = parser.parse_args()

    h_ratings = {}
    if args.home_att:
        h_ratings["attack"] = args.home_att
    if args.home_def:
        h_ratings["defense"] = args.home_def

    a_ratings = {}
    if args.away_att:
        a_ratings["attack"] = args.away_att
    if args.away_def:
        a_ratings["defense"] = args.away_def

    res = predict_matchup(
        home_team=args.home,
        away_team=args.away,
        home_ratings=h_ratings or None,
        away_ratings=a_ratings or None,
    )
    print_prediction_report(res)
