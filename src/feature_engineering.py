"""
feature_engineering.py
-----------------------
Engineers predictive features from historical match logs and lineup player ratings.

Features computed:
1. Rolling Form (last 5 matches prior to current game):
   - Points per game, goals scored/conceded averages, goal difference, win rate.
2. Head-to-Head (H2H) History:
   - Prior match counts, win rates, and average goals scored between the two teams.
3. Positional Rating Matchups:
   - Overall difference, Home Attack vs Away Defense, Away Attack vs Home Defense,
     Midfield control differential, Attack vs Goalkeeper matchup.
4. Schedule & Rest:
   - Rest days differential.

Strict chronological ordering is enforced to avoid any data leakage.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_historical_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Process full dataset chronologically to generate rolling form, H2H,
    and rating matchup features for every match.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

    # Initialize tracking structures
    # Team history: list of dicts with match details: {date, goals_for, goals_against, points, is_win}
    team_history: Dict[str, list] = {}
    # H2H history: key is tuple(sorted([teamA, teamB])): list of {home_team, away_team, home_goals, away_goals}
    h2h_history: Dict[Tuple[str, str], list] = {}

    engineered_rows = []

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        m_date = row["date"]

        # ---------------------------------------------------------
        # 1. Rolling Form (Last N games prior to this match)
        # ---------------------------------------------------------
        home_past = team_history.get(home, [])
        away_past = team_history.get(away, [])

        def _calc_form(past_matches):
            if not past_matches:
                # Default league priors if no matches played yet
                return {
                    "ppg": 1.35,
                    "gf_avg": 1.40,
                    "ga_avg": 1.40,
                    "gd_avg": 0.0,
                    "win_rate": 0.35,
                }
            recent = past_matches[-window:]
            ppg = np.mean([m["pts"] for m in recent])
            gf = np.mean([m["gf"] for m in recent])
            ga = np.mean([m["ga"] for m in recent])
            gd = gf - ga
            win_rate = np.mean([1.0 if m["pts"] == 3 else 0.0 for m in recent])
            return {
                "ppg": round(float(ppg), 3),
                "gf_avg": round(float(gf), 3),
                "ga_avg": round(float(ga), 3),
                "gd_avg": round(float(gd), 3),
                "win_rate": round(float(win_rate), 3),
            }

        h_form = _calc_form(home_past)
        a_form = _calc_form(away_past)

        # ---------------------------------------------------------
        # 2. Head-to-Head (H2H) Prior Matchups
        # ---------------------------------------------------------
        pair_key = tuple(sorted([home, away]))
        h2h_past = h2h_history.get(pair_key, [])

        h2h_count = len(h2h_past)
        if h2h_count > 0:
            h_wins = 0
            a_wins = 0
            draws = 0
            h_goals_tot = 0
            a_goals_tot = 0

            for m in h2h_past:
                if m["home"] == home:
                    hg, ag = m["h_goals"], m["a_goals"]
                else:
                    hg, ag = m["a_goals"], m["h_goals"]

                h_goals_tot += hg
                a_goals_tot += ag
                if hg > ag:
                    h_wins += 1
                elif hg < ag:
                    a_wins += 1
                else:
                    draws += 1

            h2h_home_win_rate = round(h_wins / h2h_count, 3)
            h2h_away_win_rate = round(a_wins / h2h_count, 3)
            h2h_home_avg_goals = round(h_goals_tot / h2h_count, 3)
            h2h_away_avg_goals = round(a_goals_tot / h2h_count, 3)
        else:
            h2h_home_win_rate = 0.45  # typical home win advantage
            h2h_away_win_rate = 0.30
            h2h_home_avg_goals = 1.50
            h2h_away_avg_goals = 1.15

        # ---------------------------------------------------------
        # 3. Positional Lineup Rating Matchups
        # ---------------------------------------------------------
        diff_overall = row["home_rating_overall"] - row["away_rating_overall"]
        diff_attack_defense_h = row["home_rating_attack"] - row["away_rating_defense"]
        diff_attack_defense_a = row["away_rating_attack"] - row["home_rating_defense"]
        diff_midfield = row["home_rating_midfield"] - row["away_rating_midfield"]
        diff_att_gk_h = row["home_rating_attack"] - row["away_rating_gk"]
        diff_att_gk_a = row["away_rating_attack"] - row["home_rating_gk"]

        # Rest differential
        rest_diff = row["home_rest_days"] - row["away_rest_days"]

        # Collect complete feature vector
        feat_dict = {
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            # Player ratings
            "home_rating_overall": row["home_rating_overall"],
            "home_rating_attack": row["home_rating_attack"],
            "home_rating_midfield": row["home_rating_midfield"],
            "home_rating_defense": row["home_rating_defense"],
            "home_rating_gk": row["home_rating_gk"],
            "away_rating_overall": row["away_rating_overall"],
            "away_rating_attack": row["away_rating_attack"],
            "away_rating_midfield": row["away_rating_midfield"],
            "away_rating_defense": row["away_rating_defense"],
            "away_rating_gk": row["away_rating_gk"],
            # Rating differentials
            "diff_overall": round(diff_overall, 2),
            "diff_attack_defense_h": round(diff_attack_defense_h, 2),
            "diff_attack_defense_a": round(diff_attack_defense_a, 2),
            "diff_midfield": round(diff_midfield, 2),
            "diff_att_gk_h": round(diff_att_gk_h, 2),
            "diff_att_gk_a": round(diff_att_gk_a, 2),
            # Rolling form
            "home_ppg_last5": h_form["ppg"],
            "home_gf_last5": h_form["gf_avg"],
            "home_ga_last5": h_form["ga_avg"],
            "home_gd_last5": h_form["gd_avg"],
            "home_win_rate_last5": h_form["win_rate"],
            "away_ppg_last5": a_form["ppg"],
            "away_gf_last5": a_form["gf_avg"],
            "away_ga_last5": a_form["ga_avg"],
            "away_gd_last5": a_form["gd_avg"],
            "away_win_rate_last5": a_form["win_rate"],
            # H2H metrics
            "h2h_matches_prior": h2h_count,
            "h2h_home_win_rate": h2h_home_win_rate,
            "h2h_away_win_rate": h2h_away_win_rate,
            "h2h_home_avg_goals": h2h_home_avg_goals,
            "h2h_away_avg_goals": h2h_away_avg_goals,
            # Rest days
            "home_rest_days": row["home_rest_days"],
            "away_rest_days": row["away_rest_days"],
            "rest_diff": rest_diff,
            # Targets
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "result": row["result"],
        }
        engineered_rows.append(feat_dict)

        # ---------------------------------------------------------
        # Update running history AFTER feature extraction (No leakage!)
        # ---------------------------------------------------------
        h_pts = 3 if row["home_goals"] > row["away_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0)
        a_pts = 3 if row["away_goals"] > row["home_goals"] else (1 if row["home_goals"] == row["away_goals"] else 0)

        if home not in team_history:
            team_history[home] = []
        team_history[home].append({
            "date": m_date,
            "gf": row["home_goals"],
            "ga": row["away_goals"],
            "pts": h_pts,
        })

        if away not in team_history:
            team_history[away] = []
        team_history[away].append({
            "date": m_date,
            "gf": row["away_goals"],
            "ga": row["home_goals"],
            "pts": a_pts,
        })

        if pair_key not in h2h_history:
            h2h_history[pair_key] = []
        h2h_history[pair_key].append({
            "home": home,
            "away": away,
            "h_goals": row["home_goals"],
            "a_goals": row["away_goals"],
        })

    feat_df = pd.DataFrame(engineered_rows)
    return feat_df


# List of explicit numeric feature columns for modeling
FEATURE_COLUMNS = [
    # Player ratings
    "home_rating_overall",
    "home_rating_attack",
    "home_rating_midfield",
    "home_rating_defense",
    "home_rating_gk",
    "away_rating_overall",
    "away_rating_attack",
    "away_rating_midfield",
    "away_rating_defense",
    "away_rating_gk",
    # Tactical differentials
    "diff_overall",
    "diff_attack_defense_h",
    "diff_attack_defense_a",
    "diff_midfield",
    "diff_att_gk_h",
    "diff_att_gk_a",
    # Historical rolling form
    "home_ppg_last5",
    "home_gf_last5",
    "home_ga_last5",
    "home_gd_last5",
    "home_win_rate_last5",
    "away_ppg_last5",
    "away_gf_last5",
    "away_ga_last5",
    "away_gd_last5",
    "away_win_rate_last5",
    # H2H stats
    "h2h_matches_prior",
    "h2h_home_win_rate",
    "h2h_away_win_rate",
    "h2h_home_avg_goals",
    "h2h_away_avg_goals",
    # Fatigue
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
]


def build_single_match_features(
    home_team: str,
    away_team: str,
    home_ratings: Dict[str, float],
    away_ratings: Dict[str, float],
    home_form: Dict[str, float],
    away_form: Dict[str, float],
    h2h_stats: Dict[str, float],
    home_rest_days: int = 7,
    away_rest_days: int = 7,
) -> pd.DataFrame:
    """
    Constructs a feature row for inference when predicting an upcoming match.
    """
    diff_overall = home_ratings["overall"] - away_ratings["overall"]
    diff_adh = home_ratings["attack"] - away_ratings["defense"]
    diff_ada = away_ratings["attack"] - home_ratings["defense"]
    diff_mid = home_ratings["midfield"] - away_ratings["midfield"]
    diff_att_gk_h = home_ratings["attack"] - away_ratings["gk"]
    diff_att_gk_a = away_ratings["attack"] - home_ratings["gk"]
    rest_diff = home_rest_days - away_rest_days

    row = {
        "home_rating_overall": home_ratings["overall"],
        "home_rating_attack": home_ratings["attack"],
        "home_rating_midfield": home_ratings["midfield"],
        "home_rating_defense": home_ratings["defense"],
        "home_rating_gk": home_ratings["gk"],
        "away_rating_overall": away_ratings["overall"],
        "away_rating_attack": away_ratings["attack"],
        "away_rating_midfield": away_ratings["midfield"],
        "away_rating_defense": away_ratings["defense"],
        "away_rating_gk": away_ratings["gk"],
        "diff_overall": diff_overall,
        "diff_attack_defense_h": diff_adh,
        "diff_attack_defense_a": diff_ada,
        "diff_midfield": diff_mid,
        "diff_att_gk_h": diff_att_gk_h,
        "diff_att_gk_a": diff_att_gk_a,
        "home_ppg_last5": home_form.get("ppg", 1.35),
        "home_gf_last5": home_form.get("gf_avg", 1.40),
        "home_ga_last5": home_form.get("ga_avg", 1.40),
        "home_gd_last5": home_form.get("gd_avg", 0.0),
        "home_win_rate_last5": home_form.get("win_rate", 0.35),
        "away_ppg_last5": away_form.get("ppg", 1.35),
        "away_gf_last5": away_form.get("gf_avg", 1.40),
        "away_ga_last5": away_form.get("ga_avg", 1.40),
        "away_gd_last5": away_form.get("gd_avg", 0.0),
        "away_win_rate_last5": away_form.get("win_rate", 0.35),
        "h2h_matches_prior": h2h_stats.get("matches", 0),
        "h2h_home_win_rate": h2h_stats.get("home_win_rate", 0.45),
        "h2h_away_win_rate": h2h_stats.get("away_win_rate", 0.30),
        "h2h_home_avg_goals": h2h_stats.get("home_avg_goals", 1.50),
        "h2h_away_avg_goals": h2h_stats.get("away_avg_goals", 1.15),
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_diff": rest_diff,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


if __name__ == "__main__":
    raw_path = "data/raw/matches_with_player_ratings.csv"
    if pd.io.common.file_exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        feat_df = compute_historical_features(raw_df)
        out_path = "data/processed/features.csv"
        feat_df.to_csv(out_path, index=False)
        print(f"[Feature Engineering] Processed {len(feat_df)} rows. Saved to: {out_path}")
        print(f"Features engineered: {len(FEATURE_COLUMNS)} columns.")
