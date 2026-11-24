"""
data_generator.py
-----------------
Generates a realistic historical dataset of football matches including both:
  1. Indian Super League (ISL) matches
  2. Premier League matches
Includes team match stats, starting lineup player ratings (Attack, Midfield,
Defense, Goalkeeper, Overall), match dates, rest days, and final scores.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Premier League Clubs Configuration
PREMIER_LEAGUE_CONFIG = {
    "Manchester City": {"att": 88, "mid": 89, "def": 86, "gk": 88, "tier": 1, "league": "Premier League"},
    "Arsenal": {"att": 86, "mid": 86, "def": 86, "gk": 86, "tier": 1, "league": "Premier League"},
    "Liverpool": {"att": 87, "mid": 85, "def": 85, "gk": 89, "tier": 1, "league": "Premier League"},
    "Aston Villa": {"att": 82, "mid": 81, "def": 80, "gk": 86, "tier": 2, "league": "Premier League"},
    "Tottenham": {"att": 83, "mid": 82, "def": 81, "gk": 82, "tier": 2, "league": "Premier League"},
    "Chelsea": {"att": 82, "mid": 82, "def": 80, "gk": 80, "tier": 2, "league": "Premier League"},
    "Newcastle": {"att": 82, "mid": 81, "def": 81, "gk": 84, "tier": 2, "league": "Premier League"},
    "Manchester United": {"att": 81, "mid": 80, "def": 79, "gk": 81, "tier": 2, "league": "Premier League"},
    "West Ham": {"att": 79, "mid": 78, "def": 77, "gk": 80, "tier": 3, "league": "Premier League"},
    "Brighton": {"att": 78, "mid": 79, "def": 77, "gk": 79, "tier": 3, "league": "Premier League"},
    "Bournemouth": {"att": 77, "mid": 76, "def": 75, "gk": 77, "tier": 3, "league": "Premier League"},
    "Crystal Palace": {"att": 77, "mid": 76, "def": 76, "gk": 79, "tier": 3, "league": "Premier League"},
    "Fulham": {"att": 76, "mid": 77, "def": 76, "gk": 81, "tier": 3, "league": "Premier League"},
    "Wolves": {"att": 76, "mid": 76, "def": 75, "gk": 80, "tier": 3, "league": "Premier League"},
    "Everton": {"att": 75, "mid": 76, "def": 77, "gk": 82, "tier": 3, "league": "Premier League"},
    "Brentford": {"att": 76, "mid": 75, "def": 74, "gk": 78, "tier": 4, "league": "Premier League"},
    "Nottingham Forest": {"att": 75, "mid": 75, "def": 75, "gk": 78, "tier": 4, "league": "Premier League"},
    "Leicester City": {"att": 74, "mid": 74, "def": 73, "gk": 75, "tier": 4, "league": "Premier League"},
    "Southampton": {"att": 73, "mid": 73, "def": 72, "gk": 76, "tier": 4, "league": "Premier League"},
    "Ipswich Town": {"att": 72, "mid": 72, "def": 71, "gk": 74, "tier": 4, "league": "Premier League"},
}

# Indian Super League (ISL) Clubs Configuration
ISL_CONFIG = {
    "Mohun Bagan SG": {"att": 74, "mid": 72, "def": 71, "gk": 73, "tier": 1, "league": "Indian Super League"},
    "Mumbai City FC": {"att": 73, "mid": 73, "def": 71, "gk": 72, "tier": 1, "league": "Indian Super League"},
    "Bengaluru FC": {"att": 72, "mid": 71, "def": 71, "gk": 74, "tier": 1, "league": "Indian Super League"},
    "FC Goa": {"att": 72, "mid": 71, "def": 70, "gk": 70, "tier": 1, "league": "Indian Super League"},
    "Kerala Blasters FC": {"att": 71, "mid": 70, "def": 69, "gk": 68, "tier": 2, "league": "Indian Super League"},
    "Odisha FC": {"att": 72, "mid": 71, "def": 68, "gk": 71, "tier": 2, "league": "Indian Super League"},
    "East Bengal FC": {"att": 70, "mid": 69, "def": 68, "gk": 69, "tier": 2, "league": "Indian Super League"},
    "Chennaiyin FC": {"att": 69, "mid": 68, "def": 68, "gk": 68, "tier": 2, "league": "Indian Super League"},
    "NorthEast United FC": {"att": 70, "mid": 68, "def": 67, "gk": 68, "tier": 3, "league": "Indian Super League"},
    "Jamshedpur FC": {"att": 69, "mid": 68, "def": 67, "gk": 68, "tier": 3, "league": "Indian Super League"},
    "Punjab FC": {"att": 68, "mid": 67, "def": 66, "gk": 67, "tier": 3, "league": "Indian Super League"},
    "Mohammedan SC": {"att": 67, "mid": 66, "def": 66, "gk": 66, "tier": 3, "league": "Indian Super League"},
    "Hyderabad FC": {"att": 66, "mid": 65, "def": 65, "gk": 66, "tier": 3, "league": "Indian Super League"},
}

# Unified Teams Configuration
TEAMS_CONFIG = {**PREMIER_LEAGUE_CONFIG, **ISL_CONFIG}


def _calculate_overall(att: float, mid: float, df: float, gk: float) -> float:
    """Calculate weighted starting XI overall rating."""
    return round(0.27 * att + 0.33 * mid + 0.30 * df + 0.10 * gk, 2)


def generate_match_dataset(
    num_seasons: int = 4,
    start_year: int = 2021,
    random_seed: int = 42,
    output_path: str = "data/raw/matches_with_player_ratings.csv",
) -> pd.DataFrame:
    """
    Generate realistic multi-season league match data including both
    Indian Super League (ISL) and Premier League fixtures.
    """
    np.random.seed(random_seed)
    matches = []
    match_counter = 1

    leagues = [
        ("Premier League", PREMIER_LEAGUE_CONFIG, datetime(start_year, 8, 14)),
        ("Indian Super League", ISL_CONFIG, datetime(start_year, 9, 21)),
    ]

    for season_idx in range(num_seasons):
        season_year = start_year + season_idx
        season_name = f"{season_year}-{season_year + 1}"

        for league_name, league_teams_dict, base_start_date in leagues:
            current_date = datetime(season_year, base_start_date.month, base_start_date.day)
            team_names = list(league_teams_dict.keys())
            last_played = {team: current_date - timedelta(days=7) for team in team_names}

            # Build double round-robin schedule within this league
            pairings = []
            for home in team_names:
                for away in team_names:
                    if home != away:
                        pairings.append((home, away))

            np.random.shuffle(pairings)
            fixtures_per_round = len(team_names) // 2

            for i, (home, away) in enumerate(pairings):
                round_idx = i // fixtures_per_round
                match_date = current_date + timedelta(days=round_idx * 5 + int(np.random.choice([0, 1, 2])))

                home_rest = max(3, (match_date - last_played[home]).days)
                away_rest = max(3, (match_date - last_played[away]).days)
                last_played[home] = match_date
                last_played[away] = match_date

                h_cfg = TEAMS_CONFIG[home]
                a_cfg = TEAMS_CONFIG[away]

                h_att = round(np.random.normal(h_cfg["att"], 1.4), 1)
                h_mid = round(np.random.normal(h_cfg["mid"], 1.3), 1)
                h_def = round(np.random.normal(h_cfg["def"], 1.3), 1)
                h_gk = round(np.random.normal(h_cfg["gk"], 1.5), 1)
                h_ovr = _calculate_overall(h_att, h_mid, h_def, h_gk)

                a_att = round(np.random.normal(a_cfg["att"], 1.4), 1)
                a_mid = round(np.random.normal(a_cfg["mid"], 1.3), 1)
                a_def = round(np.random.normal(a_cfg["def"], 1.3), 1)
                a_gk = round(np.random.normal(a_cfg["gk"], 1.5), 1)
                a_ovr = _calculate_overall(a_att, a_mid, a_def, a_gk)

                fatigue_h = -0.08 if home_rest < 4 else 0.0
                fatigue_a = -0.10 if away_rest < 4 else 0.0

                # ISL has high tactical pace with average goals around 2.75
                base_h = 1.52 if league_name == "Indian Super League" else 1.50
                base_a = 1.18 if league_name == "Indian Super League" else 1.15

                lambda_home = base_h + 0.07 * (h_att - a_def) + 0.04 * (h_mid - a_mid) - 0.03 * (a_gk - 70 if league_name == "Indian Super League" else a_gk - 80) + fatigue_h
                lambda_away = base_a + 0.07 * (a_att - h_def) + 0.04 * (a_mid - h_mid) - 0.03 * (h_gk - 70 if league_name == "Indian Super League" else h_gk - 80) + fatigue_a

                lambda_home = max(0.2, lambda_home)
                lambda_away = max(0.15, lambda_away)

                h_goals = int(np.random.poisson(lambda_home))
                a_goals = int(np.random.poisson(lambda_away))

                res = "H" if h_goals > a_goals else ("D" if h_goals == a_goals else "A")

                matches.append({
                    "match_id": f"M{match_counter:05d}",
                    "season": season_name,
                    "league": league_name,
                    "date": match_date.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "home_rating_overall": h_ovr,
                    "home_rating_attack": h_att,
                    "home_rating_midfield": h_mid,
                    "home_rating_defense": h_def,
                    "home_rating_gk": h_gk,
                    "away_rating_overall": a_ovr,
                    "away_rating_attack": a_att,
                    "away_rating_midfield": a_mid,
                    "away_rating_defense": a_def,
                    "away_rating_gk": a_gk,
                    "home_rest_days": home_rest,
                    "away_rest_days": away_rest,
                    "home_goals": h_goals,
                    "away_goals": a_goals,
                    "result": res,
                })
                match_counter += 1

    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    isl_count = len(df[df["league"] == "Indian Super League"])
    epl_count = len(df[df["league"] == "Premier League"])
    print(f"[Data Generator] Successfully generated {len(df)} total matches ({isl_count} ISL matches, {epl_count} Premier League matches).")
    print(f"[Data Generator] Saved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_match_dataset()
