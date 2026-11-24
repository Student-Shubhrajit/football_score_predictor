"""
app.py
------
Interactive Football Match Score Prediction Dashboard built with Streamlit.

Features:
- Fixture Selection: Select Home and Away teams.
- Lineup Rating Customization: Sliders for Attack, Midfield, Defense, and Goalkeeper.
- Live Predicted Scoreboard & Expected Goals (xG).
- Outcome Probability Distribution (Home Win %, Draw %, Away Win %).
- 8x8 Score Probability Matrix Heatmap.
- Tactical Positional Comparison & Form Guide.
"""

import sys
from pathlib import Path

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_generator import TEAMS_CONFIG, ISL_CONFIG, PREMIER_LEAGUE_CONFIG
from src.predict import predict_matchup, get_latest_team_form, get_h2h_stats

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Football Score Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #00d26a;
        margin-bottom: 12px;
    }
    .score-board {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 2px solid #374151;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 15px 0px 25px 0px;
    }
    .score-title {
        color: #9ca3af;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }
    .score-digits {
        font-size: 52px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 4px;
    }
    .team-name {
        font-size: 24px;
        font-weight: 700;
        color: #f3f4f6;
    }
    .xg-sub {
        font-size: 15px;
        color: #10b981;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("⚽ AI Football Score & Match Predictor")
st.markdown(
    "Predict the football match scorelines."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Competition & Fixture Setup")
league_choice = st.sidebar.selectbox(
    "🏆 Select League",
    ["Indian Super League (ISL)", "Premier League", "All Leagues"],
    index=0
)

if league_choice == "Indian Super League (ISL)":
    team_list = sorted(list(ISL_CONFIG.keys()))
    default_home = "Mohun Bagan SG" if "Mohun Bagan SG" in team_list else team_list[0]
    default_away = "East Bengal FC" if "East Bengal FC" in team_list else team_list[1]
elif league_choice == "Premier League":
    team_list = sorted(list(PREMIER_LEAGUE_CONFIG.keys()))
    default_home = "Arsenal" if "Arsenal" in team_list else team_list[0]
    default_away = "Chelsea" if "Chelsea" in team_list else team_list[1]
else:
    team_list = sorted(list(TEAMS_CONFIG.keys()))
    default_home = team_list[0]
    default_away = team_list[1]

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    home_team = st.selectbox("Home Team", team_list, index=team_list.index(default_home) if default_home in team_list else 0)
with col_s2:
    away_options = [t for t in team_list if t != home_team]
    target_away = default_away if default_away in away_options else away_options[0]
    away_team = st.selectbox("Away Team", away_options, index=away_options.index(target_away))

st.sidebar.markdown("---")
st.sidebar.subheader(f"🛡️ {home_team} Lineup Ratings")
h_default = TEAMS_CONFIG.get(home_team, {"att": 72, "mid": 71, "def": 70, "gk": 72})
h_att = st.sidebar.slider(f"{home_team} Attack", 50, 99, int(h_default["att"]), key="h_att")
h_mid = st.sidebar.slider(f"{home_team} Midfield", 50, 99, int(h_default["mid"]), key="h_mid")
h_def = st.sidebar.slider(f"{home_team} Defense", 50, 99, int(h_default["def"]), key="h_def")
h_gk = st.sidebar.slider(f"{home_team} Goalkeeper", 50, 99, int(h_default["gk"]), key="h_gk")

st.sidebar.markdown("---")
st.sidebar.subheader(f"⚔️ {away_team} Lineup Ratings")
a_default = TEAMS_CONFIG.get(away_team, {"att": 70, "mid": 69, "def": 68, "gk": 70})
a_att = st.sidebar.slider(f"{away_team} Attack", 50, 99, int(a_default["att"]), key="a_att")
a_mid = st.sidebar.slider(f"{away_team} Midfield", 50, 99, int(a_default["mid"]), key="a_mid")
a_def = st.sidebar.slider(f"{away_team} Defense", 50, 99, int(a_default["def"]), key="a_def")
a_gk = st.sidebar.slider(f"{away_team} Goalkeeper", 50, 99, int(a_default["gk"]), key="a_gk")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Schedule & Fatigue")
col_r1, col_r2 = st.sidebar.columns(2)
with col_r1:
    h_rest = st.number_input("Home Rest Days", min_value=2, max_value=21, value=7)
with col_r2:
    a_rest = st.number_input("Away Rest Days", min_value=2, max_value=21, value=7)

# Predict Button
run_pred = st.sidebar.button("🔮 Run Match Prediction", type="primary", use_container_width=True)

# Execute Prediction
with st.spinner("Analyzing tactical matchups and computing score distribution..."):
    h_ratings = {"attack": h_att, "midfield": h_mid, "defense": h_def, "gk": h_gk}
    a_ratings = {"attack": a_att, "midfield": a_mid, "defense": a_def, "gk": a_gk}

    pred_res = predict_matchup(
        home_team=home_team,
        away_team=away_team,
        home_ratings=h_ratings,
        away_ratings=a_ratings,
        home_rest_days=h_rest,
        away_rest_days=a_rest,
    )

# Extract prediction values
mls = pred_res["most_likely_score"]
mls_prob = pred_res["most_likely_score_prob"]
xg = pred_res["expected_goals"]
probs = pred_res["probabilities"]

# Display Match Header & Scoreboard
st.markdown(
    f"""
    <div class="score-board">
        <div class="score-title">Predicted Most Likely Scoreline (Poisson-Dixon Coles Model)</div>
        <div style="display: flex; justify-content: space-around; align-items: center; margin-top: 12px;">
            <div style="flex: 1; text-align: right; padding-right: 25px;">
                <div class="team-name">{home_team}</div>
                <div class="xg-sub">Expected Goals: {xg['home']} xG</div>
            </div>
            <div style="flex: 0 0 auto;">
                <div class="score-digits">{mls[0]}  -  {mls[1]}</div>
                <div style="font-size: 13px; color: #9ca3af;">{mls_prob}% Probability</div>
            </div>
            <div style="flex: 1; text-align: left; padding-left: 25px;">
                <div class="team-name">{away_team}</div>
                <div class="xg-sub">Expected Goals: {xg['away']} xG</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Outcome Probabilities Bar
st.subheader("📊 Match Outcome Probabilities (1X2)")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric(f"{home_team} Win", f"{probs['home_win']}%")
    st.progress(probs["home_win"] / 100.0)
with col_p2:
    st.metric("Draw", f"{probs['draw']}%")
    st.progress(probs["draw"] / 100.0)
with col_p3:
    st.metric(f"{away_team} Win", f"{probs['away_win']}%")
    st.progress(probs["away_win"] / 100.0)

st.markdown("---")

# Main Analysis Columns
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("🎯 Exact Score Probability Heatmap")
    st.caption("Distribution of scoreline probabilities across 0 to 6 goals per team.")

    matrix = pred_res["score_matrix"][:7, :7] * 100.0  # percentage
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar=True,
        ax=ax,
        xticklabels=[str(i) for i in range(7)],
        yticklabels=[str(i) for i in range(7)],
    )
    ax.set_xlabel(f"{away_team} Goals (Away)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{home_team} Goals (Home)", fontsize=11, fontweight="bold")
    ax.set_title("Exact Score Probabilities (%)", fontsize=12, fontweight="bold")
    st.pyplot(fig)

with col_right:
    st.subheader("🏆 Top Most Probable Scorelines")
    top_scores = pred_res["top_scorelines"]
    scores_df = pd.DataFrame(top_scores)[["score", "probability"]]
    scores_df.columns = ["Scoreline", "Probability (%)"]
    st.dataframe(scores_df, use_container_width=True, hide_index=True)

    st.subheader("⚡ Betting & Match Dynamics")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Over / Under 2.5 Goals**")
        st.write(f"• Over 2.5: **{pred_res['over_under_2_5']['over']}%**")
        st.write(f"• Under 2.5: **{pred_res['over_under_2_5']['under']}%**")
    with col_b2:
        st.markdown("**Both Teams To Score (BTTS)**")
        st.write(f"• Yes: **{pred_res['btts']['yes']}%**")
        st.write(f"• No: **{pred_res['btts']['no']}%**")

    st.subheader("🛡️ Tactical Matchup Comparison")
    tactical_data = {
        "Metric": ["Overall Rating", "Attack", "Midfield", "Defense", "Goalkeeper"],
        home_team: [
            pred_res["home_ratings"]["overall"],
            pred_res["home_ratings"]["attack"],
            pred_res["home_ratings"]["midfield"],
            pred_res["home_ratings"]["defense"],
            pred_res["home_ratings"]["gk"],
        ],
        away_team: [
            pred_res["away_ratings"]["overall"],
            pred_res["away_ratings"]["attack"],
            pred_res["away_ratings"]["midfield"],
            pred_res["away_ratings"]["defense"],
            pred_res["away_ratings"]["gk"],
        ],
    }
    st.table(pd.DataFrame(tactical_data).set_index("Metric"))

# Historical Form Section
st.markdown("---")
st.subheader("📈 Recent Historical Form (Last 5 Games Prior)")
col_f1, col_f2 = st.columns(2)
h_form = pred_res["home_form"]
a_form = pred_res["away_form"]

with col_f1:
    st.markdown(f"**{home_team} Form Guide**")
    st.write(f"- Points Per Game: **{h_form.get('ppg', 'N/A')}**")
    st.write(f"- Goals Scored Avg: **{h_form.get('gf_avg', 'N/A')}**")
    st.write(f"- Goals Conceded Avg: **{h_form.get('ga_avg', 'N/A')}**")
    st.write(f"- Win Rate: **{int(h_form.get('win_rate', 0.35) * 100)}%**")

with col_f2:
    st.markdown(f"**{away_team} Form Guide**")
    st.write(f"- Points Per Game: **{a_form.get('ppg', 'N/A')}**")
    st.write(f"- Goals Scored Avg: **{a_form.get('gf_avg', 'N/A')}**")
    st.write(f"- Goals Conceded Avg: **{a_form.get('ga_avg', 'N/A')}**")
    st.write(f"- Win Rate: **{int(a_form.get('win_rate', 0.35) * 100)}%**")
