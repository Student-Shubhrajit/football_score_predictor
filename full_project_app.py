"""
================================================================================
FOOTBALL MATCH SCORE PREDICTOR - ALL-IN-ONE FULL STACK APPLICATION
================================================================================
Components included in this single file:
  1. DATABASE LAYER : SQLite + SQLAlchemy (Teams, Players, Matches, PredictionLogs)
     - Includes both Indian Super League (ISL) and Premier League clubs & lineups!
  2. MACHINE LEARNING ENGINE : Poisson Goal Expectancy + Dixon-Coles Bivariate Adjustment
  3. BACKEND REST API : Flask web server with prediction & query endpoints
  4. FRONTEND UI : Modern responsive dark-themed Single Page Application (HTML/CSS/JS)
     - Features competition switcher (ISL vs Premier League), rating sliders & heatmaps!

RUNNING THIS APP:
  python full_project_app.py
Then open your browser at:
  http://127.0.0.1:5000
================================================================================
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Flask for Backend API & Frontend Serving
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# SQLAlchemy for SQLite Database
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


# ==============================================================================
# 1. DATABASE LAYER (SQLite & SQLAlchemy)
# ==============================================================================

DB_FILE = "football_all_in_one.db"
DB_URL = f"sqlite:///{DB_FILE}"

Base = declarative_base()
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    short_code = Column(String(10), nullable=False)
    stadium = Column(String(100), nullable=False)
    league = Column(String(50), default="Premier League")
    tier = Column(Integer, default=1)

    players = relationship("Player", back_populates="team", cascade="all, delete-orphan")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    name = Column(String(100), nullable=False)
    position = Column(String(20), nullable=False)  # "FWD", "MID", "DEF", "GK"
    jersey_number = Column(Integer, default=10)
    rating = Column(Float, nullable=False)
    pace = Column(Integer, default=75)
    shooting = Column(Integer, default=75)
    passing = Column(Integer, default=75)
    defending = Column(Integer, default=75)

    team = relationship("Team", back_populates="players")


class MatchRecord(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(DateTime, default=datetime.utcnow)
    season = Column(String(20), default="2023-2024")
    league = Column(String(50), default="Indian Super League")
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    home_ovr = Column(Float, nullable=False)
    away_ovr = Column(Float, nullable=False)
    result = Column(String(5), nullable=False)  # "H", "D", "A"


class PredictionEntry(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    predicted_score = Column(String(20), nullable=False)
    home_xg = Column(Float, nullable=False)
    away_xg = Column(Float, nullable=False)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    btts_prob = Column(Float, nullable=False)
    over_2_5_prob = Column(Float, nullable=False)


# Real Clubs, Squads & Ratings for Indian Super League (ISL) & European giants
INITIAL_TEAMS_DATA = [
    # --- INDIAN SUPER LEAGUE (ISL) ---
    {
        "name": "Mohun Bagan SG",
        "short_code": "MBSG",
        "stadium": "Salt Lake Stadium, Kolkata",
        "league": "Indian Super League",
        "tier": 1,
        "squad": [
            ("Vishal Kaith", "GK", 1, 73, 40, 20, 68, 74),
            ("Subhasish Bose", "DEF", 15, 71, 70, 52, 68, 72),
            ("Alberto Rodriguez", "DEF", 2, 72, 68, 45, 69, 73),
            ("Tom Aldred", "DEF", 32, 70, 65, 40, 64, 71),
            ("Asish Rai", "DEF", 44, 69, 78, 50, 66, 68),
            ("Anirudh Thapa", "MID", 7, 71, 72, 66, 73, 67),
            ("Sahal Abdul Samad", "MID", 18, 72, 75, 70, 74, 58),
            ("Greg Stewart", "MID", 10, 73, 70, 74, 75, 52),
            ("Liston Colaco", "FWD", 17, 72, 82, 72, 68, 48),
            ("Jason Cummings", "FWD", 35, 74, 76, 76, 69, 50),
            ("Dimitri Petratos", "FWD", 9, 75, 74, 77, 76, 52),
        ],
    },
    {
        "name": "Bengaluru FC",
        "short_code": "BFC",
        "stadium": "Sree Kanteerava Stadium, Bengaluru",
        "league": "Indian Super League",
        "tier": 1,
        "squad": [
            ("Gurpreet Singh Sandhu", "GK", 1, 74, 42, 20, 70, 75),
            ("Rahul Bheke", "DEF", 2, 70, 68, 48, 65, 71),
            ("Aleksandar Jovanovic", "DEF", 4, 71, 65, 42, 64, 72),
            ("Naorem Roshan Singh", "DEF", 32, 70, 78, 60, 71, 68),
            ("Chinglensana Singh", "DEF", 5, 69, 66, 40, 64, 70),
            ("Suresh Singh Wangjam", "MID", 8, 70, 74, 62, 70, 71),
            ("Alberto Noguera", "MID", 10, 72, 69, 71, 75, 64),
            ("Pedro Capo", "MID", 6, 71, 65, 64, 72, 72),
            ("Ryan Williams", "FWD", 7, 72, 79, 70, 70, 52),
            ("Sunil Chhetri", "FWD", 11, 73, 72, 76, 71, 48),
            ("Jorge Pereyra Diaz", "FWD", 9, 72, 75, 73, 68, 54),
        ],
    },
    {
        "name": "Mumbai City FC",
        "short_code": "MCFC",
        "stadium": "Mumbai Football Arena",
        "league": "Indian Super League",
        "tier": 1,
        "squad": [
            ("Phurba Lachenpa", "GK", 1, 72, 40, 20, 68, 73),
            ("Mehtab Singh", "DEF", 4, 71, 69, 45, 65, 72),
            ("Tiri", "DEF", 5, 72, 64, 42, 68, 73),
            ("Valpuia", "DEF", 22, 68, 70, 40, 63, 68),
            ("Nathan Rodrigues", "DEF", 14, 67, 68, 40, 62, 67),
            ("Yoell van Nieff", "MID", 8, 73, 68, 72, 76, 71),
            ("Jon Toral", "MID", 10, 72, 70, 73, 74, 58),
            ("Brandon Fernandes", "MID", 7, 71, 70, 68, 75, 55),
            ("Lallianzuala Chhangte", "FWD", 17, 74, 84, 72, 70, 48),
            ("Nikolaos Karelis", "FWD", 9, 73, 74, 75, 66, 50),
            ("Bipin Singh", "FWD", 29, 71, 80, 69, 67, 50),
        ],
    },
    {
        "name": "Kerala Blasters FC",
        "short_code": "KBFC",
        "stadium": "Jawaharlal Nehru Stadium, Kochi",
        "league": "Indian Super League",
        "tier": 2,
        "squad": [
            ("Sachin Suresh", "GK", 1, 68, 40, 20, 64, 69),
            ("Pritam Kotal", "DEF", 20, 69, 66, 45, 65, 70),
            ("Milos Drincic", "DEF", 15, 70, 64, 42, 63, 71),
            ("Sandeep Singh", "DEF", 2, 67, 70, 40, 62, 67),
            ("Aibanbha Dohling", "DEF", 3, 68, 72, 45, 64, 67),
            ("Vibin Mohanan", "MID", 8, 68, 68, 62, 69, 65),
            ("Danish Farooq", "MID", 6, 68, 69, 64, 68, 68),
            ("Adrian Luna", "MID", 10, 74, 75, 73, 77, 62),
            ("Noah Sadaoui", "FWD", 7, 73, 81, 74, 72, 50),
            ("Jesus Jimenez", "FWD", 9, 72, 74, 73, 68, 52),
            ("Rahul KP", "FWD", 17, 70, 80, 67, 65, 52),
        ],
    },
    {
        "name": "East Bengal FC",
        "short_code": "EBFC",
        "stadium": "Salt Lake Stadium, Kolkata",
        "league": "Indian Super League",
        "tier": 2,
        "squad": [
            ("Prabhsukhan Gill", "GK", 1, 69, 40, 20, 65, 70),
            ("Anwar Ali", "DEF", 4, 71, 70, 45, 68, 72),
            ("Hector Yuste", "DEF", 5, 70, 60, 45, 66, 71),
            ("Mohamad Rakip", "DEF", 12, 67, 72, 40, 62, 66),
            ("Lalchungnunga", "DEF", 22, 68, 69, 42, 63, 68),
            ("Saul Crespo", "MID", 21, 71, 68, 68, 72, 70),
            ("Jeakson Singh", "MID", 6, 70, 66, 62, 69, 71),
            ("Madih Talal", "MID", 10, 73, 75, 72, 76, 56),
            ("Mahesh Singh Naorem", "FWD", 11, 71, 78, 69, 71, 52),
            ("Cleiton Silva", "FWD", 9, 72, 71, 74, 70, 52),
            ("Nandhakumar Sekar", "FWD", 17, 70, 77, 68, 67, 50),
        ],
    },
    {
        "name": "FC Goa",
        "short_code": "FCG",
        "stadium": "Fatorda Stadium, Goa",
        "league": "Indian Super League",
        "tier": 1,
        "squad": [
            ("Laxmikant Kattimani", "GK", 1, 70, 40, 20, 66, 71),
            ("Sandesh Jhingan", "DEF", 5, 72, 64, 45, 64, 73),
            ("Odei Onaindia", "DEF", 4, 71, 62, 40, 65, 72),
            ("Jay Gupta", "DEF", 32, 69, 75, 52, 67, 68),
            ("Seriton Fernandes", "DEF", 20, 68, 71, 45, 64, 67),
            ("Carl McHugh", "MID", 6, 71, 66, 65, 71, 72),
            ("Rowllin Borges", "MID", 8, 69, 67, 64, 68, 70),
            ("Borja Herrera", "MID", 10, 72, 72, 73, 75, 60),
            ("Mohammad Yasir", "FWD", 27, 69, 76, 66, 68, 50),
            ("Armando Sadiku", "FWD", 9, 73, 74, 75, 67, 54),
            ("Iker Guarrotxena", "FWD", 34, 73, 75, 74, 73, 52),
        ],
    },
    {
        "name": "Odisha FC",
        "short_code": "OFC",
        "stadium": "Kalinga Stadium, Bhubaneswar",
        "league": "Indian Super League",
        "tier": 2,
        "squad": [
            ("Amrinder Singh", "GK", 1, 71, 42, 20, 67, 72),
            ("Mourtada Fall", "DEF", 5, 72, 62, 55, 64, 74),
            ("Carlos Delgado", "DEF", 4, 71, 63, 40, 66, 72),
            ("Jerry Lalrinzuala", "DEF", 3, 68, 72, 45, 64, 67),
            ("Amey Ranawade", "DEF", 12, 69, 76, 45, 65, 68),
            ("Ahmed Jahouh", "MID", 10, 74, 60, 72, 81, 73),
            ("Puitea", "MID", 8, 69, 68, 62, 70, 68),
            ("Isak Vanlalruatfela", "MID", 7, 70, 76, 68, 70, 50),
            ("Roy Krishna", "FWD", 21, 74, 76, 77, 72, 54),
            ("Diego Mauricio", "FWD", 9, 73, 77, 76, 68, 52),
            ("Jerry Mawihmingthanga", "FWD", 17, 70, 78, 68, 68, 48),
        ],
    },
    {
        "name": "NorthEast United FC",
        "short_code": "NEUFC",
        "stadium": "Indira Gandhi Athletic Stadium, Guwahati",
        "league": "Indian Super League",
        "tier": 3,
        "squad": [
            ("Gurmeet Singh", "GK", 1, 68, 40, 20, 63, 69),
            ("Asheer Akhtar", "DEF", 4, 67, 65, 40, 63, 67),
            ("Michel Zabaco", "DEF", 3, 70, 62, 40, 64, 71),
            ("Buanthanglun Samte", "DEF", 13, 66, 70, 42, 62, 66),
            ("Dinesh Singh", "DEF", 2, 67, 72, 40, 63, 66),
            ("Mohammed Ali Bemammer", "MID", 6, 71, 64, 64, 71, 72),
            ("Mayakkannan", "MID", 8, 67, 66, 60, 68, 67),
            ("Nestor Albiach", "MID", 10, 71, 71, 72, 73, 55),
            ("Jithin MS", "FWD", 11, 71, 84, 68, 67, 46),
            ("Alaaeddine Ajaraie", "FWD", 9, 74, 80, 76, 70, 52),
            ("Parthib Gogoi", "FWD", 7, 71, 78, 71, 66, 48),
        ],
    },
    # --- PREMIER LEAGUE ---
    {
        "name": "Arsenal",
        "short_code": "ARS",
        "stadium": "Emirates Stadium",
        "league": "Premier League",
        "tier": 1,
        "squad": [
            ("David Raya", "GK", 1, 85, 45, 20, 78, 86),
            ("William Saliba", "DEF", 2, 87, 82, 40, 72, 88),
            ("Gabriel Magalhaes", "DEF", 6, 86, 75, 45, 68, 87),
            ("Ben White", "DEF", 4, 83, 78, 62, 79, 83),
            ("Jurrien Timber", "DEF", 12, 82, 81, 58, 77, 82),
            ("Declan Rice", "MID", 41, 87, 76, 74, 82, 86),
            ("Martin Odegaard", "MID", 8, 89, 75, 83, 89, 68),
            ("Thomas Partey", "MID", 5, 83, 68, 72, 81, 82),
            ("Bukayo Saka", "FWD", 7, 88, 86, 84, 83, 65),
            ("Kai Havertz", "FWD", 29, 83, 80, 81, 80, 58),
            ("Gabriel Martinelli", "FWD", 11, 84, 89, 78, 76, 52),
        ],
    },
    {
        "name": "Manchester City",
        "short_code": "MCI",
        "stadium": "Etihad Stadium",
        "league": "Premier League",
        "tier": 1,
        "squad": [
            ("Ederson", "GK", 31, 88, 60, 30, 91, 87),
            ("Ruben Dias", "DEF", 3, 88, 66, 42, 70, 89),
            ("Manuel Akanji", "DEF", 25, 84, 80, 54, 75, 84),
            ("Josko Gvardiol", "DEF", 24, 85, 81, 68, 78, 85),
            ("Kyle Walker", "DEF", 2, 83, 89, 60, 76, 82),
            ("Rodri", "MID", 16, 91, 68, 82, 86, 89),
            ("Kevin De Bruyne", "MID", 17, 90, 72, 87, 94, 65),
            ("Bernardo Silva", "MID", 20, 88, 78, 78, 87, 72),
            ("Phil Foden", "FWD", 47, 88, 85, 86, 88, 60),
            ("Erling Haaland", "FWD", 9, 91, 89, 93, 70, 50),
            ("Jeremy Doku", "FWD", 11, 82, 92, 72, 77, 42),
        ],
    },
    {
        "name": "Liverpool",
        "short_code": "LIV",
        "stadium": "Anfield",
        "league": "Premier League",
        "tier": 1,
        "squad": [
            ("Alisson Becker", "GK", 1, 89, 52, 25, 85, 90),
            ("Virgil van Dijk", "DEF", 4, 89, 78, 60, 72, 90),
            ("Ibrahima Konate", "DEF", 5, 84, 82, 45, 66, 85),
            ("Trent Alexander-Arnold", "DEF", 66, 86, 76, 75, 90, 79),
            ("Andrew Robertson", "DEF", 26, 84, 82, 65, 81, 81),
            ("Ryan Gravenberch", "MID", 38, 83, 80, 75, 82, 78),
            ("Alexis Mac Allister", "MID", 10, 85, 74, 80, 84, 78),
            ("Dominik Szoboszlai", "MID", 8, 84, 83, 84, 83, 69),
            ("Mohamed Salah", "FWD", 11, 89, 87, 88, 83, 50),
            ("Luis Diaz", "FWD", 7, 85, 89, 79, 76, 52),
            ("Diogo Jota", "FWD", 20, 84, 81, 83, 76, 55),
        ],
    },
    {
        "name": "Chelsea",
        "short_code": "CHE",
        "stadium": "Stamford Bridge",
        "league": "Premier League",
        "tier": 2,
        "squad": [
            ("Robert Sanchez", "GK", 1, 80, 50, 20, 72, 80),
            ("Levi Colwill", "DEF", 6, 81, 75, 45, 74, 82),
            ("Wesley Fofana", "DEF", 29, 81, 79, 45, 68, 82),
            ("Malo Gusto", "DEF", 27, 81, 85, 55, 76, 78),
            ("Marc Cucurella", "DEF", 3, 81, 78, 60, 76, 80),
            ("Moises Caicedo", "MID", 25, 84, 76, 68, 78, 85),
            ("Enzo Fernandez", "MID", 8, 83, 72, 76, 84, 78),
            ("Cole Palmer", "MID", 20, 86, 81, 85, 85, 58),
            ("Noni Madueke", "FWD", 11, 80, 87, 76, 74, 46),
            ("Nicolas Jackson", "FWD", 15, 81, 85, 79, 72, 48),
            ("Pedro Neto", "FWD", 7, 81, 88, 77, 78, 48),
        ],
    },
]


def init_db():
    """Initializes tables and seeds starting data if empty."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        if session.query(Team).count() == 0:
            print("[Database] Creating SQLite database & seeding clubs and Starting XIs...")
            for c_data in INITIAL_TEAMS_DATA:
                t = Team(
                    name=c_data["name"],
                    short_code=c_data["short_code"],
                    stadium=c_data["stadium"],
                    league=c_data.get("league", "Indian Super League"),
                    tier=c_data["tier"],
                )
                session.add(t)
                session.flush()

                for p_info in c_data["squad"]:
                    p_name, pos, num, rating, pace, sho, pas, df = p_info
                    player = Player(
                        team_id=t.id,
                        name=p_name,
                        position=pos,
                        jersey_number=num,
                        rating=float(rating),
                        pace=pace,
                        shooting=sho,
                        passing=pas,
                        defending=df,
                    )
                    session.add(player)

            session.commit()

            # Seed realistic historical match encounters
            teams = session.query(Team).all()
            base_date = datetime(2023, 8, 1)
            for i, ht in enumerate(teams):
                for j, at in enumerate(teams):
                    if ht.id != at.id and ht.league == at.league:
                        h_rating = float(np.mean([p.rating for p in ht.players]))
                        a_rating = float(np.mean([p.rating for p in at.players]))
                        lam_h = max(0.3, 1.45 + 0.08 * (h_rating - a_rating))
                        lam_a = max(0.2, 1.10 + 0.08 * (a_rating - h_rating))
                        hg = int(np.random.poisson(lam_h))
                        ag = int(np.random.poisson(lam_a))
                        res = "H" if hg > ag else ("D" if hg == ag else "A")

                        rec = MatchRecord(
                            match_date=base_date + timedelta(days=(i * 10 + j)),
                            season="2023-2024",
                            league=ht.league,
                            home_team=ht.name,
                            away_team=at.name,
                            home_goals=hg,
                            away_goals=ag,
                            home_ovr=round(h_rating, 1),
                            away_ovr=round(a_rating, 1),
                            result=res,
                        )
                        session.add(rec)
            session.commit()
            print("[Database] Initialization complete. Seeded clubs and match history.")
    finally:
        session.close()


# ==============================================================================
# 2. MACHINE LEARNING ENGINE (Poisson & Dixon-Coles Predictor)
# ==============================================================================

class FootballPredictorEngine:
    """
    Computes Expected Goals (xG) and full Score Distribution matrix using
    Poisson goal expectancy and Dixon-Coles adjustment.
    """

    def __init__(self, rho: float = -0.06):
        self.rho = rho
        self.base_home_xg = 1.48
        self.base_away_xg = 1.12
        self.att_def_weight = 0.075
        self.mid_control_weight = 0.035
        self.gk_impact_weight = -0.030

    def compute_expected_goals(
        self,
        h_ratings: Dict[str, float],
        a_ratings: Dict[str, float],
        h_form_ppg: float = 1.5,
        a_form_ppg: float = 1.5,
        h_rest: int = 7,
        a_rest: int = 7,
    ) -> Tuple[float, float]:
        """Calculates expected goals for home and away sides."""
        h_att_diff = h_ratings["attack"] - a_ratings["defense"]
        a_att_diff = a_ratings["attack"] - h_ratings["defense"]
        mid_diff = h_ratings["midfield"] - a_ratings["midfield"]

        # Adapt goalkeeper baseline dynamically to tournament rating tier
        gk_base = 70.0 if (h_ratings.get("gk", 80) < 78 and a_ratings.get("gk", 80) < 78) else 80.0
        h_gk_diff = a_ratings["gk"] - gk_base
        a_gk_diff = h_ratings["gk"] - gk_base

        form_delta = (h_form_ppg - a_form_ppg) * 0.12
        fatigue_h = -0.10 if h_rest < 4 else 0.0
        fatigue_a = -0.12 if a_rest < 4 else 0.0

        lambda_h = (
            self.base_home_xg
            + self.att_def_weight * h_att_diff
            + self.mid_control_weight * mid_diff
            + self.gk_impact_weight * h_gk_diff
            + form_delta
            + fatigue_h
        )

        lambda_a = (
            self.base_away_xg
            + self.att_def_weight * a_att_diff
            - self.mid_control_weight * mid_diff
            + self.gk_impact_weight * a_gk_diff
            - form_delta
            + fatigue_a
        )

        lambda_h = float(np.clip(lambda_h, 0.20, 5.50))
        lambda_a = float(np.clip(lambda_a, 0.15, 5.00))
        return round(lambda_h, 2), round(lambda_a, 2)

    def calculate_score_matrix(self, lambda_h: float, lambda_a: float, max_goals: int = 6) -> np.ndarray:
        """Bivariate Poisson score matrix with Dixon-Coles adjustment."""
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            prob_i = poisson.pmf(i, lambda_h)
            for j in range(max_goals + 1):
                prob_j = poisson.pmf(j, lambda_a)
                matrix[i, j] = prob_i * prob_j

        if self.rho != 0.0:
            tau_00 = max(0.0, 1.0 - lambda_h * lambda_a * self.rho)
            tau_01 = max(0.0, 1.0 + lambda_h * self.rho)
            tau_10 = max(0.0, 1.0 + lambda_a * self.rho)
            tau_11 = max(0.0, 1.0 - self.rho)

            matrix[0, 0] *= tau_00
            matrix[0, 1] *= tau_01
            matrix[1, 0] *= tau_10
            matrix[1, 1] *= tau_11

        total = np.sum(matrix)
        if total > 0:
            matrix /= total

        return matrix

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        h_ratings: Dict[str, float],
        a_ratings: Dict[str, float],
        h_form_ppg: float = 1.5,
        a_form_ppg: float = 1.5,
        h_rest: int = 7,
        a_rest: int = 7,
    ) -> Dict[str, Any]:
        """Runs full prediction for fixture."""
        lh, la = self.compute_expected_goals(h_ratings, a_ratings, h_form_ppg, a_form_ppg, h_rest, a_rest)
        matrix = self.calculate_score_matrix(lh, la, max_goals=6)

        home_win_prob = float(np.sum(np.tril(matrix, -1)))
        draw_prob = float(np.sum(np.diag(matrix)))
        away_win_prob = float(np.sum(np.triu(matrix, 1)))

        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        most_likely_score = (int(max_idx[0]), int(max_idx[1]))
        mls_prob = float(matrix[max_idx])

        flattened = np.argsort(matrix, axis=None)[::-1]
        top_scores = []
        for idx in flattened[:5]:
            h, a = np.unravel_index(idx, matrix.shape)
            top_scores.append({
                "score": f"{h} - {a}",
                "home": int(h),
                "away": int(a),
                "probability": round(float(matrix[h, a]) * 100, 2),
            })

        under_2_5 = sum(matrix[i, j] for i in range(7) for j in range(7) if i + j < 2.5)
        over_2_5 = 1.0 - under_2_5
        btts_yes = float(np.sum(matrix[1:, 1:]))
        btts_no = 1.0 - btts_yes
        matrix_pct = (matrix * 100.0).round(2).tolist()

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_goals": {"home": lh, "away": la},
            "most_likely_score": most_likely_score,
            "most_likely_score_prob": round(mls_prob * 100, 2),
            "probabilities": {
                "home_win": round(home_win_prob * 100, 1),
                "draw": round(draw_prob * 100, 1),
                "away_win": round(away_win_prob * 100, 1),
            },
            "top_scorelines": top_scores,
            "over_under_2_5": {
                "over": round(over_2_5 * 100, 1),
                "under": round(under_2_5 * 100, 1),
            },
            "btts": {
                "yes": round(btts_yes * 100, 1),
                "no": round(btts_no * 100, 1),
            },
            "score_matrix": matrix_pct,
        }


ml_engine = FootballPredictorEngine()


# ==============================================================================
# 3. FRONTEND UI TEMPLATE (HTML5 + CSS3 + Modern JavaScript SPA)
# ==============================================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Football Score Predictor (ISL & Premier League)</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --accent-green: #10b981;
      --accent-blue: #3b82f6;
      --accent-orange: #f97316;
      --accent-red: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      padding: 24px;
      min-height: 100vh;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 24px; }
    header h1 { font-size: 32px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
    header p { color: var(--text-muted); margin-top: 6px; font-size: 15px; }
    .badge-isl { background: #ea580c; color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; margin-left: 8px; }
    .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width: 900px) { .dashboard-grid { grid-template-columns: 1fr; } }
    .card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card-title { font-size: 17px; font-weight: 700; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
    .form-group { margin-bottom: 12px; }
    label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 4px; }
    select, input[type="range"] { width: 100%; }
    select {
      background: #0f172a;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
    }
    .slider-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .slider-val { font-weight: 700; color: var(--accent-green); font-size: 14px; min-width: 32px; text-align: right; }
    .btn-predict {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #ffffff;
      border: none;
      padding: 14px 20px;
      font-size: 16px;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      width: 100%;
      margin-top: 15px;
      transition: all 0.2s ease;
    }
    .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3); }
    .scoreboard {
      grid-column: 1 / -1;
      background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
      border: 2px solid #3b82f6;
      border-radius: 14px;
      padding: 24px;
      text-align: center;
    }
    .score-display { display: flex; justify-content: space-around; align-items: center; margin: 16px 0; }
    .team-box { flex: 1; }
    .team-box.home { text-align: right; padding-right: 20px; }
    .team-box.away { text-align: left; padding-left: 20px; }
    .team-box h2 { font-size: 26px; font-weight: 800; }
    .team-box .xg-text { color: var(--accent-green); font-size: 14px; font-weight: 600; margin-top: 4px; }
    .score-center { background: #0f172a; border: 1px solid #334155; padding: 10px 24px; border-radius: 10px; }
    .score-digits { font-size: 54px; font-weight: 900; letter-spacing: 4px; color: #fff; }
    .score-conf { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    .prob-bar-container { display: flex; height: 24px; border-radius: 8px; overflow: hidden; margin: 15px 0 8px 0; }
    .prob-seg { display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #fff; transition: width 0.5s ease; }
    .prob-home { background-color: var(--accent-blue); }
    .prob-draw { background-color: #64748b; }
    .prob-away { background-color: var(--accent-red); }
    .prob-legend { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: var(--text-muted); }
    .matrix-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .matrix-table th, .matrix-table td { border: 1px solid #334155; padding: 6px; text-align: center; font-size: 11px; }
    .matrix-table th { background: #0f172a; color: var(--text-muted); }
    .top-score-row { display: flex; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid #334155; font-size: 14px; }
    .top-score-row:last-child { border-bottom: none; }
    .top-score-row span:last-child { font-weight: 700; color: var(--accent-green); }
    .history-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    .history-table th, .history-table td { padding: 10px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }
    .history-table th { color: var(--text-muted); font-size: 12px; text-transform: uppercase; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>⚽ Football Match Score Predictor <span class="badge-isl">ISL Included</span></h1>
      <p>Integrated Full-Stack: Indian Super League (ISL) & Premier League • SQLite DB • ML Engine • Live UI</p>
    </header>

    <!-- Scoreboard Result -->
    <div class="scoreboard" id="scoreboard">
      <div style="font-size: 12px; text-transform: uppercase; color: #94a3b8; letter-spacing: 2px;">Predicted Scoreline</div>
      <div class="score-display">
        <div class="team-box home">
          <h2 id="lbl-home-team">Mohun Bagan SG</h2>
          <div class="xg-text" id="lbl-home-xg">Expected Goals: 1.85 xG</div>
        </div>
        <div class="score-center">
          <div class="score-digits" id="lbl-score">2 - 1</div>
          <div class="score-conf" id="lbl-conf">18.4% Probability</div>
        </div>
        <div class="team-box away">
          <h2 id="lbl-away-team">East Bengal FC</h2>
          <div class="xg-text" id="lbl-away-xg">Expected Goals: 1.05 xG</div>
        </div>
      </div>

      <!-- Probability bar -->
      <div class="prob-bar-container">
        <div class="prob-seg prob-home" id="bar-home" style="width: 55%;">55%</div>
        <div class="prob-seg prob-draw" id="bar-draw" style="width: 25%;">25%</div>
        <div class="prob-seg prob-away" id="bar-away" style="width: 20%;">20%</div>
      </div>
      <div class="prob-legend">
        <span id="txt-home-win">Home Win: 55%</span>
        <span id="txt-draw">Draw: 25%</span>
        <span id="txt-away-win">Away Win: 20%</span>
      </div>
    </div>

    <!-- League Selector Bar -->
    <div class="card" style="margin-top: 20px; padding: 14px 20px;">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
          <span style="font-size: 15px; font-weight: 700;">🏆 Select Football League / Tournament:</span>
          <span style="font-size: 13px; color: #94a3b8; margin-left: 8px;">Filter match predictions by league</span>
        </div>
        <select id="sel-league-filter" onchange="filterClubsByLeague()" style="max-width: 320px;">
          <option value="Indian Super League" selected>🇮🇳 Indian Super League (ISL)</option>
          <option value="Premier League">🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League</option>
          <option value="ALL">🌐 All Competitions (Mixed)</option>
        </select>
      </div>
    </div>

    <div class="dashboard-grid" style="margin-top: 20px;">
      <!-- Home Club Setup -->
      <div class="card">
        <div class="card-title">🛡️ Home Team Setup</div>
        <div class="form-group">
          <label>Select Club</label>
          <select id="sel-home-team" onchange="onHomeTeamChange()"></select>
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Attack Rating</label>
            <span class="slider-val" id="val-h-att">74</span>
          </div>
          <input type="range" id="sl-h-att" min="50" max="99" value="74" oninput="updateVal('h-att')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Midfield Rating</label>
            <span class="slider-val" id="val-h-mid">72</span>
          </div>
          <input type="range" id="sl-h-mid" min="50" max="99" value="72" oninput="updateVal('h-mid')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Defense Rating</label>
            <span class="slider-val" id="val-h-def">71</span>
          </div>
          <input type="range" id="sl-h-def" min="50" max="99" value="71" oninput="updateVal('h-def')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Goalkeeper Rating</label>
            <span class="slider-val" id="val-h-gk">73</span>
          </div>
          <input type="range" id="sl-h-gk" min="50" max="99" value="73" oninput="updateVal('h-gk')">
        </div>
      </div>

      <!-- Away Club Setup -->
      <div class="card">
        <div class="card-title">⚔️ Away Team Setup</div>
        <div class="form-group">
          <label>Select Club</label>
          <select id="sel-away-team" onchange="onAwayTeamChange()"></select>
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Attack Rating</label>
            <span class="slider-val" id="val-a-att">70</span>
          </div>
          <input type="range" id="sl-a-att" min="50" max="99" value="70" oninput="updateVal('a-att')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Midfield Rating</label>
            <span class="slider-val" id="val-a-mid">69</span>
          </div>
          <input type="range" id="sl-a-mid" min="50" max="99" value="69" oninput="updateVal('a-mid')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Defense Rating</label>
            <span class="slider-val" id="val-a-def">68</span>
          </div>
          <input type="range" id="sl-a-def" min="50" max="99" value="68" oninput="updateVal('a-def')">
        </div>
        <div class="form-group">
          <div class="slider-row">
            <label>Goalkeeper Rating</label>
            <span class="slider-val" id="val-a-gk">69</span>
          </div>
          <input type="range" id="sl-a-gk" min="50" max="99" value="69" oninput="updateVal('a-gk')">
        </div>
      </div>

      <!-- Action Button -->
      <div style="grid-column: 1 / -1;">
        <button class="btn-predict" onclick="triggerPrediction()">🔮 Predict Match Score Now</button>
      </div>

      <!-- Top Scorelines & Betting Dynamics -->
      <div class="card">
        <div class="card-title">🏆 Top Probable Scorelines</div>
        <div id="top-scores-container"></div>
        <div style="margin-top: 15px; padding-top: 12px; border-top: 1px solid #334155;">
          <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px;">
            <span>Over 2.5 Goals: <strong id="lbl-over-25" style="color:#10b981;">50%</strong></span>
            <span>Under 2.5 Goals: <strong id="lbl-under-25" style="color:#3b82f6;">50%</strong></span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 13px;">
            <span>Both Teams to Score (BTTS): <strong id="lbl-btts" style="color:#f59e0b;">50%</strong></span>
          </div>
        </div>
      </div>

      <!-- 6x6 Score Probability Heatmap -->
      <div class="card">
        <div class="card-title">🎯 Score Probability Matrix (0 to 5 Goals)</div>
        <div style="overflow-x: auto;">
          <table class="matrix-table" id="matrix-table"></table>
        </div>
      </div>

      <!-- Database Logs -->
      <div class="card" style="grid-column: 1 / -1;">
        <div class="card-title">💾 Recent Predictions Logged in SQLite Database</div>
        <table class="history-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Fixture</th>
              <th>Predicted Score</th>
              <th>xG (Home - Away)</th>
              <th>Home Win %</th>
              <th>Draw %</th>
              <th>Away Win %</th>
            </tr>
          </thead>
          <tbody id="history-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let allClubsData = [];
    let filteredClubs = [];

    function updateVal(key) {
      document.getElementById('val-' + key).innerText = document.getElementById('sl-' + key).value;
    }

    async function loadTeams() {
      try {
        const res = await fetch('/api/teams');
        allClubsData = await res.json();
        filterClubsByLeague();
        loadHistory();
      } catch (err) {
        console.error("Failed to load clubs:", err);
      }
    }

    function filterClubsByLeague() {
      const selectedLeague = document.getElementById('sel-league-filter').value;
      if (selectedLeague === 'ALL') {
        filteredClubs = allClubsData;
      } else {
        filteredClubs = allClubsData.filter(c => c.league === selectedLeague);
      }

      const selHome = document.getElementById('sel-home-team');
      const selAway = document.getElementById('sel-away-team');
      selHome.innerHTML = '';
      selAway.innerHTML = '';

      filteredClubs.forEach((club) => {
        selHome.add(new Option(club.name, club.name));
        selAway.add(new Option(club.name, club.name));
      });

      if (filteredClubs.length > 1) {
        // Default pairs
        if (selectedLeague === 'Indian Super League') {
          selHome.value = "Mohun Bagan SG";
          selAway.value = "East Bengal FC";
        } else {
          selHome.value = filteredClubs[0].name;
          selAway.value = filteredClubs[1].name;
        }
      }
      onHomeTeamChange();
      onAwayTeamChange();
      triggerPrediction();
    }

    function onHomeTeamChange() {
      const name = document.getElementById('sel-home-team').value;
      const c = filteredClubs.find(x => x.name === name);
      if (c) {
        document.getElementById('sl-h-att').value = c.ratings.attack;
        document.getElementById('sl-h-mid').value = c.ratings.midfield;
        document.getElementById('sl-h-def').value = c.ratings.defense;
        document.getElementById('sl-h-gk').value = c.ratings.gk;
        updateVal('h-att'); updateVal('h-mid'); updateVal('h-def'); updateVal('h-gk');
      }
    }

    function onAwayTeamChange() {
      const name = document.getElementById('sel-away-team').value;
      const c = filteredClubs.find(x => x.name === name);
      if (c) {
        document.getElementById('sl-a-att').value = c.ratings.attack;
        document.getElementById('sl-a-mid').value = c.ratings.midfield;
        document.getElementById('sl-a-def').value = c.ratings.defense;
        document.getElementById('sl-a-gk').value = c.ratings.gk;
        updateVal('a-att'); updateVal('a-mid'); updateVal('a-def'); updateVal('a-gk');
      }
    }

    async function triggerPrediction() {
      const payload = {
        home_team: document.getElementById('sel-home-team').value,
        away_team: document.getElementById('sel-away-team').value,
        home_ratings: {
          attack: parseFloat(document.getElementById('sl-h-att').value),
          midfield: parseFloat(document.getElementById('sl-h-mid').value),
          defense: parseFloat(document.getElementById('sl-h-def').value),
          gk: parseFloat(document.getElementById('sl-h-gk').value),
        },
        away_ratings: {
          attack: parseFloat(document.getElementById('sl-a-att').value),
          midfield: parseFloat(document.getElementById('sl-a-mid').value),
          defense: parseFloat(document.getElementById('sl-a-def').value),
          gk: parseFloat(document.getElementById('sl-a-gk').value),
        }
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderPrediction(data);
        loadHistory();
      } catch (err) {
        console.error("Prediction failed:", err);
      }
    }

    function renderPrediction(data) {
      document.getElementById('lbl-home-team').innerText = data.home_team;
      document.getElementById('lbl-away-team').innerText = data.away_team;
      document.getElementById('lbl-home-xg').innerText = `Expected Goals: ${data.expected_goals.home} xG`;
      document.getElementById('lbl-away-xg').innerText = `Expected Goals: ${data.expected_goals.away} xG`;

      const score = data.most_likely_score;
      document.getElementById('lbl-score').innerText = `${score[0]} - ${score[1]}`;
      document.getElementById('lbl-conf').innerText = `${data.most_likely_score_prob}% Probability`;

      // Probabilities
      const p = data.probabilities;
      document.getElementById('bar-home').style.width = p.home_win + '%';
      document.getElementById('bar-home').innerText = p.home_win + '%';
      document.getElementById('bar-draw').style.width = p.draw + '%';
      document.getElementById('bar-draw').innerText = p.draw + '%';
      document.getElementById('bar-away').style.width = p.away_win + '%';
      document.getElementById('bar-away').innerText = p.away_win + '%';

      document.getElementById('txt-home-win').innerText = `${data.home_team} Win: ${p.home_win}%`;
      document.getElementById('txt-draw').innerText = `Draw: ${p.draw}%`;
      document.getElementById('txt-away-win').innerText = `${data.away_team} Win: ${p.away_win}%`;

      // Top Scores
      const topDiv = document.getElementById('top-scores-container');
      topDiv.innerHTML = '';
      data.top_scorelines.forEach(item => {
        const row = document.createElement('div');
        row.className = 'top-score-row';
        row.innerHTML = `<span>Score <strong>${item.score}</strong></span><span>${item.probability}%</span>`;
        topDiv.appendChild(row);
      });

      document.getElementById('lbl-over-25').innerText = `${data.over_under_2_5.over}%`;
      document.getElementById('lbl-under-25').innerText = `${data.over_under_2_5.under}%`;
      document.getElementById('lbl-btts').innerText = `${data.btts.yes}%`;

      // Render Matrix Table
      const matrixTable = document.getElementById('matrix-table');
      matrixTable.innerHTML = '';
      let headerRow = '<tr><th>H \\ A</th>';
      for(let j = 0; j <= 5; j++) headerRow += `<th>${j}</th>`;
      headerRow += '</tr>';
      matrixTable.innerHTML += headerRow;

      for (let i = 0; i <= 5; i++) {
        let rowHtml = `<tr><th>${i}</th>`;
        for (let j = 0; j <= 5; j++) {
          const val = data.score_matrix[i][j];
          const opacity = Math.min(1.0, val / 15.0);
          const bg = `rgba(16, 185, 129, ${opacity * 0.7})`;
          rowHtml += `<td style="background-color: ${bg}; font-weight: ${val > 10 ? '700' : '400'};">${val}%</td>`;
        }
        rowHtml += '</tr>';
        matrixTable.innerHTML += rowHtml;
      }
    }

    async function loadHistory() {
      try {
        const res = await fetch('/api/predictions');
        const list = await res.json();
        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';
        list.forEach(item => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${item.created_at}</td>
            <td><strong>${item.home_team}</strong> vs <strong>${item.away_team}</strong></td>
            <td style="color:#10b981; font-weight:700;">${item.predicted_score}</td>
            <td>${item.home_xg} - ${item.away_xg}</td>
            <td>${item.home_win_prob}%</td>
            <td>${item.draw_prob}%</td>
            <td>${item.away_win_prob}%</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error("Failed to load history:", err);
      }
    }

    window.onload = loadTeams;
  </script>
</body>
</html>
"""


# ==============================================================================
# 4. BACKEND REST API (Flask)
# ==============================================================================

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    """Serves the complete frontend dashboard."""
    return render_template_string(FRONTEND_HTML)


@app.route("/api/teams", methods=["GET"])
def api_get_teams():
    """Returns all clubs with tactical attributes & league from SQLite."""
    session = SessionLocal()
    try:
        teams = session.query(Team).all()
        data = []
        for t in teams:
            fwds = [p.rating for p in t.players if p.position == "FWD"]
            mids = [p.rating for p in t.players if p.position == "MID"]
            defs = [p.rating for p in t.players if p.position == "DEF"]
            gks = [p.rating for p in t.players if p.position == "GK"]

            att = round(float(np.mean(fwds)), 1) if fwds else 72.0
            mid = round(float(np.mean(mids)), 1) if mids else 71.0
            df = round(float(np.mean(defs)), 1) if defs else 70.0
            gk = round(float(np.mean(gks)), 1) if gks else 72.0
            ovr = round(0.27 * att + 0.33 * mid + 0.30 * df + 0.10 * gk, 1)

            data.append({
                "id": t.id,
                "name": t.name,
                "short_code": t.short_code,
                "stadium": t.stadium,
                "league": t.league,
                "ratings": {
                    "overall": ovr,
                    "attack": att,
                    "midfield": mid,
                    "defense": df,
                    "gk": gk,
                },
            })
        return jsonify(data)
    finally:
        session.close()


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Receives fixture and player ratings payload, computes prediction,
    logs prediction to SQLite, and returns JSON response.
    """
    req = request.get_json() or {}
    home_team = req.get("home_team", "Mohun Bagan SG")
    away_team = req.get("away_team", "East Bengal FC")

    h_ratings = req.get("home_ratings", {"attack": 74, "midfield": 72, "defense": 71, "gk": 73})
    a_ratings = req.get("away_ratings", {"attack": 70, "midfield": 69, "defense": 68, "gk": 69})

    # Execute ML prediction
    prediction = ml_engine.predict_match(
        home_team=home_team,
        away_team=away_team,
        h_ratings=h_ratings,
        a_ratings=a_ratings,
        h_form_ppg=1.8,
        a_form_ppg=1.4,
    )

    # Log to SQLite database
    session = SessionLocal()
    try:
        log_entry = PredictionEntry(
            home_team=home_team,
            away_team=away_team,
            predicted_score=f"{prediction['most_likely_score'][0]} - {prediction['most_likely_score'][1]}",
            home_xg=prediction["expected_goals"]["home"],
            away_xg=prediction["expected_goals"]["away"],
            home_win_prob=prediction["probabilities"]["home_win"],
            draw_prob=prediction["probabilities"]["draw"],
            away_win_prob=prediction["probabilities"]["away_win"],
            btts_prob=prediction["btts"]["yes"],
            over_2_5_prob=prediction["over_under_2_5"]["over"],
        )
        session.add(log_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"[Warning] Failed to log prediction: {e}")
    finally:
        session.close()

    return jsonify(prediction)


@app.route("/api/predictions", methods=["GET"])
def api_get_predictions():
    """Returns recent prediction logs from SQLite."""
    session = SessionLocal()
    try:
        logs = (
            session.query(PredictionEntry)
            .order_by(PredictionEntry.created_at.desc())
            .limit(10)
            .all()
        )
        data = [
            {
                "id": l.id,
                "created_at": l.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "home_team": l.home_team,
                "away_team": l.away_team,
                "predicted_score": l.predicted_score,
                "home_xg": l.home_xg,
                "away_xg": l.away_xg,
                "home_win_prob": l.home_win_prob,
                "draw_prob": l.draw_prob,
                "away_win_prob": l.away_win_prob,
            }
            for l in logs
        ]
        return jsonify(data)
    finally:
        session.close()


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   AI FOOTBALL SCORE PREDICTOR - ALL-IN-ONE FULL STACK SYSTEM   ")
    print("      Includes Indian Super League (ISL) & Premier League       ")
    print("=" * 70)
    # 1. Initialize SQLite Database
    init_db()

    # 2. Start Web Server
    host = "127.0.0.1"
    port = 5000
    print(f"\n[Server] Starting Full-Stack Application at: http://{host}:{port}")
    print("[Server] Press Ctrl+C to terminate.")
    print("=" * 70 + "\n")
    app.run(host=host, port=port, debug=False)
