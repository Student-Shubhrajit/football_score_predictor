"""
database.py
-----------
Database layer using SQLite and SQLAlchemy.
Models:
  1. Team: Club metadata and baseline attributes
  2. Player: Individual footballers with positions, overall ratings, and tactical attributes
  3. MatchHistory: Past league match records with scores and lineup ratings
  4. PredictionLog: Historical record of predictions made by users
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
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

DATABASE_FILE = "football_database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    short_code = Column(String(10), nullable=False)
    stadium = Column(String(100), nullable=False)
    league = Column(String(50), default="Premier League")
    tier = Column(Integer, default=2)

    # Relationships
    players = relationship("Player", back_populates="team", cascade="all, delete-orphan")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    name = Column(String(100), nullable=False)
    position = Column(String(20), nullable=False)  # "FWD", "MID", "DEF", "GK"
    jersey_number = Column(Integer, default=10)
    rating = Column(Float, nullable=False)  # 60.0 to 99.0
    pace = Column(Integer, default=75)
    shooting = Column(Integer, default=75)
    passing = Column(Integer, default=75)
    defending = Column(Integer, default=75)

    team = relationship("Team", back_populates="players")


class MatchHistory(Base):
    __tablename__ = "match_history"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(DateTime, default=datetime.utcnow)
    season = Column(String(20), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_goals = Column(Integer, nullable=False)
    away_goals = Column(Integer, nullable=False)
    home_rating_overall = Column(Float, nullable=False)
    away_rating_overall = Column(Float, nullable=False)
    result = Column(String(5), nullable=False)  # "H", "D", "A"


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    home_team = Column(String(100), nullable=False)
    away_team = Column(String(100), nullable=False)
    predicted_score = Column(String(20), nullable=False)  # e.g. "2 - 1"
    home_xg = Column(Float, nullable=False)
    away_xg = Column(Float, nullable=False)
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    btts_prob = Column(Float, nullable=False)
    over_2_5_prob = Column(Float, nullable=False)


# Baseline real-world club profiles and starting XI stars
SAMPLE_CLUBS_DATA = [
    {
        "name": "Arsenal",
        "short_code": "ARS",
        "stadium": "Emirates Stadium",
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
    {
        "name": "Real Madrid",
        "short_code": "RMA",
        "stadium": "Santiago Bernabeu",
        "tier": 1,
        "squad": [
            ("Thibaut Courtois", "GK", 1, 89, 48, 20, 75, 90),
            ("Antonio Rudiger", "DEF", 22, 87, 82, 55, 72, 87),
            ("Eder Militao", "DEF", 3, 85, 83, 50, 70, 85),
            ("Dani Carvajal", "DEF", 2, 85, 81, 65, 80, 83),
            ("Ferland Mendy", "DEF", 23, 82, 89, 62, 74, 82),
            ("Aurelien Tchouameni", "MID", 14, 86, 76, 72, 80, 86),
            ("Federico Valverde", "MID", 8, 88, 88, 84, 85, 82),
            ("Jude Bellingham", "MID", 5, 90, 80, 86, 86, 82),
            ("Rodrygo", "FWD", 11, 86, 88, 82, 81, 48),
            ("Kylian Mbappe", "FWD", 9, 91, 97, 90, 80, 45),
            ("Vinicius Junior", "FWD", 7, 90, 95, 84, 81, 44),
        ],
    },
    {
        "name": "Barcelona",
        "short_code": "BAR",
        "stadium": "Camp Nou",
        "tier": 1,
        "squad": [
            ("Marc-Andre ter Stegen", "GK", 1, 88, 50, 25, 87, 88),
            ("Pau Cubarsi", "DEF", 2, 82, 73, 40, 79, 83),
            ("Inigo Martinez", "DEF", 5, 82, 68, 52, 74, 83),
            ("Jules Kounde", "DEF", 23, 85, 83, 55, 76, 85),
            ("Alejandro Balde", "DEF", 3, 82, 91, 55, 75, 78),
            ("Pedri", "MID", 8, 87, 78, 77, 88, 72),
            ("Gavi", "MID", 6, 84, 76, 74, 80, 78),
            ("Dani Olmo", "MID", 20, 85, 78, 83, 84, 58),
            ("Lamine Yamal", "FWD", 19, 86, 86, 80, 84, 45),
            ("Robert Lewandowski", "FWD", 9, 88, 72, 89, 78, 48),
            ("Raphinha", "FWD", 11, 86, 89, 83, 82, 55),
        ],
    },
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
]


def init_database() -> None:
    """Creates database tables and seeds initial teams and footballers."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    try:
        # Check if teams already seeded
        existing_teams = session.query(Team).count()
        if existing_teams == 0:
            print("[Database] Seeding real clubs, stadiums, and Starting XI players...")

            for club_info in SAMPLE_CLUBS_DATA:
                team = Team(
                    name=club_info["name"],
                    short_code=club_info["short_code"],
                    stadium=club_info["stadium"],
                    league=club_info.get("league", "Premier League"),
                    tier=club_info["tier"],
                )
                session.add(team)
                session.flush()

                # Add squad players
                for p in club_info["squad"]:
                    name, pos, num, rating, pace, sho, pas, df = p
                    player = Player(
                        team_id=team.id,
                        name=name,
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

            # Seed realistic historical matches between these clubs
            teams = session.query(Team).all()
            base_date = datetime(2023, 8, 15)
            match_id_counter = 1

            for i, h_t in enumerate(teams):
                for j, a_t in enumerate(teams):
                    if h_t.id != a_t.id:
                        m_date = base_date + timedelta(days=match_id_counter * 3)
                        # Realistic score generation based on ratings
                        h_ratings = [p.rating for p in h_t.players] if h_t.players else [80.0]
                        a_ratings = [p.rating for p in a_t.players] if a_t.players else [80.0]
                        h_avg = float(np.mean(h_ratings))
                        a_avg = float(np.mean(a_ratings))

                        lam_h = max(0.4, 1.45 + 0.08 * (h_avg - a_avg))
                        lam_a = max(0.2, 1.10 + 0.08 * (a_avg - h_avg))

                        hg = int(np.random.poisson(lam_h))
                        ag = int(np.random.poisson(lam_a))
                        res = "H" if hg > ag else ("D" if hg == ag else "A")

                        match_rec = MatchHistory(
                            match_date=m_date,
                            season="2023-2024",
                            home_team_id=h_t.id,
                            away_team_id=a_t.id,
                            home_goals=hg,
                            away_goals=ag,
                            home_rating_overall=round(float(h_avg), 1),
                            away_rating_overall=round(float(a_avg), 1),
                            result=res,
                        )
                        session.add(match_rec)
                        match_id_counter += 1

            session.commit()
            print(f"[Database] Successfully initialized and seeded SQLite database at {DATABASE_FILE}.")
        else:
            print(f"[Database] SQLite database already exists with {existing_teams} clubs.")
    finally:
        session.close()


def get_all_teams() -> List[Dict[str, Any]]:
    """Retrieve all clubs with aggregated lineup rating statistics."""
    session = SessionLocal()
    try:
        teams = session.query(Team).all()
        result = []
        for t in teams:
            players = t.players
            fwds = [p.rating for p in players if p.position == "FWD"]
            mids = [p.rating for p in players if p.position == "MID"]
            defs = [p.rating for p in players if p.position == "DEF"]
            gks = [p.rating for p in players if p.position == "GK"]

            att_rating = round(float(np.mean(fwds)), 1) if fwds else 78.0
            mid_rating = round(float(np.mean(mids)), 1) if mids else 78.0
            def_rating = round(float(np.mean(defs)), 1) if defs else 78.0
            gk_rating = round(float(np.mean(gks)), 1) if gks else 80.0
            ovr_rating = round(0.27 * att_rating + 0.33 * mid_rating + 0.30 * def_rating + 0.10 * gk_rating, 1)

            result.append({
                "id": t.id,
                "name": t.name,
                "short_code": t.short_code,
                "stadium": t.stadium,
                "league": t.league,
                "ratings": {
                    "overall": ovr_rating,
                    "attack": att_rating,
                    "midfield": mid_rating,
                    "defense": def_rating,
                    "gk": gk_rating,
                },
                "squad_count": len(players),
            })
        return result
    finally:
        session.close()


def get_team_squad(team_name: str) -> List[Dict[str, Any]]:
    """Retrieve full starting XI and squad for a club."""
    session = SessionLocal()
    try:
        team = session.query(Team).filter(Team.name == team_name).first()
        if not team:
            return []
        return [
            {
                "id": p.id,
                "name": p.name,
                "position": p.position,
                "jersey_number": p.jersey_number,
                "rating": p.rating,
                "pace": p.pace,
                "shooting": p.shooting,
                "passing": p.passing,
                "defending": p.defending,
            }
            for p in team.players
        ]
    finally:
        session.close()


def log_prediction(pred_data: Dict[str, Any]) -> int:
    """Save user prediction record into SQLite database."""
    session = SessionLocal()
    try:
        record = PredictionLog(
            home_team=pred_data["home_team"],
            away_team=pred_data["away_team"],
            predicted_score=f"{pred_data['most_likely_score'][0]} - {pred_data['most_likely_score'][1]}",
            home_xg=pred_data["expected_goals"]["home"],
            away_xg=pred_data["expected_goals"]["away"],
            home_win_prob=pred_data["probabilities"]["home_win"],
            draw_prob=pred_data["probabilities"]["draw"],
            away_win_prob=pred_data["probabilities"]["away_win"],
            btts_prob=pred_data["btts"]["yes"],
            over_2_5_prob=pred_data["over_under_2_5"]["over"],
        )
        session.add(record)
        session.commit()
        return record.id
    finally:
        session.close()


def get_recent_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve logged prediction history from database."""
    session = SessionLocal()
    try:
        logs = (
            session.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
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
                "btts_prob": l.btts_prob,
                "over_2_5_prob": l.over_2_5_prob,
            }
            for l in logs
        ]
    finally:
        session.close()


if __name__ == "__main__":
    init_database()
    teams = get_all_teams()
    print(f"Total clubs in SQLite DB: {len(teams)}")
    for t in teams:
        print(f"  - {t['name']} ({t['short_code']}): Overall {t['ratings']['overall']}, Attack {t['ratings']['attack']}, Defense {t['ratings']['defense']}")
