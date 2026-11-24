@echo off
echo Starting Football Score Predictor...
cd /d %~dp0
python -m streamlit run app.py
pause
