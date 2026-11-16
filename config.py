"""Proje ayarlari: yollar, ligler, sezonlar."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
DB_PATH = DATA / "futbol.sqlite"

# Son 5 sezon + icinde bulunulan sezon (sezon = baslangic yili)
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
START_DATE = "2021-07-01"

# Ortak lig kodlari. Her kaynagin kendi kimligi yaninda.
LEAGUES = {
    "ENG": {"name": "Premier League", "xgabora": "E0", "tm": "GB1", "understat": "EPL", "fotmob": 47, "sofascore": 17},
    "ESP": {"name": "La Liga", "xgabora": "SP1", "tm": "ES1", "understat": "La_liga", "fotmob": 87, "sofascore": 8},
    "GER": {"name": "Bundesliga", "xgabora": "D1", "tm": "L1", "understat": "Bundesliga", "fotmob": 54, "sofascore": 35},
    "ITA": {"name": "Serie A", "xgabora": "I1", "tm": "IT1", "understat": "Serie_A", "fotmob": 55, "sofascore": 23},
    "FRA": {"name": "Ligue 1", "xgabora": "F1", "tm": "FR1", "understat": "Ligue_1", "fotmob": 53, "sofascore": 34},
    "TUR": {"name": "Super Lig", "xgabora": "T1", "tm": "TR1", "understat": None, "fotmob": 71, "sofascore": 52},
    "UCL": {"name": "Champions League", "xgabora": None, "tm": "CL", "understat": None, "fotmob": 42, "sofascore": 7},
}

# Kibar kazima: istekler arasi bekleme (saniye)
MIN_DELAY = 3.0
MAX_DELAY = 7.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0 Safari/537.36"
)
