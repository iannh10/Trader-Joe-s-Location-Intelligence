import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
YELP_DIR      = DATA_DIR / "yelp"
CENSUS_DIR    = DATA_DIR / "census"
TJ_DIR        = DATA_DIR / "trader_joes"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = ROOT_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
TABLES_DIR    = REPORTS_DIR / "tables"

# ─────────────────────────────────────────────
# API KEYS (loaded from .env — never hardcoded)
# ─────────────────────────────────────────────
YELP_API_KEY   = os.getenv("YELP_API_KEY", "")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

# ─────────────────────────────────────────────
# YELP SETTINGS
# ─────────────────────────────────────────────
YELP_SEARCH_RADIUS_M = 10000   # 10km radius per search point
YELP_MAX_RESULTS     = 50      # Yelp API max per request
YELP_REQUEST_DELAY   = 0.5     # seconds between API calls
YELP_MAX_REVIEWS     = 3       # free tier limit

# ─────────────────────────────────────────────
# CENSUS SETTINGS
# ─────────────────────────────────────────────
CENSUS_YEAR       = 2022
CENSUS_STATE_FIPS = "06"       # California

# ─────────────────────────────────────────────
# MODELING SETTINGS
# ─────────────────────────────────────────────
TRAIN_RATIO            = 0.80
RANDOM_STATE           = 42
CV_FOLDS               = 5
NEGATIVE_SAMPLE_RATIO  = 3     # 3 non-TJ zips per TJ zip

# ─────────────────────────────────────────────
# COMPETITOR SEARCH CONFIG
# ─────────────────────────────────────────────
COMPETITOR_CATEGORIES = [
    "grocery",
    "organicstores",
    "healthmarkets",
    "farmersmarket",
    "convenience",
]

COMPETITOR_BRANDS = [
    "Whole Foods", "Sprouts", "Safeway", "Vons", "Ralphs",
    "Pavilions", "Smart & Final", "WinCo", "Stater Bros",
    "Food 4 Less", "Gelson's", "Bristol Farms", "Lazy Acres",
    "99 Ranch Market", "H Mart", "Grocery Outlet", "Aldi",
]

# ─────────────────────────────────────────────
# CALIFORNIA SEARCH GRID
# (lat, lon, label) — major population centers
# ─────────────────────────────────────────────
CA_SEARCH_GRID = [
    (34.0522, -118.2437, "Los Angeles"),
    (34.1478, -118.1445, "Pasadena"),
    (33.9425, -118.4081, "Inglewood"),
    (34.0195, -118.4912, "Santa Monica"),
    (34.2220, -118.4695, "Northridge"),
    (33.8353, -118.3401, "Torrance"),
    (33.7701, -118.1937, "Long Beach"),
    (33.6846, -117.8265, "Irvine"),
    (33.7879, -117.8531, "Anaheim"),
    (33.6595, -117.9988, "Huntington Beach"),
    (33.5427, -117.7854, "Mission Viejo"),
    (32.7157, -117.1611, "San Diego"),
    (32.8328, -117.2713, "La Jolla"),
    (32.7795, -117.0359, "El Cajon"),
    (37.7749, -122.4194, "San Francisco"),
    (37.8716, -122.2727, "Berkeley"),
    (37.5485, -121.9886, "Fremont"),
    (37.3382, -121.8863, "San Jose"),
    (37.6879, -122.4702, "Daly City"),
    (37.9577, -122.0522, "Walnut Creek"),
    (37.4419, -122.1430, "Palo Alto"),
    (37.5630, -122.0530, "Dublin"),
    (38.5816, -121.4944, "Sacramento"),
    (38.6785, -121.7733, "Woodland"),
    (38.7521, -121.2880, "Roseville"),
    (36.7378, -119.7871, "Fresno"),
    (35.3733, -119.0187, "Bakersfield"),
    (37.9577, -121.2908, "Stockton"),
    (34.4208, -119.6982, "Santa Barbara"),
    (33.9806, -117.3755, "Riverside"),
]