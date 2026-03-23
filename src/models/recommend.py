# Location Recommendation
# Scores all CA zip codes and recommends the best new TJ location.

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


ALL_FEATURES_PATH = Path("data/processed/all_features.csv")
TJ_ALL_PATH = Path("data/trader_joes/tj_locations_ca.csv")
MODEL_PATH = Path("reports/models/best_model.pkl")
OUTPUT_PATH = Path("reports/tables/location_scores.csv")
TOP_PATH = Path("reports/tables/top_recommendations.csv")

FEATURE_COLS = [
    "competitor_count",
    "avg_competitor_rating",
    "market_saturation_score",
    "opportunity_score",
    "avg_price_tier",
    "total_reviews",
    "total_population",
    "median_age",
    "median_household_income",
    "per_capita_income",
    "pct_bachelors_plus",
    "poverty_rate",
    "diversity_index",
    "pct_hispanic",
    "median_gross_rent",
    "median_home_value",
    "housing_occupancy_rate",
    "unemployment_rate",
    "income_rent_ratio",
    "pct_transit_commuters",
    "total_households",
]


def load_data():
    all_f = pd.read_csv(ALL_FEATURES_PATH)
    tj = pd.read_csv(TJ_ALL_PATH)
    model = joblib.load(MODEL_PATH)
    return all_f, tj, model


def score_all_zips(all_f, model):
    X = all_f[FEATURE_COLS].copy().fillna(all_f[FEATURE_COLS].median())
    probs = model.predict_proba(X)[:, 1]
    scores = all_f[["zip_code", "has_tj"]].copy()
    scores["zip_code"] = scores["zip_code"].astype(str).str.zfill(5)
    scores["tj_probability"] = probs.round(4)
    scores = scores.sort_values("tj_probability", ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index + 1
    return scores


def get_candidates(scores, tj):
    tj["zip_code"] = tj["zip_code"].astype(str).str.zfill(5)
    existing_zips = set(tj["zip_code"].tolist())

    candidates = scores[scores["has_tj"] == 0].copy()
    candidates = candidates[~candidates["zip_code"].isin(existing_zips)]
    candidates = candidates.sort_values("tj_probability", ascending=False)
    return candidates


if __name__ == "__main__":
    all_f, tj, model = load_data()

    scores = score_all_zips(all_f, model)
    candidates = get_candidates(scores, tj)

    scores.to_csv(OUTPUT_PATH, index=False)
    candidates.head(20).to_csv(TOP_PATH, index=False)

    print("Top 10 Recommended Locations:")
    print(f"{'Rank':<6} {'Zip Code':<12} {'TJ Probability'}")
    print("-" * 35)
    for _, row in candidates.head(10).iterrows():
        print(f"{int(row['rank']):<6} {row['zip_code']:<12} {row['tj_probability']:.4f}")

    top = candidates.iloc[0]
    print(f"\nTop recommendation : ZIP {top['zip_code']}")
    print(f"TJ probability : {top['tj_probability']:.4f}")