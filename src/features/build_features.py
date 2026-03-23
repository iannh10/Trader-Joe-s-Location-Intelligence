# Feature Engineering
# Combines and processes raw datasets to create a clean feature matrix for modeling.

import pandas as pd
import numpy as np
from pathlib import Path

#  File paths

YELP_PATH = Path("data/yelp/zip_features.csv")
CENSUS_PATH = Path("data/census/ca_demographics.csv")
TJ_TRAIN_PATH = Path("data/trader_joes/tj_train.csv")
TJ_TEST_PATH = Path("data/trader_joes/tj_test.csv")
TJ_ALL_PATH = Path("data/trader_joes/tj_locations_ca.csv")

OUTPUT_TRAIN = Path("data/processed/train_features.csv")
OUTPUT_TEST = Path("data/processed/test_features.csv")
OUTPUT_ALL = Path("data/processed/all_features.csv")

CA_ZIP_PREFIXES = tuple(str(i) for i in range(900, 967))

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


# Helper function to filter for CA zip codes

def is_ca_zip(series):
    return series.str.startswith(CA_ZIP_PREFIXES)


# Loads all raw datasets

def load_data():
    yelp = pd.read_csv(YELP_PATH)
    census = pd.read_csv(CENSUS_PATH)
    tj_all = pd.read_csv(TJ_ALL_PATH)
    train = pd.read_csv(TJ_TRAIN_PATH)
    test = pd.read_csv(TJ_TEST_PATH)
    return yelp, census, tj_all, train, test


# Data cleaning for each dataset.

def clean_yelp(df):
    df = df.copy()
    df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)
    df = df.dropna(subset=["zip_code"])
    df = df[df["zip_code"].str.match(r'^\d{5}$')]
    df = df[is_ca_zip(df["zip_code"])]
    return df.reset_index(drop=True)


def clean_census(df):
    df = df.copy()
    df = df.rename(columns={
        "zip":         "zip_code",
        "black_alone": "african_american_population",
    })
    df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)
    df = df[df["zip_code"].str.match(r'^\d{5}$')]
    df = df[is_ca_zip(df["zip_code"])]
    return df.reset_index(drop=True)


def clean_tj(df):
    df = df.copy()
    df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)
    df = df[df["zip_code"].str.match(r'^\d{5}$')]
    df = df[is_ca_zip(df["zip_code"])]
    return df.reset_index(drop=True)


# Labels: 1 if zip has TJ, 0 otherwise

def build_labels(tj_df, all_zips):
    tj_zips = set(tj_df["zip_code"].tolist())
    labels = pd.DataFrame({"zip_code": all_zips})
    labels["has_tj"] = labels["zip_code"].apply(
        lambda z: 1 if z in tj_zips else 0
    )
    return labels


# Merges all features into one matrix

def build_feature_matrix(yelp, census, tj_zips_df):
    df = census.copy()

    df = df.merge(yelp, on="zip_code", how="left")

    yelp_cols = [
        "competitor_count", "avg_competitor_rating",
        "median_competitor_rating", "avg_review_count",
        "total_reviews", "avg_price_tier", "open_count",
        "market_saturation_score", "opportunity_score",
    ]
    df[yelp_cols] = df[yelp_cols].fillna(0)

    labels = build_labels(tj_zips_df, df["zip_code"].tolist())
    df = df.merge(labels, on="zip_code", how="left")

    return df


# Feature selection

def select_features(df):
    cols = ["zip_code"] + FEATURE_COLS + ["has_tj"]
    df = df[[c for c in cols if c in df.columns]].copy()

    # Fill remaining nulls with column median
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# Run the feature engineering pipeline

if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    yelp, census, tj_all, train, test = load_data()

    yelp = clean_yelp(yelp)
    census = clean_census(census)
    tj_all = clean_tj(tj_all)
    train = clean_tj(train)
    test = clean_tj(test)

    train_matrix = build_feature_matrix(yelp, census, train)
    test_matrix = build_feature_matrix(yelp, census, test)
    all_matrix = build_feature_matrix(yelp, census, tj_all)

    train_matrix = select_features(train_matrix)
    test_matrix = select_features(test_matrix)
    all_matrix = select_features(all_matrix)

    train_matrix.to_csv(OUTPUT_TRAIN, index=False)
    test_matrix.to_csv(OUTPUT_TEST, index=False)
    all_matrix.to_csv(OUTPUT_ALL, index=False)

    print(f"Train matrix: {train_matrix.shape}")
    print(f"Test matrix: {test_matrix.shape}")
    print(f"All matrix: {all_matrix.shape}")
    print(f"TJ in train: {train_matrix['has_tj'].sum()}")
    print(f"TJ in test: {test_matrix['has_tj'].sum()}")
    print(f"Train nulls: {train_matrix.isnull().sum().sum()}")
    print(f"Test nulls: {test_matrix.isnull().sum().sum()}")
    print(f"All nulls: {all_matrix.isnull().sum().sum()}")