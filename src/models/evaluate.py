# Model Evaluation
# Cross validation results, test set performance, and scoring summary.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib


TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
ALL_PATH = Path("data/processed/all_features.csv")
TJ_ALL_PATH = Path("data/trader_joes/tj_locations_ca.csv")
TJ_TEST_PATH = Path("data/trader_joes/tj_test.csv")
MODEL_PATH = Path("reports/models/best_model.pkl")
TABLES_DIR = Path("reports/tables")

RANDOM_STATE = 42
TARGET = "has_tj"

FEATURE_COLS = [
    "competitor_count", "avg_competitor_rating", "market_saturation_score",
    "opportunity_score", "avg_price_tier", "total_reviews", "total_population",
    "median_age", "median_household_income", "per_capita_income",
    "pct_bachelors_plus", "poverty_rate", "diversity_index", "pct_hispanic",
    "median_gross_rent", "median_home_value", "housing_occupancy_rate",
    "unemployment_rate", "income_rent_ratio", "pct_transit_commuters",
    "total_households",
]



def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    all_f = pd.read_csv(ALL_PATH)
    tj_all = pd.read_csv(TJ_ALL_PATH)
    tj_test = pd.read_csv(TJ_TEST_PATH)
    model = joblib.load(MODEL_PATH)
    return train, test, all_f, tj_all, tj_test, model


def prepare(df):
    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())
    y = df[TARGET].copy()
    return X, y



def run_cross_validation(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    print("\n--- 5-Fold Cross Validation Results ---")
    for i, s in enumerate(scores):
        print(f"  Fold {i+1} : {s:.4f}")
    print(f"  Mean : {scores.mean():.4f}")
    print(f"  Std : {scores.std():.4f}")
    print(f"  Min : {scores.min():.4f}")
    print(f"  Max : {scores.max():.4f}")

    cv_df = pd.DataFrame({
        "fold": [f"Fold {i+1}" for i in range(len(scores))] + ["Mean", "Std"],
        "auc": list(scores) + [scores.mean(), scores.std()],
    })
    return cv_df, scores.mean()



def evaluate_test_set(model, X_test, y_test, test_df, tj_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- Test Set Performance ---")
    print(f"  Test AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No TJ', 'Has TJ'])}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Score all test zip codes
    test_df = test_df.copy()
    test_df["zip_code"] = test_df["zip_code"].astype(str).str.zfill(5)
    test_df["tj_probability"] = y_prob.round(4)

    # Get real TJ test locations and their scores
    tj_test["zip_code"] = tj_test["zip_code"].astype(str).str.zfill(5)
    tj_zips = set(tj_test["zip_code"].tolist())

    tj_scores = test_df[test_df["zip_code"].isin(tj_zips)][
        ["zip_code", "has_tj", "tj_probability"]
    ].sort_values("tj_probability", ascending=False)

    print(f"\n--- Real TJ Test Locations — Model Scores ---")
    print(f"  Total real TJ locations in test : {len(tj_scores)}")
    print(f"  Scored above 0.5 (found) : {(tj_scores['tj_probability'] >= 0.5).sum()}")
    print(f"  Scored below 0.5 (missed) : {(tj_scores['tj_probability'] < 0.5).sum()}")
    print(f"  Mean probability : {tj_scores['tj_probability'].mean():.4f}")
    print(f"  Max probability : {tj_scores['tj_probability'].max():.4f}")
    print(f"  Min probability : {tj_scores['tj_probability'].min():.4f}")
    print(f"\n{tj_scores.to_string(index=False)}")

    return tj_scores, auc



def evaluate_new_locations(model, all_f, tj_all):
    all_f = all_f.copy()
    tj_all = tj_all.copy()

    all_f["zip_code"]  = all_f["zip_code"].astype(str).str.zfill(5)
    tj_all["zip_code"] = tj_all["zip_code"].astype(str).str.zfill(5)

    X = all_f[FEATURE_COLS].fillna(all_f[FEATURE_COLS].median())
    probs = model.predict_proba(X)[:, 1]

    all_f["tj_probability"] = probs.round(4)

    existing_zips = set(tj_all["zip_code"].tolist())
    new_locs = all_f[
        (all_f["has_tj"] == 0) &
        (~all_f["zip_code"].isin(existing_zips))
    ].sort_values("tj_probability", ascending=False)

    print(f"\n--- Top 20 New Location Recommendations ---")
    print(f"{'Rank':<6} {'Zip Code':<12} {'TJ Probability'}")
    print("-" * 35)
    for i, (_, row) in enumerate(new_locs.head(20).iterrows(), 1):
        print(f"{i:<6} {row['zip_code']:<12} {row['tj_probability']:.4f}")

    return new_locs.head(20)[["zip_code", "tj_probability"]]



def save_results(cv_df, tj_scores, new_locs):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cv_df.to_csv(TABLES_DIR / "cross_validation_results.csv", index=False)
    tj_scores.to_csv(TABLES_DIR / "test_tj_scores.csv",       index=False)
    new_locs.to_csv(TABLES_DIR  / "new_location_scores.csv",  index=False)



if __name__ == "__main__":
    train, test, all_f, tj_all, tj_test, model = load_data()

    X_train, y_train = prepare(train)
    X_test, y_test = prepare(test)

    cv_df, cv_mean = run_cross_validation(model, X_train, y_train)
    tj_scores, auc = evaluate_test_set(model, X_test, y_test, test, tj_test)
    new_locs = evaluate_new_locations(model, all_f, tj_all)

    save_results(cv_df, tj_scores, new_locs)