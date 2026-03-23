# Baseline Model — Logistic Regression
# Establishes a performance baseline before trying advanced models.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,confusion_matrix, RocCurveDisplay)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib


TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
MODEL_PATH = Path("reports/models/baseline_lr.pkl")
FIGURES_DIR = Path("reports/figures")
TABLES_DIR = Path("reports/tables")

RANDOM_STATE = 42
TARGET = "has_tj"

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
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def prepare(df):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()
    X = X.fillna(X.median())
    return X, y


def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(class_weight="balanced",max_iter=1000,random_state=RANDOM_STATE,solver="lbfgs"))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test, X_train, y_train):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"Train AUC : {train_auc:.4f}")
    print(f"Test AUC : {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No TJ", "Has TJ"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob, test_auc


def feature_importance(pipeline):
    coeffs     = pipeline.named_steps["model"].coef_[0]
    importance = pd.DataFrame({
        "feature":     FEATURE_COLS,
        "coefficient": coeffs,
        "abs_coeff":   np.abs(coeffs),
    }).sort_values("abs_coeff", ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    return importance


def plot_roc(pipeline, X_test, y_test):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name="Logistic Regression")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title("ROC Curve — Baseline Logistic Regression")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_roc_curve.png", dpi=150)
    plt.close()


def plot_feature_importance(importance):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top10  = importance.head(10)
    colors = ["#2E75B6" if c > 0 else "#C0392B" for c in top10["coefficient"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10["feature"][::-1], top10["coefficient"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (positive = more likely to have TJ)")
    ax.set_title("Baseline Model — Top 10 Feature Coefficients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_feature_importance.png", dpi=150)
    plt.close()


def save_results(pipeline, importance, y_prob, test_df, auc):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    importance.to_csv(TABLES_DIR / "baseline_feature_importance.csv", index=False)

    results = test_df[["zip_code"]].copy()
    results["true_label"] = test_df[TARGET].values
    results["predicted_prob"] = y_prob
    results["predicted_label"] = (y_prob >= 0.5).astype(int)
    results = results.sort_values("predicted_prob", ascending=False)
    results.to_csv(TABLES_DIR / "baseline_predictions.csv", index=False)

    summary = pd.DataFrame([{
        "model": "Logistic Regression",
        "test_auc": round(auc, 4),
        "notes": "Baseline — class_weight=balanced, StandardScaler",
    }])
    summary.to_csv(TABLES_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    train, test = load_data()

    X_train, y_train = prepare(train)
    X_test,  y_test  = prepare(test)

    print("Training Logistic Regression")
    pipeline = train_model(X_train, y_train)

    y_prob, auc = evaluate(pipeline, X_test, y_test, X_train, y_train)

    importance = feature_importance(pipeline)

    plot_roc(pipeline, X_test, y_test)
    plot_feature_importance(importance)

    save_results(pipeline, importance, y_prob, test, auc)

    print(f"\nSaved model : {MODEL_PATH}")
    print(f"Saved figures : {FIGURES_DIR}")
    print(f"Saved tables : {TABLES_DIR}")