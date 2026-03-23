# Intermediate Models — Ridge, Lasso, KNN
# Builds on the baseline with regularization and non-linear models.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import joblib

TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
MODELS_DIR = Path("reports/models")
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
    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())
    y = df[TARGET].copy()
    return X, y

def build_ridge(C=0.1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(penalty="l2", C=C, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"))
    ])

def build_lasso(C=0.1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(penalty="l1", C=C, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="liblinear"))
    ])

def build_knn(n_neighbors=11):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  KNeighborsClassifier(n_neighbors=n_neighbors,weights="distance",metric="euclidean"))
    ])

def tune_c(build_fn, X_train, y_train, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    c_vals = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_c, best_score = 0.1, 0

    for c in c_vals:
        pipeline = build_fn(C=c)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean = scores.mean()
        if mean > best_score:
            best_score = mean
            best_c = c

    print(f"{name} best C={best_c} :  CV AUC={best_score:.4f}")
    return best_c

def tune_knn(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    k_vals  = [3, 5, 7, 9, 11, 15, 21]
    best_k, best_score = 11, 0

    for k in k_vals:
        pipeline = build_knn(n_neighbors=k)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean = scores.mean()
        if mean > best_score:
            best_score = mean
            best_k = k

    print(f"KNN best k={best_k} : CV AUC={best_score:.4f}")
    return best_k

def evaluate_model(pipeline, X_train, y_train, X_test, y_test, name):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*45}")
    print(f"{name}")
    print(f"{'='*45}")
    print(f"Train AUC : {train_auc:.4f}")
    print(f"Test AUC  : {test_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No TJ', 'Has TJ'])}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob, train_auc, test_auc

def get_feature_importance(pipeline, name):
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coeffs = model.coef_[0]
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "coefficient": coeffs,
            "abs_coeff": np.abs(coeffs),
        }).sort_values("abs_coeff", ascending=False)
        print(f"\n{name} — Top 10 features:")
        print(importance.head(10).to_string(index=False))
        return importance
    return None

def get_lasso_zeroed(pipeline):
    coeffs = pipeline.named_steps["model"].coef_[0]
    zeroed = [FEATURE_COLS[i] for i, c in enumerate(coeffs) if c == 0]
    nonzero = [FEATURE_COLS[i] for i, c in enumerate(coeffs) if c != 0]
    print(f"\nLasso zeroed out {len(zeroed)} features: {zeroed}")
    print(f"Lasso kept {len(nonzero)} features: {nonzero}")
    return zeroed, nonzero

def plot_roc_comparison(models_data, X_test, y_test):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["#2E75B6", "#E67E22", "#27AE60", "#8E44AD"]
    for i, (name, pipeline, auc) in enumerate(models_data):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Intermediate Models vs Baseline")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intermediate_roc_curves.png", dpi=150)
    plt.close()


def plot_auc_comparison(results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    colors = ["#95A5A6" if m == "Logistic Regression"
                 else "#2E75B6" for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, test_aucs, color=colors, width=0.5)
    ax.set_ylim(0.7, 1.0)
    ax.axhline(0.8467, color="#95A5A6", linestyle="--", linewidth=1, label="Baseline AUC")
    for bar, auc in zip(bars, test_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{auc:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Test AUC")
    ax.set_title("Model Comparison — Test AUC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intermediate_auc_comparison.png", dpi=150)
    plt.close()


def save_results(results, models_data):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for name, pipeline, _ in models_data:
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, MODELS_DIR / f"{fname}.pkl")

    comparison = pd.DataFrame(results)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print(f"\nModel comparison saved to {TABLES_DIR}/model_comparison.csv")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    train, test = load_data()

    X_train, y_train = prepare(train)
    X_test, y_test = prepare(test)

    print("Tuning hyperparameters")
    best_c_ridge = tune_c(build_ridge, X_train, y_train, "Ridge")
    best_c_lasso = tune_c(build_lasso, X_train, y_train, "Lasso")
    best_k = tune_knn(X_train, y_train)

    ridge = build_ridge(C=best_c_ridge)
    lasso = build_lasso(C=best_c_lasso)
    knn = build_knn(n_neighbors=best_k)

    print("\nTraining and evaluating models")

    ridge_prob, ridge_train_auc, ridge_test_auc = evaluate_model(ridge, X_train, y_train, X_test, y_test, "Ridge (L2 Regularization)")
    lasso_prob, lasso_train_auc, lasso_test_auc = evaluate_model(lasso, X_train, y_train, X_test, y_test, "Lasso (L1 Regularization)")
    knn_prob, knn_train_auc, knn_test_auc = evaluate_model(knn, X_train, y_train, X_test, y_test, f"KNN (k={best_k})")

    get_feature_importance(ridge, "Ridge")
    get_feature_importance(lasso, "Lasso")
    get_lasso_zeroed(lasso)

    results = [
        {"model": "Logistic Regression", "train_auc": 0.8844, "test_auc": 0.8467, "notes": "Baseline"},
        {"model": "Ridge", "train_auc": round(ridge_train_auc, 4), "test_auc": round(ridge_test_auc, 4), "notes": f"L2, C={best_c_ridge}"},
        {"model": "Lasso", "train_auc": round(lasso_train_auc, 4), "test_auc": round(lasso_test_auc, 4), "notes": f"L1, C={best_c_lasso}"},
        {"model": f"KNN", "train_auc": round(knn_train_auc, 4), "test_auc": round(knn_test_auc, 4), "notes": f"k={best_k}, distance weights"},
    ]

    models_data = [
        ("Logistic Regression", ridge, 0.8467),
        (f"Ridge (C={best_c_ridge})", ridge, ridge_test_auc),
        (f"Lasso (C={best_c_lasso})", lasso, lasso_test_auc),
        (f"KNN (k={best_k})", knn, knn_test_auc),
    ]

    plot_roc_comparison(models_data, X_test, y_test)
    plot_auc_comparison(results)
    save_results(results, [
        ("ridge", ridge, ridge_test_auc),
        ("lasso", lasso, lasso_test_auc),
        (f"knn_k{best_k}", knn, knn_test_auc),
    ])