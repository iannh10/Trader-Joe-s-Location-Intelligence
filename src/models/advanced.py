# Advanced Models — Random Forest and Gradient Boosting
# Tree-based ensemble methods with built-in feature importance.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import joblib


TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
ALL_PATH = Path("data/processed/all_features.csv")
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
    test  = pd.read_csv(TEST_PATH)
    all_f = pd.read_csv(ALL_PATH)
    return train, test, all_f

def prepare(df):
    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())
    y = df[TARGET].copy()
    return X, y

def tune_random_forest(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = [
        {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 8},
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 15},
        {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 10},
    ]
    best_params, best_score = param_grid[0], 0

    for params in param_grid:
        model  = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mean   = scores.mean()
        print(f"RandomForest {params} : CV AUC={mean:.4f}")
        if mean > best_score:
            best_score  = mean
            best_params = params

    print(f"Best RandomForest: {best_params} :  CV AUC={best_score:.4f}")
    return best_params


def build_random_forest(params):
    return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params)

def tune_gradient_boosting(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,  "subsample": 0.8, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.01,  "subsample": 0.7, "min_samples_leaf": 10},
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1,   "subsample": 0.6, "min_samples_leaf": 8},
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05,  "subsample": 0.7, "min_samples_leaf": 10},
    ]
    best_params, best_score = param_grid[0], 0

    for params in param_grid:
        model = GradientBoostingClassifier(random_state=RANDOM_STATE,**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        mean = scores.mean()
        print(f"GradientBoosting {params} : CV AUC={mean:.4f}")
        if mean > best_score:
            best_score  = mean
            best_params = params

    print(f"Best GradientBoosting: {best_params} : CV AUC={best_score:.4f}")
    return best_params


def build_gradient_boosting(params):
    return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Train AUC : {train_auc:.4f}")
    print(f"Test AUC : {test_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No TJ', 'Has TJ'])}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob, train_auc, test_auc

def get_feature_importance(model, name):
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n{name} — Top 10 features:")
    print(importance.head(10).to_string(index=False))
    return importance

def plot_feature_importance(importance, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top10 = importance.head(10)
    fname = name.lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10["feature"][::-1], top10["importance"][::-1], color="#2E75B6")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{name} — Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{fname}_feature_importance.png", dpi=150)
    plt.close()


def plot_roc_all(X_test, y_test, rf_model, gb_model, rf_auc, gb_auc):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = MODELS_DIR / "baseline_lr.pkl"
    ridge_path = MODELS_DIR / "ridge.pkl"

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ["#95A5A6", "#3498DB", "#27AE60", "#8E44AD"]

    if baseline_path.exists():
        baseline = joblib.load(baseline_path)
        y_prob = baseline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        base_auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[0], linewidth=2, label=f"Logistic Regression (AUC={base_auc:.4f})")

    if ridge_path.exists():
        ridge = joblib.load(ridge_path)
        y_prob = ridge.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ridge_auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[1], linewidth=2, label=f"Ridge (AUC={ridge_auc:.4f})")

    y_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[2], linewidth=2, label=f"Random Forest (AUC={rf_auc:.4f})")

    y_prob = gb_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[3], linewidth=2, label=f"Gradient Boosting (AUC={gb_auc:.4f})")

    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models_roc_curves.png", dpi=150)
    plt.close()


def plot_final_comparison(results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    colors = ["#95A5A6","#3498DB","#E67E22","#E74C3C","#27AE60","#8E44AD"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(models, test_aucs, color=colors, width=0.6)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.8467, color="#95A5A6", linestyle="--", linewidth=1.5, label="Baseline AUC (0.8467)")
    for bar, auc in zip(bars, test_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f"{auc:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test AUC")
    ax.set_title("Complete Model Comparison — Test AUC")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "complete_model_comparison.png", dpi=150)
    plt.close()


def save_best_model(results, rf_model, gb_model):
    best = max(results, key=lambda x: x["test_auc"])
    print(f"\nBest model: {best['model']} (Test AUC={best['test_auc']})")

    if best["model"] == "Random Forest":
        joblib.dump(rf_model, MODELS_DIR / "best_model.pkl")
    elif best["model"] == "Gradient Boosting":
        joblib.dump(gb_model, MODELS_DIR / "best_model.pkl")
    elif (MODELS_DIR / "ridge.pkl").exists():
        ridge = joblib.load(MODELS_DIR / "ridge.pkl")
        joblib.dump(ridge, MODELS_DIR / "best_model.pkl")
    elif (MODELS_DIR / "baseline_lr.pkl").exists():
        baseline = joblib.load(MODELS_DIR / "baseline_lr.pkl")
        joblib.dump(baseline, MODELS_DIR / "best_model.pkl")

    print(f"Saved best model: {MODELS_DIR}/best_model.pkl")
    return best


def save_results(results, rf_model, gb_model, rf_importance, gb_importance):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(gb_model, MODELS_DIR / "gradient_boosting.pkl")

    rf_importance.to_csv(TABLES_DIR / "rf_feature_importance.csv",  index=False)
    gb_importance.to_csv(TABLES_DIR / "gb_feature_importance.csv",  index=False)

    comparison = pd.DataFrame(results)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    print("\nFinal Model Comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    train, test, all_f = load_data()

    X_train, y_train = prepare(train)
    X_test,  y_test  = prepare(test)

    print("Tuning Random Forest")
    rf_params = tune_random_forest(X_train, y_train)

    print("\nTuning Gradient Boosting")
    gb_params = tune_gradient_boosting(X_train, y_train)

    rf_model = build_random_forest(rf_params)
    gb_model = build_gradient_boosting(gb_params)

    print("\nEvaluating models")
    rf_prob, rf_train_auc, rf_test_auc = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    gb_prob, gb_train_auc, gb_test_auc = evaluate_model(gb_model, X_train, y_train, X_test, y_test, "Gradient Boosting")

    rf_importance = get_feature_importance(rf_model, "Random Forest")
    gb_importance = get_feature_importance(gb_model, "Gradient Boosting")

    plot_feature_importance(rf_importance, "Random Forest")
    plot_feature_importance(gb_importance, "Gradient Boosting")

    results = [
        {"model": "Logistic Regression", "train_auc": 0.8844,"test_auc": 0.8467, "notes": "Baseline"},
        {"model": "Ridge","train_auc": 0.8848,"test_auc": 0.8478, "notes": "L2, C=10"},
        {"model": "Lasso","train_auc": 0.8780,"test_auc": 0.8363, "notes": "L1, C=0.1"},
        {"model": "KNN","train_auc": 1.0000,"test_auc": 0.4923, "notes": "k=21, failed"},
        {"model": "Random Forest","train_auc": round(rf_train_auc, 4),"test_auc": round(rf_test_auc, 4), "notes": str(rf_params)},
        {"model": "Gradient Boosting","train_auc": round(gb_train_auc, 4),"test_auc": round(gb_test_auc, 4), "notes": str(gb_params)},
    ]

    plot_roc_all(X_test, y_test, rf_model, gb_model, rf_test_auc, gb_test_auc)

    plot_final_comparison(results)

    save_results(results, rf_model, gb_model, rf_importance, gb_importance)

    save_best_model(results, rf_model, gb_model)

    print(f"\nSaved models: {MODELS_DIR}")
    print(f"Saved figures: {FIGURES_DIR}")
    print(f"Saved tables : {TABLES_DIR}")