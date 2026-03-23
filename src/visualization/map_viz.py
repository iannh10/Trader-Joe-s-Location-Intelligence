# Visualizations
# Generates maps and charts to analyze model results and recommendations.

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import folium

# Configuration

ALL_FEATURES_PATH = Path("data/processed/all_features.csv")
TJ_ALL_PATH = Path("data/trader_joes/tj_locations_ca.csv")
MODEL_PATH = Path("reports/models/best_model.pkl")
FIGURES_DIR = Path("reports/figures")

FEATURE_COLS = [
    "competitor_count", "avg_competitor_rating", "market_saturation_score",
    "opportunity_score", "avg_price_tier", "total_reviews", "total_population",
    "median_age", "median_household_income", "per_capita_income",
    "pct_bachelors_plus", "poverty_rate", "diversity_index", "pct_hispanic",
    "median_gross_rent", "median_home_value", "housing_occupancy_rate",
    "unemployment_rate", "income_rent_ratio", "pct_transit_commuters",
    "total_households",
]


# Loading data and model.

def load_data():
    all_f = pd.read_csv(ALL_FEATURES_PATH)
    tj_all = pd.read_csv(TJ_ALL_PATH)
    model = joblib.load(MODEL_PATH)
    return all_f, tj_all, model


# Scoring zip codes with the model to get probabilities for mapping and analysis.

def score_all(all_f, model):
    all_f = all_f.copy()
    all_f["zip_code"] = all_f["zip_code"].astype(str).str.zfill(5)
    X = all_f[FEATURE_COLS].fillna(all_f[FEATURE_COLS].median())
    probs = model.predict_proba(X)[:, 1]
    all_f["tj_probability"] = probs.round(4)
    return all_f


# Mapping California zip codes.

def plot_california_map(all_f, tj_all):

    tj_all = tj_all.copy()
    tj_all["zip_code"] = tj_all["zip_code"].astype(str).str.zfill(5)

    biz_path = Path("data/yelp/businesses_raw.csv")
    if not biz_path.exists():
        print("  No business data for map centroids.")
        return

    biz = pd.read_csv(biz_path)
    biz["zip_code"] = biz["zip_code"].astype(str).str.zfill(5)
    zip_centroids   = biz.groupby("zip_code").agg(
        lat=("latitude",  "mean"),
        lon=("longitude", "mean")
    ).reset_index()

    scored = all_f.merge(zip_centroids, on="zip_code", how="inner")
    scored = scored.dropna(subset=["lat","lon"])
    scored = scored[
        (scored["lat"].between(32, 42)) &
        (scored["lon"].between(-125, -114))
    ]

    # Folium map centered on California.
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=6,
                   tiles="CartoDB positron")

    # Color coding based on TJ probability.
    def get_color(p):
        if p >= 0.90:   
            return "#C0392B"
        elif p >= 0.75: 
            return "#E74C3C"
        elif p >= 0.60: 
            return "#E67E22"
        elif p >= 0.45: 
            return "#F1C40F"
        elif p >= 0.30: 
            return "#2ECC71"
        else: 
            return "#3498DB"

    for _, row in scored.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=get_color(row["tj_probability"]),
            fill=True, fill_opacity=0.7,
            popup=f"ZIP {row['zip_code']}: {row['tj_probability']:.4f}",
        ).add_to(m)

    # Add existing Trader Joe's locations.
    tj_coords = tj_all.dropna(subset=["latitude","longitude"])
    for _, row in tj_coords.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"Trader Joe's — {row['city']}",
            icon=folium.Icon(color="black", icon="shopping-cart", prefix="fa")
        ).add_to(m)

    # Highlight the top recommendation (ZIP 92101).
    top = all_f[all_f["has_tj"]==0].sort_values("tj_probability", ascending=False).iloc[0]
    top_coords = scored[scored["zip_code"] == str(top["zip_code"]).zfill(5)]
    if not top_coords.empty:
        folium.Marker(
            location=[top_coords.iloc[0]["lat"], top_coords.iloc[0]["lon"]],
            popup="TOP RECOMMENDATION: ZIP 92101 — Downtown San Diego",
            icon=folium.Icon(color="red", icon="star", prefix="fa")
        ).add_to(m)

    # Add legend.
    legend_html = """
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
                background:white;padding:15px;border-radius:8px;
                border:1px solid #ccc;font-family:Arial;font-size:13px;">
        <b>TJ Location Probability</b><br>
        <span style="color:#C0392B">&#9679;</span> 0.90+ Very High<br>
        <span style="color:#E74C3C">&#9679;</span> 0.75-0.90 High<br>
        <span style="color:#E67E22">&#9679;</span> 0.60-0.75 Medium<br>
        <span style="color:#F1C40F">&#9679;</span> 0.45-0.60 Low<br>
        <span style="color:#2ECC71">&#9679;</span> 0.30-0.45 Very Low<br>
        <span style="color:#3498DB">&#9679;</span> Below 0.30<br>
        <b>&#9733;</b> Top Recommendation<br>
        <b>&#9906;</b> Existing TJ Store
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map as an HTML file.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m.save(str(FIGURES_DIR / "california_tj_probability_map.html"))


# Model comparison bar chart

def plot_model_comparison():

    models = ["Logistic\nRegression","Ridge","Lasso","KNN","Random\nForest","Gradient\nBoosting"]
    test_aucs = [0.8467, 0.8478, 0.8363, 0.4923, 0.8435, 0.8142]
    colors = ["#95A5A6","#2E75B6","#E67E22","#E74C3C","#27AE60","#8E44AD"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, test_aucs, color=colors, width=0.55,
                     edgecolor="white", linewidth=1.5)

    ax.axhline(0.8467, color="#95A5A6", linestyle="--",
               linewidth=1.5, label="Baseline AUC (0.8467)")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Test AUC", fontsize=13)
    ax.set_title("Complete Model Comparison — Test AUC\nTrader Joe's California Location Recommendation",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    for bar, auc in zip(bars, test_aucs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{auc:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    bars[1].set_edgecolor("#1F4E79")
    bars[1].set_linewidth(3)
    ax.text(bars[1].get_x() + bars[1].get_width()/2,
            bars[1].get_height() + 0.05,
            "WINNER", ha="center", fontsize=10,
            color="#1F4E79", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# Feature importance chart.

def plot_feature_importance():

    model  = joblib.load(MODEL_PATH)
    lr     = model.named_steps["model"]
    coeffs = lr.coef_[0]

    importance = pd.DataFrame({
        "feature":     FEATURE_COLS,
        "coefficient": coeffs,
    }).sort_values("coefficient", key=abs, ascending=True).tail(15)

    colors = ["#2E75B6" if c > 0 else "#E74C3C"
              for c in importance["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance["feature"], importance["coefficient"],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (positive = increases TJ probability)", fontsize=12)
    ax.set_title("Ridge Regression — Top 15 Feature Coefficients\nTrader Joe's Location Model",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    pos = mpatches.Patch(color="#2E75B6", label="Positive — increases TJ probability")
    neg = mpatches.Patch(color="#E74C3C", label="Negative — decreases TJ probability")
    ax.legend(handles=[pos, neg], fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ROC curves for all models.

def plot_roc_curves():
    from sklearn.metrics import roc_curve, roc_auc_score

    test_f = pd.read_csv("data/processed/test_features.csv")
    X_test = test_f[FEATURE_COLS].fillna(test_f[FEATURE_COLS].median())
    y_test = test_f["has_tj"]

    models_to_plot = [
        ("Logistic Regression","reports/models/baseline_lr.pkl","#95A5A6"),
        ("Ridge (Best)", "reports/models/ridge.pkl","#2E75B6"),
        ("Random Forest", "reports/models/random_forest.pkl","#27AE60"),
        ("Gradient Boosting", "reports/models/gradient_boosting.pkl", "#8E44AD"),
    ]

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, path, color in models_to_plot:
        if not Path(path).exists():
            continue
        m           = joblib.load(path)
        y_prob      = m.predict_proba(X_test)[:, 1]
        auc         = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.4f})")

    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random baseline (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models\nTrader Joe's Location Recommendation",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# Top Recommendations bar chart

def plot_top_recommendations(all_f, tj_all):

    # Get top 20 zip codes with highest TJ probability that don't have an existing store.
    all_f["zip_code"]  = all_f["zip_code"].astype(str).str.zfill(5)
    tj_all["zip_code"] = tj_all["zip_code"].astype(str).str.zfill(5)
    existing  = set(tj_all["zip_code"].tolist())

    top20 = all_f[
        (all_f["has_tj"] == 0) &
        (~all_f["zip_code"].isin(existing))
    ].sort_values("tj_probability", ascending=False).head(20)

    # Highlight the top recommendation (ZIP 92101) in a different color.
    colors = ["#C0392B" if i == 0 else "#2E75B6"
              for i in range(len(top20))]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        top20["zip_code"].iloc[::-1],
        top20["tj_probability"].iloc[::-1],
        color=colors[::-1], edgecolor="white"
    )

    # Annotate bars with probability values.
    ax.set_xlabel("TJ Location Probability", fontsize=12)
    ax.set_title("Top 20 Recommended New Trader Joe's Locations in California",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.90, 1.01)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    for bar, (_, row) in zip(bars[::-1], top20.iterrows()):
        ax.text(row["tj_probability"] + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{row['tj_probability']:.4f}",
                va="center", fontsize=9)

    top_patch = mpatches.Patch(color="#C0392B",
                               label="Top recommendation (ZIP 92101)")
    ax.legend(handles=[top_patch], fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_recommendations.png", dpi=150, bbox_inches="tight")
    plt.close()


# Demographic profile comparison.

def plot_demographic_profile(all_f):

    all_f = all_f.copy()
    all_f["zip_code"] = all_f["zip_code"].astype(str).str.zfill(5)

    top_rec = all_f[all_f["zip_code"] == "92101"]
    if top_rec.empty:
        print("  ZIP 92101 not found.")
        return

    top_rec = top_rec.iloc[0]
    ca_med = all_f[FEATURE_COLS].median()

    features = [
        "pct_bachelors_plus", "diversity_index",
        "median_household_income", "median_gross_rent",
        "pct_transit_commuters", "total_households",
    ]
    labels = [
        "College\nEducated %", "Diversity\nIndex",
        "Median\nIncome", "Median\nRent",
        "Transit\nCommuters %", "Total\nHouseholds",
    ]

    vals_92101, vals_ca = [], []
    for f in features:
        col_min = all_f[f].min()
        col_max = all_f[f].max()
        rng = col_max - col_min if col_max != col_min else 1
        vals_92101.append((top_rec[f]  - col_min) / rng)
        vals_ca.append((ca_med[f] - col_min) / rng)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, vals_92101, width,
           label="ZIP 92101 (Top Recommendation)",
           color="#2E75B6", alpha=0.9)
    ax.bar(x + width/2, vals_ca, width,
           label="California Median",
           color="#95A5A6", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Normalized Score (0-1 across CA zip codes)", fontsize=11)
    ax.set_title("ZIP 92101 vs California Median — Key Demographics\nWhy Downtown San Diego is the Top Recommendation",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demographic_profile_92101.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# Run all visualizations

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_f, tj_all, model = load_data()
    scored = score_all(all_f, model)

    plot_california_map(scored, tj_all)
    plot_model_comparison()
    plot_feature_importance()
    plot_roc_curves()
    plot_top_recommendations(scored, tj_all)
    plot_demographic_profile(scored)