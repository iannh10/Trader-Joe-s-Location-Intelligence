# Product Recommendations
# Uses competitor business categories from Yelp data to recommend top 5 TJ product categories for each location.
# No hardcoded weights or thresholds — driven entirely by actual competitor category distribution near each zip code.

import pandas as pd
from pathlib import Path
from collections import Counter
import joblib


ALL_FEATURES_PATH = Path("data/processed/all_features.csv")
TEST_FEATURES_PATH = Path("data/processed/test_features.csv")
YELP_BIZ_PATH = Path("data/yelp/businesses_raw.csv")
TJ_ALL_PATH = Path("data/trader_joes/tj_locations_ca.csv")
TJ_TEST_PATH = Path("data/trader_joes/tj_test.csv")
MODEL_PATH = Path("reports/models/best_model.pkl")
OUTPUT_PATH = Path("reports/tables/product_recommendations.csv")

FEATURE_COLS = [
    "competitor_count", "avg_competitor_rating", "market_saturation_score",
    "opportunity_score", "avg_price_tier", "total_reviews", "total_population",
    "median_age", "median_household_income", "per_capita_income",
    "pct_bachelors_plus", "poverty_rate", "diversity_index", "pct_hispanic",
    "median_gross_rent", "median_home_value", "housing_occupancy_rate",
    "unemployment_rate", "income_rent_ratio", "pct_transit_commuters",
    "total_households",
]

# Yelp category to TJ product mapping 
# Maps Yelp business category aliases to TJ product categories
# Derived from Yelp's official category taxonomy

CATEGORY_MAP = {
    # Organic & Natural
    "organic_stores": "Organic & Natural Foods",
    "healthmarkets": "Organic & Natural Foods",
    "farmersmarket": "Organic & Natural Foods",
    "juicebars": "Organic & Natural Foods",
    "herbshops": "Organic & Natural Foods",

    # International & Ethnic Foods
    "asianfusion": "International & Ethnic Foods",
    "chinese": "International & Ethnic Foods",
    "japanese": "International & Ethnic Foods",
    "korean": "International & Ethnic Foods",
    "thai": "International & Ethnic Foods",
    "indian": "International & Ethnic Foods",
    "mexican": "International & Ethnic Foods",
    "mediterranean": "International & Ethnic Foods",
    "middleeastern": "International & Ethnic Foods",
    "vietnamese": "International & Ethnic Foods",
    "latin": "International & Ethnic Foods",
    "filipino": "International & Ethnic Foods",
    "halal": "International & Ethnic Foods",
    "importedfood": "International & Ethnic Foods",
    "ethnic_grocery": "International & Ethnic Foods",

    # Wine, Beer & Spirits
    "wine_bars": "Wine, Beer & Spirits",
    "beer_bar": "Wine, Beer & Spirits",
    "breweries": "Wine, Beer & Spirits",
    "wineries": "Wine, Beer & Spirits",
    "winetastingroom": "Wine, Beer & Spirits",
    "beer_and_wine": "Wine, Beer & Spirits",
    "distilleries": "Wine, Beer & Spirits",
    "cocktailbars": "Wine, Beer & Spirits",

    # Frozen & Convenience
    "convenience": "Frozen Meals & Convenience",
    "servicestations": "Frozen Meals & Convenience",

    # Plant-Based & Vegan
    "vegan": "Plant-Based & Vegan",
    "vegetarian": "Plant-Based & Vegan",
    "rawfood": "Plant-Based & Vegan",

    # Premium Snacks
    "gourmet": "Premium Snacks & Nuts",
    "candy": "Premium Snacks & Nuts",
    "chocolate": "Premium Snacks & Nuts",
    "popcorn": "Premium Snacks & Nuts",
    "nuts": "Premium Snacks & Nuts",

    # Fresh Produce & Flowers
    "grocery": "Fresh Produce & Flowers",
    "florists": "Fresh Produce & Flowers",
    "gardens": "Fresh Produce & Flowers",
    "csa": "Fresh Produce & Flowers",

    # Health & Supplements
    "vitaminssupplements": "Health & Supplements",
    "nutritionists": "Health & Supplements",
    "herbshops": "Health & Supplements",
    "sportsnewtrition": "Health & Supplements",

    # Cheese & Specialty Dairy
    "cheese": "Cheese & Specialty Dairy",
    "creameries": "Cheese & Specialty Dairy",
    "icecream": "Cheese & Specialty Dairy",

    # Ready-to-Eat
    "delis": "Ready-to-Eat & Prepared Foods",
    "salad": "Ready-to-Eat & Prepared Foods",
    "sandwiches": "Ready-to-Eat & Prepared Foods",
    "soup": "Ready-to-Eat & Prepared Foods",
    "hotdogs": "Ready-to-Eat & Prepared Foods",
    "wraps": "Ready-to-Eat & Prepared Foods",

    # Coffee & Tea
    "coffee": "Coffee & Tea",
    "coffeeroasteries": "Coffee & Tea",
    "tea": "Coffee & Tea",
    "bubbletea": "Coffee & Tea",
    "cafes": "Coffee & Tea",

    # Bakery & Bread
    "bakeries": "Bakery & Bread",
    "bagels": "Bakery & Bread",
    "donuts": "Bakery & Bread",
    "cupcakes": "Bakery & Bread",
    "patisserie": "Bakery & Bread",
    "pretzels": "Bakery & Bread",

    # Seafood & Sushi
    "seafood": "Seafood & Sushi",
    "sushi": "Seafood & Sushi",
    "fishnchips": "Seafood & Sushi",
    "rawbar": "Seafood & Sushi",
    "poke": "Seafood & Sushi",

    # Gluten-Free
    "gluten_free": "Gluten-Free Products",

    # Seasonal & Holiday
    "giftshops": "Seasonal & Holiday Items",
    "holiday": "Seasonal & Holiday Items",
}


def load_data():
    all_f = pd.read_csv(ALL_FEATURES_PATH)
    test_f = pd.read_csv(TEST_FEATURES_PATH)
    tj_all = pd.read_csv(TJ_ALL_PATH)
    tj_test = pd.read_csv(TJ_TEST_PATH)
    model = joblib.load(MODEL_PATH)
    biz = pd.read_csv(YELP_BIZ_PATH)
    return all_f, test_f, tj_all, tj_test, model, biz


def get_zip_categories(zip_code, biz):
    zip_code = str(zip_code).zfill(5)
    zip_biz  = biz[biz["zip_code"].astype(str).str.zfill(5) == zip_code]

    # Expand to nearby zips if fewer than 5 businesses
    if len(zip_biz) < 5:
        prefix  = zip_code[:3]
        zip_biz = biz[biz["zip_code"].astype(str).str.zfill(5).str.startswith(prefix)]

    if zip_biz.empty:
        return Counter()

    # Extract all category aliases
    all_aliases = []
    for cats in zip_biz["category_aliases"].dropna():
        aliases = [c.strip().lower() for c in str(cats).split(",")]
        all_aliases.extend(aliases)

    return Counter(all_aliases)


def recommend_products(zip_code, biz):
    category_counts = get_zip_categories(zip_code, biz)

    if not category_counts:
        return [], 0

    # Map Yelp categories to TJ products
    tj_product_counts = Counter()
    for yelp_cat, count in category_counts.items():
        tj_product = CATEGORY_MAP.get(yelp_cat)
        if tj_product:
            tj_product_counts[tj_product] += count

    total_businesses = sum(category_counts.values())
    top5 = tj_product_counts.most_common(5)
    return top5, total_businesses



def get_top_new_locations(all_f, tj_all, model, n=5):
    all_f["zip_code"] = all_f["zip_code"].astype(str).str.zfill(5)
    tj_all["zip_code"] = tj_all["zip_code"].astype(str).str.zfill(5)
    X = all_f[FEATURE_COLS].fillna(all_f[FEATURE_COLS].median())
    probs = model.predict_proba(X)[:, 1]
    all_f["tj_probability"] = probs.round(4)
    existing = set(tj_all["zip_code"].tolist())
    return all_f[
        (all_f["has_tj"] == 0) &
        (~all_f["zip_code"].isin(existing))
    ].sort_values("tj_probability", ascending=False).head(n)


def get_top_test_locations(test_f, tj_test, model, n=5):
    test_f["zip_code"] = test_f["zip_code"].astype(str).str.zfill(5)
    tj_test["zip_code"] = tj_test["zip_code"].astype(str).str.zfill(5)
    X = test_f[FEATURE_COLS].fillna(test_f[FEATURE_COLS].median())
    probs = model.predict_proba(X)[:, 1]
    test_f["tj_probability"] = probs.round(4)
    tj_zips = set(tj_test["zip_code"].tolist())
    return test_f[test_f["zip_code"].isin(tj_zips)].sort_values("tj_probability", ascending=False).head(n)


def process_locations(locations, label, biz):
    results = []

    print(f"\n{'='*60}")
    print(f"{label.upper()}")
    print(f"{'='*60}")

    for _, loc in locations.iterrows():
        zip_code = str(loc["zip_code"]).zfill(5)
        prob = loc["tj_probability"]
        products, total_biz = recommend_products(zip_code, biz)

        print(f"\nZIP {zip_code}  (TJ Probability: {prob:.4f})")
        print(f"Competitor businesses analyzed: {total_biz}")

        if not products:
            print("No category data available for this zip.")
            continue

        print(f"Top 5 Recommended TJ Products:")
        for i, (product, count) in enumerate(products, 1):
            print(f"  {i}. {product}  ({count} competitor mentions)")

        for i, (product, count) in enumerate(products, 1):
            results.append({
                "location_type": label,
                "zip_code": zip_code,
                "tj_probability": prob,
                "rank": i,
                "product": product,
                "competitor_mentions": count,
            })

    return results



if __name__ == "__main__":
    all_f, test_f, tj_all, tj_test, model, biz = load_data()

    new_locs = get_top_new_locations(all_f,  tj_all,  model, n=5)
    test_locs = get_top_test_locations(test_f, tj_test, model, n=5)

    new_results = process_locations(new_locs, "Top 5 New Locations",  biz)
    test_results = process_locations(test_locs, "Top 5 Test Locations", biz)

    all_results = new_results + test_results
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")