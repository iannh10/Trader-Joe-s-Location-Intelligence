import sys
import time
import requests
import pandas as pd
from tqdm import tqdm
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(curr_dir))
sys.path.append(parent_dir)

from config import (
    YELP_API_KEY, YELP_DIR,
    YELP_SEARCH_RADIUS_M, YELP_REQUEST_DELAY,
    YELP_MAX_RESULTS, YELP_MAX_REVIEWS,
    COMPETITOR_CATEGORIES, COMPETITOR_BRANDS,
    CA_SEARCH_GRID
)

BASE_API = "https://api.yelp.com/v3"


def build_headers():
    # need Bearer with a space before the key otherwise it wont work
    h = {
        "Authorization": "Bearer " + YELP_API_KEY,
        "Accept": "application/json",
    }
    return h


def fetch_businesses(lat, lon, term=None, category=None):
    params = {
        "latitude": lat,
        "longitude": lon,
        "radius": YELP_SEARCH_RADIUS_M,
        "limit": YELP_MAX_RESULTS,
        "sort_by": "rating",
    }

    if term is not None:
        params["term"] = term
    if category is not None:
        params["categories"] = category

    url = BASE_API + "/businesses/search"

    try:
        r = requests.get(url, headers=build_headers(), params=params, timeout=10)

        if r.status_code == 200:
            data = r.json()
            return data
        elif r.status_code == 429:
            # 429 means we are sending too many requests
            # learned this the hard way, need to back off
            print("rate limited, sleep 15s")
            time.sleep(15)
            return None
        else:
            print("got status code", r.status_code, "for", url)
            return None

    except Exception as e:
        print("error:", e)
        return None


def fetch_reviews(biz_id):
    url = BASE_API + "/businesses/" + str(biz_id) + "/reviews"

    try:
        r = requests.get(
            url,
            headers=build_headers(),
            params={"limit": YELP_MAX_REVIEWS},
            timeout=10
        )

        if r.status_code == 200:
            data = r.json()
            reviews = data.get("reviews", [])
            return reviews
        else:
            return []

    except:
        return []


def extract_business(b, grid_name):
    # pull out all the fields we need from the api response
    # the nesting is kind of annoying

    cats = []
    aliases = []
    cat_list = b.get("categories", [])
    for i in range(len(cat_list)):
        c = cat_list[i]
        cat_title = c.get("title", "")
        cat_alias = c.get("alias", "")
        cats.append(cat_title)
        aliases.append(cat_alias)

    # join them with comma
    cats_str = ", ".join(cats)
    aliases_str = ", ".join(aliases)

    coords = b.get("coordinates", {})
    loc = b.get("location", {})

    result = {
        "business_id": b.get("id"),
        "name": b.get("name"),
        "rating": b.get("rating"),
        "review_count": b.get("review_count"),
        "price": b.get("price"),
        "is_closed": b.get("is_closed"),
        "latitude": coords.get("latitude"),
        "longitude": coords.get("longitude"),
        "address": loc.get("address1"),
        "city": loc.get("city"),
        "state": loc.get("state"),
        "zip_code": loc.get("zip_code"),
        "categories": cats_str,
        "category_aliases": aliases_str,
        "phone": b.get("phone"),
        "distance_meters": b.get("distance"),
        "search_area": grid_name,
    }

    return result


def run_yelp_collection():
    # using a dict so we dont store the same business twice
    all_biz = {}
    all_reviews = []

    print("start scraping yelp")
    print("grid points:", len(CA_SEARCH_GRID))
    print("categories:", len(COMPETITOR_CATEGORIES))
    print("brands:", len(COMPETITOR_BRANDS))

    # go through each point in the search grid
    for item in tqdm(CA_SEARCH_GRID):
        lat = item[0]
        lon = item[1]
        grid_name = item[2]

        # search by category
        for j in range(len(COMPETITOR_CATEGORIES)):
            cat = COMPETITOR_CATEGORIES[j]
            res = fetch_businesses(lat, lon, category=cat)

            if res is not None and "businesses" in res:
                biz_list = res["businesses"]
                for k in range(len(biz_list)):
                    b = biz_list[k]
                    # double check it's actually in california
                    b_state = b.get("location", {}).get("state")
                    if b_state == "CA":
                        bid = b.get("id")
                        if bid is not None and bid not in all_biz:
                            all_biz[bid] = extract_business(b, grid_name)

            time.sleep(YELP_REQUEST_DELAY)

        # also search by brand name
        for j in range(len(COMPETITOR_BRANDS)):
            brand = COMPETITOR_BRANDS[j]
            res = fetch_businesses(lat, lon, term=brand)

            if res is not None and "businesses" in res:
                biz_list = res["businesses"]
                for k in range(len(biz_list)):
                    b = biz_list[k]
                    b_state = b.get("location", {}).get("state")
                    if b_state == "CA":
                        bid = b.get("id")
                        if bid is not None and bid not in all_biz:
                            all_biz[bid] = extract_business(b, grid_name)

            time.sleep(YELP_REQUEST_DELAY)

    print("total unique businesses:", len(all_biz))
    print("now fetching reviews...")

    # get reviews for each business
    biz_ids = list(all_biz.keys())
    for i in tqdm(range(len(biz_ids))):
        bid = biz_ids[i]
        revs = fetch_reviews(bid)

        for j in range(len(revs)):
            r = revs[j]
            txt = r.get("text", "")
            if isinstance(txt, str):
                txt = txt.strip()
            else:
                txt = ""

            # get the user name if it exists
            user_info = r.get("user")
            if user_info is not None:
                uname = user_info.get("name")
            else:
                uname = None

            review_row = {
                "business_id": bid,
                "review_id": r.get("id"),
                "rating": r.get("rating"),
                "text": txt,
                "time_created": r.get("time_created"),
                "user_name": uname,
            }
            all_reviews.append(review_row)

        time.sleep(YELP_REQUEST_DELAY)

    # turn the dict values into a list for the dataframe
    biz_rows = []
    for bid in all_biz:
        biz_rows.append(all_biz[bid])

    biz_df = pd.DataFrame(biz_rows)
    rev_df = pd.DataFrame(all_reviews)

    return biz_df, rev_df


def build_zip_features(df):
    if len(df) == 0:
        print("no data to build features from")
        return pd.DataFrame()

    df = df.copy()

    # only keep california rows
    df = df[df["state"] == "CA"]

    # need both zip and rating to be valid
    df = df.dropna(subset=["zip_code", "rating"])
    print("building features for", len(df), "businesses")

    # convert dollar signs to numbers
    # $ = 1, $$ = 2, etc
    def convert_price(p):
        if p == "$":
            return 1
        elif p == "$$":
            return 2
        elif p == "$$$":
            return 3
        elif p == "$$$$":
            return 4
        else:
            return None

    price_nums = []
    for i in range(len(df)):
        p = df.iloc[i]["price"]
        price_nums.append(convert_price(p))
    df["price_num"] = price_nums

    # turn is_closed boolean into 0/1 int for the open ones
    open_vals = []
    for i in range(len(df)):
        closed = df.iloc[i]["is_closed"]
        if closed == True:
            open_vals.append(0)
        else:
            open_vals.append(1)
    df["is_open_int"] = open_vals

    # group by zip and calculate features
    gp = df.groupby("zip_code")

    out = pd.DataFrame()
    out["competitor_count"] = gp["business_id"].count()
    out["avg_competitor_rating"] = gp["rating"].mean()
    out["median_competitor_rating"] = gp["rating"].median()
    out["avg_review_count"] = gp["review_count"].mean()
    out["total_reviews"] = gp["review_count"].sum()
    out["avg_price_tier"] = gp["price_num"].mean()
    out["open_count"] = gp["is_open_int"].sum()

    out = out.reset_index()

    # market saturation = how crowded and well-rated the area is
    out["market_saturation_score"] = out["competitor_count"] * out["avg_competitor_rating"] / 5.0
    out["market_saturation_score"] = out["market_saturation_score"].round(3)

    # opportunity = inverse of saturation (higher = less competition)
    # add 0.1 so we dont divide by zero
    out["opportunity_score"] = 1 / (out["market_saturation_score"] + 0.1)
    out["opportunity_score"] = out["opportunity_score"].round(3)

    return out


def add_sentiment(df):
    if len(df) == 0:
        return pd.DataFrame()

    # simple keyword based sentiment
    # not perfect but good enough for now
    pos_words = ["fresh", "organic", "clean", "friendly", "quality", "great", "love"]
    neg_words = ["expensive", "crowded", "poor", "bad", "dirty", "slow"]

    def score(txt):
        if not isinstance(txt, str):
            return 0

        t = txt.lower()
        s = 0

        # check each positive word
        for i in range(len(pos_words)):
            w = pos_words[i]
            if w in t:
                s = s + 1

        # check each negative word
        for i in range(len(neg_words)):
            w = neg_words[i]
            if w in t:
                s = s - 1

        return s

    df = df.copy()

    scores = []
    for i in range(len(df)):
        txt = df.iloc[i]["text"]
        sc = score(txt)
        scores.append(sc)

    df["sentiment_score"] = scores
    return df


if __name__ == "__main__":
    if not YELP_API_KEY:
        print("no api key! check the config file")
        sys.exit(1)

    if not os.path.exists(YELP_DIR):
        os.makedirs(YELP_DIR)
        print("created", YELP_DIR)

    biz_df, rev_df = run_yelp_collection()

    print("businesses:", len(biz_df))
    print("reviews:", len(rev_df))

    if len(biz_df) > 0:
        biz_path = os.path.join(YELP_DIR, "businesses_raw.csv")
        biz_df.to_csv(biz_path, index=False)
        print("saved businesses to", biz_path)

        zip_df = build_zip_features(biz_df)
        if len(zip_df) > 0:
            zip_path = os.path.join(YELP_DIR, "zip_features.csv")
            zip_df.to_csv(zip_path, index=False)
            print("saved zip features to", zip_path)

    if len(rev_df) > 0:
        rev_path = os.path.join(YELP_DIR, "reviews_raw.csv")
        rev_df.to_csv(rev_path, index=False)
        print("saved reviews to", rev_path)

        sent_df = add_sentiment(rev_df)
        if len(sent_df) > 0:
            sent_path = os.path.join(YELP_DIR, "reviews_with_sentiment.csv")
            sent_df.to_csv(sent_path, index=False)
            print("saved sentiment reviews to", sent_path)

    print("done")
