import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# file paths
IN_FILE = "data/trader_joes/tj_locations_raw.csv"
OUT_FILE = "data/trader_joes/tj_locations_ca.csv"
TRAIN_FILE = "data/trader_joes/tj_train.csv"
TEST_FILE = "data/trader_joes/tj_test.csv"


def make_output_folder():
    folder_path = "data/trader_joes"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("made folder:", folder_path)


def load_raw_file():
    df = pd.read_csv(IN_FILE)
    print("loaded file with", len(df), "rows")
    return df


# only care about California stores for this project
def keep_ca_only(df):
    ca_df = df[df["state"] == "CA"]
    ca_df = ca_df.copy()
    ca_df = ca_df.reset_index(drop=True)
    print("kept", len(ca_df), "california stores")
    return ca_df


# sometimes pandas/excel drops the leading 0 in zip codes
# this happened to me once and messed up the merge later
def fix_zip_code(df):
    df["zip_code"] = df["zip_code"].astype(str)
    # go through each row and pad with zeros if too short
    for i in range(len(df)):
        current_zip = df.iloc[i]["zip_code"]
        while len(current_zip) < 5:
            current_zip = "0" + current_zip
        df.at[df.index[i], "zip_code"] = current_zip
    return df


# only data of CA
def get_ca_data():
    data = load_raw_file()
    data = keep_ca_only(data)
    data = fix_zip_code(data)
    return data


# combine address parts into one string for geocoding
def build_search_text(row):
    # just smash them together with commas
    street_part = str(row["street"])
    city_part = str(row["city"])
    state_part = str(row["state"])
    zip_part = str(row["zip_code"])

    address_text = street_part + ", " + city_part + ", " + state_part + " " + zip_part
    return address_text


# limit the request speed so we dont get banned
def get_geocoder():
    geo = Nominatim(user_agent="my_tj_script")
    limiter = RateLimiter(geo.geocode, min_delay_seconds=1)
    return limiter


def get_coords(df):
    geocode_func = get_geocoder()

    lat_list = []
    lon_list = []

    total_rows = len(df)
    current_num = 1

    for i in range(len(df)):
        row = df.iloc[i]
        search_text = build_search_text(row)
        print("getting location", current_num, "of", total_rows, ":", search_text)

        try:
            result = geocode_func(search_text)
            if result is None:
                print("  could not find this one")
                lat_list.append(None)
                lon_list.append(None)
            else:
                lat_list.append(result.latitude)
                lon_list.append(result.longitude)
        except Exception as e:
            # if the api fails, just put None and move on
            print("error finding", search_text)
            print(e)
            lat_list.append(None)
            lon_list.append(None)

        current_num = current_num + 1

    df["latitude"] = lat_list
    df["longitude"] = lon_list
    return df


def need_geocode(df):
    # check if we already have latitude column
    if "latitude" not in df.columns:
        return True
    # check if its all empty
    all_null = df["latitude"].isnull().all()
    if all_null:
        return True
    return False


def get_unique_zip_table(df):
    # get just the unique zip codes into their own dataframe
    zip_list = []
    seen = set()
    for i in range(len(df)):
        z = df.iloc[i]["zip_code"]
        if z not in seen:
            zip_list.append(z)
            seen.add(z)

    zip_df = pd.DataFrame({"zip_code": zip_list})

    # shuffle them so the split is random
    zip_df = zip_df.sample(frac=1, random_state=42)
    zip_df = zip_df.reset_index(drop=True)
    return zip_df


def split_zip_list(zip_df):
    # roughly 80:20 split for train and test
    total = len(zip_df)
    split_point = int(total * 0.8)
    print("split point at index", split_point, "out of", total, "zips")

    train_zip_list = []
    test_zip_list = []

    for i in range(len(zip_df)):
        z = zip_df.iloc[i]["zip_code"]
        if i < split_point:
            train_zip_list.append(z)
        else:
            test_zip_list.append(z)

    return train_zip_list, test_zip_list


def make_train_test(df):
    # split by zip code so no zip leaks between train and test
    zip_df = get_unique_zip_table(df)
    train_zip_list, test_zip_list = split_zip_list(zip_df)

    train_df = df[df["zip_code"].isin(train_zip_list)].copy()
    test_df = df[df["zip_code"].isin(test_zip_list)].copy()

    # add a column so we know which split each row is in
    train_df["split"] = "train"
    test_df["split"] = "test"

    return train_df, test_df


def save_outputs(full_df, train_df, test_df):
    full_df.to_csv(OUT_FILE, index=False)
    print("saved full data to", OUT_FILE)
    train_df.to_csv(TRAIN_FILE, index=False)
    print("saved train to", TRAIN_FILE)
    test_df.to_csv(TEST_FILE, index=False)
    print("saved test to", TEST_FILE)


if __name__ == "__main__":
    make_output_folder()

    print("loading data")
    data = get_ca_data()
    print("got", len(data), "ca stores")

    if need_geocode(data):
        print("start geocoding")
        data = get_coords(data)
        missing_count = data["latitude"].isna().sum()
        if missing_count > 0:
            print("failed to find", missing_count, "addresses")
        else:
            print("all addresses found!")
    else:
        print("skip geocode, already have coords")

    train_data, test_data = make_train_test(data)
    save_outputs(data, train_data, test_data)

    print("all done")
    print("total:", len(data))
    print("train set:", len(train_data))
    print("test set:", len(test_data))
