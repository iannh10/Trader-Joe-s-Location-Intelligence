import os
import time
import pandas as pd
import numpy as np
from census import Census
from dotenv import load_dotenv

# load the api key from .env
load_dotenv()
API_KEY = os.getenv("CENSUS_API_KEY")

# columns names 
col_mapping = {
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B19013_001E": "median_household_income",
    "B19301_001E": "per_capita_income",
    "B17001_002E": "population_below_poverty",
    "B15003_022E": "bachelors_degree_holders",
    "B15003_023E": "masters_degree_holders",
    "B15003_025E": "doctorate_holders",
    "B15003_001E": "education_population_base",
    "B02001_002E": "white_alone",
    "B02001_003E": "black_alone",
    "B02001_005E": "asian_alone",
    "B03001_003E": "hispanic_or_latino",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B25002_002E": "occupied_housing_units",
    "B25002_001E": "total_housing_units",
    "B08301_001E": "total_commuters",
    "B08301_010E": "public_transit_commuters",
    "B11001_001E": "total_households",
    "B23025_002E": "labor_force",
    "B23025_005E": "unemployed",
}


def get_raw_census_data():
    c = Census(API_KEY, year=2022)
    vars_to_get = list(col_mapping.keys())

    # i think the api only lets you do 45 at a time so splitting it up
    chunk_size = 45
    chunks = []
    start = 0
    while start < len(vars_to_get):
        end = start + chunk_size
        one_chunk = vars_to_get[start:end]
        chunks.append(one_chunk)
        start = end

    all_parts = []

    for i in range(len(chunks)):
        print("getting chunk", i + 1)
        try:
            data = c.acs5.get(["NAME"] + chunks[i], {"for": "zip code tabulation area:*"})
            df_part = pd.DataFrame(data)
            all_parts.append(df_part)
            time.sleep(0.3)
        except Exception as e:
            print("problem on chunk", i + 1)
            print(e)

    if len(all_parts) == 0:
        print("no data came back, returning empty")
        return pd.DataFrame()

    # start with first chunk then merge in the rest
    merged = all_parts[0]

    for i in range(1, len(all_parts)):
        merged = pd.merge(
            merged,
            all_parts[i],
            on=["zip code tabulation area"],
            how="outer",
            suffixes=("", "_dup"),
        )

        # get rid of duplicate columns that show up from merge
        dup_cols = []
        for col in merged.columns:
            if col.endswith("_dup"):
                dup_cols.append(col)

        if len(dup_cols) > 0:
            merged = merged.drop(columns=dup_cols)

    return merged


def process_features(df_input):
    df = df_input.copy()

    # rename the zip column first
    df = df.rename(columns={"zip code tabulation area": "zip"})

    # then rename the census codes to readable names
    df = df.rename(columns=col_mapping)

    # make sure zip is always 5 digits with leading zeros
    df["zip"] = df["zip"].astype(str)
    for i in range(len(df)):
        while len(df.iloc[i, df.columns.get_loc("zip")]) < 5:
            df.iloc[i, df.columns.get_loc("zip")] = "0" + df.iloc[i, df.columns.get_loc("zip")]

    # dont need the NAME column anymore
    if "NAME" in df.columns:
        df = df.drop(columns=["NAME"])

    # convert all the number columns to actual numbers
    for col in df.columns:
        if col == "zip":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # the census api uses these weird negative numbers for missing data
    # found this out the hard way lol
    df = df.replace(-666666666, np.nan)
    df = df.replace(-999999999, np.nan)
    df = df.replace(-888888888, np.nan)

    # only keep california zips (start with 9)
    ca_mask = df["zip"].str.startswith("9")
    df = df[ca_mask]
    print("california zips:", len(df))

    # remove places with barely any people
    df = df[df["total_population"] > 100]
    print("after removing tiny places:", len(df))

    # fill education NaN with 0 before adding them up
    # otherwise the whole sum becomes NaN which is annoying
    df["bachelors_degree_holders"] = df["bachelors_degree_holders"].fillna(0)
    df["masters_degree_holders"] = df["masters_degree_holders"].fillna(0)
    df["doctorate_holders"] = df["doctorate_holders"].fillna(0)

    # calculate percent with bachelors or higher
    edu_total = df["bachelors_degree_holders"] + df["masters_degree_holders"] + df["doctorate_holders"]
    df["pct_bachelors_plus"] = edu_total / df["education_population_base"]

    # poverty rate
    df["poverty_rate"] = df["population_below_poverty"] / df["total_population"]

    # unemployment rate
    df["unemployment_rate"] = df["unemployed"] / df["labor_force"]

    # percent that take public transit to work
    df["pct_transit_commuters"] = df["public_transit_commuters"] / df["total_commuters"]

    # what percent of housing is actually occupied
    df["housing_occupancy_rate"] = df["occupied_housing_units"] / df["total_housing_units"]

    # how many years of rent to equal yearly income basically
    # multiply rent by 12 to get yearly rent
    yearly_rent = df["median_gross_rent"] * 12
    df["income_rent_ratio"] = df["median_household_income"] / yearly_rent

    # hispanic percent
    df["pct_hispanic"] = df["hispanic_or_latino"] / df["total_population"]

    # diversity index using herfindahl type formula
    # 1 minus sum of squared proportions
    white_filled = df["white_alone"].fillna(0)
    black_filled = df["black_alone"].fillna(0)
    asian_filled = df["asian_alone"].fillna(0)
    hispanic_filled = df["hispanic_or_latino"].fillna(0)

    total_race = white_filled + black_filled + asian_filled + hispanic_filled

    white_share = white_filled / total_race
    black_share = black_filled / total_race
    asian_share = asian_filled / total_race
    hispanic_share = hispanic_filled / total_race

    sum_of_squares = white_share**2 + black_share**2 + asian_share**2 + hispanic_share**2
    df["diversity_index"] = 1 - sum_of_squares

    # clean up any infinity values from dividing by zero
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # make sure percentages stay between 0 and 1
    pct_cols = [
        "pct_bachelors_plus",
        "poverty_rate",
        "unemployment_rate",
        "pct_transit_commuters",
        "housing_occupancy_rate",
        "pct_hispanic",
        "diversity_index",
    ]

    for col in pct_cols:
        # cap at 1
        mask_high = df[col] > 1
        df.loc[mask_high, col] = 1.0
        # floor at 0
        mask_low = df[col] < 0
        df.loc[mask_low, col] = 0.0

    return df


if __name__ == "__main__":
    out_dir = "data/census"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("created directory:", out_dir)

    out_file = os.path.join(out_dir, "ca_demographics.csv")

    print("start fetching")
    raw_df = get_raw_census_data()
    print("got", len(raw_df), "rows from census")

    print("start cleaning")
    final_df = process_features(raw_df)
    print("final dataframe has", len(final_df), "rows and", len(final_df.columns), "columns")

    final_df.to_csv(out_file, index=False)
    print("file saved to", out_file)
    print("done!")
