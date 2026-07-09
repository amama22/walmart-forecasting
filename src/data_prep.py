import pandas as pd


def load_raw_data(data_dir="data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    features = pd.read_csv(f"{data_dir}/features.csv")
    stores = pd.read_csv(f"{data_dir}/stores.csv")

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    return train, test, features, stores


def merge_all(sales_df, features_df, stores_df):
    df = sales_df.merge(features_df, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    df = df.merge(stores_df, on="Store", how="left")

    if "IsHoliday_feat" in df.columns:
        df = df.drop(columns=["IsHoliday_feat"])

    return df
