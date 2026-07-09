import pandas as pd


def add_lag_features(df, lags=[1, 2, 4, 52]):

    df = df.sort_values(["Store", "Dept", "Date"]).copy()

    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)

    return df


def add_rolling_features(df, windows=[4, 8]):

    df = df.sort_values(["Store", "Dept", "Date"]).copy()

    for w in windows:
        shifted = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
        df[f"sales_rolling_mean_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
        df[f"sales_rolling_std_{w}"] = shifted.groupby([df["Store"], df["Dept"]]).transform(
            lambda s: s.rolling(window=w, min_periods=1).std()
        )

    return df


def add_holiday_proximity(df):

    df = df.copy()

    holiday_dates = pd.to_datetime([
        "2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08",   # Super Bowl
        "2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06",   # Labor Day
        "2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29",   # Thanksgiving
        "2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27",   # Christmas
    ])

    def days_to_nearest_holiday(date):
        diffs = (holiday_dates - date).days
        return diffs[abs(diffs).argmin()]

    df["days_to_nearest_holiday"] = df["Date"].apply(days_to_nearest_holiday)
    return df


def add_expanding_dept_avg(df):

    df = df.copy()

    dept_weekly = df.groupby(["Dept", "Date"])["Weekly_Sales"].mean().reset_index()
    dept_weekly = dept_weekly.sort_values(["Dept", "Date"])

    dept_weekly["dept_avg_sales_expanding"] = (
        dept_weekly.groupby("Dept")["Weekly_Sales"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        dept_weekly[["Dept", "Date", "dept_avg_sales_expanding"]],
        on=["Dept", "Date"], how="left"
    )
    return df


def add_expanding_store_avg(df):

    df = df.copy()

    store_weekly = df.groupby(["Store", "Date"])["Weekly_Sales"].mean().reset_index()
    store_weekly = store_weekly.sort_values(["Store", "Date"])

    store_weekly["store_avg_sales_expanding"] = (
        store_weekly.groupby("Store")["Weekly_Sales"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        store_weekly[["Store", "Date", "store_avg_sales_expanding"]],
        on=["Store", "Date"], how="left"
    )
    return df


def build_all_features(df):
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_holiday_proximity(df)
    df = add_expanding_dept_avg(df)
    df = add_expanding_store_avg(df)
    return df
