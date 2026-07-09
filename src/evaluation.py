import numpy as np
import pandas as pd


def wmae(y_true, y_pred, is_holiday):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_holiday = np.asarray(is_holiday)

    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def walk_forward_splits(df, date_col="Date", n_splits=3, horizon_weeks=38):

    dates = pd.to_datetime(df[date_col])
    unique_dates = np.sort(dates.unique())

    horizon_days = horizon_weeks * 7
    splits = []
    last_date = unique_dates[-1]

    for i in range(n_splits):
        val_end = last_date - pd.Timedelta(days=horizon_days * i)
        val_start = val_end - pd.Timedelta(days=horizon_days)

        train_mask = dates < val_start
        val_mask = (dates >= val_start) & (dates <= val_end)

        if train_mask.sum() == 0:
            continue

        splits.append((train_mask.values, val_mask.values))

    return list(reversed(splits))
