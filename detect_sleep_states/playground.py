import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import matplotlib as plt

from data import import_data, reduce_dataset, extend_data_to_length

data_full, mappings = import_data()
data = reduce_dataset(data_full.copy())
data = extend_data_to_length(data, mappings)

# Some plotting

# mean enmo patient vise
data[data["event"] == 0].groupby(["series_id"])["enmo"].mean().hist(bins=50)
data[data["event"] == 0].groupby(["series_id"])["enmo"].max().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["enmo"].mean().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["enmo"].max().hist(bins=50)

data[data["event"] == 0].groupby(["series_id"])["anglez"].mean().hist(bins=50)
data[data["event"] == 0].groupby(["series_id"])["anglez"].max().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["anglez"].mean().hist(bins=50)
data[data["event"] == 1].groupby(["series_id"])["anglez"].max().hist(bins=50)

# ToDo: is max and avg correlated? Does it make sense to normalize?
# -> we could also add as feature and calculate feature importance

sid = data[data.series_id == 1].copy().reset_index(drop=True)
sid = sid[sid.event != -1]
sid["enmo_norm"] = sid["enmo"] / sid["enmo"].max()
sid["anglez_norm"] = (sid["anglez"] - sid["anglez"].min()) / (
    sid["anglez"].max() - sid["anglez"].min()
)
sid.loc[:40000][["event", "enmo_norm", "anglez_norm"]].plot()


# Feature engineering
# Normalize values grouped by series_id
# take differences of two consecutive timepoints


# To try:
# https://github.com/angus924/rocket

# https://pyts.readthedocs.io/en/stable/generated/pyts.classification.TimeSeriesForest.html#pyts.classification.TimeSeriesForest.fit

# xgboost and efficient way to extract data from dataset
