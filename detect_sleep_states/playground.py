import pandas as pd
import numpy as np

import matplotlib as plt

from import_data import import_data 

data, mappings = import_data()

# data prep
# could we shrink the dataset by taking average of each minute and try to predict just on the
# minutes scale? Does we loose a lot? -> Depends on accuracies we can reach with 5 sec steps

# Some plotting

# mean enmo patient vise
data[data['event'] == 0].groupby(['series_id'])['enmo'].mean().hist(bins=50)
data[data['event'] == 0].groupby(['series_id'])['enmo'].max().hist(bins=50)
data[data['event'] == 1].groupby(['series_id'])['enmo'].mean().hist(bins=50)
data[data['event'] == 1].groupby(['series_id'])['enmo'].max().hist(bins=50)

data[data['event'] == 0].groupby(['series_id'])['anglez'].mean().hist(bins=50)
data[data['event'] == 0].groupby(['series_id'])['anglez'].max().hist(bins=50)
data[data['event'] == 1].groupby(['series_id'])['anglez'].mean().hist(bins=50)
data[data['event'] == 1].groupby(['series_id'])['anglez'].max().hist(bins=50)

# ToDo: is max and avg correlated? Does it make sense to normalize?
# -> we could also add as feature and calculate feature importance

sid = data[data.series_id == 1].copy().reset_index(drop=True)
sid = sid[sid.event != -1]
sid['enmo_norm'] = sid['enmo'] / sid['enmo'].max()
sid['anglez_norm'] = (sid['anglez'] - sid['anglez'].min()) / (sid['anglez'].max() - sid['anglez'].min())
sid.loc[:40000][['event', 'enmo_norm', 'anglez_norm']].plot()



# Feature engineering
# Normalize values grouped by series_id
# take differences of two consecutive timepoints


# To try:
#https://github.com/angus924/rocket

#https://pyts.readthedocs.io/en/stable/generated/pyts.classification.TimeSeriesForest.html#pyts.classification.TimeSeriesForest.fit

#xgboost and efficient way to extract data from dataset
