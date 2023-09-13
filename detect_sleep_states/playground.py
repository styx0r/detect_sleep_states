import pandas as pd

train_series = pd.read_parquet('../data/train_series.parquet')
train_events = pd.read_csv('../data/train_events.csv')

test_series = pd.read_parquet('../data/test_series.parquet')

sample_submission = pd.read_csv('../data/sample_submission.csv')

# check some general stuff
# how may series_ids?
train_series['series_id'].drop_duplicates().size
# 277

# how many days of measurements do we have per series_id?
measures_series = train_series.groupby('series_id')['step'].max() * 5 / 60 / 60 / 24
measures_series.hist()


series_id='038441c925bb'
individual = train_series[train_series['series_id'] == series_id]
individual[['step','enmo']].plot(x='step', y='enmo')
individual[['step','anglez']].plot(x='step', y='anglez')
train_events[train_events['series_id'] == series_id]

# evaluation test as preparation
from evaluation import score
column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}
tolerances = {'onset': [1.0, 0.8],
              'wakeup': [1.0, 0.8]}
score(sample_submission, sample_submission, tolerances, **column_names)


merged_df = train_series.join(train_events, on=['series_id', 'timestamp'])