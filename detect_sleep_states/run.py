import pandas as pd
import gc

import datetime

train_series = pd.read_parquet('../data/train_series.parquet')
train_events = pd.read_csv('../data/train_events.csv')


######### Transform data to memory efficient representation #########
# series_id mapping
series_ids = pd.DataFrame({'series_id': train_series.series_id.unique()})
mapping_series_ids = series_ids.reset_index().set_index('series_id')['index'].to_dict()
train_series['series_id'] = train_series['series_id'].map(mapping_series_ids)
train_series['series_id'] = train_series['series_id'].astype('uint16')

train_series['step'] = train_series['step'].astype('uint32')

train_series['seconds']=train_series['timestamp'].str[11:13].astype('uint32') * 60 * 60 + train_series['timestamp'].str[14:16].astype('uint32') * 60 + train_series['timestamp'].str[17:19].astype('uint32')
train_series['day']=train_series['timestamp'].str[8:10].astype('uint8')
train_series['month']=train_series['timestamp'].str[5:7].astype('uint8')

train_series['year']=train_series['timestamp'].str[:4].astype('uint16')

# drop timestamp and free memory
train_series.drop('timestamp', axis=1, inplace=True)
gc.collect()

train_series['weekday']=train_series.apply(lambda x:  datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)

years = pd.DataFrame({'year': sorted(train_series.year.unique())})
mapping_years = years.reset_index().set_index('year')['index'].to_dict()
train_series['year'] = train_series['year'].map(mapping_years)
train_series['year'] = train_series['year'].astype('uint8')
