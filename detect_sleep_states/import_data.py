import pandas as pd
import numpy as np
import gc

from joblib import Parallel, delayed
import multiprocessing

import datetime

from tqdm import tqdm
tqdm.pandas()

######################################### Data Import #######################################################################################################

def add_mapping(mappings, df: pd.DataFrame, name_col: str):
    unique_mappings = pd.DataFrame({name_col: sorted(df[name_col].unique())})
    mappings[name_col] = unique_mappings.reset_index().set_index(name_col)['index'].to_dict()

def map_and_convert_type(data: pd.Series, mapping_dict: dict, dtype: str):
    return data.map(mapping_dict).astype(dtype)

def extract_timestamp_features(df: pd.DataFrame, timestamp_col: str = 'timestamp', drop_timestamp_col: bool = True):
    df['seconds']=df[timestamp_col].str[11:13].astype('uint32') * 60 * 60 + df[timestamp_col].str[14:16].astype('uint32') * 60 + df[timestamp_col].str[17:19].astype('uint32')
    df['day']=df[timestamp_col].str[8:10].astype('uint8')
    df['month']=df[timestamp_col].str[5:7].astype('uint8')

    df['year']=df[timestamp_col].str[:4].astype('uint16')

    if drop_timestamp_col:
        # drop timestamp and free memory
        df.drop(timestamp_col, axis=1, inplace=True)
        gc.collect()

    def weekday_parallel(chunk):
        tqdm.pandas()
        return chunk[['year', 'month', 'day']].progress_apply(lambda x:  datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)
    df['weekday'] = pd.concat(Parallel(n_jobs=-1)(delayed(weekday_parallel)(chunk) for chunk in np.array_split(df[['year', 'month', 'day']], multiprocessing.cpu_count() - 1))).astype('uint8')    

def import_data(train_series_path: str = '../data/train_series.parquet', train_events_path: str = '../data/train_events.csv'):

    train_series = pd.read_parquet(train_series_path)
    train_events = pd.read_csv(train_events_path)

    # mappings storage
    mappings = {}

    ######### Transform train_series data to memory efficient representation #########
    # series_id mapping
    add_mapping(mappings, train_series, 'series_id')
    train_series['series_id'] = map_and_convert_type(train_series['series_id'], mappings['series_id'], 'uint16')

    train_series['step'] = train_series['step'].astype('uint32')

    extract_timestamp_features(train_series)

    add_mapping(mappings, train_series, 'year')
    train_series['year'] = map_and_convert_type(train_series['year'], mappings['year'], 'uint8')

    ######### Transform train_events data to memory efficient representation #########
    train_events = train_events.dropna()
    train_events['night'] = train_events['night'].astype('uint16')
    train_events['series_id'] = map_and_convert_type(train_events['series_id'], mappings['series_id'], 'uint16')
    add_mapping(mappings, train_events, 'event')
    train_events['event'] = map_and_convert_type(train_events['event'], mappings['event'], 'uint8')
    train_events['step'] = train_events['step'].astype('uint32')

    extract_timestamp_features(train_events)

    return train_series, train_events, mappings

##################################################################################################################
