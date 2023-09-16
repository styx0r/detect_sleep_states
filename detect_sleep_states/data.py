import pandas as pd
import numpy as np
import gc

from joblib import Parallel, delayed
import multiprocessing

from datetime import datetime, timedelta, date

from tqdm import tqdm

tqdm.pandas()


######################################### Data Import #######################################################################################################


def add_mapping(mappings, df: pd.DataFrame, name_col: str, expand: bool = False):
    unique_values = sorted(df[name_col].unique())
    if expand:
        unique_values = (
            [min(unique_values) - 1] + unique_values + [max(unique_values) + 1]
        )
    unique_mappings = pd.DataFrame({name_col: unique_values})
    mappings[name_col] = (
        unique_mappings.reset_index().set_index(name_col)["index"].to_dict()
    )
    mappings[f"{name_col}_rev"] = {v: k for k, v in mappings[name_col].items()}


def map_and_convert_type(data: pd.Series, mapping_dict: dict, dtype: str):
    return data.map(mapping_dict).astype(dtype)


def extract_timestamp_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp", drop_timestamp_col: bool = True
):
    df["second"] = (
        df[timestamp_col].str[11:13].astype("uint32") * 60 * 60
        + df[timestamp_col].str[14:16].astype("uint32") * 60
        + df[timestamp_col].str[17:19].astype("uint32")
    )
    df["minute"] = (df["second"] // 60).astype("uint16")
    df["day"] = df[timestamp_col].str[8:10].astype("uint8")
    df["month"] = df[timestamp_col].str[5:7].astype("uint8")

    df["year"] = df[timestamp_col].str[:4].astype("uint16")

    if drop_timestamp_col:
        # drop timestamp and free memory
        df.drop(timestamp_col, axis=1, inplace=True)
        gc.collect()

    def weekday_parallel(chunk):
        tqdm.pandas()
        return chunk[["year", "month", "day"]].progress_apply(
            lambda x: date(x["year"], x["month"], x["day"]).weekday(), axis=1
        )

    df["weekday"] = pd.concat(
        Parallel(n_jobs=-1)(
            delayed(weekday_parallel)(chunk)
            for chunk in np.array_split(
                df[["year", "month", "day"]], multiprocessing.cpu_count() - 1
            )
        )
    ).astype("uint8")


def import_data(
    train_series_path: str = "../data/train_series.parquet",
    train_events_path: str = "../data/train_events.csv",
):
    train_series = pd.read_parquet(train_series_path)
    train_events = pd.read_csv(train_events_path)

    # mappings storage
    mappings = {}

    ######### Transform train_series data to memory efficient representation #########
    # series_id mapping
    add_mapping(mappings, train_series, "series_id")
    train_series["series_id"] = map_and_convert_type(
        train_series["series_id"], mappings["series_id"], "uint16"
    )

    train_series["step"] = train_series["step"].astype("uint32")

    extract_timestamp_features(train_series)

    add_mapping(mappings, train_series, "year", expand=True)
    train_series["year"] = map_and_convert_type(
        train_series["year"], mappings["year"], "uint8"
    )

    ######### Transform train_events data to memory efficient representation #########
    train_events = train_events.dropna()
    train_events["night"] = train_events["night"].astype("int16")
    train_events["series_id"] = map_and_convert_type(
        train_events["series_id"], mappings["series_id"], "uint16"
    )
    add_mapping(mappings, train_events, "event")
    train_events["event"] = map_and_convert_type(
        train_events["event"], mappings["event"], "int8"
    )
    train_events["step"] = train_events["step"].astype("uint32")

    extract_timestamp_features(train_events)

    merged_df = train_series.merge(
        train_events[["series_id", "step", "event", "night"]],
        how="left",
        left_on=["series_id", "step"],
        right_on=["series_id", "step"],
    )
    merged_df[["night", "event"]] = (
        merged_df.groupby("series_id")[["night", "event"]]
        .ffill()
        .fillna({"night": 0, "event": mappings["event"]["wakeup"]})
    )

    merged_df["night"] = merged_df["night"].astype("int16")
    merged_df["event"] = merged_df["event"].astype("int8")

    return merged_df, mappings


#############################################################################################################################################################


######################################### Data Prep #########################################################################################################


# reduce dataset using avg, variance, max and min according to minutes
def reduce_dataset(
    data: pd.DataFrame,
    reduction_grouping: list[str] = [
        "series_id",
        "year",
        "month",
        "day",
        "weekday",
        "minute",
    ],
    drop_cols: list[str] = ["second"],
):
    data_reduced = (
        data.drop(drop_cols, axis=1)
        .groupby(reduction_grouping)
        .agg(
            {
                "enmo": ["mean", "var", "max", "min"],
                "anglez": ["mean", "var", "max", "min"],
            }
        )
    )
    data_reduced.columns = [
        "_".join(col).strip() for col in data_reduced.columns.values
    ]
    data_reduced = data_reduced.reset_index()
    return data_reduced


# harmonize data by 0 padding, always starting from weekday 0
def extend_data_to_length(data: pd.DataFrame, mappings: dict):
    replication_factor = 60 * 24
    non_metric_cols = ["series_id", "year", "month", "day", "weekday", "minute"]

    # fill left size, starting with filling the starting day
    def fill_first_day(data_subset: pd.DataFrame):
        first_minute = data_subset.minute.iloc[0]
        if first_minute > 0:
            dupl_row = data_subset.iloc[[0]].copy()
            dupl_row.loc[:, ~dupl_row.columns.isin(non_metric_cols)] = 0
            data_subset = pd.concat(
                [pd.concat([dupl_row] * first_minute), data_subset], ignore_index=True
            )
            data_subset.loc[data_subset.index[:first_minute], "minute"] = range(
                first_minute
            )
        return data_subset

    data = data.groupby("series_id").apply(fill_first_day).reset_index(drop=True)

    # fill right size, starting with filling the final day
    def fill_last_day(data_subset: pd.DataFrame):
        last_minute = data_subset.minute.iloc[data_subset.shape[0] - 1]
        max_minute = replication_factor - 1
        if last_minute < max_minute:
            dupl_row = data_subset.iloc[[data_subset.shape[0] - 1]].copy()
            dupl_row.loc[:, ~dupl_row.columns.isin(non_metric_cols)] = 0
            df_concat = pd.concat([dupl_row] * (max_minute - last_minute))
            df_concat["minute"] = np.uint16(
                range(
                    df_concat["minute"].min() + 1,
                    df_concat.shape[0] + df_concat["minute"].min() + 1,
                )
            )
            data_subset = pd.concat(
                [data_subset, df_concat],
                ignore_index=True,
            )
        return data_subset

    data = data.groupby("series_id").apply(fill_last_day).reset_index(drop=True)

    # fill in the days from weekday 0
    def date_before_after(date_data: pd.Series, before: bool = True):
        date_data = date_data.astype("uint16")
        date_data["year"] = mappings["year_rev"][date_data["year"]]
        actual_date = datetime(**date_data)
        if before:
            modified_date = actual_date - timedelta(days=1)
        else:
            modified_date = actual_date + timedelta(days=1)
        return pd.Series(
            {
                "year": mappings["year"][modified_date.year],
                "month": modified_date.month,
                "day": modified_date.day,
                "weekday": modified_date.weekday(),
            }
        )

    def add_days(data_subset: pd.DataFrame, days=1, before: bool = True):
        dupl_row = data_subset.iloc[[0 if before else data_subset.shape[0] - 1]].copy()
        dupl_row.loc[:, ~dupl_row.columns.isin(non_metric_cols)] = 0

        actual_day = dupl_row.iloc[0][["year", "month", "day"]]

        new_day_data = pd.concat([dupl_row] * replication_factor * days).reset_index(
            drop=True
        )
        new_day_data.loc[:, "minute"] = np.uint16(
            list(range(replication_factor)) * days
        )

        days_i = list(range(days))
        if before:
            days_i.reverse()
        for i in days_i:
            actual_day = date_before_after(
                actual_day.drop("weekday")
                if "weekday" in actual_day.index
                else actual_day,
                before,
            )
            for col, value in actual_day.items():
                new_day_data.loc[
                    (i * replication_factor) : ((i + 1) * replication_factor - 1),
                    [col],
                ] = value

        data_subset = pd.concat(
            [new_day_data, data_subset] if before else [data_subset, new_day_data],
            ignore_index=True,
        )
        return data_subset

    def fill_to_weekday_0(data_subset: pd.DataFrame):
        while data_subset["weekday"].iloc[0] > 0:
            data_subset = add_days(data_subset, days=data_subset["weekday"].iloc[0])
        return data_subset

    def fill_to_n_days(data_subset: pd.DataFrame, n_days=90):
        days = int(data_subset.shape[0] / replication_factor)
        if days < n_days:
            data_subset = add_days(data_subset, n_days - days, False)
        return data_subset

    data = data.groupby("series_id").apply(fill_to_weekday_0).reset_index(drop=True)

    max_days = int(
        data.groupby("series_id").apply(lambda x: x.shape[0] / replication_factor).max()
    )
    chunks = [
        data[data.series_id == series_id] for series_id in data.series_id.unique()
    ]
    data = pd.concat(
        Parallel(n_jobs=-1)(
            delayed(fill_to_n_days)(chunk, max_days) for chunk in chunks
        )
    )

    return data


#############################################################################################################################################################

if __name__ == "__main__":
    merged_df, mappings = import_data()
