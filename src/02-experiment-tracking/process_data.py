import os
import pickle
import click
import polars as pl

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pl.scan_parquet(filename)

    df = df.with_columns(
        duration=(
            (pl.col('lpep_dropoff_datetime') - pl.col('lpep_pickup_datetime')).dt.total_seconds() / 60
        )
    )

    df = df.filter(
        pl.col('duration').is_between(1, 60)
    )

    df = df.with_columns(
        pu_do=pl.concat_str(
            [ # pl.concat_str will cast to str automatically
                pl.col('PULocationID'),#.cast(str),
                pl.col('DOLocationID'),#cast(str),
            ],
            separator='_'
        )        
    )

    categorical = ('pu_do',)
    numerical = ('trip_distance',)
    target = 'duration'

    df_input = df.select(
        pl.col(categorical),
        pl.col(numerical),
    ).collect() #to_dicts()

    df_target = df.select(
        pl.col(target)
    ).collect()

    return df_input, df_target


def preprocess(df: pl.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dicts()
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    # Load parquet files
    df_x_train, df_y_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet")
    )
    # print(df_x_train)
    # print(df_y_train)
    df_x_val, df_y_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet")
    )
    df_x_test, df_y_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
    )

    # Transform the target to numpy array
    y_train = df_y_train.to_numpy()
    y_val = df_y_val.to_numpy()
    y_test = df_y_test.to_numpy()

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_x_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_x_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_x_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()