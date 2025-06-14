#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle

import polars as pl


def run(year, month, model_file="model.bin"):
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/yellow_tripdata_{year:04d}-{month:02d}.parquet"

    if not os.path.exists("/output"):
        os.makedirs("output")

    with open(model_file, "rb") as f_in:
        dv, lr = pickle.load(f_in)

    def read_data(filename):
        df = pl.scan_parquet(filename)

        month_of_year = f"{year:04d}/{month:02d}"

        df = df.with_row_index(name="index")
        df = df.select(
            duration=(
                (
                    pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60
            ),
            ride_id=pl.concat_str(
                pl.lit(month_of_year),
                pl.col("index").cast(str),
                separator="_",
            ),
            PULocationID=pl.col("PULocationID").cast(pl.String),
            DOLocationID=pl.col("DOLocationID").cast(pl.String),
        )

        df = df.filter(pl.col("duration").is_between(1, 60))

        # df = df.with_columns(
        #     pu_do=pl.concat_str(
        #         [  # pl.concat_str will cast to str automatically
        #             pl.col("PULocationID"),  # .cast(str),
        #             pl.col("DOLocationID"),  # cast(str),
        #         ],
        #         separator="_",
        #     )
        # )

        categorical = ("PULocationID", "DOLocationID")
        target = ("ride_id", "duration")

        df_input = df.select(
            pl.col(categorical),
            # pl.col(numerical),
        ).collect()  # to_dicts()

        df_target = df.select(pl.col(target)).collect()

        return df_input, df_target

    # df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    # return df

    # In[7]:

    df_input, df_target = read_data(input_file)
    dicts = df_input.to_dicts()

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"Q5: Mean prediction: {y_pred.mean()}")

    df_result = df_target.with_columns(
        pl.col("ride_id"),
        pl.Series("predicted_duration", y_pred),
    )

    df_result.write_parquet(
        output_file,
        compression=None,
        # row_group_size=1000000,
        use_pyarrow=True,
        compression_level=None,
        # file_options={"version": "2.6"},
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        model_file = sys.argv[3]
    else:
        year = 2023
        month = 3
        model_file = "model.bin"
    run(year, month, model_file)
