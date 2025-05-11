# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars==1.29.0",
#     "scikit-learn"
# ]
# ///

import polars as pl

df = pl.read_parquet('./data/yellow_tripdata_2023-01.parquet')

# Read the data for January. How many columns are there?
print(df.shape[1])

# What's the standard deviation of the trips duration in January?
df = df.with_columns(
    duration=(
        (pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60
    )
)

result = df.select(
    std_duration=pl.std('duration')
)

print(result.item())

# What fraction of the records left after you dropped the outliers?
rows_before = df.height
df = df.filter(
    pl.col('duration').is_between(1, 60)
)

result = df.height / rows_before
print(result)

# What's the dimensionality of this matrix (number of columns)?
train_loc_dicts = df.select(
    pl.col('PULocationID').cast(str),
    pl.col("DOLocationID").cast(str),
).to_dicts()

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=True, ) #dtype=int
X_train = vec.fit_transform(train_loc_dicts)
print(X_train.shape[1])

# What's the RMSE on train?
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

y_train = df.select(pl.col('duration')).to_numpy()

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_train)
print(root_mean_squared_error(y_train, y_pred))


# What's the RMSE on validation?
df_feb = pl.read_parquet('./data/yellow_tripdata_2023-02.parquet')
df_feb = df_feb.with_columns(
    duration=(
        (pl.col('tpep_dropoff_datetime') - pl.col('tpep_pickup_datetime')).dt.total_seconds() / 60
    )
)

df_feb = df_feb.filter(
    pl.col('duration').is_between(1, 60)
)

test_loc_dicts = df_feb.select(
    pl.col('PULocationID').cast(str),
    pl.col("DOLocationID").cast(str),
).to_dicts()

X_test = vec.transform(test_loc_dicts)
y_test = df_feb.select(pl.col('duration')).to_numpy()

y_pred2 = linreg.predict(X_test)
print(root_mean_squared_error(y_test, y_pred2))
