import os
import yaml
import pickle
import click
import mlflow
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_pickle(filename: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./data/middle",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--config_file",
    default="src/02-experiment-tracking/rfr_config.yaml",
    help="Path to config file. It should containg MLFlow metadata, tags and hyperparameters"
)
def run_train(data_path: str, config_file: str):
    config = load_config(config_file)
    # mlflow.create_experiment(con)
    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment']['name'])
    mlflow.sklearn.autolog(extra_tags=config['model']['tags'])

    with mlflow.start_run() as this_run:
        # mlflow.set_tags(config['tags'])
        # mlflow.log_params(config['hyp_params'])

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(**config['model']['hyp_params'])
        rf.fit(X_train, y_train)

        metrics = {
            # 'validation_': root_mean_squared_error(y_train, rf.predict(X_train)),
            'validation_root_mean_squared_error': root_mean_squared_error(y_val, rf.predict(X_val))
        }

        mlflow.log_metrics(metrics, synchronous=True)
    
    last_run = mlflow.get_run(this_run.info.run_id)
    print(last_run.data.metrics, last_run.data.params)


if __name__ == '__main__':
    run_train()