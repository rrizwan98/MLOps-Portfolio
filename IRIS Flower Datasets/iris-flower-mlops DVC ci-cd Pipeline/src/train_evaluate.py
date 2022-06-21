import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow

def eval_metrics(actual, pred):
    rmse = metrics.mean_absolute_error(actual, pred)
    mae = metrics.mean_squared_error(actual, pred)
    accuracy = metrics.accuracy_score(actual, pred)
    return rmse, mae, accuracy

def train_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        rf = RandomForestClassifier(
            n_estimators = n_estimators)
        rf.fit(train_x, train_y)

        predicted_qualities = rf.predict(test_x)
        
        (rmse, mae, accuracy) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
            rf, 
            "model", 
            registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(rf, "model")

    ###################### reports & scores ###############################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "accuracy": accuracy
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
        }
        json.dump(params, f, indent=4)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_evaluate(config_path=parsed_args.config)

