import logging
import json

import pandas as pd
from flaml import AutoML
import more_itertools as m_iter
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import traceback

import joblib
# Monkey patch pandas index
import pandas as pd 
pd.Int64Index = pd.Index
pd.Float64Index = pd.Index

from tqdm import tqdm

from pathlib import Path

from cst_utils import set_seeds
from dataset import load_dataset, get_fold_timeseries, split

RANDOM_STATE = 42
from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH, FULLPRED_PATH, MODELS_PATH

import argparse

import torch
import time

def windowed(ts, window, context, reshape = True):
    windowed_ts = list(m_iter.windowed(ts, n=window, step=1))
    X = pd.DataFrame(windowed_ts).iloc[:, 0:context].to_numpy()
    y = pd.DataFrame(windowed_ts).iloc[:, context:window].to_numpy()

    # useful for NN input shape
    if reshape:
        X = np.expand_dims(X, axis=-1)
    
    return X, y


# python -u flaml_tsf_predict.py 2>&1 | tee logs/flaml_tsf_predict.log

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    set_seeds(RANDOM_STATE)

    # --- experiment ---
    exp_metadata = []

    horizon = WINDOW-CONTEXT

    # --- path init ---
    pred_path = PRED_PATH / "forecasting"
    pred_path.mkdir(exist_ok=True, parents=True)

    # set cpus to available number - 10%
    import os
    n_cpus = os.cpu_count()
    n_cpus = max(1, int(n_cpus * 0.9))


    with open("config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    forecasting_config = config["tasks"]["timeseries-forecasting"]

    logging.info("="*50)
    logging.info(f"Meta application for task {forecasting_config['task_type']} ...")

    for dataset in forecasting_config['datasets']:

        if dataset['dataset_id'] == "WSD_3k":
            continue

        logging.info("-"*50)
        logging.info(f"-> Meta application for dataset {dataset['dataset_id']} ...")

        df = load_dataset(dataset['dataset_id'])
        
        # Transform dt# into datetime
        df = df.sort_values(["ts", "dt#"])
        df["dt#"] = df.groupby("ts").cumcount()
        df["dt#"] = df.apply(
            lambda row: pd.Timestamp("2020-01-01") + pd.to_timedelta(row["dt#"], unit=str(dataset['frequency']).lower()),
            axis=1
        )
        

        logging.info(f"Dataset {dataset['dataset_id']} loaded with shape {df.shape}")
        
        if dataset['dataset_id'] == "WSD_3k":
            df["ts"] = df['ts'].replace("WSD__real-world#", "", regex=True) # fix for WSD_3k dataset ids mismatch among dataset and metrics_df
        
        
        df.set_index(['ts'], inplace=True)

        # iterate across time series of the dataset
        for idx, (ts_id, sub_df) in enumerate(df.groupby("ts")):

            logging.info(f"Starting predict for {dataset['dataset_id']=} {ts_id=}")

            # sort by datetime
            sub_df = sub_df.sort_values(by=["dt#"], ascending=True)

            train_ts, test_ts = split(sub_df, TEST_SIZE)
            val_ts, test_ts = split(test_ts, VAL_SIZE)

            mt = {
                'dataset': dataset['dataset_id'],
                'ts': ts_id,
                'model': 'FLAML'
            }

            model_path = MODELS_PATH / f"flaml_model_AzFuncInvoke_{ts_id}.pkl"
            model = joblib.load(model_path)
            logging.info(f"Loaded model from {model_path}")

            logging.info(f"List of columns in training dataframe: {train_ts.columns}")

            history_df = pd.concat([val_ts[-WINDOW:], test_ts])
            X_test, y_test = windowed(ts=history_df.y.values, context=WINDOW+CONTEXT, window=WINDOW+WINDOW, reshape=False)

            logging.info(f"Shape of windowed testing set {X_test.shape=} {y_test.shape=}")

            best_estimator_obj = model.model  # this is the fitted BaseEstimator  
            best_estimator_name = model.best_estimator  # string name of estimator  
            best_config = model.best_config

            warm_start_config = model.best_config_per_estimator

            
            predictions = []
            start_prediction_time = datetime.now()
            window_idx = 0
            refit_times = []
            pred_times = []
            for X_window in tqdm(X_test, total=len(X_test), desc=f"Predictions for {ts_id}"):
                # Retrain model on this window with previous hyperparameters

                #new_data = pd.concat([train_ts, val_ts, test_ts[:CONTEXT+window_idx]], ignore_index=True)
                new_data = pd.concat([val_ts, test_ts[:CONTEXT+window_idx]], ignore_index=True)
                logging.info(f"Len of new train, val and test {len(train_ts)=} {len(val_ts)=} {len(test_ts[:CONTEXT+window_idx])=}")
                
                _temp_X_df = pd.DataFrame({
                    'dt#': [pd.Timestamp(new_data["dt#"].values[-1]) + pd.to_timedelta(i + 1, unit=str("min").lower()) for i in range(len(X_window))],
                    'y': [None for i in range(len(X_window))],
                    'ts': ts_id,
                }).reset_index()
                
                start_refit_time = datetime.now()
                model.fit(
                    dataframe=new_data,
                    label='y',
                    time_col='dt#',
                    eval_method="holdout",
                    group_ids=None,
                    period=50,
                    time_budget=1,
                    estimator_list=[model.best_estimator],  # reuse the found model type
                    task='ts_forecast',
                    max_iter=1,
                    log_file_name=None,  
                    verbose=3,
                    n_jobs=n_cpus,
                    split_ratio=1.0,
                    retrain_full=False,
                    starting_points = warm_start_config          
                )
                end_refit_time = datetime.now()
                refit_times.append((end_refit_time - start_refit_time).total_seconds() * 1000)

                start_pred_time = datetime.now()
                preds = model.predict(_temp_X_df[:50]).to_numpy()
                end_pred_time = datetime.now()
                pred_times.append((end_pred_time - start_pred_time).total_seconds() * 1000)
                predictions.append(preds)
                window_idx += 1

            mt.update({'refit_time': sum(refit_times)})
            mt.update({'pred_time': sum(pred_times)})

            exp_metadata.append(mt)

            res_df = pd.concat([
                pd.DataFrame(X_test, columns=[f"X_test_{i}" for i in range(X_test.shape[1])]), 
                pd.DataFrame(np.array(predictions), columns=[f"y_pred_{i}" for i in range(horizon)]), 
                pd.DataFrame(y_test, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["dataset"] = dataset['dataset_id']
            res_df["ts"] = ts_id
            res_df["model"] = "FLAML"
            res_df.reset_index(names = "test_window", inplace=True)
            pred_filename = f"dataset_{dataset['dataset_id']}#ts_{ts_id}#model_FLAML.csv"
            res_df.set_index(["dataset", "ts", "model", "test_window"]).to_csv(pred_path / pred_filename)

            logging.info(f"Prediction saved as: {pred_filename}")

            # break

        save_in = RES_PATH
        save_in.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(exp_metadata).to_csv(save_in / f"flaml_exp_predict_{dataset['dataset_id']}.csv", index=None)

        
        