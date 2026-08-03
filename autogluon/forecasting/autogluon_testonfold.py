from kfold import create_kfold
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from datetime import datetime
import logging
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from cst_utils import set_seeds
from dataset import split, get_fold_timeseries
import shutil

from dataset import load_dataset

from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH, FULLPRED_PATH

RANDOM_STATE = 42

def proc_data(ts, describe = False):

    ts = np.array(ts)

    train_ts, test_ts = split(ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")

    return {
        "ts": [train_ts, val_ts, test_ts],
    }

def context_horizon_split(df, context_length, horizon_length, step=1):
    X_windows = []
    y_windows = []

    for item_id, g in df.groupby("ts"):
        g = g.sort_values("dt#").reset_index(drop=True)
        n = len(g)

        win_idx = 0
        # Now the rolling size is context_length + horizon_length
        total_window = context_length + horizon_length

        for start in range(0, n - total_window + 1, step):
            # Full window
            win = g.iloc[start:start + total_window].copy()

            # Split into context and horizon
            context = win.iloc[:context_length].copy()
            horizon = win.iloc[context_length:].copy()

            # Add new item_id for uniqueness
            context["window"] = f"{item_id}_{win_idx}"
            horizon["window"] = f"{item_id}_{win_idx}"

            X_windows.append(context)
            y_windows.append(horizon)

            win_idx += 1

    X_df = pd.concat(X_windows, ignore_index=True)
    y_df = pd.concat(y_windows, ignore_index=True)

    return X_df, y_df

# python -u forecasting/autogluon_testonfold.py --task timeseries-forecasting 2>&1 | tee forecasting/logs/forecasting_autogluon_testonfold.log

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    set_seeds(RANDOM_STATE)

    
    # --- args ---
    parser = argparse.ArgumentParser(description="Test autogluon on a single fold")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open("config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        logging.info("="*50)
        logging.info(f"Meta application for task {task['task_type']} ...")

        for dataset in task['datasets']:

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
                df["ts"] = df['ts'].replace("WSD__real-world#", "", regex=True) # fix for WSD_3k dataset ids mismatch
            df.set_index(['ts'], inplace=True)

            metrics_df = pd.read_csv(dataset['metric_path'], index_col='id')
            metrics_df.index = metrics_df.index.map(str)
            
            train_ids, test_ids = get_fold_timeseries(metrics_df, group_split=dataset['group_split'])

            # number of training time series
            logging.info(f"Number of training time series: {len(train_ids)} out of {len(metrics_df.index.unique())}")

            # filter df to only include train_ids
            train_df = df[df.index.isin(train_ids)]
            logging.info(f"Training dataset shape: {train_df.shape}")

            test_df = df[df.index.isin(test_ids)]
            logging.info(f"Testing dataset shape: {test_df.shape}")

            # --- experiment ---
            exp_metadata = []

            (window, context) = (WINDOW, CONTEXT)
            horizon = window - context
            
            # define TimeSeriesDataFrame with X_train and y_train
            train_data = TimeSeriesDataFrame.from_data_frame(
                train_df.reset_index(),
                id_column="ts",
                timestamp_column="dt#"
            )

            # Train and dump a model using the training data of a single fold
            predictor_path = f"forecasting/global-autogluon-logs/autogluon-{dataset['dataset_id']}-fold0"

            # model = TimeSeriesPredictor(
            #     prediction_length=horizon,
            #     path=predictor_path,
            #     target="y",
            #     eval_metric="SMAPE",
            #     verbosity=4,
            #     freq=dataset['frequency']
            # )

            # # check nans
            # assert train_data.isna().sum().sum() == 0, "DataFrame contains NaNs"

            # # dump train_data to csv for inspection
            # train_data.to_data_frame().to_csv(f"forecasting_train_data_{dataset['dataset_id']}_fold0.csv", index=True)

            # logging.info(f"Start fitting for AutoGluon and dataset {dataset['dataset_id']} ...")
            # logging.info(f"AutoGluon parameters: freq={dataset['frequency']}, prediction_length={horizon}, target='y', eval_metric='SMAPE', verbosity=2, presets='medium_quality', time_limit=None")
            # start_fitting_time = datetime.now()
            # model.fit(
            #     train_data,
            #     presets="medium_quality", # Can significantly impact predictive accuracy, memory footprint, inference latency of trained models, and various other properties of the returned predictor.
            #     time_limit=None, # Approximately how long fit() will run (wall-clock time in seconds). If not specified, fit() will run until all models have completed training.
            #     refit_full=True  # refit the model on the full data (including the last validation window)
            # )
            # end_fitting_time = datetime.now()
            # logging.info(f"End fitting for AutoGluon and dataset {dataset['dataset_id']}")

            # fitting_time = end_fitting_time - start_fitting_time
            # logging.info(f"Fitting time: {fitting_time}")

            # # the predictor is already saved in predictor_path during model.fit()
            # logging.info(f"Predictor saved to {predictor_path}")

            # test on target_ts_test_df testing data
            
            for ts_id in test_ids:
                logging.info(f"Processing test time series {ts_id} ...")
                ts_data = test_df.loc[ts_id].sort_values("dt#")

                target_ts_train_df, target_ts_test_df = split(ts_data, TEST_SIZE)

                target_ts_train_df = TimeSeriesDataFrame.from_data_frame(
                    target_ts_train_df.reset_index(),
                    id_column="ts",
                    timestamp_column="dt#"
                )

                exp_info = {
                    'dataset': dataset['dataset_id'],
                    'ts': ts_id,
                    'model': 'AutoGluon'
                }

                predictor_path_specialized = f"forecasting/global-autogluon-logs/autogluon-{dataset['dataset_id']}-fold0-{ts_id}"
                shutil.copytree(predictor_path, predictor_path_specialized)

                # it's not possible to fit again, only predict. Attempting to fit again will raise an error. Reload the model instead.
                model = TimeSeriesPredictor.load(predictor_path_specialized)
                if model == None:
                    logging.error(f"Failed to load model from {predictor_path_specialized}")
                else:
                    logging.info(f"Successfully loaded model from {predictor_path_specialized}")

                # fit a specialized model for this single time series
                logging.info(f"Start fitting specialized model for time series {ts_id} ...")
                start_fitting_time = datetime.now()
                model.fit(
                    target_ts_train_df,
                    presets="medium_quality", # Can significantly impact predictive accuracy, memory footprint, inference latency of trained models, and various other properties of the returned predictor.
                    time_limit=None, # Approximately how long fit() will run (wall-clock time in seconds). If not specified, fit() will run until all models have completed training.
                    refit_full=True  # refit the model on the full data
                )

                end_fitting_time = datetime.now()
                logging.info(f"End fitting specialized model for time series {ts_id}")
                fitting_time = end_fitting_time - start_fitting_time
                logging.info(f"Fitting time for specialized model: {fitting_time}")
                exp_info['fitting_time'] = fitting_time.total_seconds()

                # Cutoff evaluation
                step = 1
                for cutoff in range(-context, 0, step):
                    logging.info(f"Evaluating time series {ts_id} with cutoff {cutoff} ...")

                    # generate predictions using the cutoff context
                    y_pred = model.predict(target_ts_test_df, cutoff=cutoff)
                    y_pred.to_csv(f"forecasting/global-autogluon-logs/autogluon-{dataset['dataset_id']}-fold0-{ts_id}-cutoff{cutoff}-predictions.csv")
                    break
                
                break

            break