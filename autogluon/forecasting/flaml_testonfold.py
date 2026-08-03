import logging
import json

import pandas as pd
from flaml import AutoML
import more_itertools as m_iter
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime

import traceback

# Monkey patch pandas index
import pandas as pd 
pd.Int64Index = pd.Index
pd.Float64Index = pd.Index

from pathlib import Path

from cst_utils import set_seeds
from dataset import load_dataset, get_fold_timeseries, split

RANDOM_STATE = 42
from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH, FULLPRED_PATH

import argparse

from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
import torch
import time

# Initialize once (to avoid re-instantiation per call)
_smape_metric = SymmetricMeanAbsolutePercentageError()


# python -u forecasting/flaml_testonfold.py --task timeseries-forecasting 2>&1 | tee forecasting/logs/forecasting_flaml_testonfold.log

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    set_seeds(RANDOM_STATE)

    # --- args ---
    parser = argparse.ArgumentParser(description="Test FLAML on a single fold")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    # --- experiment ---
    exp_metadata = []

    (window, context) = (WINDOW, CONTEXT)
    horizon = window - context

    # --- path init ---
    pred_path = PRED_PATH / "automl"
    pred_path.mkdir(exist_ok=True, parents=True)
    full_pred_path = Path(FULLPRED_PATH)
    full_pred_path.mkdir(exist_ok=True, parents=True)

    # set cpus to available number - 10%
    import os
    n_cpus = os.cpu_count()
    n_cpus = max(1, int(n_cpus * 0.9))


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

            # iterate across time series
            for idx, (ts_id, sub_df) in enumerate(df.groupby("ts")):

                if idx >= 5:
                    break

                logging.info(f"Analysis of {dataset['dataset_id']=} {ts_id=}")

                # sort by datetime
                sub_df = sub_df.sort_values(by=["dt#"], ascending=True)

                train_ts, test_ts = split(sub_df, TEST_SIZE)

                mt = {
                    'dataset': ts_id,
                    'ts': ts_id,
                    'model': 'FLAML'
                }

                # --- FLAML ---
                model = AutoML()

                automl_settings = {
                    "time_budget": 10**9,  # total running time in seconds
                    "metric": 'mape',  # primary metrics can be chosen from: ['mape', 'smape', 'mae', 'rmse', 'r2']
                    "task": 'ts_forecast',
                    "period": horizon,  # frequency of time series
                    "eval_method": 'auto',
                    'early_stop': True,
                    'log_type': 'all',
                    'n_jobs': n_cpus,
                    "log_file_name": f"forecasting/logs/flaml_{dataset['dataset_id']}_{ts_id}.log",
                }

                logging.info("Columns in the training data: {}".format(train_ts.columns.tolist()))
                logging.info("Starting FLAML training ...")
                start_fitting_time = datetime.now()
                try:
                    model.fit(
                        dataframe=train_ts.reset_index(),
                        label='y',
                        time_col='dt#',
                        group_ids=None,
                        **automl_settings
                    )
                except Exception as e:
                    logging.error("FLAML training failed with exception:")
                    logging.error(traceback.format_exc())
                    continue
                end_fitting_time = datetime.now()
                logging.info(f"FLAML training time: {end_fitting_time - start_fitting_time}")
                logging.info("FLAML training completed.")

                # dump the predictor
                model_path = RES_PATH / f"flaml_model_{dataset['dataset_id']}_{ts_id}.pkl"
                model_path.parent.mkdir(exist_ok=True, parents=True)
                model.save_best_model(model_path)
                logging.info(f"FLAML model saved to {model_path}")

                # append training time to metadata
                mt['training_time'] = (end_fitting_time - start_fitting_time).total_seconds()

                exp_metadata.append(mt)

            
            save_in = RES_PATH
            save_in.mkdir(exist_ok=True, parents=True)
            pd.DataFrame(exp_metadata).to_csv(save_in / f"flaml_exp_{dataset['dataset_id']}.csv", index=None)

            break