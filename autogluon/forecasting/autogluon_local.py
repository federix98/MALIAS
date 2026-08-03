
import os

# # Safely remove environment variables if they exist
# for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
#     os.environ.pop(var, None)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import numpy as np
import pandas as pd
import argparse

from pathlib import Path

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import split, windowed
from datetime import datetime

from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH, FULLPRED_PATH
from cst_utils import set_seeds

set_seeds()

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

# python -u forecasting/autogluon_local.py --dataset-path data/azure10min.csv 2>&1 | tee autogluon_forecasting.log

if __name__ == "__main__":

    # --- args ---
    parser = argparse.ArgumentParser(description="AutoGluon for time series forecasting experiment")
    parser.add_argument('--dataset-path', default='dataset.csv', type=str, required=True)
    args = parser.parse_args()

    df_path = Path(args.dataset_path)
    dataset_name = df_path.stem
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    # --- path init ---
    pred_path = PRED_PATH / "automl"
    pred_path.mkdir(exist_ok=True, parents=True)
    full_pred_path = Path(FULLPRED_PATH)
    full_pred_path.mkdir(exist_ok=True, parents=True)

    # --- experiment ---
    exp_metadata = []

    (window, context) = (WINDOW, CONTEXT)
    horizon = window - context

    for idx, (ts_id, sub_df) in enumerate(df.groupby("ts")):

        # if idx >= 5:
        #     break

        logging.info(f"Analysis of {dataset_name=} {ts_id=}")

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values

        processed = proc_data(ts, describe=True)

        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]
        ts_context = np.concatenate((ts_train, ts_eval))
        logging.info(f"{ts_context.shape=}")


        mt = {
            'dataset': dataset_name,
            'ts': ts_id,
            'model': 'AutoGluon'
        }

        model = TimeSeriesPredictor(
            prediction_length=horizon,
            path=f"autogluon-logs/autogluon-{ts_id}",
            target="y",
            eval_metric="SMAPE",
            verbosity=4
        )

        # Convert numpy array to pandas DataFrame
        ts_context_df = pd.DataFrame({
            "ts": ts_id,  # identifier for the series (can be multiple)
            "dt#": pd.date_range("2020-01-01", periods=len(ts_context), freq="10min"),
            "y": ts_context
        })

        # Convert to AutoGluon's TimeSeriesDataFrame
        ts_context_df = TimeSeriesDataFrame.from_data_frame(
            ts_context_df,
            id_column="ts",
            timestamp_column="dt#"
        )
        

        test_len = len(ts_test)
        start_time = pd.Timestamp("2021-01-01 00:00:00")  # choose an appropriate start
        time_index = pd.date_range(start=start_time, periods=test_len, freq="10min")

        # Build pandas DataFrame in the format required
        ts_test_df = pd.DataFrame({
            "ts": ts_id,       # if only one series, constant ID
            "dt#": time_index,
            "y": ts_test
        })

        X_test, y_test = context_horizon_split(ts_test_df, context_length=context, horizon_length=horizon, step=1)

        windowed_test = TimeSeriesDataFrame.from_data_frame(
            X_test,
            id_column="window",
            timestamp_column="dt#"
        )

        logging.info(f"Experiment with AutoGluon")

        logging.info(f"Start fitting for AutoGluon")
        start_fitting_time = datetime.now()
        model.fit(
            ts_context_df,
            presets="medium_quality", # Can significantly impact predictive accuracy, memory footprint, inference latency of trained models, and various other properties of the returned predictor.
            time_limit=None, # Approximately how long fit() will run (wall-clock time in seconds). If not specified, fit() will run until all models have completed training.
        )
        end_fitting_time = datetime.now()
        logging.info(f"End fitting for AutoGluon")
        fitting_time = end_fitting_time - start_fitting_time

        logging.info(f"Start zero-shot predict for AutoGluon")
        start_prediction_time = datetime.now()
        y_pred = model.predict(windowed_test)
        end_prediction_time = datetime.now()
        prediction_time = end_prediction_time - start_prediction_time
        logging.info(f"End predict for AutoGluon")

        model.leaderboard().to_csv(f"autogluon-logs/AutoGluonLeaderboard_dataset#{dataset_name}#ts_{ts_id}.csv")

        # logging.info(f"Predictions {y_pred}")
        
        # dump full predictions
        y_pred = y_pred.assign(dataset=dataset_name, ts=ts_id, model="AutoGluon")
        y_pred.to_csv(full_pred_path / f"dataset_{dataset_name}#ts_{ts_id}#model_AutoGluon.csv")

        # With AutoGluon the predictions are in long format and they include quantiles.
        # To be coherent with the other experiments, it is needed to reshape the dataframe in a wide format

        y_pred = y_pred.reset_index()[['dataset', 'model', 'ts', 'item_id', 'timestamp', 'mean']].rename(columns={'mean': 'y', 'item_id': 'window'})
        y_pred['window'] = y_pred['window'].str.split("_").str[-1].astype(int)
        y_pred.sort_values(["dataset", "ts", "model", "window", "timestamp"], inplace=True)
        y_pred["step"] = y_pred.groupby(["dataset", "ts", "model", "window"]).cumcount()
        wide_predictions = pd.DataFrame(y_pred).pivot_table(
            index=["dataset", "ts", "model", "window"],
            columns="step",
            values="y"
        ).reset_index().rename(columns={"window": "test_window"}).rename(columns={i: f"y_pred_{i}" for i in range(50)})

        # print(wide_predictions)

        y_test = y_test.assign(dataset='azure10min', ts=ts_id, model="AutoGluon")[['dataset', 'ts', 'window', 'dt#', 'y']]
        y_test['window'] = y_test['window'].str.split("_").str[-1].astype(int)
        y_test = y_test.sort_values(["dataset", "ts", "window", "dt#"])
        y_test["step"] = y_test.groupby(["dataset", "ts", "window"]).cumcount()
        wide_test = y_test.pivot_table(
            index=["dataset", "ts", "window"],
            columns="step",
            values="y"
        ).reset_index().rename(columns={"window": "test_window"}).rename(columns={i: f"y_{i}" for i in range(50)})

        X_test = X_test.assign(dataset='azure10min', ts=ts_id, model="AutoGluon")[['dataset', 'ts', 'window', 'dt#', 'y']]
        X_test['window'] = X_test['window'].str.split("_").str[-1].astype(int)
        X_test = X_test.sort_values(["dataset", "ts", "window", "dt#"])
        X_test["step"] = X_test.groupby(["dataset", "ts", "window"]).cumcount()
        wide_x_test = X_test.pivot_table(
            index=["dataset", "ts", "window"],
            columns="step",
            values="y"
        ).reset_index().rename(columns={"window": "test_window"}).rename(columns={i: f"X_test_{i}" for i in range(50)})

        formatted_predictions = pd.merge(wide_x_test, wide_test, on=['dataset', 'ts', 'test_window'])
        formatted_predictions = pd.merge(formatted_predictions, wide_predictions, on=['dataset', 'ts', 'test_window'])
        formatted_predictions.to_csv(pred_path / f"dataset_{dataset_name}#ts_{ts_id}#model_AutoGluon.csv", index=None)

        exp_metadata.append(mt)

    save_in = RES_PATH
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"autogluon_exp_{dataset_name}.csv", index=None)