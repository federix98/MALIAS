
import os

# Safely remove environment variables if they exist
for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.pop(var, None)

import socket

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

from forecasting_models import Chronos, TFM

from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH
from cst_utils import set_seeds

set_seeds()

def create_models(window, context):
    chronos_model = Chronos(window=window, context=context, max_context=512)
    timesfm_model = TFM(window=window, context=context, frequency=0, max_context=512)

    return chronos_model, timesfm_model

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

if __name__ == "__main__":

    # --- args ---
    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--dataset-path', default='dataset.csv', type=str, required=True)
    args = parser.parse_args()

    df_path = Path(args.dataset_path)
    dataset_name = df_path.stem
    logging.info(f"Dataset loading from {df_path}")
    df = pd.read_csv(df_path)
    logging.info(f"Dataset loaded.")

    exp_metadata = []

    (window, context) = (WINDOW, CONTEXT)
    horizon = window - context
    models = create_models(window = window, context = context)

    for (ts_id), sub_df in df.groupby("ts"):

        logging.info(f"Analysis of {dataset_name=} {ts_id=}")

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values

        processed = proc_data(ts, describe=True)

        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]
        ts_context = np.concatenate((ts_train, ts_eval))
        logging.info(f"{ts_context.shape=}")

        X_test, y_test = windowed(ts_test, window=window, context=context, reshape=False)

        # experiment using models
        for model in models:

            mt = {
                'dataset': dataset_name,
                'ts': ts_id,
                'model': model.get_name()
            }

            logging.info(f"Experiment with {model.get_name()}")

            logging.info(f"Start zero-shot predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict(ts_context, ts_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")

            logging.info(f"Predictions {y_pred}")
            
            # dump predictions
            pred_path = PRED_PATH / "pretrained"
            pred_path.mkdir(exist_ok=True, parents=True)
            res_df = pd.concat([pd.DataFrame(X_test, columns=[f"X_test_{i}" for i in range(context)]), pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(horizon)]), pd.DataFrame(y_test, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["dataset"] = dataset_name
            res_df["ts"] = ts_id
            res_df["model"] = model.get_name()
            res_df.reset_index(names = "test_window", inplace=True)
            res_df.set_index(["dataset", "ts", "model", "test_window"]).to_csv(pred_path / f"dataset_{dataset_name}#ts_{ts_id}#model_{model.get_name()}.csv")

            mt.update({'fit_time': None, 'pred_time': prediction_time.total_seconds() * 1000})
            exp_metadata.append(mt)

    save_in = RES_PATH
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"pretrained_exp_{dataset_name}.csv", index=None)