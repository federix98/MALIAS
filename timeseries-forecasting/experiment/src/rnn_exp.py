import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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

from sklearn.preprocessing import StandardScaler

from dataset import split, windowed
from datetime import datetime

from ts_utils import get_sarima_params

from forecasting_models import RNN

from cfg import TEST_SIZE, VAL_SIZE, WINDOW, CONTEXT, PRED_PATH, RES_PATH
from cst_utils import set_seeds

set_seeds()

def create_models(window, context):
    rnn_model = RNN(model = "RNN", window = window, context = context)
    lstm_model = RNN(model = "LSTM", window = window, context = context)
    gru_model = RNN(model = "GRU", window = window, context = context)

    return rnn_model, lstm_model, gru_model

def proc_data(ts, scaler, window, context, describe = False):

    ts = np.array(ts)
    scaled_ts = scaler.fit_transform(ts.reshape(-1, 1))
    scaled_ts = scaled_ts.reshape(scaled_ts.shape[0])

    train_ts, test_ts = split(scaled_ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    X_train, y_train = windowed(train_ts, window, context)
    X_val, y_val = windowed(val_ts, window, context)
    X_test, y_test = windowed(test_ts, window, context)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")
        logging.info(f"Train splits shape: [X={X_train.shape}, y={y_train.shape}]")
        logging.info(f"Val splits shape: [X={X_val.shape}, y={y_val.shape}]")
        logging.info(f"Test splits shape: [X={X_test.shape}, y={y_test.shape}]")


    return {
        "ts": [train_ts, val_ts, test_ts],
        "X": [X_train, X_val, X_test],
        "y": [y_train, y_val, y_test]
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
    
    for (ts_id), sub_df in df.groupby("ts"):

        logging.info(f"Analysis of {dataset_name=} {ts_id=}")

        ts = sub_df.sort_values(by=["dt#"], ascending=True).y.values
        
        seasonality = get_sarima_params(dataset_name=dataset_name, ts_id=ts_id)["s"]
        if seasonality == 0:
            seasonality = 1

        (window, context) = (WINDOW, CONTEXT)
        horizon = window - context

        ml_models = create_models(window, context)

        scaler = StandardScaler()
        processed = proc_data(ts, scaler, window = window, context = context, describe = True)

        X_train, X_val, X_test = processed["X"][0], processed["X"][1], processed["X"][2]
        y_train, y_val, y_test = processed["y"][0], processed["y"][1], processed["y"][2]

        # experiment using models
        for model in ml_models:

            mt = {
                'dataset': dataset_name,
                'ts': ts_id,
                'model': model.get_name()
            }

            logging.info(f"Experiment with {model.get_name()}")

            # model.load()

            logging.info(f"Start fitting for {model.get_name()}")
            start_fitting_time = datetime.now()
            model.fit(X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val, verbose = 1, save_param=f"dataset_{dataset_name}#ts_{ts_id}")
            end_fitting_time = datetime.now()
            logging.info(f"End fitting for {model.get_name()}")
            fitting_time = end_fitting_time - start_fitting_time

            logging.info(f"Start predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict(X_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")
            y_pred = scaler.inverse_transform(y_pred)

            X_test_rescaled = scaler.inverse_transform(X_test.squeeze())
            y_true_rescaled = scaler.inverse_transform(y_test)
            
            # dump predictions
            pred_path = PRED_PATH / "rnn_exp"
            pred_path.mkdir(exist_ok=True, parents=True)
            res_df = pd.concat([pd.DataFrame(X_test_rescaled, columns=[f"X_test_{i}" for i in range(horizon)]), pd.DataFrame(y_pred, columns=[f"y_pred_{i}" for i in range(horizon)]), pd.DataFrame(y_true_rescaled, columns=[f"y_{i}" for i in range(horizon)])], axis = 1)
            res_df["dataset"] = dataset_name
            res_df["ts"] = ts_id
            res_df["model"] = model.get_name()
            res_df.reset_index(names = "test_window", inplace=True)
            res_df.set_index(["dataset", "ts", "model", "test_window"]).to_csv(pred_path / f"dataset_{dataset_name}#ts_{ts_id}#model_{model.get_name()}.csv")

            mt.update({'fit_time': fitting_time.total_seconds() * 1000, 'pred_time': prediction_time.total_seconds() * 1000})
            exp_metadata.append(mt)

    save_in = RES_PATH
    save_in.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(exp_metadata).to_csv(save_in / f"rnn_exp_{dataset_name}.csv", index=None)