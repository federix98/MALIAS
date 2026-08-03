import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from pathlib import Path

import logging
import argparse

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import windowed, proc_data
from datetime import datetime

from forecasting_models import SNAIVE, SMM, SARIMA, ProphetWrapper, ETS

from ts_utils import get_sarima_params

from cfg import WINDOW, CONTEXT, PRED_PATH, RES_PATH
from cst_utils import set_seeds

set_seeds()

def create_models(window, context, seasonality):
    # baselines
    snaive_model = SNAIVE(window = window, context = context)
    smm_model = SMM(window = window, context = context, seasonality = seasonality, context_perdiods = 4)

    return snaive_model, smm_model

def create_statistical_models(dataset_name, ts_id, window, context, seasonality):

    # statistical models
    # prophet_model = ProphetWrapper(window=window, context=context, seasonality = seasonality)
    
    if seasonality > 1:
        ets_model = ETS(seasonality = seasonality, window = window, context = context)
    else:
        ets_model = ETS(seasonality = None, window = window, context = context)

    best_sarima_params = get_sarima_params(dataset_name=dataset_name, ts_id=ts_id)
    sarima_model = SARIMA(
        seasonality = seasonality, 
        window = window, 
        context = context, 
        p = best_sarima_params['p'], 
        d = best_sarima_params["d"], 
        q = best_sarima_params["q"], 
        P = best_sarima_params["P"], 
        D = best_sarima_params["D"], 
        Q = best_sarima_params["Q"]
    )

    # return ets_model, sarima_model
    return sarima_model,

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

        if "AZ" in dataset_name.upper():
            seasonality = 6 # Set hourly seasonality for azure
        print("Using seasonality", seasonality)

        (window, context) = (WINDOW, CONTEXT)
        horizon = window - context

        models = create_statistical_models(dataset_name = dataset_name, ts_id = ts_id, window = window, context = context, seasonality = seasonality) + create_models(window = window, context = context, seasonality = seasonality)

        processed = proc_data(ts, describe=True)

        ts_train, ts_eval, ts_test = processed["ts"][0], processed["ts"][1], processed["ts"][2]

        ts_context = np.concatenate((ts_train, ts_eval))
        logging.info(f"{ts_context.shape=}")

        X_test, y_test = windowed(ts_test, window=window, context=context, reshape=False)

        for model in models:

            mt = {
                'dataset': dataset_name,
                'ts': ts_id,
                'model': model.get_name()
            }

            logging.info(f"Experiment with {model.get_name()}")

            model.fit(ts_context)

            logging.info(f"Start predict for {model.get_name()}")
            start_prediction_time = datetime.now()
            y_pred = model.predict(ts_test)
            end_prediction_time = datetime.now()
            prediction_time = end_prediction_time - start_prediction_time
            logging.info(f"End predict for {model.get_name()}")
            
            # dump predictions
            pred_path = PRED_PATH / "stat_exp"
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
    pd.DataFrame(exp_metadata).to_csv(save_in / f"stat_exp_{dataset_name}.csv", index=None)