
from multiprocessing import context
import os
from tracemalloc import start

# # Safely remove environment variables if they exist
# for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
#     os.environ.pop(var, None)

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularDataset, TabularPredictor

import numpy as np
import pandas as pd
import argparse

from pathlib import Path

import more_itertools as m_iter

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import json

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

def anomaly_label_window(labels, window_size):
    n = len(labels)
    n_windows = n // window_size
    if n % window_size != 0:
        n_windows += 1

    windowed_labels = []
    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n)
        window = labels[start:end]
        # If any point in the window is anomalous (label == 1), label the whole window as anomalous
        windowed_label = 1 if np.any(window == 1) else 0
        windowed_labels.append(windowed_label)

    return np.array(windowed_labels)

def load_ad_dataset(dataset):
    ds_path = Path(f"data/{dataset.upper()}")
    dfs = []
    for ts_path in ds_path.glob("*"):
        logging.info(f"[{dataset.upper()} processing...] loading time series from {ts_path}")
        # read training set
        training_series = np.load(ts_path / "train.npy")
        training_labels = np.load(ts_path / "train_label.npy")
        dfs.append(
            pd.DataFrame({
                'ts': ts_path.name,
                'dt#': np.arange(len(training_series)), 
                'y': training_series,
                'label': training_labels,
                'split': 'train'
            })
        )

        testing_series = np.load(ts_path / "test.npy")
        testing_labels = np.load(ts_path / "test_label.npy")
        dfs.append(
            pd.DataFrame({
                'ts': ts_path.name,
                'dt#': np.arange(len(testing_series)), 
                'y': testing_series,
                'label': testing_labels,
                'split': 'test'
            })
        )
    assert dfs, f"No time series found in {ds_path}"
    return pd.concat(dfs)

# python -u classification/autogluon_local.py --task anomaly-detection 2>&1 | tee classification/logs/autogluon_local.log

if __name__ == "__main__":
    
    # --- args ---
    parser = argparse.ArgumentParser(description="Test autogluon (local) on anomaly-detection task")
    parser.add_argument('--task', default='anomaly-detection', type=str, required=True)
    args = parser.parse_args()

    # set cpus to available number - 10%
    n_cpus = os.cpu_count()
    n_cpus = max(1, int(n_cpus * 0.9))

    # --- path init ---
    pred_path = PRED_PATH / "automl"
    pred_path.mkdir(exist_ok=True, parents=True)
    full_pred_path = Path(FULLPRED_PATH)
    full_pred_path.mkdir(exist_ok=True, parents=True)

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

            df = load_ad_dataset(dataset['dataset_id'])
            logging.info(f"Loaded dataset with shape {df.shape} and columns {df.columns.tolist()}")

            # check for nans or infs
            if df.isnull().values.any() or np.isinf(df.select_dtypes(include=[np.number])).values.any():
                raise ValueError(f"Dataset {dataset['dataset_id']} contains NaNs or Infs. Please clean the data before proceeding.")
            logging.info("Dataset passed NaN/Inf check.")

            for idx, (ts_id, sub_df) in enumerate(df.groupby("ts")):

                mt = {
                    'dataset': dataset['dataset_id'],
                    'ts': ts_id,
                    'model': 'AutoGluon'
                }

                logging.info(f"Analysis of {dataset['dataset_id']} {ts_id=}. Total length: {len(sub_df)}")

                train_df, test_df = sub_df[sub_df['split']=='train'], sub_df[sub_df['split']=='test']
                logging.info(f"Train/test shapes: {train_df.shape}/{test_df.shape}")

                train_df = train_df.drop(columns=['ts', 'split'])
                test_df = test_df.drop(columns=['ts', 'split'])

                # check if test_df has two labels
                classification_labels_test = test_df['label'].unique()
                if len(classification_labels_test) == 2:
                    logging.info(f"Two labels detected: {classification_labels_test}. Processing time series.")
                else:
                    logging.info(f"Test set does not contain both the labels: {classification_labels_test}. Skipping time series.")
                    continue

                # perform windowing of the time series and label the window as anomalous if any point in the window is anomalous
                window_size = 100

                X_train = pd.DataFrame(list(m_iter.windowed(train_df['y'].values, n=window_size, step=1)))
                X_test = pd.DataFrame(list(m_iter.windowed(test_df['y'].values, n=window_size, step=1)))

                # generate windowed labels
                windowed_labels_train = pd.DataFrame(list(m_iter.windowed(train_df['label'].values, n=window_size, step=1)))
                windowed_labels_test = pd.DataFrame(list(m_iter.windowed(test_df['label'].values, n=window_size, step=1)))
                
                # generate one label for each window (1 if any point in the window is anomalous, else 0)
                # iterate over each row and apply the logic
                y_train = windowed_labels_train.apply(lambda row: 1 if np.any(row == 1) else 0, axis=1).values
                y_test = windowed_labels_test.apply(lambda row: 1 if np.any(row == 1) else 0, axis=1).values

                logging.info(f"Windowed train/test shapes: {X_train.shape}/{X_test.shape}")
                logging.info(f"Windowed train/test labels shapes: {y_train.shape}/{y_test.shape}")

                X_train['label'] = y_train
                X_test['label'] = y_test

                # assign column names
                X_train.columns = [f'x{i}' for i in range(X_train.shape[1]-1)] + ['label']
                X_test.columns = [f'x{i}' for i in range(X_test.shape[1]-1)] + ['label']

                model = TabularPredictor(
                    label='label',
                    path=f"autogluon-logs/{task['task_type']}/autogluon--{dataset['dataset_id']}--{ts_id}", 
                    problem_type='binary', 
                    # eval_metric='roc_auc', 
                    verbosity=2
                )
                
                start_time = datetime.now()
                logging.info(f"Training started at {start_time} ...")

                model.fit(
                    train_data = X_train,
                    time_limit = None, # unlimited time
                    presets = 'medium_quality',
                    num_cpus=n_cpus,     # limit CPU cores
                    num_gpus=0,     # or more, if you have GPUs
                    verbosity=3,
                )
                end_time = datetime.now()
                fitting_time = end_time - start_time
                logging.info(f"Training ended at {end_time}, duration: {fitting_time}")

                logging.info("Model training completed.")

                proba_df = model.predict_proba(X_test.drop(columns=['label']))
                y_pred = proba_df[1].values  # probability of the positive class
                y_pred_labels = (y_pred >= 0.5).astype(int)  # threshold at 0.5

                logging.info(f"Predicted labels: {y_pred_labels}")
                logging.info(f"True labels: {y_test}")

                # dump predictions with ts_id
                save_pred_to = Path(PRED_PATH) / task['task_type']
                save_pred_to.mkdir(exist_ok=True, parents=True)
                res_path = save_pred_to / f"autogluon-{dataset['dataset_id']}-{ts_id}.csv"
                res_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({ 
                    'y_true': y_test,
                    'y_pred_proba': y_pred,
                    'y_pred': y_pred_labels,
                    'ts': ts_id
                }).to_csv(res_path, index=False)
                logging.info(f"Predictions saved to {res_path}")

            # only the first dataset
            break