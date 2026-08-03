
import os

import numpy as np
import pandas as pd

from pathlib import Path

import more_itertools as m_iter

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import load_ad_dataset
from datetime import datetime

from cfg import PRED_PATH
from cst_utils import set_seeds

from FLAMLWrapper import FLAMLAnomalyDetectionClassifier

set_seeds()

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

# python -u flaml_tsad.py 2>&1 | tee logs/flaml_tsad.log

if __name__ == "__main__":

    # set cpus to available number - 10%
    n_cpus = os.cpu_count()
    n_cpus = max(1, int(n_cpus * 0.9))

    task = 'anomaly-detection'
    dataset_id = 'AIOps'

    logging.info("="*50)
    logging.info(f"Meta application for task {task} ...")

    logging.info("-"*50)
    logging.info(f"-> Meta application for dataset {dataset_id} ...")

    df = load_ad_dataset(dataset_id)
    logging.info(f"Loaded dataset with shape {df.shape} and columns {df.columns.tolist()}")

    df = df.sort_values(["ts", "dt#"])
    df["dt#"] = df.groupby("ts").cumcount()
    df["dt#"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(df["dt#"], unit="s")

    logging.info(f"Datetime dt# column converted to pandas datetime")


    # check for nans or infs
    if df.isnull().values.any() or np.isinf(df.select_dtypes(include=[np.number])).values.any():
        raise ValueError(f"Dataset {dataset_id} contains NaNs or Infs. Please clean the data before proceeding.")
    logging.info("Dataset passed NaN/Inf check.")

    res = []

    for idx, (ts_id, sub_df) in enumerate(df.groupby("ts")):

        mt = {
            'dataset': dataset_id,
            'ts': ts_id,
            'model': 'FLAML'
        }

        logging.info(f"Analysis of {dataset_id} {ts_id=}. Total length: {len(sub_df)}")

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

        X_train, X_test = train_df[["dt#", "y"]], test_df[["dt#", "y"]]
        y_train, y_test = train_df['label'], test_df['label']

        # build FLAMLClassifier
        clf = FLAMLAnomalyDetectionClassifier(ts_id = ts_id)
        
        start_time = datetime.now()
        logging.info(f"Training started at {start_time} ...")
        clf.fit(X_train=X_train, y_train=y_train)
        end_time = datetime.now()
        fitting_time = end_time - start_time
        mt["train_time"] = fitting_time.total_seconds()
        logging.info(f"Training ended at {end_time}, duration: {fitting_time}")

        logging.info("Model training completed.")

        clf.dump(path = f"./results/models/anomaly-detection/FLAML_tsad_AIOps#{ts_id}.pkl")

        print("Best ML leaner:", clf.model.best_estimator)
        print("Best hyperparmeter config:", clf.model.best_config)
        print(f"Best mape on validation data: {clf.model.best_loss}")
        print(f"Training duration of best run: {clf.model.best_config_train_time}s")

        start_time = datetime.now()
        y_pred_labels = clf.predict(X_test=X_test).tolist()
        end_time = datetime.now()
        test_time = end_time - start_time
        print(y_pred_labels)
        

        mt["test_time"] = test_time.total_seconds()

        # dump predictions with ts_id
        save_pred_to = Path(PRED_PATH) / task
        save_pred_to.mkdir(exist_ok=True, parents=True)
        res_path = save_pred_to / f"FLAML-{dataset_id}-{ts_id}.csv"
        res_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ 
            'y_true': y_test,
            'y_pred': y_pred_labels,
            'ts': ts_id
        }).to_csv(res_path, index=False)
        logging.info(f"Predictions saved to {res_path}")

        res.append(mt)

    metrics_path = "results/flaml_tsad_metrics.csv"
    pd.DataFrame(res).to_csv(metrics_path, index=False)
    logging.info(f"Metrics saved to {metrics_path}")