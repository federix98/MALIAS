import os
import joblib
import logging

from flaml import AutoML

from cfg import MODELS_PATH

import pandas as pd
import numpy as np
import more_itertools as m_iter

from tqdm import tqdm

from datetime import datetime

def windowed(ts, window, context, reshape = True):
    windowed_ts = list(m_iter.windowed(ts, n=window, step=1))
    X = pd.DataFrame(windowed_ts).iloc[:, 0:context].to_numpy()
    y = pd.DataFrame(windowed_ts).iloc[:, context:window].to_numpy()

    # useful for NN input shape
    if reshape:
        X = np.expand_dims(X, axis=-1)
    
    return X, y

class FLAMLSteadyStateClassifier:

    def __init__(self, fold : int = None):

        # set cpus to available number - 10%
        self.n_cpus = os.cpu_count()
        self.n_cpus = max(1, int(self.n_cpus * 0.9))

        if fold == None:
            self.model = None
            self.fold = None
            self.settings = None
        else:
            self.fold = fold
            self.model = AutoML()

            self.settings = {
                "time_budget": 33*531,  # This is the budget of VMD dataset
                "task": 'classification',
                "log_file_name": f"logs/flaml_ssd_fold#{self.fold}.log",
                "log_type": "all",
                "n_jobs": self.n_cpus,  # parallel processing
                "seed": 42,  # for reproducibility,
                # "max_iter": 1000
            }

    def fit(self, X_train, y_train):
        logging.info(f"Start {self.__class__.__name__} training")
        logging.info(f"Shapes of training {X_train.shape}, testing {y_train.shape}")
        self.model.fit(X_train=X_train, y_train=y_train, **self.settings)
        logging.info(f"Training {self.__class__.__name__} completed")

    def predict_proba(self, X_test):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        try:
            proba = self.model.predict_proba(X_test)
        except AttributeError:
            raise NotImplementedError("Underlying model does not support predict_proba.")
        
        return proba

    def dump(self, path : str):
        joblib.dump(self.model, path)
        logging.info(f"{self.__class__.__name__} stored: {path}")

    def load(self, path : str, fold : int):
        self.fold = fold
        self.model = joblib.load(path)

        self.settings = {
            "time_budget": 1733,  # This is the budget of AIOps dataset
            "task": 'classification',
            "log_file_name": f"logs/flaml_tsad_{self.fold}.log",
            "log_type": "all",
            "n_jobs": self.n_cpus,  # parallel processing
            "seed": 42,  # for reproducibility
        }


class FLAMLAnomalyDetectionClassifier:

    def __init__(self, ts_id : str = None):

        # set cpus to available number - 10%
        self.n_cpus = os.cpu_count()
        self.n_cpus = max(1, int(self.n_cpus * 0.9))

        if ts_id == None:
            self.model = None
            self.ts_id = None
            self.settings = None
        else:
            self.ts_id = ts_id
            self.model = AutoML()

            self.settings = {
                "time_budget": 1733,  # This is the budget of AIOps dataset
                "task": 'classification',
                "log_file_name": f"logs/flaml_tsad_{self.ts_id}.log",
                "log_type": "all",
                "n_jobs": self.n_cpus,  # parallel processing
                "seed": 42,  # for reproducibility,
                "max_iter": 1000
            }

    def fit(self, X_train, y_train):
        logging.info(f"Start {self.__class__.__name__} training")
        logging.info(f"Shapes of training {X_train.shape}, testing {y_train.shape}")
        self.model.fit(X_train=X_train, y_train=y_train, **self.settings)
        logging.info(f"Training {self.__class__.__name__} completed")

    def predict_proba(self, X_test):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        try:
            proba = self.model.predict_proba(X_test)
        except AttributeError:
            raise NotImplementedError("Underlying model does not support predict_proba.")
        
        return proba

    def dump(self, path):
        joblib.dump(self.model, path)
        logging.info(f"{self.__class__.__name__} stored: {path}")

    def load(self, path, ts_id):
        self.ts_id = ts_id
        self.model = joblib.load(path)

        self.settings = {
            "time_budget": 1733,  # This is the budget of AIOps dataset
            "task": 'classification',
            "log_file_name": f"logs/flaml_tsad_{self.ts_id}.log",
            "log_type": "all",
            "n_jobs": self.n_cpus,  # parallel processing
            "seed": 42,  # for reproducibility
        }




class FLAMLForecaster:

    def __init__(self, ts_id, horizon = 50):
        # set cpus to available number - 10%
        self.n_cpus = os.cpu_count()
        self.n_cpus = max(1, int(self.n_cpus * 0.9))

        self.horizon = horizon

        if ts_id == None:
            self.model = None
            self.ts_id = None
            self.settings = None
        else:
            self.ts_id = ts_id
            self.model = AutoML()

            self.settings = {
                "time_budget": 169,  # This is the budget of AIOps dataset
                "task": 'ts_forecast',
                "period": self.horizon,
                "log_file_name": f"logs/flaml_forecasting#AzFuncInvoke_{ts_id}.log",
                "log_type": "all",
                "n_jobs": self.n_cpus,  # parallel processing
                "seed": 42,  # for reproducibility,
                "max_iter": 1000,
                "eval_method": 'auto',
                'early_stop': True,
            }

    def fit(self, train_dataset) -> float:
        logging.info(f"Start {self.__class__.__name__} training")
        logging.info(f"Shapes of training dataset {train_dataset.shape}, testing {y_train.shape}")
        start_fitting_time = datetime.now()
        self.model.fit(
            dataframe=train_dataset.reset_index(),
            label='y',
            time_col='dt#',
            group_ids=None,
            **self.settings
        )
        end_fitting_time = datetime.now()
        logging.info(f"Training {self.__class__.__name__} completed")
        return (end_fitting_time - start_fitting_time).total_seconds() * 1000
    
    def predict(self, history_df, window, context):
        X_test, y_test = windowed(ts=history_df.y.values, context=window+context, window=window*2, reshape=False)
        
        warm_start_config = self.model.best_config_per_estimator
        refit_times = []
        pred_times = []
        for X_window in tqdm(X_test, total=len(X_test), desc=f"Predictions for {self.ts_id}"):
            pass
        pass

    def dump(self, path):
        joblib.dump(self.model, path)
        logging.info(f"{self.__class__.__name__} stored: {path}")

    def load(self):
        model_path = MODELS_PATH / f"flaml_model_AzFuncInvoke_{self.ts_id}.pkl"
        self.model = joblib.load(model_path)