import os
import numpy as np

from pathlib import Path

import logging
from autogluon.tabular import TabularPredictor

import pandas as pd

class AutoGluonClassifier:

    def __init__(self, dir : str = "./results/_tmp") -> None:

        # set cpus to available number - 10%
        self.n_cpus = os.cpu_count()
        self.n_cpus = max(1, int(self.n_cpus * 0.9))

        if dir == None:
            self.dir = None
            self.model = None
            logging.info(f"AutoGluon Classifier without DIR")
        else:        
            # create _tmp folder if not exists
            self.dir = Path(dir)
            self.dir.mkdir(exist_ok=True, parents=True)
            self.model = TabularPredictor(
                label='label',
                path=self.dir, 
                problem_type='binary', 
                # eval_metric='roc_auc', # use default metric for binary classification 
                verbosity=3
            )
            logging.info(f"AutoGluon Classifier built ({self.dir})")

    # def fit(self, x_train, y_train, x_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    def fit(self, X_train, y_train):
        # train_data contains both cols and label data 

        train_data = pd.DataFrame(X_train)
        train_data["label"] = y_train

        # shuffle
        train_data = train_data.sample(frac=1).reset_index(drop=True)

        logging.info("Head and tail of training dataset")
        logging.info(train_data.head())
        logging.info("-----------------------------")
        logging.info(str(train_data.tail()) + "\n\n")

        self.model.fit(
            train_data = train_data,
            time_limit = None,
            presets = "medium_quality",
            num_cpus = self.n_cpus,
            num_gpus = 0,
            verbosity = 3
        )


    def predict(self, x):
        x_input = pd.DataFrame(x)
        proba_df = self.model.predict_proba(x_input)
        probas = proba_df[1].values
        y = (probas >= 0.5).astype(int)
        return y

    def predict_proba(self, x):
        x_input = pd.DataFrame(x)
        proba_df = self.model.predict_proba(x_input)
        probas = proba_df[1].values
        return probas
    
    def load(self, path):
        self.dir = Path(path)
        self.model = TabularPredictor.load(path)
        logging.info(f"Loaded model from {path}")