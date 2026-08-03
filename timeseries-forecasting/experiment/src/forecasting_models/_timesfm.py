import timesfm
from tqdm import tqdm

import torch
import os

NUM_CPUS = os.cpu_count()
NUM_THREADS = max(1, NUM_CPUS - 5) 
torch.set_num_threads(NUM_THREADS)

import logging
# from transformers import logging as hf_logging

# hf_logging.set_verbosity_debug()
# logging.basicConfig(level=logging.DEBUG)

from dataset import windowed

import numpy as np

class TFM():

    def __init__(self, window = 28, context = 14, frequency = 0, max_context = 512):
        # Initialize the model
        self.window = window
        self.context = context
        self.horizon = self.window - self.context

        self.max_context = max_context

        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch")

        self.frequency = frequency
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=self.horizon,
            ),
            checkpoint=checkpoint
        )

    def tune(self, X, y):
        raise NotImplementedError(f"tune() is not supported by {self.__class__.__name__}")
        
    def fit(self, X_train, y_train, X_val, y_val, epochs, verbose):
        # perform model training on the dataset passing training and validation data.
        pass
        
    # def predict(self, X_test):
    #     X_test = X_test.squeeze()
    #     point_forecast, experimental_quantile_forecast = self.model.forecast(
    #         X_test,
    #         freq=[self.frequency for elem in X_test],
    #     )

    #     return point_forecast

    def predict(self, train_ts, test_ts):
        X_test, y_test = windowed(test_ts, window=self.window, context=self.context, reshape=False)

        _past_ctx = train_ts.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        input_ctxs = []
        for i, test_window in enumerate(y_test):
            input_ctxs.append(_past_ctx[-len(train_ts.tolist()):])
            y_true.append(test_window)
            _past_ctx.append(test_window[0])

        input_ctxs = np.array(input_ctxs).squeeze()

        logging.info(f"TimesFM contexts {input_ctxs}")

        point_forecast, experimental_quantile_forecast = self.model.forecast(
            input_ctxs,
            freq=[self.frequency for elem in input_ctxs],
        )

        return point_forecast
        
    def predict_proba(self, X):
        # Predict class probabilities for given data.
        raise NotImplementedError(f"predict_proba() is not supported by {self.__class__.__name__}")

    def dump(self, path, parameter = ""):
        # Save the model to a path.
        pass
        # raise NotImplementedError(f"dump() is not supported by {self.__class__.__name__}")

    def load(self, filename):
        # Load the model from path
        raise NotImplementedError(f"load() is not supported by {self.__class__.__name__}")

    def set_seeds(self, seed):
        # Set the seed for all necessary random number generators.
        raise NotImplementedError(f"set_seeds() is not supported by {self.__class__.__name__}")

    def get_name(self):
        return "TimesFM"