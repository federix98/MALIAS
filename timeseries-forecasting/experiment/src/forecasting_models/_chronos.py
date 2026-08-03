'''Chronos forecasting'''
import pandas as pd  # requires: pip install pandas
import torch
import os

NUM_CPUS = os.cpu_count()
NUM_THREADS = max(1, NUM_CPUS - 5) 
torch.set_num_threads(NUM_THREADS)

from chronos import BaseChronosPipeline

from dataset import windowed

import numpy as np

from tqdm import tqdm


import logging

class Chronos():

    def __init__(self, window = 28, context = 14, max_context = 512):
        # Initialize the model
        self.context = context
        self.window = window
        self.horizon = self.window - self.context
        self.max_context = max_context
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-mini", # "amazon/chronos-bolt-small",  
            device_map="cpu",  # use "cpu" for CPU inference
            torch_dtype=torch.bfloat16,
        )

    def fit(self, X):
        # Fit the model from path
        raise NotImplementedError(f"fit() is not supported by {self.__class__.__name__}. The model is already pre-trained.")
        
    def predict(self, train_ts, test_ts):
        X_test, y_test = windowed(test_ts, window=self.window, context=self.context, reshape=False)

        _past_ctx = train_ts.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        input_ctxs = []
        for i, test_window in enumerate(y_test):
            input_ctxs.append(_past_ctx[-self.max_context:])
            y_true.append(test_window)
            _past_ctx.append(test_window[0])

        input_ctxs = np.array(input_ctxs).squeeze()

        logging.info(f"Chronos contexts shape {np.array(input_ctxs).shape}")

        pred_batch_size = 30

        num_samples = input_ctxs.shape[0]
        results = []

        # Process input_ctxs in batches
        for i in tqdm(range(0, num_samples, pred_batch_size), desc="Predicting"):
            batch_ctxs = input_ctxs[i:i + pred_batch_size]

            logging.info(f"Predicting shape {np.array(batch_ctxs).shape}")
            
            y_pred_quantiles, y_pred_mean = self.pipeline.predict_quantiles(
                context=torch.tensor(batch_ctxs),
                prediction_length=self.horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            
            y_pred = y_pred_mean.numpy().squeeze()
            results.append(y_pred)

        # Concatenate all batch results
        final_predictions = np.concatenate(results, axis=0)
        return final_predictions

        # y_pred_quantiles, y_pred_mean = self.pipeline.predict_quantiles(
        #     context=torch.tensor(input_ctxs),
        #     prediction_length=self.horizon,
        #     quantile_levels=[0.1, 0.5, 0.9],
        # )

        # y_pred = y_pred_mean.numpy().squeeze()
        # return y_pred


        # for i, test_window in tqdm(enumerate(y_test), total=len(y_test), desc="Predicting test windows with Chronos Pre-trained model"):

        #     # cut the input window if it exceeds the max context
        #     y_pred_quantiles, y_pred_mean = self.pipeline.predict_quantiles(
        #         context=torch.tensor(_past_ctx[max(0, len(_past_ctx) - self.max_context):]),
        #         prediction_length=self.horizon,
        #         quantile_levels=[0.1, 0.5, 0.9],
        #     )

        #     ## to get quantiles
        #     # y_pred_low, y_pred_median, y_pred_high = y_pred_quantiles[0, :, 0], y_pred_quantiles[0, :, 1], y_pred_quantiles[0, :, 2]

        #     y_true.append(test_window)
        #     y_pred.append(y_pred_mean.numpy().squeeze())

        #     _past_ctx.append(test_window[0])

        # return np.array(y_pred)

    def predict_proba(self, X):
        # Predict class probabilities for given data.
        raise NotImplementedError(f"predict_proba() is not supported by {self.__class__.__name__}")

    def dump(self, filename):
        # Save the model to a path.
        raise NotImplementedError(f"dump() is not supported by {self.__class__.__name__}")

    def load(self, filename):
        # Load the model from path
        raise NotImplementedError(f"load() is not supported by {self.__class__.__name__}")

    def get_name(self):
        return "Chronos"
