'''Exponential smoothing (ETS)'''
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

from ts_utils import find_trend_type, test_boxcox_suitability

from dataset import windowed
import numpy as np

from tqdm import tqdm

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ETS():

    def __init__(self, seasonality = None, window = 28, context = 14, max_context = 512):
        # Initialize the model
        self.window = window
        self.context = context
        self.horizon = self.window - self.context
        self.max_context = max_context

        self.seasonality = seasonality

    def get_name(self):
        return self.__class__.__name__

    def fit(self, X):
        # perform model training on the dataset passing training and validation data.
        # raise NotImplementedError(f"fit() is not supported by {self.__class__.__name__}")
        self.X = X

        if self.seasonality:
            # we exploit the fit method to get the best params
            self.trend_type = find_trend_type(self.X, self.seasonality)

            self.boxcox = test_boxcox_suitability(self.X)

    def predict(self, X):
        X_test, y_test = windowed(X, window=self.window, context=self.context, reshape=False)

        _past_ctx = self.X.tolist() + X_test[0].tolist()

        y_true, y_pred = [], []

        for i, test_window in tqdm(enumerate(y_test), total = len(y_test), desc = 'Predicting with ETS'):

            if self.seasonality == None:
                self.model = ExponentialSmoothing(
                    _past_ctx, 
                    seasonal=None,
                    trend = 'add',
                    initialization_method='estimated'
                )
                self.model_fit = self.model.fit()
            else:
                try:
                    self.model = ExponentialSmoothing(
                        _past_ctx, 
                        seasonal="add", 
                        seasonal_periods=self.seasonality,
                        trend=self.trend_type, 
                        use_boxcox=self.boxcox,
                        initialization_method='estimated'
                    )
                    self.model_fit = self.model.fit()
                except Exception as e:
                    logging.warning(f"{e}. Using additive model")
                    print(_past_ctx)
                    self.model = ExponentialSmoothing(
                        _past_ctx, 
                        seasonal="add", 
                        seasonal_periods=self.seasonality,
                        trend='additive', 
                        use_boxcox=self.boxcox,
                        initialization_method='estimated'
                    )
                    self.model_fit = self.model.fit()
            y_pred_ith = self.model_fit.forecast(steps = self.horizon)

            y_true.append(test_window)
            y_pred.append(y_pred_ith)

            _past_ctx.append(test_window[0])
            _past_ctx = _past_ctx[-self.max_context:]

        return np.array(y_pred)

    def predict_proba(self, X):
        # Predict class probabilities for given data.
        raise NotImplementedError(f"predict_proba() is not supported by {self.__class__.__name__}")

    def dump(self, filename):
        # Save the model to a path.
        raise NotImplementedError(f"dump() is not supported by {self.__class__.__name__}")

    def load(self, filename):
        # Load the model from path
        raise NotImplementedError(f"load() is not supported by {self.__class__.__name__}")