import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

from scipy.stats import boxcox

from cfg import RES_PATH

def get_sarima_params(dataset_name, ts_id):
    sarima_results_path = RES_PATH / "sarima-tuning" / dataset_name
    sarima_params = pd.read_csv(sarima_results_path / f"dataset_{dataset_name}#ts_{ts_id}.csv")
    sarima_params.sort_values(by=["AIC"], ascending=True, inplace=True)
    best_params = sarima_params.iloc[0]
    return {
        "p" : best_params["p"],
        "d" : best_params["d"],
        "q" : best_params["q"],
        "P" : best_params["P"],
        "D" : best_params["D"],
        "Q" : best_params["Q"],
        "s" : best_params["s"]
    }

def test_boxcox_suitability(time_series):
    """
    Test whether Box-Cox transformation is suitable for the time series.
    """
    try:
        _, lmbda = boxcox(time_series)
        return True
    except ValueError:
        return False


# def find_seasonality(ts, sampling_rate=1):
#     # Remove the mean to focus on oscillations
#     ts_detrended = ts - np.mean(ts)
    
#     # Perform Fourier Transform
#     fft = np.fft.fft(ts_detrended)
#     frequencies = np.fft.fftfreq(len(ts), d=sampling_rate)  # Frequencies in the same units as sampling_rate
#     magnitudes = np.abs(fft)
    
#     # Filter positive frequencies only
#     positive_freqs = frequencies[frequencies > 0]
#     positive_magnitudes = magnitudes[frequencies > 0]
    
#     # Find the dominant frequency
#     dominant_frequency = positive_freqs[np.argmax(positive_magnitudes)]

#     print(dominant_frequency)
    
    
#     seasonality = int(round(1 / dominant_frequency))
#     return seasonality

def find_seasonality(data, threshold=0.5, max_lag=None):
    """
    Detects the seasonality of a NumPy array based on autocorrelation.
    
    Parameters:
        data (numpy.ndarray): The input array of numerical values.
        threshold (float): The minimum autocorrelation value to consider significant seasonality.
        max_lag (int, optional): The maximum lag to check for seasonality. Defaults to len(data)//2.
    
    Returns:
        int or None: The seasonality period (lag with highest significant autocorrelation) 
                     or None if no seasonality is detected.
    """
    if len(data) < 2:
        return None  # Not enough data to detect seasonality
    
    # Determine the maximum lag to test
    if max_lag is None:
        max_lag = len(data) // 2
    
    # Compute autocorrelations
    autocorr = acf(data, nlags=max_lag, fft=True)
    
    # Find the lag with the highest significant autocorrelation
    for lag in range(1, len(autocorr)):
        if autocorr[lag] > threshold:
            return lag
    
    # If no significant autocorrelation is found, return 1
    return 1

# def find_seasonality(data, threshold=0.5, max_lag=None):
#     if len(data) < 2:
#         return 1
    
#     if max_lag is None:
#         max_lag = len(data) // 2
    
#     autocorr = acf(data, nlags=max_lag, fft=True)
    
#     # Exclude lag 0, find best lag above threshold
#     valid_lags = [(lag, val) for lag, val in enumerate(autocorr[1:], start=1) if val > threshold]
    
#     if not valid_lags:
#         return 1  # No significant seasonality detected
    
#     # Return lag with highest autocorrelation
#     best_lag = max(valid_lags, key=lambda x: x[1])[0]
#     return best_lag


def find_trend_type(time_series, period):
    """
    Identifies if a time series has an additive or mult trend.

    Parameters:
        time_series (pd.Series): The input time series data.
        period (int): The seasonal period of the time series.

    Returns:
        str: 'Additive' if the trend appears to be additive,
             'Multiplicative' if the trend appears to be multiplicative,
             'No clear trend' if inconclusive.
    """

    if period < 2:
        return "add" # no seasonal component
    # Decompose the time series using STL decomposition
    stl = STL(time_series, period=period, robust=True)
    result = stl.fit()

    # Extract components
    trend = result.trend
    residual = result.resid

    # Check for multiplicative trend: residuals scaled to the level of the series
    residual_ratio = residual / trend

    # Evaluate standard deviation of residuals
    std_residual = np.std(residual)
    std_residual_ratio = np.std(residual_ratio)

    # Decide trend type based on residual consistency
    if std_residual_ratio < std_residual * 0.8:
        return 'mul'
    elif std_residual_ratio > std_residual * 1.2:
        return 'add'
    else:
        return 'unk'

# Helper function to check stationarity
def check_stationarity(time_series, significance_level=0.05):
    result = adfuller(time_series)
    is_stationary = result[1] < significance_level
    return bool(is_stationary)