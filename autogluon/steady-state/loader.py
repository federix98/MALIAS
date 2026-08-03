from multiprocessing import Pool
from functools import partial

import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

from constants import WINDOW_SIZE
from util import get_light_benchmark_list, get_benchmark_list, load_timeseries, load_classification, foreach_fork



# rebalance datapoints of a fork
def _resample(df):
    # 100 segments per fork (50 per class)
    sample_size_per_class =  50

    idx_per_class = {}
    for class_ in [0, 1]:
    # Get min and max starts_at values for majority class
        starts_at_min = df[df.y == class_].starts_at.min()
        starts_at_max = df[df.y == class_].starts_at.max()
        idx = np.linspace(starts_at_min, starts_at_max, sample_size_per_class, dtype=int)
        idx_per_class[class_] = np.unique(idx)

    for class_, idx in idx_per_class.items():
        #  Correct sample size if necessary
        if len(idx) < sample_size_per_class:
            other_class = 1 - class_
            sample_size = 2*sample_size_per_class - len(idx)
            starts_at_min = df[df.y == other_class].starts_at.min()
            starts_at_max = df[df.y == other_class].starts_at.max()
            idx_per_class[other_class] = np.linspace(starts_at_min, starts_at_max, sample_size, dtype=int)

    # Get mask
    mask0 = (df.y == 0) & (df.starts_at.isin(idx_per_class[0]))
    mask1 = (df.y == 1) & (df.starts_at.isin(idx_per_class[1]))
        
    # Return resampled dataframe
    return df[mask0 | mask1]


# rebalance dataset
def resample(df):
    # resample each fork
    return df.groupby(['benchmark_id', 'no_fork'], group_keys=False).apply(_resample)



def process_fork(i, ts, st):
    rows = []
    windows = sliding_window_view(ts, WINDOW_SIZE)

    for starts_at, window in enumerate(windows):
        end_at = starts_at + WINDOW_SIZE - 1
        clas = 0
        if (0 <= st < len(ts)) and (starts_at > st):
            clas = 1
        
        # Scale values without losing precision
        window = window * 10**6
        # Standardize data
        window = (window - window.mean()) / window.std()
        # Handle cases where std is equal to 0
        window[np.isnan(window) | np.isinf(window)] = 0

        row = [i, starts_at, end_at, *window.tolist(), clas]
        rows.append(row)

    return rows


def process_benchmark(benchmark, steady_state_only=False):
    print('Processing ' + benchmark + ' ...')

    timeseries = load_timeseries(benchmark)
    classification = load_classification(benchmark)

    rows = []
    for i, ts, st in foreach_fork(timeseries, classification):
        # Skip non steady state forks if steady_state_only is True
        if steady_state_only is True and st == -1:
            continue
        newrows = process_fork(i, ts, st)
        rows += newrows

    x_columns = ["x{}".format(i) for i in range(WINDOW_SIZE)]
    columns = ['no_fork', 'starts_at', 'end_at'] + x_columns + ['y']
    df = pd.DataFrame(rows, columns=columns)
    df['benchmark_id'] = benchmark

    print('Ending processing of ' + benchmark)
    return df

def load_dataset(steady_state_only=False, sort=False, stratify=False):
    with Pool() as pool:
        fn = partial(process_benchmark, steady_state_only=steady_state_only)
        dfs = pool.map(fn, get_benchmark_list())
        df =  pd.concat(dfs, ignore_index=True)

        if stratify:
            df = resample(df)

        if sort:
            df.sort_values(["benchmark_id", "no_fork", "starts_at"], inplace=True)

        return df

def lightload_dataset(steady_state_only=False, sort=False, stratify=False, cut = 10):
    with Pool() as pool:
        fn = partial(process_benchmark, steady_state_only=steady_state_only)
        dfs = pool.map(fn, get_light_benchmark_list(cut = cut))
        df =  pd.concat(dfs, ignore_index=True)

        if stratify:
            df = resample(df)

        if sort:
            df.sort_values(["benchmark_id", "no_fork", "starts_at"], inplace=True)

        return df
