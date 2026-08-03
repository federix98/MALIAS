from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
from dask.diagnostics import ProgressBar


import pandas as pd
import numpy as np
import json
import re
import traceback

from multiprocessing import Pool
from functools import partial

from glob import glob

import dask.dataframe as dd

from constants import TS_PATH

from tqdm import tqdm



'''
Methodology
-> extract the statistical features with tsfresh from all
the benchmarks and forks.

-> to apply meta-learning model pick a random
fork (which is not the one of the segment to be classified)
and use the corresponding statistical features.

This requires Python >= 3.9.2 (known issue with dask in previous versions)
'''


def load_timeseries(benchmark):
    return load(TS_PATH, benchmark)


def load(dirpath, benchmark):
    path = '{}/{}.json'.format(dirpath, benchmark)
    with open(path) as f:
        return json.load(f)

def process_timeseries(benchmark):
    print('Processing ' + benchmark + ' ...')

    timeseries = load_timeseries(benchmark)

    dfs = []
    for f, ts_ in enumerate(timeseries):
        df = pd.DataFrame(ts_, columns = ["y"])
        df["benchmark_id"] = benchmark
        df["no_fork"] = f
        df["dt#"] = range(len(ts_))
        dfs.append(df)

    print('Ending processing of ' + benchmark, "len of dfs = ", len(dfs))
    
    if dfs:
        return pd.concat(dfs)
    return pd.DataFrame()

def get_benchmark_list():
    benchmarks = []
    for path in glob(TS_PATH + '/*.json'):
        benchmark = re.sub(r'\.json$', '', path.split('/')[-1])
        benchmarks.append(benchmark)

    return np.unique(benchmarks).tolist()

def load_dataset(sort=False, stratify=False):
    with Pool() as pool:
        fn = partial(process_timeseries)
        dfs = pool.map(fn, get_benchmark_list())
        df =  pd.concat(dfs, ignore_index=True)

        if sort:
            df.sort_values(["benchmark_id", "no_fork"], inplace=True)

        return df

if __name__ == "__main__":

    ts = load_dataset(sort = True)

    ts["id"] = ts["benchmark_id"] + "---" + ts["no_fork"].astype(str)
    ts = ts[["id", "dt#", "y"]]

    assert ts["id"].isnull().sum() == 0
    assert ts["dt#"].isnull().sum() == 0
    assert ts.duplicated(subset=["id", "dt#"]).sum() == 0

    # extraction_settings = ComprehensiveFCParameters()
    extraction_settings = EfficientFCParameters()

    # Assume `df` has columns: id, time, value
    grouped = ts.groupby("id")
    results = []

    # print(ts.y.dtype)
    # exit()

    for group_id, group_df in tqdm(grouped, total=len(grouped), desc="Extracting features from forks"):
        try:
            _ddf = dd.from_pandas(group_df, npartitions=20)
            extracted_features = extract_features(_ddf, column_id='id', column_sort='dt#', column_value='y',
                        default_fc_parameters=extraction_settings,
                        # we impute = remove all NaN features automatically
                        # impute_function=impute, 
                        pivot=False, n_jobs = -1, disable_progressbar=False)
            
            # convert to pandas
            extracted_features = extracted_features.compute()
            
            # remove those that have the same value for all the rows
            extracted_features = extracted_features.loc[:, extracted_features.nunique() > 1]

            # transform the index in the 'id' column
            extracted_features = extracted_features.reset_index().rename(columns={"index": "id"})

            results.append(extracted_features)
            # print(f"Successfully processed id: {group_id}")
        except Exception as e:
            print(f"Failed to process id {group_id}: {e}")
            print(traceback.format_exc())

    final_features = pd.concat(results)

    final_features.to_csv("extracted_features.csv", index=None)