from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from dask.diagnostics import ProgressBar


import pandas as pd
import numpy as np
import json
import re
import argparse
import traceback

from multiprocessing import Pool
from functools import partial

from glob import glob
from pathlib import Path

import dask.dataframe as dd

from tqdm import tqdm

from dataset import split

from cfg import TEST_SIZE
from cst_utils import set_seeds



'''
Methodology
-> extract the statistical features with tsfresh from all
the benchmarks and forks.

-> to apply meta-learning model pick a random
fork (which is not the one of the segment to be classified)
and use the corresponding statistical features.

This requires Python >= 3.9.2 (known issue with dask in previous versions)
'''

TS_PATH = "../data/"

def load_dataset(dataset):
    df = pd.read_csv(Path(TS_PATH) / f"{dataset}.csv")
    train_dfs = []
    for (ts_id), sub_df in df.groupby("ts"):
        sub_df.sort_values(by="dt#", inplace=True)
        train_df, test_df = split(sub_df, TEST_SIZE)
        train_dfs.append(train_df)

    return pd.concat(train_dfs)


if __name__ == "__main__":

    # --- args ---
    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--dataset-name', default='dataset.csv', type=str, required=True)
    args = parser.parse_args()

    ts_df = load_dataset(args.dataset_name)

    print(ts_df.head(), len(ts_df), ts_df.ts.unique())

    assert ts_df["ts"].isnull().sum() == 0
    assert ts_df["dt#"].isnull().sum() == 0
    assert ts_df.duplicated(subset=["ts", "dt#"]).sum() == 0

    extraction_settings = EfficientFCParameters()
    # extraction_settings = MinimalFCParameters()

    grouped = ts_df.groupby("ts")
    results = []

    # print(ts.y.dtype)
    # exit()

    for group_id, group_df in tqdm(grouped, total=len(grouped), desc="Extracting features from TSAD"):
        try:
            _ddf = dd.from_pandas(group_df, npartitions=20)
            extracted_features = extract_features(_ddf, column_id='ts', column_sort='dt#', column_value='y',
                        default_fc_parameters=extraction_settings,
                        # we impute = remove all NaN features automatically
                        # impute_function=impute, 
                        pivot=False, n_jobs = -1, disable_progressbar=False)
            
            # convert to pandas
            extracted_features = extracted_features.compute()
            
            # remove those that have the same value for all the rows
            extracted_features = extracted_features.loc[:, extracted_features.nunique() > 1]

            # transform the index in the 'id' column
            extracted_features = extracted_features.reset_index().rename(columns={"index": "ts"})

            results.append(extracted_features)
            # print(f"Successfully processed id: {group_id}")
        except Exception as e:
            print(f"Failed to process id {group_id}: {e}")
            print(traceback.format_exc())

    final_features = pd.concat(results)

    final_features.to_csv(f"../results/extracted_features_{args.dataset_name}.csv", index=None)