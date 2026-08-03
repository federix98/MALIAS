import pandas as pd
import numpy as np

import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from kfold import create_predefined_split

import argparse


'''
Changes:
- kfold: from 10 to 3 splits
- random foreest estimators: from 1000 to 10
- RFECV step: from 1 to 20
'''

if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Formatting features")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open("config.json") as c_file:
        config = json.loads(c_file.read())

    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Formatting features formatting features {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Feature selection for dataset {dataset['dataset_id']} ...")

            df_features = pd.read_csv(dataset['original_feature_path'])

            df_features.rename(columns={"ts": "id"}, inplace=True)

            # drop 'kind' since it has unique value and 'level_2' columns
            df_features.drop(columns=["kind", "level_2"], inplace=True)

            df_features = df_features.pivot_table(
                values = "value",
                index = "id",
                columns = ["variable"]
            )

            # remove null features
            df_features = df_features.dropna(axis=1, how='any')

            df_features.to_csv(dataset["feature_path"])