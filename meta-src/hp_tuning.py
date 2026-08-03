import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from kfold import create_kfold

import random
import numpy as np

from cfg import RANDOM_STATE, CONFIG_PATH

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


'''
Changes:
n_estimators: from np.arange(1000, 3001, 1000) to np.arange(100, 301, 100)
'''

def search_best_params(X, y, kf):
    seed = RANDOM_STATE

    rf = RandomForestRegressor(random_state=seed)

    # param_distributions = {
    #     'n_estimators': [100, 500, 1000],        # number of trees
    #     'max_depth': [None, 3, 10],            # maximum depth of the tree
    #     'min_samples_split': [2, 5],            # minimum samples required to split
    #     'min_samples_leaf': [1, 2],             # minimum samples required at each leaf node
    #     'max_features': [None, 'sqrt', 'log2'],
    #     'bootstrap': [True, False]             # whether bootstrap samples are used
    # }

    # shorter param
    param_distributions = {
        'n_estimators': [10, 100, 1000],        # number of trees
        'max_depth': [None, 5, 20],            # maximum depth of the tree
        'criterion': ['squared_error', 'absolute_error']
    }

    search = GridSearchCV(
        estimator=rf,
        param_grid=param_distributions,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=50,
        verbose=2)
    
    search.fit(X, y)

    best_params = search.best_params_
    best_params['n_estimators'] = int(best_params['n_estimators'])

    return best_params

if __name__ == '__main__':
    
    # --- args ---
    parser = argparse.ArgumentParser(description="Hp tuning meta")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Hp tuning for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Hp tuning for dataset {dataset['dataset_id']} ...")

            # df_features = pd.read_csv(dataset['feature_path'], index_col='id')
            df_features = pd.read_csv(dataset['selected_feature_path'], index_col='id')
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            features = df_features.columns
            models = df_metrics.columns

            # normalize mae
            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            # join features and mae
            df = df_features.join(df_metrics, how = 'inner')

            print(df.head())

            X = df[features] 
            y = df[models] 

            kf = create_kfold(df = df, group_split = dataset["group_split"], split = "val")

            best_params = search_best_params(X, y, kf)
            print(f"[{datetime.now()}] Done")

            with open(CONFIG_PATH / f"best_params_{dataset['dataset_id']}.json", 'w') as f:
                json.dump(best_params, f)
