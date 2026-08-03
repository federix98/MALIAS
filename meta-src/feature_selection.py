import json
import argparse
from datetime import datetime
import numpy as np

import pandas as pd
import sklearn
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from kfold import create_kfold

from cfg import RANDOM_STATE, CONFIG_PATH

def multicollinearity_filter(X, kf, features):
    threshold = 0.8
    print(f"[{datetime.now()}] Feature selection > Applying Pearson Correlation Filter (pearson threshold {threshold})")

    selected_sets = []

    for train_idx, _ in kf:
        X_train_fold = X.copy().iloc[train_idx][features]
        X_train_fold = X_train_fold.loc[:, X_train_fold.nunique() > 1]

        corr_matrix = X_train_fold.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        sorted_features = sorted(X_train_fold.columns)
        X_train_fold = X_train_fold[sorted_features]

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        X_train_fold = X_train_fold.drop(columns=to_drop)

        selected_sets.append(set(X_train_fold.columns.tolist()))

    for i, s in enumerate(selected_sets):
        print(f"Fold {i}: Selected {len(s)} features ({len(features) - len(s)}) removed.")
    globally_selected = sorted(list(set.intersection(*selected_sets)))
    return globally_selected

def select_features(X, y, kf):

    seed = RANDOM_STATE

    n_estimators = 100
    min_features_to_select=5
    step=1

    print(f"[{datetime.now()}] Feature selection > Applying RFECV filter")
    print(f"=> RFECV Parameters:")
    print(f"===> RandomForestRegressor: {n_estimators=}, {seed=}")
    print(f"===> RFECV: {min_features_to_select=}, {step=}")
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=5)
    rfe = RFECV(estimator=rf, min_features_to_select=min_features_to_select, step=step, cv=kf, scoring='neg_mean_absolute_error', n_jobs=8, verbose=1)
    rfe.fit(X, y)

    return X.columns[rfe.support_]

if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Feature selection for meta-learner")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Feature selection for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Feature selection for dataset {dataset['dataset_id']} ...")

            df_features = pd.read_csv(dataset['feature_path'], index_col='id')
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            features = df_features.columns
            models = df_metrics.columns

            # normalize mae
            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            # join features and mae
            df = df_features.join(df_metrics, how = 'inner')

            
            indep_features = multicollinearity_filter(X=df[features], kf=create_kfold(df = df, group_split = dataset["group_split"], split = "val"), features=features)
            print(f"Feature selection > Pearson Correlation Filter: Selected {len(indep_features)} features")

            features_ = select_features(X=df[indep_features], y=df[models], kf=create_kfold(df = df, group_split = dataset["group_split"], split = "val"))
            print(f"Feature selection > RFECV filter: Selected {len(features_)} features")

            print(f"[{datetime.now()}] Feature selection completed. {len(features_)} features selected")

            print(f"[{datetime.now()}] : Final selection {features_}")

            df[features_].to_csv(dataset['selected_feature_path'])

            print(f"[{datetime.now()}] Selected features succesfully dumped.")
            