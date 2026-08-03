import json
import argparse
from datetime import datetime
import numpy as np
import yaml
import random

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression

from kfold import create_kfold

from cfg import ABLATION_FEATURES_SELECT, CONFIG_PATH


'''
Changes:
- kfold: from 10 to 3 splits
- random foreest estimators: from 1000 to 10
- RFECV step: from 1 to 5
'''

def __all_selection(features):
    return [{
        "fs_name": "all",
        "fs_variant": "default",
        "selected_features": features 
    }]

def __random_selection(features, fs_params):
    results = []
    seed = fs_params["seed"]
    elements = fs_params["n"]

    random.seed(seed)

    for element in elements:
        selected = random.sample(features.tolist(), element)
        results.append({
            "fs_name": "random",
            "fs_variant": f"k={element}",
            "selected_features": selected 
        })
    return results

def __multi_target_mutual_info_regression(X, Y):
    scores = []
    for i in range(Y.shape[1]):
        vals = mutual_info_regression(X, Y[:, i])
        scores.append(vals)
    avg_scores = np.mean(scores, axis=0)
    return avg_scores, np.ones_like(avg_scores)

def __multi_target_f_regression(X, Y):
    scores = []
    for i in range(Y.shape[1]):
        f_vals, _ = f_regression(X, Y[:, i])
        scores.append(f_vals)
    avg_scores = np.mean(scores, axis=0)
    return avg_scores, np.ones_like(avg_scores)

def __KBest_selection(features, df, group_split, X, y, fs_params, score_func=f_regression):
    results = []

    for k in fs_params["k_values"]:
        print(f"[{datetime.now()}] k",  k)
        
        for function in fs_params["function"]:
            print(f"[{datetime.now()}] function",  function)

            ids = X.index

            kf = create_kfold(df = df, group_split = group_split, split = "val")

            all_scores = []

            for train_index, _ in kf:
                train_ids = ids[train_index]


                X_train = X.loc[train_ids, features]
                y_train = y.loc[train_ids, models]

                # Fit SelectKBest on training set
                if function == "f_regr":
                    selector = SelectKBest(score_func=__multi_target_mutual_info_regression, k='all')
                elif function == "mutual_info_regression":
                    selector = SelectKBest(score_func=__multi_target_f_regression, k='all')
        
                selector.fit(X_train, y_train)
                all_scores.append(selector.scores_)

            mean_scores = np.mean(all_scores, axis=0)
            top_k_indices = np.argsort(mean_scores)[::-1][:k]
            top_k_features = np.array(X.columns)[top_k_indices]  # if X is a DataFrame

            print(f"{len(top_k_features)} featues globally selected")

            results.append({
                "fs_name": "SelectKBest",
                "fs_variant": f"function={function}___k={k}",
                "selected_features": top_k_features
            })
        print(f"[{datetime.now()}] Done.")

    return results

def select_features(features, df, group_split, X, y, fs_config):
    results = []

    for fs_name, fs_params in fs_config.items():
        print(f"\n🔍 Applying Feature Selection: {fs_name}")
        if fs_name == "select_k_best":
            results += __KBest_selection(features, df, group_split, X, y, fs_params)
        elif fs_name == "all":
            results += __all_selection(features)
        elif fs_name == "random":
            results += __random_selection(features, fs_params)

    if not len(results):
        print("No method applied.")

    return results

if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    with open(CONFIG_PATH / "feature_selection.yaml") as f:
        fs_config = yaml.safe_load(f)["selectors"]

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

            fs_results = select_features(features=features, df=df, group_split=dataset["group_split"], X=df[features], y=df[models], fs_config=fs_config)
            
            dest_path = ABLATION_FEATURES_SELECT
            dest_path.mkdir(exist_ok=True, parents=True)

            for fs_result in fs_results:
                # print("Selected", fs_result["selected_features"])
                df[fs_result["selected_features"]].to_csv(dest_path / f"selected#{dataset['dataset_id']}---{fs_result['fs_name']}---{fs_result['fs_variant']}.csv")