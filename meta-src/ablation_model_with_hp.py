import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV

from kfold import create_kfold
from pathlib import Path

import yaml
import importlib

import random
import numpy as np

from cfg import RANDOM_STATE, CONFIG_PATH, ABLATION_MODELS

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

ABLATION_MODELS_TUNED = ABLATION_MODELS.parent / (ABLATION_MODELS.name + "_tuned")


def train_and_predict(df, group_split, features, models, mdl_config):

    results = {}
    tuning_info = {}

    for mdl_name, mdl_params in mdl_config.items():
        print(f"\n🔍 Model: {mdl_name}")

        module = importlib.import_module(mdl_params["module"])
        ModelClass = getattr(module, mdl_params["class"])

        seed = RANDOM_STATE

        ids = df.index

        kf_val = list(create_kfold(df=df, group_split=group_split, split="val"))

        if mdl_params.get("use_random_state", False):
            base_estimator = ModelClass(random_state=seed)
        else:
            base_estimator = ModelClass()

        estimator = base_estimator
        # estimator = MultiOutputRegressor(base_estimator)

        param_grid = mdl_params.get("param_grid", {})

        X_full = df[features]
        y_full = df[models]

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=kf_val,
            n_jobs=50,
            verbose=2)

        search.fit(X_full, y_full)

        best_params = search.best_params_
        best_score = float(search.best_score_)

        print(f"Best params for {mdl_name}: {best_params}")
        print(f"Best CV score for {mdl_name}: {best_score}")

        tuning_info[mdl_name] = {
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_params.items()},
            "cv_score": best_score,
        }

        kf = create_kfold(df=df, group_split=group_split, split="test")

        pred = pd.DataFrame(index=ids, columns=models)

        for fold, (train_index, test_index) in enumerate(kf):
            train_ids = ids[train_index]
            test_index = ids[test_index]

            X_train = df.loc[train_ids, features]
            y_train = df.loc[train_ids, models]

            X_test = df.loc[test_index, features]

            if mdl_params.get("use_random_state", False):
                base = ModelClass(random_state=seed)
            else:
                base = ModelClass()

            regressor = base
            # rf = MultiOutputRegressor(base)
            regressor.set_params(**best_params)

            print("fitting with features", X_train.columns)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            pred.loc[test_index, models] = y_pred

        results[mdl_name] = pred

    if not len(results):
        print("No method applied.")

    return results, tuning_info


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    with open(CONFIG_PATH / "models_with_hp.yaml") as f:
        mdl_config = yaml.safe_load(f)["models"]

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Ablation study (models) for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Ablation study (models) for dataset {dataset['dataset_id']} ...")

            df_features = pd.read_csv(dataset['selected_feature_path'], index_col='id')
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            features = df_features.columns
            models = df_metrics.columns

            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            df = df_features.join(df_metrics, how='inner')

            mdls_results, tuning_info = train_and_predict(features=features, models=models, df=df, group_split=dataset["group_split"], mdl_config=mdl_config)

            dest_path = ABLATION_MODELS_TUNED
            dest_path.mkdir(exist_ok=True, parents=True)

            with open(dest_path / f"tuning_results#{dataset['dataset_id']}.json", 'w') as f:
                json.dump(tuning_info, f, indent=2)

            for mdl_name, mdl_result in mdls_results.items():

                mdl_result = mdl_result.dropna()

                mdl_result = mdl_result.astype(float)

                if task["objective"] == "maximize":
                    best_models = mdl_result.idxmax(axis=1).to_frame('model')
                else:
                    best_models = mdl_result.idxmin(axis=1).to_frame('model')

                dest_path = ABLATION_MODELS_TUNED
                dest_path.mkdir(exist_ok=True, parents=True)

                best_models.to_csv(dest_path / f"ablation_models#{dataset['dataset_id']}---{mdl_name}.csv")