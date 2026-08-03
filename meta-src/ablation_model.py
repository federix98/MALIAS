import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from kfold import create_kfold
from pathlib import Path

import yaml
import importlib

from cfg import RANDOM_STATE, CONFIG_PATH, ABLATION_MODELS



def train_and_predict(df, group_split, features, models, mdl_config):

    results = {}

    for mdl_name, mdl_params in mdl_config.items():
        print(f"\n🔍 Model: {mdl_name}")

        module = importlib.import_module(mdl_params["module"])
        ModelClass = getattr(module, mdl_params["class"])


        seed = RANDOM_STATE

        ids = df.index

        kf = create_kfold(df = df, group_split = group_split, split = "test")

        pred = pd.DataFrame(index=ids, columns=models)

        for fold, (train_index, test_index) in enumerate(kf):
            # for the final application, all the training set is used
            train_ids = ids[train_index]
            test_index = ids[test_index]

            X_train = df.loc[train_ids, features]
            y_train = df.loc[train_ids, models]

            X_test = df.loc[test_index, features]

            if mdl_params.get("use_random_state", False):
                rf = ModelClass(random_state=seed)
            else:
                rf = ModelClass()
            print("fitting with features", X_train.columns)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            pred.loc[test_index, models] = y_pred

        results[mdl_name] = pred

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

    with open(CONFIG_PATH / "models.yaml") as f:
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

            # normalize mae
            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            # join features and mae
            df = df_features.join(df_metrics, how = 'inner')

            mdls_results = train_and_predict(features=features, models=models, df=df, group_split=dataset["group_split"], mdl_config=mdl_config)
            
            dest_path = ABLATION_MODELS
            dest_path.mkdir(exist_ok=True, parents=True)

            for mdl_name, mdl_result in mdls_results.items():

                mdl_result = mdl_result.dropna()

                mdl_result = mdl_result.astype(float)

                if task["objective"] == "maximize":
                    best_models = mdl_result.idxmax(axis=1).to_frame('model')
                else:
                    best_models = mdl_result.idxmin(axis=1).to_frame('model')
                
                dest_path = ABLATION_MODELS
                dest_path.mkdir(exist_ok=True, parents=True)
                # print("Selected", mdl_result["selected_features"])
                best_models.to_csv(dest_path / f"ablation_models#{dataset['dataset_id']}---{mdl_name}.csv")