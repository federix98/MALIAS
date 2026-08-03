import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from kfold import create_kfold
from pathlib import Path

import joblib

from cfg import RANDOM_STATE, CHECKPOINTS_PATH, CONFIG_PATH, META_PREDICTIONS


def load_best_params(dataset_id):
    # with open(f"best_params.json", 'r') as file:
    with open(CONFIG_PATH / f"best_params_{dataset_id}.json", 'r') as file:
        best_params = json.load(file)
    return best_params


def train_and_predict(df, dataset_id, group_split, features, models):

    seed = RANDOM_STATE

    ids = df.index

    best_params = load_best_params(dataset_id)

    kf = create_kfold(df = df, group_split = group_split, split = "test")

    pred = pd.DataFrame(index=ids, columns=models)

    for fold, (train_index, test_index) in enumerate(kf):
        # for the final application, all the training set is used
        train_ids = ids[train_index]
        test_index = ids[test_index]

        X_train = df.loc[train_ids, features]
        y_train = df.loc[train_ids, models]

        X_test = df.loc[test_index, features]

        rf = RandomForestRegressor(**best_params, random_state=seed, n_jobs=30)
        print("fitting with features", X_train.columns)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # dump model
        joblib.dump(rf, CHECKPOINTS_PATH / f"{dataset_id}___{fold}.joblib")

        pred.loc[test_index, models] = y_pred

    return pred


if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Predict meta - time series")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    CHECKPOINTS_PATH.mkdir(exist_ok=True, parents=True)

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Meta application for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Meta application for dataset {dataset['dataset_id']} ...")

            df_features = pd.read_csv(dataset['selected_feature_path'], index_col='id')
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            features = df_features.columns
            models = df_metrics.columns

            # normalize target
            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            # join features and targets
            df = df_features.join(df_metrics, how = 'inner')

            print("Shape of the joined dataset", df.shape)
            
            pred = train_and_predict(df, dataset_id = dataset['dataset_id'], group_split=dataset['group_split'], features=features, models=models)

            pred = pred.dropna()

            pred = pred.astype(float)
            # print(pred, pred.dtypes)

            print(f"[{datetime.now()}] Done")

            if task["objective"] == "maximize":
                best_models = pred.idxmax(axis=1).to_frame('model')
            else:
                best_models = pred.idxmin(axis=1).to_frame('model')
            
            best_models.to_csv(META_PREDICTIONS / f"meta_{dataset['dataset_id']}.csv")






