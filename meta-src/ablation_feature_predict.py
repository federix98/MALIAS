import json
import argparse
from datetime import datetime

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import minmax_scale

from predict_meta import train_and_predict

from cfg import CONFIG_PATH, ABLATION_FEATURES_SELECT, ABLATION_FEATURES_PRED


def load_best_params(dataset_id):
    with open(CONFIG_PATH / f"best_params_{dataset_id}.json", 'r') as file:
        best_params = json.load(file)
    return best_params

if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Meta application for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Meta application for dataset {dataset['dataset_id']} ...")
            
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')
            models = df_metrics.columns

            df_features_path = ABLATION_FEATURES_SELECT

            for feature_path in df_features_path.glob(f"*{dataset['dataset_id']}*.csv"):
                print(f"Processing {feature_path.name} ...")
                df_features = pd.read_csv(feature_path, index_col='id')

                output_name = feature_path.stem.split("#")[-1]

                features = df_features.columns
                
                df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

                # join features and mae
                df = df_features.join(df_metrics, how = 'inner')

                print(df.head())
                
                pred = train_and_predict(df, dataset_id = dataset['dataset_id'], group_split=dataset['group_split'], features=features, models=models)

                pred = pred.dropna()

                pred = pred.astype(float)
                print(pred, pred.dtypes)

                print(f"[{datetime.now()}] Done")

                if task["objective"] == "maximize":
                    best_models = pred.idxmax(axis=1).to_frame('model')
                else:
                    best_models = pred.idxmin(axis=1).to_frame('model')
                
                dest_path = ABLATION_FEATURES_PRED
                dest_path.mkdir(exist_ok=True, parents=True)

                best_models.to_csv(dest_path / f"{output_name}.csv")






