import json
import argparse
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import minmax_scale

from kfold import create_kfold

import joblib

from cfg import CONFIG_PATH, LOAD_FOLDER, GENTEST_RESULTS

# how to map training to testing datasets
dataset_mapping = {
    "jmh": "VMs",
    "VMs": "jmh",
    "AzFuncInvoke": "WSD_3k",
    "WSD_3k": "AzFuncInvoke",
    "AIOps": "WSD",
    "WSD": "AIOps"
}

def train_and_predict(df, dataset_id, group_split, features, models):

    print("features=", features)
    
    ids = df.index

    kf = create_kfold(df = df, group_split = group_split, split = "test")

    pred = pd.DataFrame(index=ids, columns=models)

    for fold, (_, test_index) in enumerate(kf):
        test_ids = ids[test_index]

        X_test = df.loc[test_ids, features]

        load_filename = f"{dataset_mapping[dataset_id]}___{fold}.joblib"
        rf = joblib.load(LOAD_FOLDER / load_filename)
        
        print(f"Using checkpoint {load_filename} for {dataset_id} ...")
        y_pred = rf.predict(X_test)

        pred.loc[test_ids, models] = y_pred

    return pred


if __name__ == '__main__':

    # --- args ---
    parser = argparse.ArgumentParser(description="Tuning of SARIMA parameters based on the training time series")
    parser.add_argument('--task', default='timeseries-forecasting', type=str, required=True)
    args = parser.parse_args()

    assert LOAD_FOLDER.exists(), "Models checkpoints not found."

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Generalization for task {task['task_type']} ...")

        for dataset in task['datasets']:

            print(f"[{datetime.now()}] Generalization for dataset {dataset['dataset_id']} ...")

            # loading mapped features
            other_dataset = [d for d in task['datasets'] if d != dataset][0]
            print("current", dataset['dataset_id'], "other_dataset", other_dataset['dataset_id'])

            other_features = pd.read_csv(other_dataset['selected_feature_path'], index_col='id').columns

            print(f"Using features of {other_dataset['selected_feature_path']} for {dataset['dataset_id']} ...")

            df_features = pd.read_csv(dataset['feature_path'], index_col='id')[other_features]
            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            features = df_features.columns
            models = df_metrics.columns

            # normalize mae
            df_metrics = pd.DataFrame(minmax_scale(df_metrics, axis=1), columns=models, index=df_metrics.index)

            # join features and mae
            df = df_features.join(df_metrics, how = 'inner')
            
            pred = train_and_predict(df, dataset_id = dataset['dataset_id'], group_split=dataset['group_split'], features=features, models=models)

            pred = pred.dropna()

            pred = pred.astype(float)
            print(pred, pred.dtypes)

            print(f"[{datetime.now()}] Done")

            if task["objective"] == "maximize":
                best_models = pred.idxmax(axis=1).to_frame('model')
            else:
                best_models = pred.idxmin(axis=1).to_frame('model')

            GENTEST_RESULTS.mkdir(exist_ok=True, parents=True)
            
            best_models.to_csv(GENTEST_RESULTS / f"meta_{dataset['dataset_id']}.csv")