import pandas as pd
from pathlib import Path

from datetime import datetime

from typing import Dict
import numpy as np

from tqdm import tqdm

from pathlib import Path

import time

import pandas as pd
from EasyTSAD.Evaluations.Protocols import EventKthPrcPA

import torch
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError

from sklearn.metrics import mean_squared_error

error_function = SymmetricMeanAbsolutePercentageError()

WINDOW, CONTEXT = 100, 50

def smape_degradation(y_pred, y_true):
    errors = {}
    for i in range(WINDOW - CONTEXT):
        err_params = (torch.tensor(y_true[f"y_{i}"]), torch.tensor(y_pred[f"y_pred_{i}"]))
        error_i = error_function(*err_params)
        errors[f"t+{i}"] = error_i.item() * 100 # percentage
        # print(error_i)
    return errors

def degradation(err_fn, y_pred, y_true):
    errors = {}
    for i in range(WINDOW - CONTEXT):
        # print(f"Computing error for step {i}")
        err_params = (y_true[f"y_{i}"], y_pred[f"y_pred_{i}"])
        # print(err_params)
        error_function = globals()[err_fn]
        error_i = error_function(*err_params)
        errors[f"t+{i}"] = error_i
        # print(error_i)
    return errors

if __name__ == "__main__":

    save_path = Path("results/FLAML")
    save_path.mkdir(exist_ok=True, parents=True)

    ### TIMESERIES FORECASTING EXPERIMENT ###

    FLAML_pred_df = pd.concat(
        [pd.read_csv(pred_file) for pred_file in Path(f"./results/pred/forecasting/").glob("*AzFuncInvoke*.csv")]
    )

    FLAML_pred_df["dataset"] = "azure10min"

    res = []

    for (dataset, ts, model), sub_df in tqdm(FLAML_pred_df.groupby(["dataset", "ts", "model"]), desc="Computing errors"):
        res_block = {"dataset": dataset, "ts": ts, "model": model, "model": model}

        y_pred = sub_df[[col for col in sub_df.columns if col.startswith("y_pred")]]
        y_true = sub_df[[col for col in sub_df.columns if col.startswith("y_") and "pred" not in col]]

        # print("Computing error for", (dataset, ts, model))

        degradation_errors = smape_degradation(y_pred = y_pred, y_true = y_true)
        res_block.update(degradation_errors)

        res.append(res_block)

    res_df = pd.DataFrame(res)

    no_cols = res_df.melt(id_vars=['dataset', 'ts', 'model'], value_name="smape", var_name='offset')

    mean_smape = no_cols.groupby(['dataset', 'ts', 'model']).smape.mean().reset_index()
    median_smape = no_cols.groupby(['dataset', 'ts', 'model']).smape.median().reset_index()
    formatted = mean_smape.pivot(index=['dataset', 'ts'], columns='model', values="smape").reset_index()

    models_to_include = ['FLAML']


    formatted[["ts"] + models_to_include].rename(columns = {"ts": "id"}).to_csv(save_path / "Azure_metrics_FLAML.csv", index=False)

    # ### ANOMALY DETECTION EXPERIMENT ###

    # iterate through predictions
    results = []
    for pred_path in Path("./results/pred/anomaly-detection").glob("*.csv"):
        print("Analyzing ", pred_path)
        pred_df = pd.read_csv(pred_path)
        ts_id = pred_path.stem.replace("FLAML-AIOps-", "")
        k = 3
        metric = EventKthPrcPA(k=k, mode='squeeze')
        scores = pred_df.y_pred_proba.values
        y_true = pred_df.y_true.values
        result = metric.calc(scores, y_true, margins=(0, 5)).to_dict()['3-th auprc under event-based pa with mode squeeze'] # margins (0, 5) used in the tool
        results.append({
            'id': ts_id,
            'FLAML': result
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path / "AIOps_metrics_FLAML.csv", index=None) 


    ## STEADY STATE EXPERIMENT ###
    FLAML_SSD_path = "./ssd_results/models_metrics_forkwise_eval.csv"

    ag_df = pd.read_csv(FLAML_SSD_path)

    id_cols = ['clf', 'benchmark', "no_fork"]
    metric_cols = [col for col in ag_df.columns if col not in id_cols + ['fold']]

    aggregated_df = ag_df.groupby(id_cols)[metric_cols].mean().reset_index()
    aggregated_df["id"] = aggregated_df["benchmark"] + '---' + aggregated_df["no_fork"].astype(str)

    aggregated_df = aggregated_df[["id", "clf", "fbeta_2"]]

    aggregated_df = aggregated_df.pivot_table(
        values = "fbeta_2",
        index = "id",
        columns = ["clf"]
    )

    ag_df = aggregated_df.reset_index()
    ag_df.rename(columns = {'FLAMLSteadyStateClassifier': "FLAML"}, inplace=True)
    ag_df.to_csv(save_path / "VMD_metrics_FLAML.csv", index=None)