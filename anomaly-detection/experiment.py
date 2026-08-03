from datetime import datetime

from typing import Dict
import numpy as np

import time

import pandas as pd

from EasyTSAD.Controller import TSADController

from EasyTSAD.Methods import AR, LSTMADalpha, LSTMADbeta, FCVAE, AE, FITS, EncDecAD

SAVE_PATH = "./experiments_metrics.csv"

def exp_instance(methods, dataset, schema):
    
    # if cfg_path is None, using default configuration
    gctrl = TSADController(cfg_path=None)

    # use the entire dataset
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="data/datasets", # The path to the parent directory of "UTS"
        datasets=dataset
    )

    methods = methods
    training_schema = schema

    for method in methods:
        print(f"[{datetime.now()}] Running method {method} on dataset {dataset}")
        # run models
        gctrl.run_exps(
            method=method,
            training_schema=training_schema
        )

if __name__ == "__main__":

    # initialize the results list
    res = []

    methods = ["AR", "LSTMADalpha", "FCVAE", "FITS", "EncDecAD"]

    experiments = [
        {
            "methods": methods,
            "schema": "naive",
            "dataset": "AIOPS"
        },
        {
            "methods": methods,
            "schema": "naive",
            "dataset": "WSD"
        }
    ]

    for experiment in experiments:

        # init metrics
        metrics = experiment

        # train the model
        print(f"[{datetime.now()}] Running AD experiment on dataset: {experiment['dataset']} ...")

        t1 = time.time()
        exp_instance(methods = experiment["methods"], dataset = experiment["dataset"], schema = experiment["schema"])
        t2 = time.time()
        exp_time = t2 - t1
        
        metrics["exp_time"] = exp_time
        print(f"[{datetime.now()}] AD experiment completed")

    # append the metrics to the results list
    res.append(metrics)

    # Save the results
    pd.DataFrame(res).to_csv(SAVE_PATH, index=False)
    print(f"[{datetime.now()}] Metrics saved to {SAVE_PATH}")