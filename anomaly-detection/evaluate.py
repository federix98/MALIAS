from datetime import datetime

from typing import Dict
import numpy as np

import time

import pandas as pd

from EasyTSAD.Controller import TSADController

from EasyTSAD.Methods import AR, LSTMADalpha, LSTMADbeta, FCVAE, AE, FITS, EncDecAD
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA, EventKthF1PA, EventKthPrcPA, PointAuprcPA, EventPrcPA, PointKthF1PA

def eval_instance(methods, dataset, schema):
    
    # if cfg_path is None, using default configuration
    gctrl = TSADController(cfg_path=None)

    
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

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
        gctrl.set_evals(
            [
                PointF1PA(),
                EventF1PA(),
                EventF1PA(mode="squeeze"),
                
                PointKthF1PA(3),
                PointKthF1PA(10),
                PointKthF1PA(20),
                PointKthF1PA(50),
                PointKthF1PA(150),
                PointAuprcPA(),
                EventKthPrcPA(3, mode="raw"),
                EventKthPrcPA(10, mode="raw"),
                EventKthPrcPA(20, mode="raw"),
                EventKthPrcPA(50, mode="raw"),
                EventKthPrcPA(150, mode="raw"),
                
                EventKthF1PA(3, mode="log", base=3),
                EventKthF1PA(10, mode="log", base=3),
                EventKthF1PA(20, mode="log", base=3),
                EventKthF1PA(50, mode="log", base=3),
                EventKthF1PA(150, mode="log", base=3),
                EventPrcPA(mode="log", base=3),
                EventKthPrcPA(3, mode="log", base=3),
                EventKthPrcPA(10, mode="log", base=3),
                EventKthPrcPA(20, mode="log", base=3),
                EventKthPrcPA(50, mode="log", base=3),
                EventKthPrcPA(150, mode="log", base=3),
                
                EventKthF1PA(3, mode="squeeze"),
                EventKthF1PA(10, mode="squeeze"),
                EventKthF1PA(20, mode="squeeze"),
                EventKthF1PA(50, mode="squeeze"),
                EventKthF1PA(150, mode="squeeze"),
                EventPrcPA(mode="squeeze"),
                EventKthPrcPA(3, mode="squeeze"),
                EventKthPrcPA(10, mode="squeeze"),
                EventKthPrcPA(20, mode="squeeze"),
                EventKthPrcPA(50, mode="squeeze"),
                EventKthPrcPA(150, mode="squeeze"),
            ]
        )

        for method in methods:
            gctrl.do_evals(
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
        print(f"[{datetime.now()}] Running AD evaluation on dataset: {experiment['dataset']} ...")

        eval_instance(methods = experiment["methods"], dataset = experiment["dataset"], schema = experiment["schema"])

        print(f"[{datetime.now()}] AD evaluation completed")