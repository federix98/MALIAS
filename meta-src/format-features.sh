#!/bin/bash

mkdir -p logs

python format.py --task anomaly-detection 2>&1 | tee logs/anomaly_detection_format.log
python format.py --task timeseries-forecasting 2>&1 | tee logs/timeseries_forecasting_format.log