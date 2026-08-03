#!/bin/bash

# !!! execute the krun-ext.ipynb notebook before to populate the data/timeseries/all folder.

python -u filter_outliers.py &> logs/filter_outliers.log # remove outliers following the approach presented in Tratt's paper

python -u changepoint_analysis.py &> logs/changepoint_analysis.log # detect changepoint indexes

python -u classify_runs.py &> logs/classify_runs.log # classify forks in [steady state | no steady state] and benchmarks

python -u rename.py &> logs/rename.log # rename files and exclude PyPy