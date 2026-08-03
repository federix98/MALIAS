#!/bin/bash

mkdir -p ../logs

### WSD Experiment ###
DATASET_NAME=WSD_3k
python -u tuning.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/tuning_${DATASET_NAME}.log
python -u stat_exp.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/stat_exp_${DATASET_NAME}.log
python -u rnn_exp.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/rnn_exp_${DATASET_NAME}.log
python -u pretrained.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/pre_trained_${DATASET_NAME}.log

### AFD ###
DATASET_NAME=AFD
python -u tuning.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/tuning_${DATASET_NAME}.log
python -u stat_exp.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/stat_exp_${DATASET_NAME}.log
python -u rnn_exp.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/rnn_exp_${DATASET_NAME}.log
python -u pretrained.py 2>&1 --dataset-path ../data/${DATASET_NAME}.csv | tee ../logs/pre_trained_${DATASET_NAME}.log