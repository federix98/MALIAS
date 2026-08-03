#!/bin/bash

mkdir -p logs/ablation
mkdir -p results/ablation/meta


# Ablation Feature Selection

python -u ablation_feature_selection.py 2>&1 --task anomaly-detection | tee logs/ablation/ablation_feature_selection_AD.log
python -u ablation_feature_selection.py 2>&1 --task timeseries-forecasting | tee logs/ablation/ablation_feature_selection_TSF.log
python -u ablation_feature_selection.py 2>&1 --task steady-state | tee logs/ablation/ablation_feature_selection_SSD.log

python -u ablation_feature_predict.py 2>&1 --task anomaly-detection | tee logs/ablation/ablation_feature_predict_AD.log
python -u ablation_feature_predict.py 2>&1 --task timeseries-forecasting | tee logs/ablation/ablation_feature_predict_TSF.log
python -u ablation_feature_predict.py 2>&1 --task steady-state | tee logs/ablation/ablation_feature_predict_SSD.log

python -u ablation_feature_eval.py 2>&1 --task anomaly-detection | tee logs/ablation/ablation_feature_eval_AD.log
python -u ablation_feature_eval.py 2>&1 --task timeseries-forecasting | tee logs/ablation/ablation_feature_eval_TSF.log
python -u ablation_feature_eval.py 2>&1 --task steady-state | tee logs/ablation/ablation_feature_eval_SSD.log


# Ablation Model Selection

##### ========= Without Hyperparameter Tuning ========= #####
# python -u ablation_model.py 2>&1 --task anomaly-detection | tee logs/ablation/ablation_model_AD.log
# python -u ablation_model.py 2>&1 --task timeseries-forecasting | tee logs/ablation/ablation_model_TSF.log
# python -u ablation_model.py 2>&1 --task steady-state | tee logs/ablation/ablation_model_SSD.log

##### ========= With Hyperparameter Tuning ========= #####
python -u ablation_model_with_hp.py 2>&1 --task anomaly-detection | tee logs/ablation/ablation_model_with_hp_AD.log
python -u ablation_model_with_hp.py 2>&1 --task timeseries-forecasting | tee logs/ablation/ablation_model_with_hp_TSF.log
python -u ablation_model_with_hp.py 2>&1 --task steady-state | tee logs/ablation/ablation_model_with_hp_SSD.log

python -u ablation_model_eval.py 2>&1 --task anomaly-detection --tuning | tee logs/ablation/ablation_model_with_hp_eval_AD.log
python -u ablation_model_eval.py 2>&1 --task timeseries-forecasting --tuning | tee logs/ablation/ablation_model_with_hp_eval_TSF.log
python -u ablation_model_eval.py 2>&1 --task steady-state --tuning | tee logs/ablation/ablation_model_with_hp_eval_SSD.log