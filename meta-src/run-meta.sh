#!/bin/bash

# Experiment notes:
# - the scripts are execution specifically for the specified task (e.g. anomaly-detection)
# - Once a script is executed, all the datasets of the specified task are considered
# - Refer to config/config.json file for the experiment configuration

mkdir -p logs

run_scripts_by_task() {
    local task_name="$1"

    # python -u assert_model_heterogeneity.py --task ${task_name} 2>&1 | tee logs/${task_name}___assert_model_heterogeneity.log
    # python -u feature_selection.py --task ${task_name} 2>&1 | tee logs/${task_name}___feature_selection.log
    python -u hp_tuning.py --task ${task_name} 2>&1 | tee logs/${task_name}___hp_tuning.log
    python -u predict_meta.py --task ${task_name} 2>&1 | tee logs/${task_name}___predict_meta.log
    # python -u evaluate.py --task ${task_name} 2>&1 | tee logs/${task_name}___evaluate.log
    # python -u gentest.py --task ${task_name} 2>&1 | tee logs/${task_name}___gentest.log
    # python -u gentest_eval.py --task ${task_name} 2>&1 | tee logs/${task_name}___gentest_eval.log
}

run_scripts_by_task anomaly-detection
# run_scripts_by_task timeseries-forecasting
# run_scripts_by_task steady-state
