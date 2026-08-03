#!/bin/bash

mkdir -p logs/generalize

run_scripts_by_task() {
    local task_name="$1"

    # python -u gentest.py --task ${task_name} 2>&1 | tee logs/generalize/${task_name}___gentest.log
    python -u gentest_eval.py --task ${task_name} 2>&1 | tee logs/generalize/${task_name}___gentest_eval.log
}

run_scripts_by_task anomaly-detection
run_scripts_by_task timeseries-forecasting
run_scripts_by_task steady-state
