#!/bin/bash

kill_tree() {
    local parent=$1
    for child in $(pgrep -P "$parent"); do
        kill_tree "$child"
    done
    kill -9 "$parent" 2>/dev/null
}

bash script.sh &> logs/script_$(date +%Y-%m-%d).log &
wrapper_pid=$!

t_sec=100
echo "Tracer wrapper PID: $wrapper_pid. You have $t_sec seconds to confirm (y/n):"

if read -t $t_sec -r answer; then
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        echo "Confirmed, letting process $wrapper_pid run."
        wait $wrapper_pid
        exit $?
    else
        echo "Not confirmed. Killing process $wrapper_pid."
        kill_tree $wrapper_pid
        exit 1
    fi
else
    echo "No input within $timeout_sec seconds. Killing process $wrapper_pid."
    pkill -9 -u $USER
    exit 1
fi