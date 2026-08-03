#!/bin/bash

# activate FLAML env

python -u flaml_tsf_fit.py 2>&1 | tee logs/flaml_tsf_fit.log
python -u flaml_tsf_predict.py 2>&1 | tee logs/flaml_tsf_predict.log

python -u flaml_tsad.py 2>&1 | tee logs/flaml_tsad.log

python -u flaml_ssd_fit.py 2>&1 | tee logs/flaml_ssd_fit.log
python -u flaml_ssd_predict.py 2>&1 | tee logs/flaml_ssd_predict.log