#!/bin/bash

python extract_features.py

python format.py

python feature_selection.py

python hp_tuning.py

python predict_meta.py