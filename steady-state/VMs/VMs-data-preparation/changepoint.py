import json
from math import log

import numpy as np
import rpy2
import rpy2.interactive.packages
import rpy2.robjects
from kneed import KneeLocator

def find_penalty(measurements):
    return 15 * log(len(measurements))


def pelt(ts):
    penalty = find_penalty(ts)
    cpt = rpy2.interactive.packages.importr('changepoint')
    measurements = rpy2.robjects.FloatVector(ts)
    changepoints = cpt.cpt_meanvar(measurements, method='PELT', penalty='Manual',
                                   pen_value=penalty)
    # List indices in R start at 1.
    return [int(cpoint - 1) for cpoint in changepoints.slots['cpts']]


def changepoint(ts_path, filtered_path, changepoints_path):
    with open(ts_path) as f_ts,  open(filtered_path) as f_filtered, open(changepoints_path, 'w') as f_cpts:
        ts_list = json.load(f_ts)
        filtered = json.load(f_filtered)
        results = []

        for ts, indexes in zip(ts_list, filtered):
            ts = np.array(ts)[indexes].tolist()
            # detect changepoints indexes
            cpts_ = pelt(ts)
            # lead back indexes to series without outliers
            cpts_ = np.array(indexes)[cpts_].tolist()

            results.append(cpts_)

        json.dump(results, f_cpts)

