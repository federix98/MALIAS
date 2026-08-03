from glob import glob
import re
import json

import numpy as np

from constants import TS_PATH, CLAS_PATH

def get_light_benchmark_list(cut=10):
    benchmarks = []
    cnt = 0
    for path in glob(CLAS_PATH + '/*.json'):
        benchmark = re.sub(r'\.json$', '', path.split('/')[-1])
        benchmarks.append(benchmark)
        cnt += 1
        if cnt >= cut:
            break

    return np.unique(benchmarks).tolist()

def get_benchmark_list():
    benchmarks = []
    for path in glob(CLAS_PATH + '/*.json'):
        benchmark = re.sub(r'\.json$', '', path.split('/')[-1])
        benchmarks.append(benchmark)

    return np.unique(benchmarks).tolist()


def load_timeseries(benchmark):
    return load(TS_PATH, benchmark)

def load_classification(benchmark):
    return load(CLAS_PATH, benchmark)


def load(dirpath, benchmark):
    path = '{}/{}.json'.format(dirpath, benchmark)
    with open(path) as f:
        return json.load(f)


def foreach_fork(ts, classification, steady_state_only=False):
    for f, ts_ in enumerate(ts):
        c = classification['forks'][f]
        st = classification['steady_state_starts'][f]
        if steady_state_only is False:
            yield f, ts_, st
        elif c == 'steady state':
            yield f, ts_, st


def get_benchmark_name(path):
    filename = path.split('/')[-1]
    return re.sub('\.csv$', '', filename)