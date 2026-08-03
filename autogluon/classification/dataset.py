import pandas as pd
import numpy as np

import logging
# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)


import more_itertools as m_iter

from kfold import create_kfold

from cfg import WINDOW, HORIZON, TEST_SIZE, VAL_SIZE

def proc_data(ts, describe = False):

    ts = np.array(ts)

    train_ts, test_ts = split(ts, TEST_SIZE)
    val_ts, test_ts = split(test_ts, VAL_SIZE)

    if describe:
        logging.info(f"Time series shape: {ts.shape}")
        logging.info(f"(train, eval, test) time series shapes: ({train_ts.shape, val_ts.shape, test_ts.shape})")

    return {
        "ts": [train_ts, val_ts, test_ts],
    }

def windowed(ts, window, context, reshape = True):
    windowed_ts = list(m_iter.windowed(ts, n=window, step=1))
    X = pd.DataFrame(windowed_ts).iloc[:, 0:context].to_numpy()
    y = pd.DataFrame(windowed_ts).iloc[:, context:window].to_numpy()

    # useful for NN input shape
    if reshape:
        X = np.expand_dims(X, axis=-1)
    
    return X, y

def split(ts, test_size = .2):
    return ts[:-int(len(ts)*test_size)], ts[-int(len(ts)*test_size):]

def extract_features(df):
    window, horizon = WINDOW, HORIZON
    y = df[["y{}".format(i) for i in range(horizon)]].to_numpy()
    X = df[["x{}".format(i) for i in range(window-horizon)]].to_numpy()
    return X.astype('float32'), y.astype('float32')

def load_dataset(dataset_id):
    if dataset_id == "WSD_3k":
        return pd.read_csv("data/WSD_3k.csv")
    elif dataset_id == "AzFuncInvoke":
        return pd.read_csv("data/azure10min.csv")
    else:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
    

def get_fold_timeseries(df, group_split):

    ids = df.index

    kf = create_kfold(df = df, group_split = group_split, split = "test")

    for fold, (train_index, test_index) in enumerate(kf):
        logging.info(f"Processing fold {fold} ...")
        # for the final application, all the training set is used
        train_ids = ids[train_index]
        test_index = ids[test_index]
        
        # print(f"Fold {fold}: Train IDs {train_ids}, Test IDs {test_index}")

        # Only process the first fold for demonstration
        return train_ids, test_index
