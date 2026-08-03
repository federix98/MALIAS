from datetime import datetime
import time
import os

import pandas as pd

from FLAMLWrapper import FLAMLSteadyStateClassifier

from loader import load_dataset
from ml import split, extract_features
from ml import CustomKFold
from constants import FOLDS_PATH, FIT_MODELS_METRICS_PATH, MODELS_PATH, RES_PATH

from cst_utils import set_seeds

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def create_classifiers(fold):
    # Initialize the classifiers
    FLAML_classifier = FLAMLSteadyStateClassifier(fold = fold)
    
    return [FLAML_classifier]

def train(clf, df_train):
    t1 = time.time()
    x_train, y_train = extract_features(df_train)

    logging.info(f"Starting training... Shapes of X_train and y_train: {x_train.shape} - {y_train.shape}")

    # fit the model
    clf.fit(x_train, y_train)
    t2 = time.time()
    train_time = t2 - t1

    return train_time

# python -u flaml_ssd_fit.py 2>&1 | tee logs/flaml_ssd_fit.log

def main():

    set_seeds(42)
    # init results and moder folder
    for folder in [RES_PATH, MODELS_PATH]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # load the dataset
    logging.info(f"Loading dataset...")
    df = load_dataset(steady_state_only=True, stratify=True)
    logging.info(f"Dataset loaded")

    logging.info(str(df.head()) + "\n\n" + str(df["benchmark_id"].unique()))

    # initialize the results list
    res = []

    # initialize k-fold
    kf = CustomKFold(df, k=5)
    # save folds
    kf.save_folds(FOLDS_PATH)

    for fold, (df_train, _test) in enumerate(kf.iter()):
        # create the classifiers
        dump_path = f"{MODELS_PATH}/autogluon--VMD--fold#{fold}.pkl"
        classifiers = create_classifiers(fold)
        for clf in classifiers:
            # get the classifier name
            clf_name = clf.__class__.__name__

            # init metrics dict
            metrics = {"fold": fold, "clf": clf_name}

            # train the model
            logging.info(f"Training {clf_name} on fold {fold} ...")
            train_time = train(clf, df_train)
            metrics["train_time"] = train_time
            logging.info(f"Training completed")

            clf.dump(path = dump_path)
            logging.info(f"Model saved to {dump_path}")

            # append the metrics to the results list
            res.append(metrics)

    # Save the results
    pd.DataFrame(res).to_csv(FIT_MODELS_METRICS_PATH, index=False)
    logging.info(f"Metrics saved to {FIT_MODELS_METRICS_PATH}")


if __name__ == "__main__":
    main()
