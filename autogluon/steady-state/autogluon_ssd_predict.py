from datetime import datetime
import time
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score, fbeta_score
import pandas as pd

from loader import load_dataset
from ml import extract_features
from ml import CustomKFold
from constants import FOLDS_PATH, MODELS_PATH, RES_PATH, N_FOLDS
import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from cst_utils import set_seeds


from AutoGluonWrapper import AutoGluonClassifier

SAVE_PATH = RES_PATH + '/models_metrics_forkwise_eval.csv'

def create_classifiers(log_dir):
    # Initialize the classifiers
    autogluon_classifier = AutoGluonClassifier(dir = log_dir)
    
    return [autogluon_classifier]

def evaluate(clf, df_test, fold):

    fold_results = []
    for (benchmark, fork), _sub_df in df_test.groupby(["benchmark_id", "no_fork"]):

        logging.info(f"Starting evaluation... {clf.__class__.__name__}, {fold}, {benchmark}, {fork}")

        # extract features
        x_test, y_test = extract_features(_sub_df)

        t1 = time.time()
        # predict
        y_probs = clf.predict_proba(x_test)

        t2 = time.time()

        y_pred = (y_probs >= .5).astype(int)

        # compute metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        fbeta_2 = fbeta_score(y_test, y_pred, beta=0.2)
        test_time = t2 - t1

        fold_results.append({'fold': fold, 'clf': clf.__class__.__name__, 'benchmark': benchmark, 'no_fork': fork, 'acc': acc, 'bal_acc': bal_acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'fbeta_2': fbeta_2, 'test_time': test_time})

    return fold_results

# python -u autogluon_ssd_predict.py 2>&1 | tee logs/autogluon_ssd_predict.log

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

    # df_ = resample(df)

    # initialize the results list
    res =[]

    # initialize k-fold
    kf = CustomKFold(df, k=N_FOLDS)
    # save folds
    kf.save_folds(FOLDS_PATH)

    for fold, (df_train, df_test) in enumerate(kf.iter()):

        # create the classifiers
        classifiers = create_classifiers(log_dir=None)

        for clf in classifiers:
            # get the classifier name
            clf_name = clf.__class__.__name__
            
            log_dir = f"{MODELS_PATH}/autogluon--VMD--fold#{fold}"
            # load the best model
            clf.load(log_dir)
            
            # evaluate the model
            print(f"[{datetime.now()}] Testing {clf_name} on fold {fold} ...")

            eval_metrics = evaluate(clf, df_test, fold)
            print(eval_metrics)
            print(f"[{datetime.now()}] Testing completed")

            # append the metrics to the results list
            res += eval_metrics

    # # Save the results
    pd.DataFrame(res).to_csv(SAVE_PATH, index=False)
    print(f"[{datetime.now()}] Metrics saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()