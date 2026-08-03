from datetime import datetime
import time
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score
import pandas as pd

from rocket import Rocket
from oscnn import OSCNN
from fcn import FCN
from loader import load_dataset, lightload_dataset
from ml import resample, split, extract_features
from ml import CustomKFold
from constants import FOLDS_PATH, EVAL_MODELS_METRICS_PATH, MODELS_PATH, RES_PATH, VALIDATION_SPLIT_SIZE, N_FOLDS, THRESHOLDS_PATH, PREDICTION_TEST_PATH

SAVE_PATH = RES_PATH + '/models_metrics_forkwise_eval.csv'

def create_classifiers():
    # Initialize the classifiers
    rocket_classifier = Rocket()
    OS_CNN_classifier = OSCNN()
    FCN_classifier = FCN()
    
    return [FCN_classifier, OS_CNN_classifier, rocket_classifier]

def evaluate(clf, df_test, threshold, fold):
    clf_name = clf.__class__.__name__

    fold_results = []
    for (benchmark, fork), _sub_df in df_test.groupby(["benchmark_id", "no_fork"]):

        print((clf.__class__.__name__, fold, benchmark, fork))

        # extract features
        x_test, y_test = extract_features(_sub_df)

        t1 = time.time()
        # predict
        y_probs = clf.predict_proba(x_test)

        t2 = time.time()

        y_pred = (y_probs >= threshold).astype(int)

        # compute metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        test_time = t2 - t1

        fold_results.append({'fold': fold, 'clf': clf.__class__.__name__, 'benchmark': benchmark, 'no_fork': fork, 'acc': acc, 'bal_acc': bal_acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'test_time': test_time})

    return fold_results

def main():
    # init results and moder folder
    for folder in [RES_PATH, MODELS_PATH]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # load the dataset
    print(f"[{datetime.now()}] Loading dataset...")
    df = load_dataset(steady_state_only=True, stratify=True)
    thresholds = pd.read_csv(THRESHOLDS_PATH, index_col=["model","fold"])
    print(f"[{datetime.now()}] Dataset loaded")

    # df_ = resample(df)

    # initialize the results list
    res =[]

    # initialize k-fold
    kf = CustomKFold(df, k=N_FOLDS)
    # save folds
    kf.save_folds(FOLDS_PATH)

    # Create predictions test directory if it does not exist
    os.makedirs(PREDICTION_TEST_PATH, exist_ok=True)

    for fold, (df_train, df_test) in enumerate(kf.iter()):

        # create the classifiers
        classifiers = create_classifiers()

        for clf in classifiers:
            # get the classifier name
            clf_name = clf.__class__.__name__

            # make the experiement reproducible
            clf.set_seeds(42)
            
            load_from = f"{MODELS_PATH}/{clf_name}_{fold}"
            # load the best model
            clf.load(load_from)
            
            # evaluate the model
            print(f"[{datetime.now()}] Testing {clf_name} on fold {fold} ...")

            if clf_name == "Rocket":
                threshold = 0.5
            else:
                threshold = thresholds.loc[(clf_name, fold), 'threshold']
            print(f"Threshold: {threshold} for ({clf_name}, {fold})")

            eval_metrics = evaluate(clf, df_test, threshold, fold)
            print(eval_metrics)
            print(f"[{datetime.now()}] Testing completed")

            # append the metrics to the results list
            res += eval_metrics

    # # Save the results
    pd.DataFrame(res).to_csv(SAVE_PATH, index=False)
    print(f"[{datetime.now()}] Metrics saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()