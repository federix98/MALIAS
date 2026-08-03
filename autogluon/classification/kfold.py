import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, KFold   

np.set_printoptions(threshold=sys.maxsize)

def save_folds_to_csv(split_iterator, out_path="fold_splits.csv"):
    """
    Save the train, val, and test indices from a split_iterator to a CSV file.

    Parameters:
    - df (pd.DataFrame): The original dataset.
    - split_iterator: An iterator that yields (train_index, val_index) or (train_val_index, test_index)
    - split (str): Either "val" (train/val) or "test" (train_val/test).
    - out_path (str): Destination CSV file path.
    """
    records = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(split_iterator):
        for idx in train_idx:
            records.append({"fold": fold, "index": idx, "split": "train"})
        for idx in val_idx:
            records.append({"fold": fold, "index": idx, "split": "val"})
        for idx in test_idx:
            records.append({"fold": fold, "index": idx, "split": "test"})

    fold_df = pd.DataFrame(records)
    fold_df.to_csv(out_path, index=False)
    print(f"Saved fold assignments to: {out_path}")

def create_kfold(df, group_split = None, split = "val"):
    if group_split:
        print(f"CustomKFold > Creating stratified splits (splitting token: '{group_split}')")
        groups = df.index.str.split(group_split).str[0]
        kf = create_predefined_split(groups, split = split)
    else:
        print("CustomKFold > Creating splits (no stratification)")
        kf = create_predefined_split(df.index, split = split, grouped=False)
    return kf

############## DIVIDE THE TRAINING SET TO CREATE THE VALIDATION SET ##################

def create_predefined_split(groups, split = "test", val_ratio=0.2, grouped=True):

    random_state = 42
    rng = np.random.default_rng(random_state)

    n_splits = 5

    if grouped:
        kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        split_iterator = kf.split(groups, groups)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = kf.split(groups)

    for i, (train_val_index, test_index) in enumerate(split_iterator):
        # Split train_val into train and validation
        shuffled_train_val_idx = rng.permutation(train_val_index)
        n_val = int(val_ratio * len(shuffled_train_val_idx))

        val_index = shuffled_train_val_idx[:n_val]
        train_index = shuffled_train_val_idx[n_val:]

        if split == "val":
            yield train_index, val_index
        elif split == "test":
            # test case: return training and validation together in train_val_index
            yield train_val_index, test_index
        else:
            # all
            yield train_index, val_index, test_index