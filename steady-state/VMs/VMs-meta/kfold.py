import numpy as np
from sklearn.model_selection import StratifiedKFold, PredefinedSplit   


def create_predefined_split(groups, split = 'test'):

    random_state = 42
    n_splits = 3

    test_fold = np.repeat(-1, len(groups))
    rng = np.random.default_rng(random_state)

    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for i, (_, test_index) in enumerate(kf.split(groups, groups)):

        selected_validation_index = rng.choice(test_index, size=len(test_index) // 2, replace=False)
        selected_test_index = np.setdiff1d(test_index, selected_validation_index)

        if split == 'test':
            test_fold[selected_test_index] = i
        elif split == 'val':
            test_fold[selected_validation_index] = i
        else:
            raise Exception("Unsupported value for 'split' parameter.")
    

    # Ensure we have at least one fold index per fold
    unique_folds = set(test_fold[test_fold != -1])
    assert len(unique_folds) == n_splits, "Not all folds have test samples!"

    ps = PredefinedSplit(test_fold)
    
    return ps