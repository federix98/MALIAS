import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import minmax_scale

from kfold import create_predefined_split


'''
Changes:
n_estimators: from np.arange(1000, 3001, 1000) to np.arange(100, 301, 100)
'''

def search_best_params(X, y, kf):
    seed = 42

    rf = RandomForestRegressor(random_state=seed)

    param_distributions = {
        'n_estimators': np.arange(100, 301, 100) ,        # number of trees
        'max_depth': [None, 10, 20],            # maximum depth of the tree
        'min_samples_split': [2, 5],            # minimum samples required to split
        'min_samples_leaf': [1, 2],             # minimum samples required at each leaf node
        # 'max_features': ['auto', 'sqrt', 'log2'],
        # 'bootstrap': [True, False]             # whether bootstrap samples are used
    }

    search = GridSearchCV(
        estimator=rf,
        param_grid=param_distributions,
        scoring='neg_mean_absolute_error',
        cv=kf,
        n_jobs=30,
        verbose=2)
    
    search.fit(X, y)

    best_params = search.best_params_
    best_params['n_estimators'] = int(best_params['n_estimators'])

    return best_params

if __name__ == '__main__':
    df_features = pd.read_csv('selected_features.csv', index_col='id')
    df_acc = pd.read_csv('bal_acc.csv', index_col='id')

    features = df_features.columns
    models = df_acc.columns

    # normalize mae
    df_acc = pd.DataFrame(minmax_scale(df_acc, axis=1), columns=models, index=df_acc.index)

    # join features and mae
    df = df_features.join(df_acc, how = 'inner')

    X = df[features] 
    y = df[models] 

    groups = df.index.str.split('_').str[0]
    kf = create_predefined_split(groups, split = 'val')

    best_params = search_best_params(X, y, kf)

    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)