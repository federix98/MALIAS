import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from kfold import create_predefined_split


'''
Changes:
- kfold: from 10 to 3 splits
- random foreest estimators: from 1000 to 10
- RFECV step: from 1 to 20
'''

def select_features(X, y, kf):
    seed = 42
    rf = RandomForestRegressor(n_estimators=10, random_state=seed, n_jobs=5)
    # bal_acc -> mean_absolute_error
    rfe = RFECV(estimator=rf, min_features_to_select=10, step=20, cv=kf, scoring='neg_mean_absolute_error', n_jobs=20, verbose=2)

    rfe.fit(X, y)

    return features[rfe.support_]


if __name__ == '__main__':

    df_features = pd.read_csv('all_features.csv', index_col='id')
    df_acc = pd.read_csv('bal_acc.csv', index_col='id')

    features = df_features.columns
    models = df_acc.columns

    # normalize mae
    df_acc = pd.DataFrame(minmax_scale(df_acc, axis=1), columns=models, index=df_acc.index)

    # join features and mae
    df = df_features.join(df_acc, how = 'inner')

    print(df.head())

    groups = df.index.str.split('#').str[0]
    kf = create_predefined_split(groups, split = 'val')

    features_ = select_features(X=df[features], y=df[models], kf=kf)

    df[features_].to_csv('selected_features.csv')