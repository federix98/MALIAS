import json

import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor

from kfold import create_predefined_split


def load_best_params():
    with open('best_params.json', 'r') as file:
        best_params = json.load(file)
    return best_params


def train_and_predict(df):
    seed = 42

    ids = df.index

    best_params = load_best_params()

    groups = df.index.str.split('_').str[0]
    kf = create_predefined_split(groups, split = 'test')

    pred = pd.DataFrame(index=ids, columns=models)

    for train_index, test_index in kf.split(ids):
        train_ids = ids[train_index]
        test_ids = ids[test_index]


        X_train = df.loc[train_ids, features]
        y_train = df.loc[train_ids, models]

        X_test = df.loc[test_ids, features]

        rf = RandomForestRegressor(**best_params, random_state=seed, n_jobs=30)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        pred.loc[test_ids, models] = y_pred

    return pred


if __name__ == '__main__':

    df_features = pd.read_csv('selected_features.csv', index_col='id')
    df_acc = pd.read_csv('bal_acc.csv', index_col='id')

    features = df_features.columns
    models = df_acc.columns

    # normalize mae
    df_acc = pd.DataFrame(minmax_scale(df_acc, axis=1), columns=models, index=df_acc.index)

    # join features and mae
    df = df_features.join(df_acc, how = 'inner')
    
    pred = train_and_predict(df)

    pred = pred.dropna()

    pred = pred.astype(float)
    print(pred, pred.dtypes)

    best_models = pred.idxmax(axis=1).to_frame('model')
    best_models.to_csv('meta.csv')






