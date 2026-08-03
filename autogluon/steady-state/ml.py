import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def _resample(df):
    # Sample size per class
    sample_size_per_class = 50
    
    # get target column
    y = df['y']

    # get value counts
    value_counts = pd.Series(y).value_counts()

    # resample if necessary
    for class_ in [0, 1]:
        if value_counts.loc[class_] < sample_size_per_class:
            # oversample minority class
            oversample = RandomOverSampler(sampling_strategy={class_: sample_size_per_class}, random_state=42)
            df, y = oversample.fit_resample(df, y)
        elif value_counts.loc[class_] > sample_size_per_class:
            # undersample majority class
            undersample = RandomUnderSampler(sampling_strategy={class_: sample_size_per_class}, random_state=42)
            df, y = undersample.fit_resample(df, y)

    return df

def resample(df):
    # resample each fork
    return df.groupby(['benchmark_id', 'no_fork'], group_keys=False).apply(_resample)

# split dataset into train and test sets
def split(df, test_size=0.2, random_state=42):   
    # get benchmarks
    benchmarks = df['benchmark_id'].unique()
    benchmarks.sort()
    
    # create dataframe
    benchmarks = pd.DataFrame(benchmarks, columns=['benchmark_id'])
    # add project column
    benchmarks['project'] = benchmarks['benchmark_id'].str.split('#').str[0]
    
    # split into train and test benchmarks
    train_bench, test_bench = train_test_split(benchmarks, test_size=test_size, random_state=random_state, stratify=benchmarks['project'])

    # split into train and test according to benchmarks split
    train = df[df['benchmark_id'].isin(train_bench['benchmark_id'])]
    test = df[df['benchmark_id'].isin(test_bench['benchmark_id'])]
    
    return train, test


# transform dataframe into X and y numpy arrays
def extract_features(df):
    y = df['y'].to_numpy()
    X = df[["x{}".format(i) for i in range(100)]].to_numpy()
    return X.astype('float32'), y.astype(int)


# Custom K-Fold cross validation
class CustomKFold:
    def __init__(self, df, k=5):
        self.k = k
        self.df = df
        # get benchmarks 
        benchmarks = df['benchmark_id'].unique()
        benchmarks.sort()
        # create folds
        self.folds = self.__create_folds(benchmarks)

    def __kfold(self, df):
        print(df)
        kf = KFold(n_splits= self.k, shuffle=True, random_state=42)    
        # add fold column
        df.insert(2, 'fold', None)

        nofold = 0

        for _, test_index in kf.split(df):
            df.iloc[test_index, 2] = nofold
            nofold += 1

        return df

    def __create_folds(self, benchmarks):
        # create dataframe for folding
        df = pd.DataFrame(benchmarks, columns=['benchmark_id'])
        df['project'] = df['benchmark_id'].str.split('#').str[0]
        print(df['project'].unique())
        df = df.groupby('project', group_keys=False).apply(lambda x: self.__kfold(x))
        df = df[['benchmark_id', 'fold']]
        return df
    

    
    def iter(self):
        for fold in range(self.k):
            test_benchmarks = self.folds[self.folds['fold'] == fold]['benchmark_id'].unique()
            mask = self.df['benchmark_id'].isin(test_benchmarks)

            train = self.df[~mask]
            test = self.df[mask]
            yield train, test

    def save_folds(self, path):
        self.folds.to_csv(path, index=False)

    def __len__(self):
        return self.k