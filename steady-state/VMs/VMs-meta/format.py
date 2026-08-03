import pandas as pd
import numpy as np

if __name__ == "__main__":

    # df = pd.read_csv("models_metrics_forkwise_eval.csv")

    # id_cols = ['clf', 'benchmark', "no_fork"]
    # metric_cols = [col for col in df.columns if col not in id_cols + ['fold']]

    # # Aggregating by ID columns using mean
    # aggregated_df = df.groupby(id_cols)[metric_cols].mean().reset_index()
    # aggregated_df["id"] = aggregated_df["benchmark"] + '---' + aggregated_df["no_fork"].astype(str)
    
    # aggregated_df = aggregated_df[["id", "clf", "bal_acc"]]

    # aggregated_df = aggregated_df.pivot_table(
    #     values = "bal_acc",
    #     index = "id",
    #     columns = ["clf"]
    # )

    # aggregated_df.to_csv("bal_acc.csv")

    # df_features = pd.read_csv('extracted_features.csv')
    
    # # drop 'kind' since it has unique value and 'level_2' columns
    # df_features.drop(columns=["kind", "level_2"], inplace=True)

    # df_features = df_features.pivot_table(
    #     values = "value",
    #     index = "id",
    #     columns = ["variable"]
    # )

    # # remove null features
    # df_features = df_features.dropna(axis=1, how='any')

    # df_features.to_csv("all_features.csv")

    # ============================================
    # PERFORM RANDOM ASSIGNMENT OF FOLDS:
    # Each fork (time series) is paired with the 
    # statistical features extracted from another 
    # randomly selected fork within the same 
    # benchmark and project.
    # ============================================

    df_features = pd.read_csv("all_features.csv")

    _tmp_df = df_features[["id"]].copy()
    _tmp_df["benchmark_id"] = _tmp_df["id"].str.split("---").str[0]
    _tmp_df["no_fork"] = _tmp_df["id"].str.split("---").str[1]

    # Create a helper dictionary: for each b, list of available f's
    b_to_f_dict = _tmp_df.groupby('benchmark_id')['no_fork'].apply(list).to_dict()

    mapping = []
    for benchmark, forks in b_to_f_dict.items():
        for fork in forks:
            candidates = [_f for _f in forks if fork != _f]

            if not candidates:
                raise ValueError(f"No other 'f' to choose from for benchmark = {benchmark}, fork = {fork}")
            
            f_random = np.random.choice(candidates)
            mapping.append({
                "id": f"{benchmark}---{fork}",
                "mapped_to": f"{benchmark}---{f_random}"
            })

    df_mapping = pd.DataFrame(mapping)
    df_mapping.to_csv("mapping.csv", index=False)

    assert not (df_mapping['id'] == df_mapping['mapped_to']).any(), "Problem with mapping (id = mapped_to)"

    df_features.rename(columns = {"id": "mapped_to"}, inplace=True)
    df_mapping = pd.merge(df_mapping, df_features, on="mapped_to").drop(columns="mapped_to")
    df_mapping.to_csv("mapped_features.csv", index=False)