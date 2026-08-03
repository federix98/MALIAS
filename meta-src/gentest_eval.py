import pandas as pd
import json
from datetime import datetime
import argparse

from cfg import CONFIG_PATH, GENTEST_EVAL, GENTEST_RESULTS

from evaluate import get_avg_rank, friedman, pairwise_wilcoxon
from visualization import stacked_barchart, error_distribution_with_wilcoxon

if __name__ == "__main__":
    
    # --- args ---
    parser = argparse.ArgumentParser(description="Evaluation of meta-learning approach")
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    with open(CONFIG_PATH / "config.json") as c_file:
        config = json.loads(c_file.read())
    assert config, "Config file is empty or failed to parse"

    GENTEST_EVAL.mkdir(exist_ok=True, parents=True)

    for task in config["tasks"]:

        task_name = task["task_type"] 

        if task["task_type"] != args.task:
            continue

        print(f"[{datetime.now()}] Evaluation for task {task['task_type']} ...")

        metric_name = task["metric_id"]

        for dataset in task['datasets']:

            dataset_name = dataset['dataset_id']

            print(f"[{datetime.now()}] Evaluation for dataset {dataset['dataset_id']} ...")

            meta_result_path = GENTEST_RESULTS / f"meta_{dataset['dataset_id']}.csv"

            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')
            meta = pd.read_csv(meta_result_path, index_col='id')

            models = df_metrics.columns.tolist()
            df_metrics['Meta'] = meta.index.to_series().map(lambda id: df_metrics.loc[id, meta.loc[id, 'model']])

            reshaped_metrics = df_metrics.reset_index().melt(id_vars = "id", var_name="model", value_name=metric_name)

            # set avg rank params
            method = "average"
            ascending = False if task["objective"] == "maximize" else True

            print(f"Average Rank parameters: {method=} {ascending=} for task {task_name} and dataset: {dataset_name}")
            ranks_df = get_avg_rank(reshaped_metrics, value_col=metric_name, method=method, ascending=ascending)

            ranks_df.describe().to_csv(GENTEST_EVAL / f"describe_rank_{task_name}___{dataset_name}.csv")
            rank_sorted_models = ranks_df.sort_values(by="avg_rank", ascending=True)["model"].tolist()

            
            # Plot stacked barchart of rank counts
            reshaped_metrics['rank'] = reshaped_metrics.groupby('id')[metric_name].rank(ascending=ascending, method=method)
            rank_counts = reshaped_metrics.groupby(['model', 'rank']).size().unstack(fill_value=0)
            rank_percentages = rank_counts.div(rank_counts.sum(axis=1), axis=0) * 100
            stacked_barchart(rank_percentages, out_path = GENTEST_EVAL / f"rank_count_{task_name}___{dataset_name}.png")


            # Dump for each model metrics 
            reshaped_metrics["model"] = pd.Categorical(reshaped_metrics["model"], categories=rank_sorted_models, ordered=True)
            reshaped_metrics.sort_values(["id", "model"], inplace = True)
            reshaped_metrics.to_csv(GENTEST_EVAL / f"reshaped_metrics_{task_name}___{dataset_name}.csv", index=False)


            ### Statistical test
            friedman_results = friedman(df = reshaped_metrics, models = rank_sorted_models, metric_name = metric_name)

            wilcoxon_results, wilcoxon_annotations = pairwise_wilcoxon(df = reshaped_metrics, metric = metric_name)
            wilcoxon_results.to_csv(GENTEST_EVAL / f"wilcoxon_posthoc_{task_name}___{dataset_name}.csv")
            
            error_distribution_with_wilcoxon(
                df_metrics = df_metrics,
                wilcoxon_results = wilcoxon_results,
                wilcoxon_annotations = wilcoxon_annotations,
                models = rank_sorted_models,
                metric = metric_name,
                task = task["task_type"],
                dataset = dataset_name,
                out_path = GENTEST_EVAL / f"combined_plot#{task_name}___{dataset_name}.png"
            )