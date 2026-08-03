import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set the color palette
def make_highlight_palette(x_values, highlight='Meta', highlight_color='#f7dc6f', other_color='whitesmoke'):
    return {x: highlight_color if x == highlight else other_color for x in x_values}

def get_avg_rank(df, value_col = "bal_acc", method = 'min', ascending = True):
    print(f"Getting avg rank trying to {method}")
    df['rank'] = df.groupby('id')[value_col].rank(ascending=ascending, method=method)
    avg_rank = df.groupby('model')['rank'].mean().reset_index()
    avg_rank.columns = ['model', 'avg_rank']

    return avg_rank

def annotate_significance(val):
    if val < 0.001:
        return '***'
    elif val < 0.01:
        return '**'
    elif val < 0.05:
        return '*'
    else:
        return ''

if __name__ == "__main__":
    
    # --- args ---
    parser = argparse.ArgumentParser(description="Evaluation of meta-learning approach")
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    with open("config/config.json") as c_file:
        config = json.loads(c_file.read())

    models_eval_path = Path("results/models")
    models_eval_path.mkdir(exist_ok=True, parents=True)

    sns.set_style("whitegrid")

    assert config, "Config file is empty or failed to parse"

    for task in config["tasks"]:

        if task["task_type"] != args.task:
            continue

        metric_name = task["metric_id"]

        for dataset in task['datasets']:

            dataset_name = dataset['dataset_id']

            print(f"[{datetime.now()}] {dataset['dataset_id']} ...")

            df_metrics = pd.read_csv(dataset['metric_path'], index_col='id')

            models = df_metrics.columns.tolist()

            for model in models:
                print(f"Mean value for {model}:", df_metrics[model].mean())

            df_metrics.dropna(inplace=True)

            method = "dense"
            ascending = False if task["objective"] == "maximize" else True
            print(f"Objective: {method=} {ascending=} for task {task['task_type']} and dataset: {dataset_name}")

            x_order = sorted(models)  
            palette = make_highlight_palette(x_order)

            plt.figure(figsize=(8, 6))
            plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})  # Use Times New Roman

            sns.boxplot(
                data=df_metrics[models], 
                showfliers=True, 
                showmeans=True, 
                palette=palette,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markeredgewidth": .5,
                    "markersize": 6
                },
                medianprops={
                    "linewidth": .5,
                    "color": "black"
                },
                whiskerprops={
                    'linestyle': '--',
                    "color": "black",
                    "linewidth": .5
                },
                boxprops={
                    "edgecolor": "black",
                    "linewidth": .5,
                    "alpha": 1
                },
                width=0.5,
                # color="whitesmoke",
                patch_artist=True
                )
            plt.ylabel(f"Metric: {task['metric_name']}")
            plt.xticks(rotation=90)

            plt.savefig(models_eval_path / f"metric_distribution_{task['task_type']}_{dataset_name}.png", bbox_inches='tight')
            plt.close()

            rank = df_metrics.iloc[:, :-1].apply(lambda x: x.rank(method=method, ascending=ascending), axis=1)

            
            ### Statistical test
            reshaped_metrics = df_metrics.reset_index().melt(id_vars = "id", var_name="model", value_name=metric_name)

            ranks_df = get_avg_rank(reshaped_metrics, value_col=metric_name, method=method, ascending=ascending)
            rank_sorted_models = ranks_df.sort_values(by="avg_rank", ascending=True)["model"].tolist()


            reshaped_metrics['rank'] = reshaped_metrics.groupby('id')[metric_name].rank(ascending=ascending, method=method)

            rank_counts = reshaped_metrics.groupby(['model', 'rank']).size().unstack(fill_value=0)
            rank_percentages = rank_counts.div(rank_counts.sum(axis=1), axis=0) * 100

            # Plot stacked bar chart
            ax = rank_percentages.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
            plt.title('Percentage of Times Each Model Achieved Each Rank')
            plt.ylabel('Percentage (%)')
            plt.xlabel('Model')
            plt.legend(title='Rank', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.savefig(models_eval_path / f"rank_count_{task['task_type']}_{dataset_name}.png")
            plt.show()

            reshaped_metrics["model"] = pd.Categorical(reshaped_metrics["model"], categories=rank_sorted_models, ordered=True)
            reshaped_metrics.sort_values(["id", "model"], inplace = True)
            reshaped_metrics.to_csv(models_eval_path / f"reshaped_metrics_{task['task_type']}_{dataset_name}.csv", index=False)

            groups = [
                reshaped_metrics.loc[reshaped_metrics['model'] == cond, metric_name].values
                for cond in rank_sorted_models
            ]

            # Apply Friedman
            stat, pval = friedmanchisquare(*groups)
            print("Friedman test:")
            print(f"  statistic = {stat:.4f}")
            print(f"  p-value   = {pval:.4e}")

            friedman_df = pd.DataFrame(columns = ["Statistic", "$p-value$"])
            friedman_df.loc[0, "Statistic"] = f"{stat:.4f}"
            friedman_df.loc[0, "$p-value$"] = f"{pval:.2e}"

            friedman_df

            alpha = 0.05
            assert pval < alpha, "Friedman test is not significant; no post-hoc comparisons needed."

            posthoc_results = sp.posthoc_wilcoxon(
                reshaped_metrics, 
                group_col='model', 
                val_col=metric_name, 
                p_adjust='holm'
            )

            print("\nPost-hoc Wilcoxon (Holm–Bonferroni) results:")
            print(posthoc_results)
            posthoc_results.to_csv(models_eval_path / f"wilcoxon_posthoc_{task['task_type']}_{dataset_name}.csv")

            # Annotate significance and create upper triangle mask
            annotations = posthoc_results.applymap(annotate_significance)
            mask = np.triu(np.ones_like(posthoc_results, dtype=bool))

            # Color settings for p-value ranges
            cmap = ListedColormap(['#062a46', '#093963', '#7aa3cb', '#d4d1e6'])  # Reversed order
            norm = BoundaryNorm([0, 0.001, 0.01, 0.05, 1], cmap.N)

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(
                posthoc_results, 
                annot=annotations,
                mask=mask,
                fmt='', 
                cmap=cmap, 
                norm=norm, 
                cbar=False,
                linewidths=0.5,
                vmin=0, vmax=0.5
            )

            # Significance level legend
            legend_labels = ['***: p < 0.001', '**: p < 0.01', '*: p < 0.05', 'No marker: p ≥ 0.05']
            legend_elements = [Patch(facecolor=cmap(norm(bound)), edgecolor='black', label=label) 
                            for bound, label in zip([0.0001, 0.001, 0.01, 0.1], legend_labels)]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=18)

            # Axis settings
            plt.yticks(rotation=0, fontsize=16)
            plt.xticks(rotation=45, fontsize=16)
            plt.tight_layout()
            plt.savefig(models_eval_path / f"wilcoxon_heatmap_{task['task_type']}_{dataset_name}.png")
            plt.close()