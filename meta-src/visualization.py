import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set the color palette
def make_highlight_palette(x_values, highlight='Meta', highlight_color='#f7dc6f', other_color='whitesmoke'):
    palette = {x: highlight_color if x == highlight else other_color for x in x_values}
    palette["ideal"] = "green"
    return palette

def annotate_significance(val):
    if val < 0.001:
        return '***'
    elif val < 0.01:
        return '**'
    elif val < 0.05:
        return '*'
    else:
        return ''

def stacked_barchart(rank_percentages_df, out_path):

    sns.set_style("whitegrid")

    ax = rank_percentages_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.title('Percentage of Times Each Model Achieved Each Rank')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Model')
    plt.legend(title='Rank', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def error_distribution_with_wilcoxon(df_metrics, wilcoxon_results, wilcoxon_annotations, models, metric, task, dataset, out_path):
    
    sns.set_style("whitegrid")
    
    x_order = sorted(['Meta'] + models)  
    palette = make_highlight_palette(x_order)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Adjust width as needed

    # Global font settings
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})  # Times New Roman
    # --- Boxplot ---
    sns.boxplot(
        ax=axs[0],
        data=df_metrics[['Meta'] + models], 
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
        patch_artist=True
    )
    axs[0].set_title(f"{task}___{dataset}")
    axs[0].set_ylabel(f"Metric: {metric}")
    axs[0].tick_params(axis='x', rotation=45, labelsize=12)

    # --- Heatmap ---
    mask = np.triu(np.ones_like(wilcoxon_results, dtype=bool))

    # Color settings for p-value ranges
    cmap = ListedColormap(['#062a46', '#093963', '#7aa3cb', '#d4d1e6'])  # Reversed order
    norm = BoundaryNorm([0, 0.001, 0.01, 0.05, 1], cmap.N)

    sns.heatmap(
        wilcoxon_results, 
        annot=wilcoxon_annotations,
        mask=mask,
        fmt='', 
        cmap=cmap, 
        norm=norm, 
        cbar=False,
        linewidths=0.5,
        vmin=0, vmax=0.5,
        ax=axs[1]
    )
    axs[1].set_title("Wilcoxon Posthoc Test")
    axs[1].tick_params(axis='x', rotation=45, labelsize=12)
    axs[1].tick_params(axis='y', rotation=0, labelsize=12)

    # Add legend for heatmap
    legend_labels = ['***: p < 0.001', '**: p < 0.01', '*: p < 0.05', 'No marker: p ≥ 0.05']
    legend_elements = [Patch(facecolor=cmap(norm(bound)), edgecolor='black', label=label) 
                    for bound, label in zip([0.0001, 0.001, 0.01, 0.1], legend_labels)]
    axs[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
