import numpy as np

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mtick

from statannotations.Annotator import Annotator

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

def models_mapping(mdl):
    if "RNN_" in mdl:
        return mdl.replace("RNN_", "")
    if mdl == "Meta":
        return "MALIAS"
    if mdl ==  "LSTMADalpha":
        return r"LSTMAD$_\alpha$"
    return mdl

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
    
    x_order = sorted(models)  
    palette = make_highlight_palette(x_order)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Adjust width as needed

    # Global font settings
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})  # Times New Roman

    
    # --- Boxplot ---
    sns.boxplot(
        ax=axs[0],
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
        patch_artist=True
    )
    axs[0].set_title("")
    axs[0].set_ylabel(f"{metric.upper()}")
    axs[0].tick_params(axis='x', rotation=45, labelsize=16)

    ticks = axs[0].get_xticks()

    orig_labels = [t.get_text() for t in axs[0].get_xticklabels()]

    # Map each original label through the function
    new_labels = [models_mapping(lbl) for lbl in orig_labels]

    # Apply the new labels
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(new_labels, rotation=45)

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
    axs[1].set_title("")
    axs[1].tick_params(axis='x', rotation=45, labelsize=16)
    axs[1].tick_params(axis='y', rotation=0, labelsize=16)

    
    ticks = axs[1].get_xticks()
    orig_labels = [t.get_text() for t in axs[1].get_xticklabels()]
    new_labels = [models_mapping(lbl) for lbl in orig_labels]

    # Apply the new labels
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(new_labels, rotation=45)

    ticks = axs[1].get_yticks()
    orig_labels = [t.get_text() for t in axs[1].get_yticklabels()]
    new_labels = [models_mapping(lbl) for lbl in orig_labels]

    # Apply the new labels
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels(new_labels)


    # Add legend for heatmap
    legend_labels = ['***: p < 0.001', '**: p < 0.01', '*: p < 0.05', 'No marker: p ≥ 0.05']
    legend_elements = [Patch(facecolor=cmap(norm(bound)), edgecolor='black', label=label) 
                    for bound, label in zip([0.0001, 0.001, 0.01, 0.1], legend_labels)]
    axs[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def annotated_boxplot(df_metrics, wilcoxon_results, wilcoxon_annotations, models, metric, task, dataset, out_path):
    

    sns.set_style("whitegrid")

    
    # Compute mean of each model to sort by
    model_means = df_metrics.mean().sort_values(ascending=False)
    
    if metric.upper() == "SMAPE":
        model_means = df_metrics.mean().sort_values(ascending=True)    
    x_order = model_means.index.tolist()
    
    # x_order = models  
    palette = make_highlight_palette(x_order, highlight=None)
    df_metrics = df_metrics[x_order]

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(ncols=1, figsize=(4, 5))  # <-- Adjust width as needed
    ax = axs  # since we have only one axis
    # Global font settings
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})  # Times New Roman

    sns.barplot(
        ax=ax,
        data=df_metrics, 
        errorbar=('ci', 95),
        palette=palette,
        capsize=0.1,  # Size of the caps on the confidence interval bars
        errwidth=1.5,  # Width of the error bars
        edgecolor='black',  # Border color of the bars
        width=0.5
    )

    highlight_model = "Meta"  # <-- Replace with your target model name
    for patch, label in zip(ax.patches, x_order):
        if label == highlight_model:
            patch.set_hatch('////')  # Options: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'

    ax.set_title("")

    ylabels_mapping = {
        "SMAPE": "SMAPE",
        "AUPRC": "AUC-PR",
        "BAL_ACC": r"F$\beta$ Score"
    }

    ax.set_ylabel(ylabels_mapping[metric.upper()], fontsize=26)

    if metric.upper() == "SMAPE":
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    ax.tick_params(axis='x', rotation=45, labelsize=18)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')


    ticks = ax.get_xticks()

    orig_labels = [t.get_text() for t in ax.get_xticklabels()]

    # Map each original label through the function
    new_labels = [models_mapping(lbl) for lbl in orig_labels]

    # Apply the new labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(new_labels, rotation=45)
    ax.tick_params(axis='y', labelsize=18)

    print(x_order)

    long_df = df_metrics.melt(id_vars=None, var_name='variable', value_name='value', ignore_index=False)

    # pairs = list(itertools.combinations(long_df.variable.unique(), 2))
    pairs = [("Meta", m) for m in x_order if m != "Meta"]

    annot = Annotator(
        ax,
        pairs=pairs,
        data=long_df,
        x="variable",  # seaborn will treat each column name as 'variable'
        y="value",     # and the data melted into a 'value' column
        order=x_order,
    )

    print(long_df['variable'].value_counts())
        
    annot.configure(
        test="Wilcoxon",
        text_format='star',
        loc="outside",
        comparisons_correction="Holm-Bonferroni",
        hide_non_significant=True,
        line_width=2,
        line_offset=0.01
    )

    annot.apply_and_annotate()

    
    # Add all spines with bold borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.8)
        spine.set_edgecolor('black')

    ax.tick_params(
        axis='both',        # Apply to both x and y axes
        which='both',       # Apply to major and minor ticks
        direction='out',    # or 'in' or 'inout'
        top=False,
        bottom=True,
        left=True,
        right=False,
        length=6,           # Length of the tick lines
        width=.8          # Thickness of the tick lines
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    # plt.show()