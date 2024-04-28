#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/2023 4:08 PM
# @Author  : xiaomanl
# @File    : visualization
# @Software: PyCharm
import pandas as pd
import logging.config
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

warnings.filterwarnings("ignore")
import logging


def create_histogram(data, column, output_file):
    # get the value counts for the specified column
    draw_data = data[column]

    value_counts = draw_data.value_counts()
    value_counts.index = value_counts.index.astype(dtype="str")

    # create DataFrame with column names
    df = pd.DataFrame({
        'col': [column] * len(value_counts),
        'feature': value_counts.index,
        'counts': value_counts.values
    })
    # create a histogram of the value counts
    plt.barh(value_counts.index, value_counts.values)
    if len(value_counts.index)>10:
        plt.tick_params(axis='x', labelrotation=270)
    # set the plot title and axis labels
    plt.title(f"Histogram of Value Counts for {column}")
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    plt.close()
    return df

def pearson(data, save_path):
    data = data.drop(columns=["SVR.CPU.Socket(s)"])
    # Plot

    plt.figure(figsize=(12, 10), dpi=80)

    sns.heatmap(data.corr(), xticklabels=data.corr().columns, yticklabels=data.corr().columns, cmap='RdYlGn', center=0,
                annot=True, annot_kws={"fontsize":15})

    # Decorations

    plt.title(f'Corr heatmap of features in testcase {"".join(configs["test_names"])}', fontsize=22)

    plt.xticks(fontsize=8)

    plt.yticks(fontsize=10)
    plt.savefig(save_path)

    plt.show()


def bubble_plot():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    data = pd.read_csv("D:/OneDrive/stunning-guacamole/src/train/processors_count.csv")
    data = data.head(22)
    sns.set(style="white", font_scale=1.2)
    plot = sns.relplot(x="Core(s) per Socket", y="CPU Base Frequency", hue="CPU Model",
                size="Counts",
                       sizes=(500, 2000),
                       alpha=.5, palette="muted",
                       # legend=False,

                height=6, data=data)

    handles, labels = plt.gca().get_legend_handles_labels()
    plot._legend.remove()
    plt.xlim(0, 55)
    exclude_labels = ['Counts', '800', '1600', '2400', '3200', '4000']
    filtered_handles = [handle for handle, label in zip(handles, labels) if label not in exclude_labels]
    filtered_labels= [label for handle, label in zip(handles, labels) if label not in exclude_labels]
    plot.fig.legend(filtered_handles, filtered_labels, ncol=1, loc='upper right',
                    bbox_to_anchor=(1, 1), frameon=False,  prop={'size': 11})

    plt.show()
    plot.savefig("bubble_plot.png")
    # Draw Plot


def violin_plot(data):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    select_columns  = [
    "SVR.CPU.L3 Cache",
    "SVR.CPU.L1d Cache",
    "SVR.CPU.L1i Cache",
    "SVR.CPU.L2 Cache",
    "SVR.Power.TDP",
    "SVR.CPU.Base Frequency",
    "SVR.CPU.Maximum Frequency",
    "CPU_all_core_max_freq",
    "SVR.CPU.Thread(s) per Core",
    "SVR.CPU.Core(s) per Socket",
    "SVR.CPU.Socket(s)",
    "SVR.Power.Frequency (MHz)",
    "SVR.CPU.CPU(s)",
    "Measure.DIMM.Num",
    "Measure.DIMM.Total"]
    select_columns2 = [
    "Measure.DIMM.Freq",
    "QDF.Thermal Design Power",
    "QDF.All Core Turbo Freq Rate",
    "QDF.Max Turbo Frequency Rate",
    "QDF.AVX2 All Core Turbo Freq Rate",
    "QDF.AVX3 Deterministic P1 Freq Rte",
    "QDF.AVX3 All Core Turbo Freq Rate",
    "QDF.TMUL Deterministic P1 Freq Rte",
    "QDF.TMUL All Core Turbo Freq Rate",
    "QDF.CLM P1 Max Freq",
    "QDF.CLM P0 Max Freq",
    "QDF.CLM Pn Max Freq",
    "QDF.AVX FMA Execution Unit Count",
    "QDF.Max UPI Port Cnt",
    "Rank"

]

    data1 = data.loc[:, select_columns]
    data2 = data.loc[:, select_columns2]

    # single_data = data.loc[:, [""]]
    fig, axes = plt.subplots(nrows=1, ncols=len(select_columns), figsize=(20, 5), gridspec_kw={'wspace': 1})

    for i, col in enumerate(data1.columns):
        sns.violinplot(y=col, data=data2, ax=axes[i], inner='box',
                       inner_kws={'color': 'white', 'marker': '^'})
        axes[i].set_ylabel('')
        if data1[col].max() > 1e5:
            formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}K'.format(x * 1e-3))
            axes[i].yaxis.set_major_formatter(formatter)
            axes[i].set_yticks([data1[col].min(), data1[col].median(), data1[col].max()])
            # axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())
            axes[i].set_xlabel([i+1])
            axes[i].tick_params(axis='y', labelsize=12)
        else:

            axes[i].set_yticks([data1[col].min(), data1[col].median(), data1[col].max()])
            # axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())
            axes[i].set_xlabel([i+1])
            axes[i].tick_params(axis='y', labelsize=12)

        for spine in axes[i].spines.values():
            spine.set_edgecolor('#dddddd')

    plt.tight_layout()

    plt.savefig("violin_plot1.png", dpi=300)

    fig, axes = plt.subplots(nrows=1, ncols=len(select_columns2), figsize=(20, 5), gridspec_kw={'wspace': 1})


    for i, col in enumerate(data2.columns):
        sns.violinplot(y=col, data=data2, ax=axes[i], inner='box',
                       inner_kws={'color': 'white', 'marker': '^'})
        axes[i].set_ylabel('')
        if data2[col].max() > 1e5:
            formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}K'.format(x * 1e-3))
            axes[i].yaxis.set_major_formatter(formatter)
            axes[i].set_yticks([data2[col].min(), data2[col].median(), data2[col].max()])
            # axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())
            axes[i].set_xlabel([i+16])
            axes[i].tick_params(axis='y', labelsize=12)
        else:

            axes[i].set_yticks([data2[col].min(), data2[col].median(), data2[col].max()])
            # axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())
            axes[i].set_xlabel([i+16])
            axes[i].tick_params(axis='y', labelsize=12)
        for spine in axes[i].spines.values():
            spine.set_edgecolor('#dddddd')

    plt.tight_layout()
    plt.savefig("violin_plot2.png", dpi=300)

def ErrorBar():
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the font to Times New Roman for all text in the plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    colors = [
        "deep rose",
        "mango",
        "marine",
        "light urple",
        "amber",
        "rose pink",
        "turquoise",
        "scarlet",
        "greenish blue",
        "grey",
        "greyish purple",
        "wisteria",
    ]

    data = pd.read_csv("D:/OneDrive/stunning-guacamole/project report/data/SPECrate2017_int_base_label_static.csv")
    data = data[data["Labels"] != "SPECrate2017_int_base"]
    benchmark = data["Labels"].tolist()
    mean = data["Mean"].tolist()
    std = data["Std"].tolist()
    plt.figure(figsize=(5, 3))
    plt.errorbar(mean, benchmark, xerr=std, fmt='>',color=sns.xkcd_palette(colors)[0],
                 #ecolor='tab:cyan',
                 elinewidth=1, capsize=4, label="SPECrate2017_int_base", ms=4)

    plt.grid(axis="x", linestyle='--', alpha=0.6, c='#d2c9eb', zorder=0)

    plt.ylabel("Benchmark")
    plt.xlabel("Throughput")
    plt.tight_layout()
    plt.savefig("errorbar.png", dpi=300)
    plt.close()


def plot_epochs_metric(hist, file_name, metric="loss"):
    try:
        history_dict = hist.history
        epochs = range(1, len(history_dict["loss"]) + 1)
        loss_values = history_dict[metric]
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes()
        ax.set_facecolor("#f2f2f2")
        ax.set_alpha(0.7)
        fig.patch.set_facecolor("#f2f2f2")
        fig.patch.set_alpha(0.7)
        plt.plot(epochs, loss_values,"b-",color="darkorange",  label='Training ' + metric,lw=2)

        if 'val_'+ metric in history_dict:
            val_loss_values = history_dict['val_'+ metric]
            val_std = np.std(val_loss_values)
            plt.plot(epochs, val_loss_values, 'b-.',color="navy", label='Validation '+metric)
            plt.fill_between(epochs, [x-x/2 for x in val_loss_values],
                             [x+x/2 for x in val_loss_values], alpha=0.2,
                             color="navy", lw=2)
        plt.title('Training and validation ' + metric)
        plt.ylabel(metric, fontsize='large')
        plt.xlabel('Epoch', fontsize='large')
        if metric == "loss":
            plt.ylim(0, 7000)
        else:
            plt.ylim(0, 100)
        plt.grid(axis="both", linestyle='--', alpha=0.6, c='#d2c9eb', zorder=0)
        plt.legend(['train', 'val'], loc='best')
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.warning(f"Error plot metric: {e}")



def plot_train_predict(y_test, y_predict, save_path, title=""):
    try:
        fig = plt.figure(figsize=(12, 6))

        ax = plt.axes()
        ax.set_facecolor("#f2f2f2")
        ax.set_alpha(0.7)
        fig.patch.set_facecolor("#f2f2f2")
        fig.patch.set_alpha(0.7)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(axis="both", linestyle='--', alpha=0.6, c='#d2c9eb', zorder=0)
        plt.plot(range(len(y_test)), y_test, "-",color="darkorange", label='True')
        plt.plot(range(len(y_predict)), y_predict, '-.',color="navy",label='Predict')
        plt.legend(loc='best')
        plt.title(title)

        plt.savefig(save_path)
        plt.close()
        print(f"Save plot to {save_path}")
    except Exception as e:
        print(f"Plot error: {e}")








if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    # bubble_plot()
    # ErrorBar()