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
    # plt.legend()
    # save the plot to a file
    plt.savefig(output_file)
    # close the plot to free up memory
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
    # plt.xlabel(fontsize=11)
    # plt.ylabel(fontsize=11)

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



    # plt.legend(data.columns, loc='center left', bbox_to_anchor=(1.1, 0.5))
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

    # plt.legend(data.columns, loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()

    plt.savefig("violin_plot2.png", dpi=300)


    # plt.show()
    # plt.close()

    # Draw Plot
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
    # plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.yticks(frontsize=6)


    plt.savefig("errorbar.png", dpi=300)
    plt.close()

def visualized_attention(or_QKo, QKo, Qi, x, w, path, name, feature_name_list, sample_num):
    a = sample_num
    for index in range(a):
        import keras.backend as K
        samples, heads, features = QKo.shape[0], QKo.shape[1], QKo.shape[2]
        # qk = QKo.sum(axis=1)
        # QK = qk.sum(axis=0) /np.max(qk.sum(axis=0))

        target_attention = QKo[sample_num,:,:,:]
        for i in range(heads):
            i_target_attention = target_attention[i,:,:]


            draw_attention_matrix(path, f"sample_{index}_head{i}_"+name, feature_name_list, i_target_attention.transpose(), sample_num)
            single_attention = i_target_attention.sum(axis=0)
            single_attention_all = np.exp(single_attention)/sum(np.exp(single_attention))

            draw_bar(single_attention_all, feature_name_list, path, f"sample_{index}_head{i}_"+name, sample_num)


    # plt.close()

def draw_attention_matrix(path, name, feature_name_list, QK, sample_num):
    # exp_attention = np.exp(QK)
    softmax_output = QK


    save_path = os.path.join(path, "model", name).replace("\\", "/")
    plt.figure(figsize=(12, 10))
    # sns.set(font_scale=1.4)
    # annot = np.array(['%.2f' % point for point in np.array(softmax_output.ravel())]).reshape(
    #     np.shape(softmax_output))
    # annot_kws = {"size": 20}
    maxv = np.max(softmax_output)
    minv = np.min(softmax_output)
    len1 = len(feature_name_list)
    sns.set(font_scale=3)


    fig = sns.heatmap(softmax_output,
                      linewidth=0.5,
                      # annot=annot,
                      # annot_kws=annot_kws,
                      fmt='',
                      yticklabels=np.arange(1,len1+1,step=1),
                      xticklabels=np.arange(1,len1+1,step=1),
                      vmax=maxv,
                      vmin=minv,
                      # cmap="vlag",
                      cmap="YlGnBu_r",
                      )
    plt.yticks(fontproperties='Times New Roman', fontsize=30)
    plt.xticks(fontproperties='Times New Roman', fontsize=30)
    # fig.ax_heatmap.set_xticklabels(fig.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def draw_bar(single_attention_all, feature_name_list, path, name, sample_num):
    # a = int(len(feature_name_list) * 2)

    plt.figure(figsize=(12, 10), dpi=300)
    import seaborn as sns

    ax = plt.axes()
    bar_colors = ["mauve", "peach", "baby blue", "grey",
                  "greyish purple", "wisteria", "dark", "mango",
                  "rose pink", "turquoise", "deep rose","fuchsia", "light urple", "scarlet", "greenish blue", "grey",
                  "light violet", "medium blue", "purplish blue", "wine", "pinkish", "vomit", "pale orange",
                  "pastel purple", "sienna", "tangerine", "raspberry", "rose pink", "orchid",
                  "emerald"
                  ]
    # bar_colors = bar_colors.reverse()
    single_attention_all = single_attention_all.tolist()[::-1]
    colors = ["#FFC0CB", "#92a6be", "#c48d60", "#7e728c", "#FF00FF", "#9400D3", "#6A5ACD", "#6495ED", "#ADD8E6", "#00FF7F", "#FFD700", "#F4A460", "#FA8072", "#FF7F50",
              "#FF6347", "#FF4500", "#FF0000", "#800000", "#8B0000", "#FF1493", "#FF69B4", "#C71585", "#DB7093", "#FFB6C1", "#FFA07A", "#FFA500", "#FF8C00"]
    plt.grid(axis="x", linestyle='--', alpha=0.6, zorder=0,)#c='#d2c9eb', )
    bars = plt.barh(range(1, len(feature_name_list) + 1), single_attention_all, color=sns.xkcd_palette(bar_colors), edgecolor='black',  lw=2, zorder=1) # #92a6be, #c48d60, ##7e728c
    plt.legend(bars, feature_name_list,prop={"family": "Times New Roman", "size":19}, ncol=1, fontsize=19, loc='lower right', frameon=False)
    plt.gca().set_facecolor('none')
    plt.xticks(fontproperties='Times New Roman', fontsize=30)
    plt.yticks([])
    plt.xlim(0, 1.1)
    plt.axvline(x=1, c="#92a6be", linestyle='--')
    # plt.title('Attention of each group', fontproperties='Times New Roman', fontsize=fontsize)
    plt.tight_layout()
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    # plt.show()

    plt.savefig(os.path.join(path, "model", f"bar_" +name).replace("\\", "/"))
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

def plot_learning_curve():
    pass


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




def model_assessments_bar(path):
    from matplotlib.ticker import PercentFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 10})


    #####
    # all features
    #####

    model_name_list = ["Lasso", "Ridge", "EN", "SVM",
                       # "RF",
                       "XGB", "LSTM", "GRU", "NCPP"]


    # bar_colors = ["#92a6be", "#c48d60", "#7e728c", "#c48d60", "#7e728c", "#c48d60", "#7e728c"]
    bar_colors = ["scarlet", "greenish blue", "marine", "grey",
                  "greyish purple", "wisteria", "dark", "mango", #"amber",
                  "rose pink", "turquoise", "deep rose"]
    benchmarks = ["SPECrate2017 Integer base", "SPECrate2017 FP base", "MLC Latency", "MLC Bandwidth", "Stream", "HPCG"] #"Linpack", ]
    len1 = 6
    label_dict = {
        "Lasso": [ "1.22_utmalte_test_lasso"]  * len1,
        "Ridge": ["1.22_utmalte_test_ridge"] * len1,
        "ElasticNet":["2220_utmalte_elasticnet"] * len1,
        "SVMModel":["220_utmalte_SVR"] * (len1-1) + ["3.17_ultimate_svr_svm"],
        # "RandomForest": ["220_utmalte_rf"] * (len1-1) + ["3.17_ultimate_RF"],
        "XgboostModel":["220_utmalte_xgb"] * len1,
        "LSTMModel":["3_17_ultimate_lstm"] + ["220_utmalte_lstm"] * (len1-1),
        "GRU":["220_utmalte_gru"] * len1,
        "GroupMultiAttenResNet":["3_17_ultimate_cgrf","pre_layer_no_drop",  "mlc-la-all-feature_001_5000", "mlc-bd-all-feature_0001_8000", "224_utmalte_CGAF", "3_17_ultimate_cgrf"]


    }
    mae_list, mse_list, mape_list, mae_std_list, mse_std_list, mape_std_list = [], [], [], [], [], []

    model_steps = len(model_name_list) + 1

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    for e, model_name in enumerate(model_name_list):

        w = 0.2

        steps = w * model_steps
        ind = np.array([0 + i * steps for i in range(len(benchmarks))]) # the x locations for the groups
          # the width of the bars
        pos = [ind - (4 * w), ind - (3 * w), ind - (2 * w), ind - (1 * w), ind, ind + (1 * w), ind + (2 * w), ind + (3 * w), ind + (4 * w)]

        p0 = axes[0].bar(pos[e], mae_list[model_name], w, bottom=0, color=sns.xkcd_palette(bar_colors)[e],
                            label=model_name, #yerr=mae_std_list[model_name],
                            error_kw=dict(capsize=2, capthick=1, ecolor="#555555"))
        axes[0].set_ylabel('MAE', fontsize=16)
        axes[0].legend(loc='upper right', frameon=True, fontsize=16)
        axes[0].set_ylim(bottom=1)

        p1 = axes[1].bar(pos[e], [item  for item in mse_list[model_name]], w, bottom=0, color=sns.xkcd_palette(bar_colors)[e],
                            label=model_name, #yerr=[item for item in mse_std_list[model_name]],
                            error_kw=dict(capsize=2, capthick=1, ecolor="#555555"))
        axes[1].set_ylabel('MSE', fontsize=16)
        axes[1].legend(loc='upper right', frameon=True, fontsize=16)

        p2 = axes[2].bar(pos[e], [item  for item in  mape_list[model_name]], w, bottom=0, color=sns.xkcd_palette(bar_colors)[e],
                            label=model_name, #yerr=[item for item in  mape_std_list[model_name]],
                            error_kw=dict(capsize=2, capthick=1, ecolor="#555555"))
        axes[2].set_ylabel('MAPE', fontsize=16)
        axes[2].legend(loc='upper right', frameon=True, fontsize=16)
        axes[2].yaxis.set_major_formatter(PercentFormatter())

        for i in range(3):
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['top'].set_visible(False)
            axes[i].set_xticks(ind - w / 2)
            axes[i].set_xticklabels([suite for suite in benchmarks], fontsize=16, rotation=10)
            # axes[i].xticks(rotation=15)
            # axes[i].set_xticklabels([suite for suite in benchmarks], fontsize=11,rotation=10)
            axes[i].grid(False)
            axes[i].grid(axis="y", color="#dddddd", ls="--", zorder=-1)


    # plt.tight_layout() # hspace is set
    plt.rc('font', family='Times New Roman')
    plt.savefig(path+".png",bbox_inches="tight",dpi=300,)
    plt.show()
    plt.close()

def model_assessments_cycle(metric = "MAE",image_name="mae.png"):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    import plotly.io as pio
    pio.renderers.default = "browser"
    # pio.renderers.default = 'png'
    # model color
    colors = ['rgba(255, 195, 36, 0.5)', 'rgba(233, 11, 48, 0.5)', 'rgba(8, 175, 63, 0.5)', 'rgba(255,69,0, 0.5)', 'rgba(0,0,128,0.5)',
         'rgba(85,107,47, 0.5)', 'rgba(138, 43, 226, 0.5)', 'rgba(0,250,154, 0.5)', 'rgba(218,165,32, 0.5)'
              ]
    edges = ['rgba(255, 195, 36, 1)', 'rgba(233, 11, 48, 1)', 'rgba(8, 175, 63, 1)', 'rgba(255,69,0, 1)', 'rgba(0,0,128,1)',
              'rgba(85,107,47, 1)', 'rgba(138, 43, 226, 1)', 'rgba(0,250,154, 1)', 'rgba(218,165,32, 1)'
             ]
    # benchmarks = ["SPECrate2017 Integer base", "SPECrate2017 FP base", "MLC Latency", "MLC Bandwidth", "Stream", "Linpack","HPCG"]
    suites = {"SPECrate2017 Integer base":["500.perlbench_r", "502.gcc_r", "505.mcf_r", "520.omnetpp_r", "523.xalancbmk_r", "525.x264_r", "531.deepsjeng_r", "541.leela_r", "548.exchange2_r", "557.xz_r", "SPECrate2017_int_base"],
              "SPECrate2017 FP base":["503.bwaves_r", "507.cactuBSSN_r", "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r", "521.wrf_r", "526.blender_r", "527.cam4_r", "538.imagick_r", "544.nab_r", "549.fotonik3d_r", "554.roms_r", "SPECrate2017_fp_base"],
              "MLC Latency":[
                        "Local socket local cluster L1 hit latency (ns)",
                        "Local socket local cluster L2 hit latency (ns)",
                        "Local socket local cluster L3 hit latency (ns)",
                        "Local socket local cluster memory latency Random (ns)",
                        "Local socket local cluster memory latency Sequential (ns)",
                        "Local socket remote cluster memory latency Random (ns)",
                        "Local socket remote cluster memory latency Sequential (ns)",
                        "Remote socket memory latency Random (ns)",
                        "Remote socket memory latency Sequential (ns)"],
              "MLC Bandwidth": [
                        "LLC Bandwidth (MB/s)",
                        "Peak Bandwidth All Sockets 100R (MB/s)",
                        "Peak Bandwidth All Sockets 100W (MB/s)",
                        "Peak Bandwidth All Sockets 2R1W NTW (MB/s)",
                        "Peak Bandwidth All Sockets 2R1W RFO (MB/s)",
                        "Peak Bandwidth Remote (MB/s)",
                        "Peak Bandwidth Remote (Reverse) (MB/s)",
                        "Peak Bandwidth Socket 0 100R (MB/s)",
                        "Peak Bandwidth Socket 1 100R (MB/s)"],
                "Stream":["Add Bandwidth (MB/s)",
                            "Copy Bandwidth (MB/s)",
                            "Scale Bandwidth (MB/s)",
                            "Triad Bandwidth (MB/s)"],
              "HPCG":["Score(GFLOPS)"]
              }
    len1 = 6
    label_dict = {
        "Lasso": ["1.22_utmalte_test_lasso"] * len1,
        "Ridge": ["1.22_utmalte_test_ridge"] * len1,
        "ElasticNet": ["2220_utmalte_elasticnet"] * len1,
        "SVMModel": ["220_utmalte_SVR"] * (len1 - 1) + ["3.17_ultimate_svr_svm"],
        # "RandomForest": ["220_utmalte_rf"] * (len1 - 1) + ["3.17_ultimate_RF"],
        "XgboostModel": ["220_utmalte_xgb"] * len1,
        "LSTMModel": ["3_17_ultimate_lstm"] + ["220_utmalte_lstm"] * (len1 - 1),
        "GRU": ["220_utmalte_gru"] * len1,
        "GroupMultiAttenResNet": ["3_17_ultimate_cgrf", "pre_layer_no_drop", "mlc-la-all-feature_001_5000",
                                  "mlc-bd-all-feature_0001_8000", "224_utmalte_CGAF", "3_17_ultimate_cgrf"]

    }
    benchmark_mapping = {'Local socket local cluster L1 hit latency (ns)': 'Ls2lc L1 hit latency',
                         'Local socket local cluster L2 hit latency (ns)': 'Ls2lc L2 hit latency',
                         'Local socket local cluster L3 hit latency (ns)': 'Ls2lc L3 hit latency',
                         'Local socket local cluster memory latency Random (ns)': 'Ls2lc memory latency Random',
                         'Local socket local cluster memory latency Sequential (ns)': 'Ls2lc memory latency Sequential',
                         'Local socket remote cluster memory latency Random (ns)': 'Ls2lr memory latency Random',
                         'Local socket remote cluster memory latency Sequential (ns)': 'Ls2lr memory latency Sequential',
                         'Remote socket memory latency Random (ns)': 'Rs memory latency Random',
                         'Remote socket memory latency Sequential (ns)': 'Rs memory latency Sequential',
                         'LLC Bandwidth (MB/s)': 'LLC Bandwidth',
                         'Peak Bandwidth All Sockets 100R (MB/s)': 'Peak Bandwidth All Sockets 100R',
                         'Peak Bandwidth All Sockets 100W (MB/s)': 'Peak Bandwidth All Sockets 100W',
                         'Peak Bandwidth All Sockets 2R1W NTW (MB/s)': 'Peak Bandwidth All Sockets 2R1W NTW',
                         'Peak Bandwidth All Sockets 2R1W RFO (MB/s)': 'Peak Bandwidth All Sockets 2R1W RFO',
                         'Peak Bandwidth Remote (MB/s)': 'Peak Bandwidth Remote',
                         'Peak Bandwidth Remote (Reverse) (MB/s)': 'Peak Bandwidth Remote (Reverse)',
                         'Peak Bandwidth Socket 0 100R (MB/s)': 'Peak Bandwidth Socket 0 100R',
                         'Peak Bandwidth Socket 1 100R (MB/s)': 'Peak Bandwidth Socket 1 100R'
                         }



    fig = make_subplots(rows=3, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}],
                                               [{"type": "polar"}, {"type": "polar"}],[{"type": "polar"}, {"type": "polar"}]],
                        subplot_titles=[suite for suite in suites])
    bar_colors = ["scarlet", "greenish blue", "marine", "grey",
                  "greyish purple", "wisteria", "dark", "mango", #"amber",
                  "rose pink", "turquoise", "deep rose"]

    max_mae = []
    metric_list = []
    for s, suite in enumerate(suites):
        length = len(suites[suite])
        benchmarks = suites[suite]
        suit_label_dict = {k: [label_dict[k][s]]*length for k in label_dict}
        # print(category)
        mae_list, mse_list, mape_list, mae_std_list, mse_std_list, mape_std_list = [], [], [], [], [], []
        if metric == "MAPE":
            metric_list = mape_list
        elif metric == "MAE":
            metric_list = mae_list
        elif metric == "MSE":
            metric_list = mse_list
        else:
            raise ValueError("metric should be mae, mse or mape")
        max_value = max(max(v) for v in metric_list.values())
        max_mae.append(max_value)

        mapped_benchmarks = [benchmark_mapping.get(bn, bn) for bn in benchmarks]
        for m, model in enumerate(metric_list):
            r = metric_list[model]
            fig.add_trace(go.Scatterpolar(
                r=[*r, r[0]],
                theta=mapped_benchmarks,
                fill='toself',
                name=model,
                fillcolor=colors[m],
                marker=dict(color=edges[m], size=8),
                showlegend=not (s),
            ), row=int(s / 2) + 1, col=s % 2 + 1)

    fig.update_layout(
        autosize=False,
        width=1200,
        height=1000,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[0]])),
        polar2=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[1]]
            )),
        polar3=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[2]]
            )),
        polar4=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[3]],
            )),
        polar5=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[4]]
            )),
        polar6=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_mae[5]]
            )),
        font=dict(
            size=15
        ),
        legend=dict(
            x=0.44,
            y=0.95,
            traceorder='normal',
            font=dict(
                size=14, ),
            bordercolor="#ccc",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=20, t=50),
        title={
            'text': (
                f'{metric} of all benchmarks'),
            'x': 0.5,
            'y': 0.60,
            'font_size': 14,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template='seaborn'

    )
    # fig.update_annotations(yshift=20, font_size=12)
    for annotation in fig.layout.annotations:
        annotation.y = annotation.y + 0.03
    fig.update_polars(angularaxis_tickfont_size=10, radialaxis_tickfont_size=10)
    pio.write_image(fig, image_name, scale=4)
    fig.show()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    model_assessments_bar(path="")
    # model_assessments_cycle(metric="MAPE", image_name="mape.png")
    # bubble_plot()
    # ErrorBar()