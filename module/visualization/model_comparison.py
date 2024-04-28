#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/25/2024 4:39 PM
# @Author  : xiaomanl
# @File    : model_comparison_bar
# @Software: PyCharm

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
def model_assessments_bar(path):


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

def model_assessments_cycle(metric="MAE", image_name="mae.png"):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    import plotly.io as pio
    pio.renderers.default = "browser"
    # pio.renderers.default = 'png'
    # ncpp color
    colors = ['rgba(255, 195, 36, 0.5)', 'rgba(233, 11, 48, 0.5)', 'rgba(8, 175, 63, 0.5)', 'rgba(255,69,0, 0.5)',
              'rgba(0,0,128,0.5)',
              'rgba(85,107,47, 0.5)', 'rgba(138, 43, 226, 0.5)', 'rgba(0,250,154, 0.5)', 'rgba(218,165,32, 0.5)'
              ]
    edges = ['rgba(255, 195, 36, 1)', 'rgba(233, 11, 48, 1)', 'rgba(8, 175, 63, 1)', 'rgba(255,69,0, 1)',
             'rgba(0,0,128,1)',
             'rgba(85,107,47, 1)', 'rgba(138, 43, 226, 1)', 'rgba(0,250,154, 1)', 'rgba(218,165,32, 1)'
             ]
    # benchmarks = ["SPECrate2017 Integer base", "SPECrate2017 FP base", "MLC Latency", "MLC Bandwidth", "Stream", "Linpack","HPCG"]
    suites = {"SPECrate2017 Integer base": ["500.perlbench_r", "502.gcc_r", "505.mcf_r", "520.omnetpp_r",
                                            "523.xalancbmk_r", "525.x264_r", "531.deepsjeng_r", "541.leela_r",
                                            "548.exchange2_r", "557.xz_r", "SPECrate2017_int_base"],
              "SPECrate2017 FP base": ["503.bwaves_r", "507.cactuBSSN_r", "508.namd_r", "510.parest_r",
                                       "511.povray_r", "519.lbm_r", "521.wrf_r", "526.blender_r", "527.cam4_r",
                                       "538.imagick_r", "544.nab_r", "549.fotonik3d_r", "554.roms_r",
                                       "SPECrate2017_fp_base"],
              "MLC Latency": [
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
              "Stream": ["Add Bandwidth (MB/s)",
                         "Copy Bandwidth (MB/s)",
                         "Scale Bandwidth (MB/s)",
                         "Triad Bandwidth (MB/s)"],
              "HPCG": ["Score(GFLOPS)"]
              }
    len1 = 6

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
                                               [{"type": "polar"}, {"type": "polar"}],
                                               [{"type": "polar"}, {"type": "polar"}]],
                        subplot_titles=[suite for suite in suites])
    bar_colors = ["scarlet", "greenish blue", "marine", "grey",
                  "greyish purple", "wisteria", "dark", "mango",  # "amber",
                  "rose pink", "turquoise", "deep rose"]

    max_mae = []
    metric_list = []
    for s, suite in enumerate(suites):
        length = len(suites[suite])
        benchmarks = suites[suite]
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