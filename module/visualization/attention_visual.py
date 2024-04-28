#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/25/2024 4:43 PM
# @Author  : xiaomanl
# @File    : attention_visual.py
# @Software: PyCharm

import pandas as pd
import logging.config
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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

    plt.savefig(os.path.join(path, "ncpp", f"bar_" +name).replace("\\", "/"))
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


    save_path = os.path.join(path, "ncpp", name).replace("\\", "/")
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