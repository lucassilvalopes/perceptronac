"""
Example:

python3 scripts/gen_bkwd_adpt_cdng_grph.py 0.5 1 \
    results/exp_1672230131/rnn_online_coding_Adaptive_Detection_of_Dim_5pages_GRURNN650_lr1e-02_batchsize64.csv \
    results/exp_1672234599/backward_adaptive_coding_Adaptive_Detection_of_Dim_5pages_lut_mean_lr1e-2_N26.csv
"""
import sys
import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import change_aspect
from perceptronac.loading_and_saving import linestyle_tuple


def label_map(orig_lbl):
    label_map_dict = {
        "LUTmean": 'ALUT',
        "MLP": "APC",
        "RNN": "ARNN"
    }
    if "MLP" in orig_lbl or "RNN" in orig_lbl:
        ky = "MLP" if "MLP" in orig_lbl else "RNN"
        coefficient = re.search(r'(?<==).*(?=e-)',orig_lbl).group()
        exponent = re.search(r'(?<=e-).*(?=$)',orig_lbl).group()
        return label_map_dict[ky] + ' $\lambda='+str(coefficient)+'\cdot10^{-'+ str(int(exponent)) + '}$'
    else:
        return label_map_dict[orig_lbl]


def linestyle_map(orig_lbl):
    linestyle_map_dict = {
        "LUTmean": "solid",
        0.1 : "dashdot",
        0.01 : "dashed", 
        0.0001 : "dotted", 
    }
    if "MLP" in orig_lbl:
        ky = float(re.search(r'[\d\.]{1,}e-[\d]{1,}',orig_lbl).group())
        return linestyle_map_dict[ky]
    elif "RNN" in orig_lbl:
        return linestyle_tuple['densely dotted']
    else:
        return linestyle_map_dict[orig_lbl]


def color_map(orig_lbl):
    color_map_dict = {
        "LUTmean": "g",
        0.1 : "r",
        0.01 : "b",
        0.0001 : "c",
    }
    if "MLP" in orig_lbl:
        ky = float(re.search(r'[\d\.]{1,}e-[\d]{1,}',orig_lbl).group())
        return color_map_dict[ky]
    elif "RNN" in orig_lbl:
        return "m"
    else:
        return color_map_dict[orig_lbl]


if __name__ == "__main__":
    
    ylim_upper = float(sys.argv[1]) # 0.5
    legend_ncol = int(sys.argv[2]) # 1
    csv_name = sys.argv[3]

    data = pd.read_csv(csv_name)

    identifiers = re.findall(r'[\d]{10}',csv_name)

    for i in range(4,len(sys.argv)):
        csv_name = sys.argv[i]
        identifiers += re.findall(r'[\d]{10}',csv_name)
        data = pd.merge(data,pd.read_csv(csv_name),on="iteration")

    data = data.set_index("iteration")
        
    xvalues = data.index
    len_data = len(xvalues)

    data = data.to_dict(orient="list")

    labels = [label_map(k) for k in sorted(data.keys())]

    linestyles = [linestyle_map(k) for k in sorted(data.keys())]

    colors = [color_map(k) for k in sorted(data.keys())]

    ylim = [0.0, ylim_upper]

    fig = plot_comparison(xvalues,data,"iteration",
        linestyles={k:ls for k,ls in zip(sorted(data.keys()),linestyles)},
        colors={k:c for k,c in zip(sorted(data.keys()),colors)},
        markers={k:"" for k in sorted(data.keys())},
        labels={k:lb for k,lb in zip(sorted(data.keys()),labels)},
        legend_ncol=legend_ncol)

    xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

    fig.axes[0].set_xticks( xticks)
    fig.axes[0].set_xticklabels( xticks)

    ax, = fig.axes
    ax.set_ylim(ylim)

    change_aspect(ax)


    exp_id = str(int(time.time()))
    save_dir = f"results/exp_{exp_id}"

    os.makedirs(save_dir)

    fname = "_".join(sorted(set(identifiers)))

    fig.savefig(f"{save_dir.rstrip('/')}/{fname}_edited_graph.png", dpi=300)