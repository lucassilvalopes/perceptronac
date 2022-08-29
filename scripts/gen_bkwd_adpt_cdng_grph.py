
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import change_aspect


def label_map(orig_lbl):
    label_map_dict = {
        "LUTmean": 'ALUT',
    }
    if "MLP" in orig_lbl:
        exponent = re.search(r'(?<=MLPlr=1e-).*(?=$)',orig_lbl).group()
        return 'APC $\lambda=10^{-'+ str(int(exponent)) + '}$'
    else:
        return label_map_dict[orig_lbl]


def linestyle_map(orig_lbl):
    linestyle_map_dict = {
        "LUTmean": "solid",
    }
    if "MLP" in orig_lbl:
        return "dashed"
    else:
        return linestyle_map_dict[orig_lbl]


def color_map(orig_lbl):
    color_map_dict = {
        "LUTmean": "g",
    }
    if "MLP" in orig_lbl:
        return "b"
    else:
        return color_map_dict[orig_lbl]


if __name__ == "__main__":
    
    csv_name = sys.argv[1]
    ylim_upper = float(sys.argv[2])
    legend_ncol = int(sys.argv[3])

    data = pd.read_csv(csv_name,index_col=0)#,header=0)
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


    fname = f"{os.path.splitext(csv_name)[0]}_edited_graph"

    fig.savefig(fname+".png", dpi=300)