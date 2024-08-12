
import sys
import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptronac.data_exportation import plot_comparison
from perceptronac.data_exportation import change_aspect
from perceptronac.data_exportation import linestyle_tuple


def label_map(orig_lbl,lr_symbol):
    label_map_dict = {
        "LUTmean": 'ALUT',
        "MLP": "APC",
        "RNN": "ARNN",
        "Xavier": "Xavier",
        "Ours": "Ours",
        "Random": "Random Initialization",
        "Pre-training": "Pre-training"
    }
    if "MLP" in orig_lbl or "RNN" in orig_lbl:
        ky = "MLP" if "MLP" in orig_lbl else "RNN"
        coefficient = re.search(r'(?<==).*(?=e-)',orig_lbl).group()
        exponent = re.search(r'(?<=e-).*(?=$)',orig_lbl).group()
        return label_map_dict[ky] + ' $'+lr_symbol+'='+str(coefficient)+'\cdot10^{-'+ str(int(exponent)) + '}$'
    else:
        return label_map_dict[orig_lbl]



def find_lr(orig_lbl):
    return float(re.search(r'[\d\.]{1,}e-[\d]{1,}',orig_lbl).group())


def linestyle_map(orig_lbl):
    linestyle_map_dict = {
        "LUTmean": "solid",
        0.1 : "dashdot",
        0.01 : "dashed", 
        0.001 : "solid", 
        0.0001 : "dotted", 
        "Ours" : "dashed", 
        "Random" : "dashed", 
        "Xavier": linestyle_tuple["densely dashed"],
        "Pre-training": linestyle_tuple['densely dashdotdotted'],
    }
    linestyle_map_dict_rnn = {
        0.05 : "solid",
        0.01 : linestyle_tuple['densely dotted'], 
        0.005 : linestyle_tuple['loosely dashed'], 
        0.001 : "dashdot", 
    }

    if "MLP" in orig_lbl:
        ky = find_lr(orig_lbl)
        return linestyle_map_dict[ky]
    elif "RNN" in orig_lbl:
        ky = find_lr(orig_lbl)
        return linestyle_map_dict_rnn[ky]
    else:
        return linestyle_map_dict[orig_lbl]


def color_map(orig_lbl):
    color_map_dict = {
        "LUTmean": "g",
        0.1 : "r",
        0.01 : "b",
        0.001 : "tab:gray",
        0.0001 : "c",
        "Ours" : "b",
        "Random" : "b",
        "Xavier": "k",
        "Pre-training": "tab:orange",
    }
    color_map_dict_rnn = {
        0.05 : "tab:red",
        0.01 : "m",
        0.005 : "k",
        0.001 : "tab:cyan",
    }


    if "MLP" in orig_lbl:
        ky = find_lr(orig_lbl)
        return color_map_dict[ky]
    elif "RNN" in orig_lbl:
        ky = find_lr(orig_lbl)
        return color_map_dict_rnn[ky]
    else:
        return color_map_dict[orig_lbl]


def read_csvs(csvs,rename=None):
    """
    Example:
    csvs = ["csv1.csv","csv2.csv","csv3.csv"]
    rename = [("MLPlr=1e-02","Xavier"),("MLPlr=1e-02","Ours"),("MLPlr=1e-02","Pre-training")]
    """

    csv_name = csvs[0]

    data = pd.read_csv(csv_name)
    if rename is not None:
        data = data.rename(columns={rename[0][0]: rename[0][1]})

    identifiers = re.findall(r'[\d]{10}',csv_name)

    for i in range(1,len(csvs)):
        csv_name = csvs[i]
        identifiers += re.findall(r'[\d]{10}',csv_name)
        partial_data = pd.read_csv(csv_name)
        if rename is not None:
            partial_data = partial_data.rename(columns={rename[i][0]: rename[i][1]})
        partial_data = partial_data.drop([c for c in data.columns if c !="iteration"],axis=1, errors='ignore')
        data = pd.merge(data,partial_data,on="iteration")
    
    data = data.rename(columns={"iteration": "sample"})

    data = data.set_index("sample")

    return data, identifiers


def plot_bkwd_adpt_cdng_grph(data,legend_ncol,columns,figsize=(4.8,4.8),lr_symbol="\lambda",new_order_sorted_keys=None,legend_loc="upper right",xlabel="sample"):

    xvalues = data.index

    data = data.to_dict(orient="list")

    data = {k:v for k,v in data.items() if k in columns}

    labels = [label_map(k,lr_symbol) for k in sorted(data.keys())]

    linestyles = [linestyle_map(k) for k in sorted(data.keys())]

    colors = [color_map(k) for k in sorted(data.keys())]

    fig = plot_comparison(xvalues,data,xlabel,#"iteration",
        linestyles={k:ls for k,ls in zip(sorted(data.keys()),linestyles)},
        colors={k:c for k,c in zip(sorted(data.keys()),colors)},
        markers={k:"" for k in sorted(data.keys())},
        labels={k:lb for k,lb in zip(sorted(data.keys()),labels)},
        legend_ncol=legend_ncol,figsize=figsize,
        new_order_sorted_keys=new_order_sorted_keys,
        legend_loc=legend_loc)

    return fig


def set_ticks(fig,len_data):

    xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

    fig.axes[0].set_xticks( xticks)
    fig.axes[0].set_xticklabels( xticks)

    return fig


def set_ylim(ax,ylim_upper):
    
    ylim = [0.0, ylim_upper]
    ax.set_ylim(ylim)


def save_bkwd_adpt_cdng_grph(save_dir,identifiers,fig):

    fname = "_".join(sorted(set(identifiers)))

    if save_dir:
        os.makedirs(save_dir)
        fig.savefig(f"{save_dir.rstrip('/')}/{fname}_edited_graph.png", dpi=300)
    else:
        fig.savefig(f"{fname}_edited_graph.png", dpi=300)




    
