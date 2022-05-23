
import torch
import numpy as np
import numbers
import pandas as pd
import os
import matplotlib.pyplot as plt


# https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = {
     'solid':                 'solid', # Same as (0, ()) or '-'
     'dotted':                'dotted', # Same as (0, (1, 1)) or ':'
     'dashed':                'dashed', # Same as '--'
     'dashdot':               'dashdot', # Same as '-.'
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),
     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}


def change_aspect(ax,ratio=0.5):
    #https://www.statology.org/matplotlib-aspect-ratio/

    #define y-unit to x-unit ratio
    #ratio = 0.5

    #get x and y limits
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    #set aspect ratio
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)


def plot_single_curve(ax,rates,xvalues,linestyle,color,label,marker):

    handle = None
    if (len(set(rates)) == 1) and (len(rates) != 1):
        marker = ""
    valid_indices = [i for i,v in enumerate(rates) if v != -1]
    if valid_indices:            
        handle,= ax.plot(
            [xvalues[i] for i in valid_indices], 
            [rates[i] for i in valid_indices], 
            linestyle=linestyle,color=color, 
            label=label,marker=marker
        )
    return handle


def plot_comparison(xvalues,data,xlabel,ylabel='bits/sample',xscale="linear",linestyles=None,colors=None,markers=None,
    labels=None,legend_ncol=None):

    if linestyles is None:
        linestyles = {"JBIG1":"dashdot","LUT":"dotted","MLP":"solid","STATIC":"dashed"}
    if colors is None:
        colors = {"JBIG1":'red',"LUT":'green',"MLP":'blue',"STATIC":'orange'}
    if markers is None:
        markers = {"JBIG1":'s',"LUT":'^',"MLP":'o',"STATIC":'v'}
    if labels is None:
        labels = {k:k for k in data.keys()}
    if legend_ncol is None:
        legend_ncol = 1
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4.8,4.8))    

    handles = []
    for k in data.keys():
        handle = plot_single_curve(ax,data[k],xvalues,linestyles[k],colors[k],labels[k],markers[k])
        if handle:
            handles.append(handle)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.legend(handles=handles,loc="upper right", ncol=legend_ncol)

    fig.tight_layout()
    plt.show()    
    return fig


def save_values(csv_name,xvalues,data,xlabel):
    csv_name = os.path.splitext(csv_name)[0]
    values = pd.DataFrame(data)
    values.index = xvalues
    values.index.name = xlabel
    values.to_csv(f"{csv_name}.csv")


def save_configs(csv_name,configs):
    csv_name = os.path.splitext(csv_name)[0]
    df=pd.DataFrame([{k:(v if isinstance(v,numbers.Number) else str(v)) for k,v in configs.items()}])
    df = df.T
    df.columns = ["value"]
    df.index.name = "key"
    df.to_csv(f"{csv_name}.csv")


def save_fig(fig_name,fig):
    fig_name = os.path.splitext(fig_name)[0]
    fig.savefig(f"{fig_name}.png", dpi=300)
    
    
def save_model(file_name,model):
    file_name = os.path.splitext(file_name)[0]
    torch.save(model.eval().state_dict(), f"{file_name}.pt")
    

def save_data(fn_prefix,xvalues,data,xlabel,ylabel='bits/sample',xscale="linear"):
    fig=plot_comparison(xvalues,data,xlabel,ylabel=ylabel,xscale=xscale)
    save_fig(f"{fn_prefix}_graph",fig)
    save_values(f"{fn_prefix}_values",xvalues,data,xlabel)