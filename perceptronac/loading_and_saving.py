
import torch
import numpy as np
import numbers
import pandas as pd
import os
import matplotlib.pyplot as plt


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


def plot_comparison(xvalues,data,xlabel,xscale="linear",linestyles=None,colors=None,markers=None):

    if linestyles is None:
        linestyles = ["dashdot","dotted","solid","dashed"]
    if colors is None:
        colors = ['red','green','blue','orange']
    if markers is None:
        markers = ['s','^','o','v']

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4.8,4.8))    

    handles = []
    for k,linestyle,color,marker in zip(sorted(data.keys()),linestyles,colors,markers):
        handle = plot_single_curve(ax,data[k],xvalues,linestyle,color,k,marker)
        if handle:
            handles.append(handle)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('bits/sample')
    ax.set_xscale(xscale)
    ax.legend(handles=handles,loc="upper right")

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
    
