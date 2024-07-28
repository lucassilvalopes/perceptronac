
import torch
import numpy as np
import numbers
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FormatStrFormatter
from math import log10, floor
# from perceptronac.convex_hull import convex_hull


# https://matplotlib.org/stable/gallery/color/named_colors.html
base_colors = OrderedDict({
    'b': (0, 0, 1),
    'g': (0, 0.5, 0),
    'r': (1, 0, 0),
    'c': (0, 0.75, 0.75),
    'm': (0.75, 0, 0.75),
    'y': (0.75, 0.75, 0),
    'k': (0, 0, 0),
    'w': (1, 1, 1)
})


# https://matplotlib.org/3.5.0/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = OrderedDict({
     'solid':                 'solid', # Same as (0, ()) or '-'
     'dotted':                'dotted', # Same as (0, (1, 1)) or ':'
     'dashed':                'dashed', # Same as '--'
     'dashdot':               'dashdot', # Same as '-.'
     'loosely dotted':        (0, (1, 10)),
     'dotted2':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed2':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),
     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
})


def find_n_decimal_places(number) -> int:
    fractional_part = abs(number)-int(abs(number))
    if fractional_part == 0:
        return 0
    base10 = log10(fractional_part)
    return abs(floor(base10))


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


def plot_comparison(
    xvalues,data,xlabel,ylabel='bits/sample',xscale="linear",linestyles=None,colors=None,markers=None,
    labels=None,legend_ncol=None,figsize=(4.8,4.8),new_order_sorted_keys=None,legend_loc="upper right",
    axes_labelsize=10,title=None):

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
    
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)    

    handles = []

    ks = sorted(data.keys())
    if new_order_sorted_keys:
        ks = np.array(ks)[new_order_sorted_keys].tolist()

    for k in ks:
        if isinstance(xvalues,dict):
            handle = plot_single_curve(ax,data[k],xvalues[k],linestyles[k],colors[k],labels[k],markers[k])
        else:
            handle = plot_single_curve(ax,data[k],xvalues,linestyles[k],colors[k],labels[k],markers[k])
        if handle:
            handles.append(handle)

    if title:
        ax.set_title(title, fontsize=axes_labelsize)

    ax.set_xlabel(xlabel, fontsize=axes_labelsize)
    ax.set_ylabel(ylabel, fontsize=axes_labelsize)
    ax.set_xscale(xscale)
    ax.legend(handles=handles,loc=legend_loc, ncol=legend_ncol)

    fig.tight_layout()
    # plt.show()    
    return fig


def save_values(csv_name,xvalues,data,xlabel,extra=None):
    csv_name = os.path.splitext(csv_name)[0]
    values = pd.DataFrame(data)
    if extra is not None:
        for k,v in extra.items():
            values[k] = v
    if isinstance(xvalues,dict):
        values.index = pd.MultiIndex.from_tuples(zip(*xvalues.values()), names=list(map( lambda k: f"{k}_index",xvalues.keys())))
    else:
        values.index = xvalues
        values.index.name = xlabel
    values.to_csv(f"{csv_name}.csv")


def load_values(csv_name):
    # columns = pd.read_csv(csv_name, nrows=0).columns.tolist()
    # index_col = []
    # for i,col in enumerate(columns):
    #     if col.endswith("_index"):
    #         index_col.append(i)
    # if len(index_col) == 0:
    #     index_col = 0
    # return pd.read_csv(csv_name,index_col=index_col)
    df = pd.read_csv(csv_name)
    d = df.to_dict(orient="list")
    return d


def save_dataframe(csv_name:str,data:pd.DataFrame,x_col:str,y_col:str,sort_by_x_col:bool = False):
    csv_name = os.path.splitext(csv_name)[0]
    if sort_by_x_col:
        data = data.sort_values(x_col)
    data.set_index(x_col).to_csv(f"{csv_name}.csv")


# def save_dataframe(fname,data,x_col,y_col):
#     save_values(
#         fname,
#         data[x_col],
#         {y_col:data[y_col]},
#         x_col,
#         extra={k:data[k] for k in data.columns if k not in [x_col,y_col]}
#     )


def save_configs(csv_name,configs):
    csv_name = os.path.splitext(csv_name)[0]
    df=pd.DataFrame([{k:(v if isinstance(v,numbers.Number) else str(v)) for k,v in configs.items()}])
    df = df.T
    df.columns = ["value"]
    df.index.name = "key"
    df.to_csv(f"{csv_name}.csv")


def save_fig(fig_name,fig):
    # https://stackoverflow.com/questions/45239261/matplotlib-savefig-text-chopped-off
    # https://stackoverflow.com/questions/53727761/black-background-behind-a-figures-labels-and-ticks-only-after-saving-figure-bu
    fig_name = os.path.splitext(fig_name)[0]
    fig.savefig(f"{fig_name}.png", dpi=300, facecolor='w', bbox_inches = "tight")
    
    
def save_model(file_name,model):
    file_name = os.path.splitext(file_name)[0]
    torch.save(model.eval().state_dict(), f"{file_name}.pt")
    

def save_data(fn_prefix,xvalues,data,xlabel,ylabel='bits/sample',xscale="linear",specify_xticks=False,extra=None,**kwargs):
    fig=plot_comparison(xvalues,data,xlabel,ylabel=ylabel,xscale=xscale,**kwargs)
    if specify_xticks:
        fig.axes[0].set_xticks(xvalues)
        fig.axes[0].set_xticklabels(xvalues)
    save_fig(f"{fn_prefix}_graph",fig)
    save_values(f"{fn_prefix}_values",xvalues,data,xlabel,extra)


# def points_in_convex_hull(data,x_col,y_col,log_x=False):
#     """
#     https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
#     https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
#     https://stackoverflow.com/questions/29188757/matplotlib-specify-format-of-floats-for-tick-labels
#     https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation
#     """


#     data = data[[x_col,y_col]]
#     normalized_data = data.copy()
#     if log_x:
#         normalized_data[x_col] = normalized_data[x_col].apply(lambda x : math.log10(x)) # complexity in logarithmic scale
    
#     scaler = MinMaxScaler()
#     normalized_data = scaler.fit_transform(normalized_data.values) # make the coordinates in the range [0,1]

#     fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(9.6,4.8), constrained_layout=True)

#     ax[0].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x") # verify that everything is alright
#     ax[1].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x") # verify that everything is alright

#     chull = data.values[convex_hull(normalized_data.tolist()),:]
#     ax[0].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x")
#     ax[0].plot(chull[:,0],chull[:,1],linestyle="solid",color="red",marker=None)

#     ax[1].plot(data.values[:,0],data.values[:,1],linestyle="",marker="x")
#     ax[1].plot(chull[:,0],chull[:,1],linestyle="solid",color="red",marker=None)


#     ax[0].set_xscale("log")
#     xvalues = ax[0].get_xticks()
#     ub = np.min(xvalues[xvalues>=np.max(data[x_col])])
#     lb = np.max(xvalues[xvalues<=np.min(data[x_col])])
#     xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]
#     ax[0].set_xticklabels([])
#     ax[0].set_xticklabels([], minor=True)
#     ax[0].set_xticks(xvalues)
#     ax[0].set_xticklabels(xvalues)

#     if np.any((np.max(chull,axis=0) - np.min(chull,axis=0)) == 0):
#         x_lb,x_ub = np.min(chull[:,0])-0.1*np.max(chull[:,0]),1.1*np.max(chull[:,0])
#         y_lb,y_ub = np.min(chull[:,1])-0.1*np.max(chull[:,1]),1.1*np.max(chull[:,1])            
#     else:
#         x_range = np.max(chull[:,0]) - np.min(chull[:,0])
#         y_range = np.max(chull[:,1]) - np.min(chull[:,1])
#         x_lb,x_ub = np.min(chull[:,0])-0.1*x_range,np.max(chull[:,0])+0.1*x_range
#         y_lb,y_ub = np.min(chull[:,1])-0.1*y_range,np.max(chull[:,1])+0.1*y_range

#     ax[1].set_xlim(x_lb,x_ub)
#     ax[1].set_ylim(y_lb,y_ub) 

#     # xvalues = ax[1].get_xticks()
#     # lb = np.min(xvalues[xvalues>=x_lb])
#     # ub = np.max(xvalues[xvalues<=x_ub])
#     # xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]

#     xvalues = np.unique(
#         [np.min(chull[:,0]),(np.max(chull[:,0])+np.min(chull[:,0]))/2,np.max(chull[:,0])])

#     if len(xvalues) ==1 :
#         xvalues = [x_lb,xvalues[0],x_ub]

#     n_xticks = 3
#     ax[1].set_xticks(xvalues[0:len(xvalues):np.max([len(xvalues)//n_xticks,1])])
#     ax[1].set_xticklabels(xvalues[0:len(xvalues):np.max([len(xvalues)//n_xticks,1])])
#     d = find_n_decimal_places(np.min(np.diff(xvalues)))
#     ax[1].xaxis.set_major_formatter(FormatStrFormatter(f'%.{d+1}f'))

#     ax[0].set_xlabel(x_col)
#     ax[0].set_ylabel(y_col)
#     ax[1].set_xlabel(x_col)
#     ax[1].set_ylabel(y_col)

#     return convex_hull(normalized_data.tolist()),fig

