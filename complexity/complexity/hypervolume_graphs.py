#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ast
import pandas as pd
import numpy as np
from collections import OrderedDict

import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt


# In[ ]:


def plot_comparison(
    xvalues,data,xlabel,ylabel,xscale,linestyles,colors,markers,
    labels,legend_ncol,figsize,legend_loc,
    axes_labelsize,title):

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)    

    handles = []

    ks = sorted(data.keys())

    for k in ks:
        handle,= ax.plot(xvalues, data[k], 
            linestyle=linestyles[k],color=colors[k],label=labels[k],marker=markers[k])
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


# In[ ]:


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


# In[ ]:


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


# In[2]:


import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pgf.preamble": [
        r"\usepackage{siunitx}",
        r"\usepackage{inputenc}",
        r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}",
        r"\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}",
#         r"\newcommand{muJ}{\ensuremath{\mu J}}"
        
    ]})


# In[3]:


def custom_df_to_dict(in_df):

    out_dict = dict()
    for hsh in set(str(in_df[c].values.tolist()) for c in in_df.columns):
        lbls = []
        for c in in_df.columns:
            if str(in_df[c].values.tolist()) == hsh:
                lbls.append(c)
        try:
            out_dict[",".join(lbls)] = ast.literal_eval(hsh.replace("nan","None"))
        except:
            raise ValueError(hsh.replace("nan","None"))
    
    return out_dict


# In[4]:


def line_func(k):
    
    lines = list(OrderedDict({
        'densely dashed':linestyle_tuple['densely dashed'],
        'solid':linestyle_tuple['solid'],
        'long dash with offset': linestyle_tuple['long dash with offset'],
        'densely dashdotdotted': linestyle_tuple['densely dashdotdotted'], # never used
        'dotted':linestyle_tuple['dotted'],
        'dashdotdotted': linestyle_tuple['dashdotdotted'],
        'dashdot':linestyle_tuple['dashdot'],
        'dashed2':linestyle_tuple['dashed2'],
        'loosely dashed':linestyle_tuple['loosely dashed']
    }).values())
    
    if k == "Sobol":
        lin3 = lines[5]
    elif k == "qNEHVI":
        lin3 = lines[6]
    elif k == "qNParEGO":
        lin3 = lines[7]
    elif "Alg.2" in k:
        lin3 = lines[0]
    elif "Alg.3" in k:
        lin3 = lines[1]
    elif "Alg.4" in k:
        lin3 = lines[2]
    elif "Alg.5" in k:
        lin3 = lines[1] #lines[3]
    elif "Alg.6" in k:
        lin3 = lines[4]
    elif "Local Max HV" in k:
        lin3 = lines[3]
    elif "Max HV" in k:
        lin3 = lines[8]
#     print(k,"line",lin3)
    return lin3

def color_func(k):
    
#     colors1 = list(base_colors.keys())
#     colors2 = list(mcolors.TABLEAU_COLORS.keys())
    
#     colors = colors1[:7] + colors2[7:]
    
    colors = list(OrderedDict({
#         'b': base_colors['b'],
#         'g': base_colors['g'],
#         'r': base_colors['r'],
#         'c': base_colors['c'],
#         'm': base_colors['m'],
#         'y': base_colors['y'],
#         'k': base_colors['k'],
#         'w': base_colors['w'],
        'tab:red': mcolors.TABLEAU_COLORS['tab:red'], 
        'tab:green': mcolors.TABLEAU_COLORS['tab:green'],
        'tab:blue': mcolors.TABLEAU_COLORS['tab:blue'],
        'tab:purple': mcolors.TABLEAU_COLORS['tab:purple'], # never used
        'k': base_colors['k'],
        'tab:orange': mcolors.TABLEAU_COLORS['tab:orange'],
#         'tab:brown': mcolors.TABLEAU_COLORS['tab:brown'],
        'tab:pink': mcolors.TABLEAU_COLORS['tab:pink'],
        'tab:gray': mcolors.TABLEAU_COLORS['tab:gray'],
        'tab:olive': mcolors.TABLEAU_COLORS['tab:olive'],
        'tab:cyan': mcolors.TABLEAU_COLORS['tab:cyan']        
    }).values())
    

    if k == "Sobol":
        color = colors[5]
    elif k == "qNEHVI":
        color = colors[6]
    elif k == "qNParEGO":
        color = colors[7]
    elif "Alg.2" in k:
        color = colors[0]
    elif "Alg.3" in k:
        color = colors[1]
    elif "Alg.4" in k:
        color = colors[2]
    elif "Alg.5" in k:
        color = colors[1] #colors[3]
    elif "Alg.6" in k:
        color = colors[4]
    elif "Local Max HV" in k:
        color = colors[8]
    elif "Max HV" in k:
        color = colors[4]
#     print(k,"color",color)
    return color


# In[5]:


def mohpo_grph(pth,title,up_to_complexity,upper_clip=None,lower_clip=None,
               y_axis_units="Hypervolume"):

    if up_to_complexity:
        df = pd.read_csv(pth,index_col="complexity").drop([
            "iters",
            "c_angle_rule",
#             "u_angle_rule",
            "c_gift_wrapping",
            "u_gift_wrapping",
            "c_tie_break"
        ],axis=1)
    else:
        df = pd.read_csv(pth,index_col=0)

    if y_axis_units=="Hypervolume":
        df1 = df.drop("max_hv",axis=1)
        df1["global_hv"] = df.shape[0]*[df["max_hv"].iloc[-1]]
        
    elif y_axis_units=="Log Hypervolume":
        df1 = df.applymap(lambda x: np.log10(x) if x else None)
    elif y_axis_units=="Hypervolume Difference":
#         df1 = (- df.drop("max_hv",axis=1).sub(df["max_hv"], axis=0))
        max_hv = df["max_hv"].iloc[-1]
        df1 = (max_hv - df.drop("max_hv",axis=1))
        
    elif y_axis_units=="Log Hypervolume Difference":
#         df1 = (- df.drop("max_hv",axis=1).sub(df["max_hv"], axis=0)).applymap(lambda x: np.log10(x) if x else None)
        max_hv = df["max_hv"].iloc[-1]
        df1 = (max_hv - df.drop("max_hv",axis=1)).applymap(np.log10)
        
    else:
        raise ValueError(y_axis_units)
    





    df2 = df1.rename(columns={
    "global_hv":"Max HV",
    "max_hv": "Local Max HV", 
    "sobol": "Sobol", 
    "ehvi": "qNEHVI",
    "parego": "qNParEGO",
    "sobol_hv_list": "Sobol", 
    "ehvi_hv_list": "qNEHVI",
    "parego_hv_list": "qNParEGO",
    "c_angle_rule": "Alg.6",
    "u_angle_rule": "Alg.5",
    "c_gift_wrapping": "Alg.3",
    "u_gift_wrapping": "Alg.2",
    "c_tie_break": "Alg.4"})

    df3 = df2[sorted(df2.columns)]

    # df4 = df3.fillna(value=-1)
    # df4 = df3

    xvalues = df3.index
    try:
        data = custom_df_to_dict(df3)
    except:
        import pdb
        pdb.set_trace()
    xlabel= f"GLCH Complexity Level ({title})" if up_to_complexity else "number of visited networks"
    ylabel=y_axis_units
    xscale= "log" if (title in ["Encoded model bits","Multiply/adds per pixel"]) and up_to_complexity else "linear"
    linestyles={k:line_func(k) for k in data.keys()}
    colors={k:color_func(k) for k in data.keys()}
    markers = {k:"" for k in sorted(data.keys())}
    labels = {k:k for k in data.keys()}
    legend_ncol=1
    figsize=(6.0,5.1) # without title : (6.0,4.8)
    
    legend_loc="best"
    axes_labelsize=16

    fig = plot_comparison(
        xvalues,
        data,
        xlabel,
        ylabel,
        xscale,
        linestyles,
        colors,
        markers,
        labels,
        legend_ncol,
        figsize,
        legend_loc,
        axes_labelsize,
        title if not up_to_complexity else None
    )
    
    if (upper_clip is not None) or (lower_clip is not None):
        ymin, ymax = fig.axes[0].get_ylim()
        if (upper_clip is not None):
            ymax = upper_clip
        if (lower_clip is not None):
            ymin = lower_clip
        fig.axes[0].set_ylim(ymin,ymax)
        
#     yfmt = ScalarFormatter(useMathText=True)
#     yfmt.set_powerlimits((-5, 5))
#     fig.axes[0].yaxis.set_major_formatter(yfmt) 
#     xfmt = ScalarFormatter(useMathText=True)
#     xfmt.set_powerlimits((-5, 5))
#     fig.axes[0].xaxis.set_major_formatter(xfmt) 
    
    return fig


# In[6]:


def gen_graphs(
    adjusted_data_folder,min_support,up_to_complexity,upper_clips=None,lower_clips=None,
    y_axis_units="Hypervolume"):
    
    if upper_clips is None:
        upper_clips = {"micro_joules_per_pixel":None,"model_bits":None,"params":None}
    if lower_clips is None:
        lower_clips = {"micro_joules_per_pixel":None,"model_bits":None,"params":None}

    for x_axis in ["micro_joules_per_pixel","model_bits","params"]:

        fldr = adjusted_data_folder
        if up_to_complexity:
            fil3 = \
            f"{x_axis}_data_bits_over_data_samples_ax_methods_avgs_adjusted_up_to_complexity_support{min_support}.csv"
        else:
            fil3 = \
            f"{x_axis}_data_bits_over_data_samples_ax_methods_avgs_adjusted_support{min_support}.csv"
        pth = fldr + fil3

        title_map = {
            "micro_joules_per_pixel":"$\SI{}{\mu\joule}$ per pixel",
            "model_bits":"Encoded model bits", 
            "params":"Multiply/adds per pixel"
        }

        fig = mohpo_grph(pth,title_map[x_axis],up_to_complexity,upper_clips[x_axis],lower_clips[x_axis],y_axis_units)
        fig.savefig(fil3.replace(".csv",f"_{y_axis_units.replace(' ','_').lower()}.png"), dpi=300)


# In[7]:


def gen_graphs_rdc(
    adjusted_data_folder,min_support,up_to_complexity,upper_clips=None,lower_clips=None,
    y_axis_units="Hypervolume"):

    if upper_clips is None:
        upper_clips = {"flops":None,"params":None}
    if lower_clips is None:
        lower_clips = {"flops":None,"params":None}
    
    for x_axis in ["params","flops"]:
        fldr = adjusted_data_folder
        if up_to_complexity:
            fil3 = f"bpp_loss_mse_loss_{x_axis}_ax_methods_avgs_adjusted_up_to_complexity_support{min_support}.csv"
        else:
            fil3 = f"bpp_loss_mse_loss_{x_axis}_ax_methods_avgs_adjusted_support{min_support}.csv"
        pth = fldr+fil3
        
        title_map = {
            "flops":"FLOPs",
            "params":"Number of Parameters",
        }
        
        fig = mohpo_grph(pth,title_map[x_axis],up_to_complexity,upper_clips[x_axis],lower_clips[x_axis],y_axis_units)
        fig.savefig(fil3.replace(".csv",f"_{y_axis_units.replace(' ','_').lower()}.png"), dpi=300)


