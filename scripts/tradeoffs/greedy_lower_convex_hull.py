# %%
import pandas as pd
import sys
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from perceptronac.convex_hull import convex_hull
from perceptronac.power_consumption import estimate_joules, get_n_pixels
from perceptronac.power_consumption import group_energy_measurements
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from glch_utils import min_max_convex_hull

from glch_BandD import GLCH



matplotlib.use("pgf")
matplotlib.rcParams.update({"pgf.texsystem": "pdflatex","pgf.preamble": [r"\usepackage{siunitx}"]})



def build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",scale_x=1,scale_y=1,debug=True,title=None):
    return GLCH(data,possible_values,x_axis,y_axis,initial_values,to_str_method,start,scale_x,scale_y,debug,title).build_tree()



# %%
def print_tree(node,file=None):
    # print("parent", node)

    if len(node.children) == 0:
        return

    children_str = ""
    for i,c in enumerate(node.children):
        if i in node.chosen_child_indices:
            prefix = ''.join(["!"] * (node.chosen_child_indices.index(i)+1) )
        else:
            prefix = ""
        
        children_str += f"{prefix}{c} "

    print(node,children_str,file=file)

    print("\n",file=file)
    
    for i,c in enumerate(node.children):
        print_tree(node.children[i],file=file)


# %%

from line_clipping import cohenSutherlandClip


def paint_tree(ax,data,node,x_axis,y_axis,x_range,y_range):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"

    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    """

    if len(node.children) == 0:
        return

    for i,c in enumerate(node.children):
        if i in node.chosen_child_indices:
            if node.chosen_child_indices.index(i) == 0:
                color = "green"
            else:
                color = "darkgoldenrod"
        else:
            color = "firebrick"
        line_x_vec = data.loc[[str(node), str(c)],x_axis].values
        line_y_vec = data.loc[[str(node), str(c)],y_axis].values

        if (x_range is not None) and (y_range is not None):

            xd = 0.01 * np.abs(x_range[1] - x_range[0])
            yd = 0.01 * np.abs(y_range[1] - y_range[0])

            x1,y1,x2,y2 = cohenSutherlandClip(
                x_range[0]+xd,x_range[1]-xd,y_range[0]+yd,y_range[1]-yd,line_x_vec[0],line_y_vec[0],line_x_vec[1],line_y_vec[1])
        else:
            x1,y1,x2,y2 = line_x_vec[0], line_y_vec[0], line_x_vec[1], line_y_vec[1]

        if (x1 is not None) and (y1 is not None) and (x2 is not None) and (y2 is not None):
            ax.annotate('',
                xytext= (x1,y1),
                xy= (x2,y2),
                arrowprops=dict(arrowstyle="->", color=color),
                # size=size
            )
    
    for i,c in enumerate(node.children):
        paint_tree(ax,data,node.children[i],x_axis,y_axis,x_range,y_range)

# %%

def tree_nodes(n, points, mode):

    for i,c in enumerate(n.children):
        if mode == "all":
            points.append(str(c))
        elif mode == "first":
            if (len(n.chosen_child_indices) > 0) and (i == n.chosen_child_indices[0]):
                points.append(str(c))
        elif mode == "second":
            if (i in n.chosen_child_indices) and (i != n.chosen_child_indices[0]):
                points.append(str(c))
        elif mode == "lch":
            if c.lch:
                points.append(str(c))
        else:
            raise ValueError(mode)

    for i,c in enumerate(n.children):
        tree_nodes(n.children[i],points,mode)

    return points




# %%


def paint_cloud(data,x_axis,y_axis,ax,marker):
    ax.plot(data.loc[:,x_axis].values,data.loc[:,y_axis].values,linestyle="",color="tab:blue",marker=marker)


def paint_root(data,r,x_axis,y_axis,ax):
    ax.plot([data.loc[str(r),x_axis]],[data.loc[str(r),y_axis]],linestyle="",color="yellow",marker="o")


def adjust_axes(x_axis,y_axis,x_range,y_range,ax,x_in_log_scale=False,x_alias=None,y_alias=None):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """
    
    if x_in_log_scale:
        ax.set_xscale("log")
        # xvalues = ax.get_xticks()
        # ub = np.min(xvalues[xvalues>=np.max(data[x_col])])
        # lb = np.max(xvalues[xvalues<=np.min(data[x_col])])
        # xvalues = xvalues[np.logical_and(xvalues>=lb,xvalues<=ub)]
        # ax.set_xticklabels([])
        # ax.set_xticklabels([], minor=True)
        # ax.set_xticks(xvalues)
        # ax.set_xticklabels(xvalues)

    ax.set_xlabel(x_alias if x_alias else x_axis)
    ax.set_ylabel(y_alias if y_alias else y_axis)
    if x_range:
        ax.set_xlim(x_range[0],x_range[1])
    if y_range:
        ax.set_ylim(y_range[0],y_range[1])


# %%

def compute_hulls(data,rs,x_axis,y_axis):

    true_hull_points = data.iloc[min_max_convex_hull(data.loc[:,[x_axis,y_axis]].values.tolist()),:]
    
    new_points = []
    for r in rs:
        new_points += [str(r)]
        new_points += tree_nodes(r,[],"all")
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[min_max_convex_hull(probe.loc[:,[x_axis,y_axis]].values.tolist()),:]

    n_trained_networks = len(set([re.sub(r'_[\d]{2}b','',pt) for pt in new_points]))

    return true_hull_points,estimated_hull_points,n_trained_networks


def paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """
    ax.plot(true_hull_points[x_axis],true_hull_points[y_axis],linestyle=(0, (5, 5)),color="orangered",marker=None)
    ax.plot(
        estimated_hull_points[x_axis],estimated_hull_points[y_axis],linestyle="dotted",color="black",marker=None)


def paint_hull_points(true_hull_points,x_axis,y_axis,ax):

    ax.plot(true_hull_points[x_axis],true_hull_points[y_axis],linestyle="",color="orangered",marker=".")


def paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """
    ax.plot(
        estimated_hull_points[x_axis],estimated_hull_points[y_axis],linestyle="",color="black",marker="o",
        markerfacecolor='none',markersize=8)
    # plt.scatter(
    #     estimated_hull_points[x_axis],estimated_hull_points[y_axis],facecolors='none', edgecolors='k')


def paint_nodes(data,r,x_axis,y_axis,ax,is_hulls_fig=False):

    if is_hulls_fig:
        lch_nodes = [str(r)] + tree_nodes(r,[], "lch")
        lch_nodes_xy = data.loc[lch_nodes,:]
        ax.plot(lch_nodes_xy[x_axis],lch_nodes_xy[y_axis],linestyle="",color="black",marker="o",
                markerfacecolor='none',markersize=8)
    else:
        first_selected_nodes = [str(r)] + tree_nodes(r,[], "first")
        second_selected_nodes = tree_nodes(r,[], "second")
        first_selected_nodes_xy = data.loc[first_selected_nodes,:]
        second_selected_nodes_xy = data.loc[second_selected_nodes,:]
        all_nodes = [str(r)] + tree_nodes(r,[], "all")
        unselected_nodes = list((set(all_nodes) - set(first_selected_nodes)) - set(second_selected_nodes) )
        unselected_nodes_xy = data.loc[unselected_nodes,:]
        ax.plot(first_selected_nodes_xy[x_axis],first_selected_nodes_xy[y_axis],linestyle="",color="green",marker=".")
        ax.plot(second_selected_nodes_xy[x_axis],second_selected_nodes_xy[y_axis],linestyle="",color="darkgoldenrod",marker=".")
        ax.plot(unselected_nodes_xy[x_axis],unselected_nodes_xy[y_axis],linestyle="",color="firebrick",marker=".")


def save_all_data(data,r,x_axis,y_axis,x_range,y_range,data_id,x_in_log_scale=False,x_alias=None,y_alias=None):

    true_hull_points,estimated_hull_points,n_trained_networks = compute_hulls(data,[r],x_axis,y_axis)

    with open(f'glch_results/tree_{data_id}.txt', 'w') as f:
        print_tree(r,file=f)
        print(f"number of trained networks : {n_trained_networks}",file=f)

    tree_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,".")
    paint_root(data,r,x_axis,y_axis,ax)
    paint_tree(ax,data,r,x_axis,y_axis,x_range,y_range)
    # paint_hull_points(true_hull_points,x_axis,y_axis,ax)
    paint_nodes(data,r,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,x_range,y_range,ax,x_in_log_scale,x_alias,y_alias)
    tree_fig.savefig(f"glch_results/tree_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    # paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    paint_nodes(data,r,x_axis,y_axis,ax,is_hulls_fig=True)
    adjust_axes(x_axis,y_axis,None,None,ax,x_in_log_scale,x_alias,y_alias)
    hulls_fig.savefig(f"glch_results/hulls_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(f"glch_results/hulls_{data_id}",true_hull_points,estimated_hull_points)



# %%
# from scipy.sparse.csgraph import connected_components


# def labaled_points_figure(data):

#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     fig.set_size_inches(27.9,21.6)
#     ax.plot(data.loc[:,"joules"].values,data.loc[:,"data_bits/data_samples"].values,linestyle="",marker="x")
#     ax.set_xlabel("joules")
#     ax.set_ylabel("data_bits/data_samples")

#     X = data.loc[:,["joules","data_bits/data_samples"]].values
#     adj_mtx = np.logical_and(
#         np.sum((np.expand_dims(X,1) - np.expand_dims(X,0))**2,axis=2) > 0,
#         np.logical_and(
#             np.abs(np.expand_dims(X[:,0],1) - np.expand_dims(X[:,0],0)) < 0.5,
#             np.abs(np.expand_dims(X[:,1],1) - np.expand_dims(X[:,1],0)) < 0.0003,
#         )
#     )

#     n_conn_comp, conn_comp_mask = connected_components(adj_mtx)

#     points_to_merge =[]
#     for cc in range(n_conn_comp):
#         if np.sum(conn_comp_mask == cc) > 1:
#             points_to_merge.append(data.index.values[conn_comp_mask == cc].tolist())

#     for top,row in data.loc[:,["joules","data_bits/data_samples"]].iterrows():
#         normal_point = True
#         for g in points_to_merge:
#             if top in g:
#                 normal_point = False
#         if normal_point:
#             ax.text(
#                 x=row["joules"], #+0.5,
#                 y=row["data_bits/data_samples"]-0.0003,
#                 s=",".join(list(map(lambda x: str(int(x)),top.split("_")[1:3]))), 
#                 # fontdict=dict(color='black',size=8),
#                 # bbox=dict(facecolor='yellow',alpha=0.5)
#             )

#     for g in points_to_merge:
#         sorted_i = np.argsort(data.loc[g,"joules"].values)
#         ax.text(
#             x=data.loc[g[sorted_i[0]],"joules"], #+0.5,
#             y=data.loc[g[sorted_i[0]],"data_bits/data_samples"]-0.0003,
#             s="/".join(list(map(lambda y: ",".join(list(map(lambda x: str(int(x)),y.split("_")[1:3]))) , np.array(g)[sorted_i].tolist()))), 
#             # fontdict=dict(color='black',size=8),
#             # bbox=dict(facecolor='yellow',alpha=0.5)
#         )

#     # fig.savefig('test2png.png', dpi=300, facecolor='w', bbox_inches = "tight")

#     return fig

# %%
from decimal import Decimal

# https://stackoverflow.com/questions/45332056/decompose-a-float-into-mantissa-and-exponent-in-base-10-without-strings

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

def limit_significant_digits(value,last_significant_digit_position):
    factor = 10**last_significant_digit_position
    return np.round(value/factor) * factor

def limit_energy_significant_digits(data,x_axis):

    mean_col=x_axis
    std_col=f"{x_axis}_std"

    last_significant_digit_position = fexp(data[std_col].max())

    data[[mean_col,std_col]] = data[[mean_col,std_col]].apply(lambda x: pd.Series({
        mean_col:limit_significant_digits(x[mean_col],last_significant_digit_position),
        std_col:limit_significant_digits(x[std_col],last_significant_digit_position)
        # mean_col:limit_significant_digits(x[mean_col],fexp(x[std_col])),
        # std_col:limit_significant_digits(x[std_col],fexp(x[std_col]))
    },index=[mean_col,std_col]), axis=1)
    return data

# %%


def save_hull_points(file_name,true_hull_points,estimated_hull_points):

    with open(f'{file_name}.txt', 'w') as f:
        print("true_hull_points",file=f)
        print(true_hull_points,file=f)
        print("estimated_hull_points",file=f)
        print(estimated_hull_points,file=f)


def get_energy_data(csv_path,remove_noise):

    data = pd.read_csv(csv_path)

    csv_path_2 = csv_path.replace("raw_values","power_draw")

    power_draw = np.loadtxt(csv_path_2)

    power_draw[:,1] = power_draw[:,1] - 16 # np.min(power_draw[:,1])

    joules = estimate_joules(data,power_draw)

    data["joules"] = joules

    data = group_energy_measurements(data).set_index("topology")

    csv_path_3 = csv_path.replace("raw_values","conf")

    n_pixels = get_n_pixels(csv_path_3)

    data["joules_per_pixel"] = data["joules"] / n_pixels

    data["joules_per_pixel_std"] = data["joules_std"] / n_pixels

    data["micro_joules_per_pixel"] = data["joules_per_pixel"] / 1e-6

    data["micro_joules_per_pixel_std"] = data["joules_per_pixel_std"] / 1e-6
    
    if remove_noise:

        limit_energy_significant_digits(data,"time")
        limit_energy_significant_digits(data,"joules")
        limit_energy_significant_digits(data,"joules_per_pixel")
        limit_energy_significant_digits(data,"micro_joules_per_pixel")

    return data


def glch_rate_vs_energy(
        csv_path,x_axis,y_axis,title,
        scale_x=None,scale_y=None,
        x_range=None,y_range=None,
        x_in_log_scale=False,remove_noise=True,
        x_alias=None,y_alias=None
    ):

    data = get_energy_data(csv_path,remove_noise)

    # data[x_axis] = data[x_axis].values/scale_x
    # data[y_axis] = data[y_axis].values/scale_y

    if scale_x is None and scale_y is None:

        scale_x = data.loc[["032_010_010_001","032_640_640_001"],x_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],x_axis].min()
        scale_y = data.loc[["032_010_010_001","032_640_640_001"],y_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],y_axis].min()

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640]
    }

    initial_values = {"h1":10,"h2":10}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths))

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y,title=title)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,title,
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias)


def glch_rate_vs_time(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_params(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


# def glch_rate_vs_params(
#         csv_path,x_axis,y_axis,title,
#         scale_x=None,scale_y=None,
#         x_range=None,y_range=None,
#         x_in_log_scale=False,
#         x_alias=None,y_alias=None
#     ):

#     data = pd.read_csv(csv_path).set_index("topology")

#     # data[x_axis] = data[x_axis].values/scale_x
#     # data[y_axis] = data[y_axis].values/scale_y

#     if scale_x is None and scale_y is None:

#         scale_x = data.loc[["032_010_010_001","032_640_640_001"],x_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],x_axis].min()
#         scale_y = data.loc[["032_010_010_001","032_640_640_001"],y_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],y_axis].min()

#     possible_values = {
#         "h1": [10,20,40,80,160,320,640],
#         "h2": [10,20,40,80,160,320,640]
#     }

#     initial_values = {"h1":10,"h2":10}

#     def to_str_method(params):
#         widths = [32,params["h1"],params["h2"],1]
#         return '_'.join(map(lambda x: f"{x:03d}",widths))

#     r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y,title=title)

#     save_all_data(data,r,x_axis,y_axis,x_range,y_range,title,
#         x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias)



def glch_rate_vs_dist(
        csv_path,x_axis,y_axis,
        scale_x=None,scale_y=None,
        x_range=None,y_range=None,
        start="left",
        x_alias=None,y_alias=None
    ):

    data = pd.read_csv(csv_path).set_index("labels")

    # data[x_axis] = data[x_axis].values/scale_x
    # data[y_axis] = data[y_axis].values/scale_y

    if scale_x is None and scale_y is None:

        scale_x = data.loc[["L5e-3N32M32","L2e-2N224M320"],x_axis].max() - data.loc[["L5e-3N32M32","L2e-2N224M320"],x_axis].min()
        scale_y = data.loc[["L5e-3N32M32","L2e-2N224M320"],y_axis].max() - data.loc[["L5e-3N32M32","L2e-2N224M320"],y_axis].min()

    possible_values = {
        "L": ["5e-3", "1e-2", "2e-2"],
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

    # x_axis = "bpp_loss"
    # y_axis = "mse_loss"

    if start == "right":
        possible_values = {k:v[::-1] for k,v in possible_values.items()}

    initial_values = {
        "L":possible_values["L"][0], 
        "N":possible_values["N"][0],
        "M":possible_values["M"][0]
    }

    def to_str_method(params):
        return f"L{params['L']}N{params['N']}M{params['M']}"
    
    if start == "right":
        r = build_tree(data,possible_values,y_axis,x_axis,initial_values,to_str_method,scale_x=scale_y,scale_y=scale_x)
    else:
        r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,f'{x_axis}_vs_{y_axis}_start_{start}',
        x_alias=x_alias,y_alias=y_alias)


def get_x_range_y_range(data,x_axis,y_axis):

    tmp_fig, tmp_ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,tmp_ax,".")
    x_range,y_range = tmp_ax.get_xlim(),tmp_ax.get_ylim()
    return x_range,y_range


def glch_rate_vs_dist_2(csv_path,x_axis,y_axis,scale_x=None,scale_y=None,x_range=None,y_range=None,start="left"):

    data = pd.read_csv(csv_path).set_index("labels")

    # data[x_axis] = data[x_axis].values/scale_x
    # data[y_axis] = data[y_axis].values/scale_y

    if scale_x is None and scale_y is None:

        scale_x = data.loc[["L5e-3N32M32","L2e-2N224M320"],x_axis].max() - data.loc[["L5e-3N32M32","L2e-2N224M320"],x_axis].min()
        scale_y = data.loc[["L5e-3N32M32","L2e-2N224M320"],y_axis].max() - data.loc[["L5e-3N32M32","L2e-2N224M320"],y_axis].min()

    brute_dict = {
        "L": ["5e-3", "1e-2", "2e-2"]
    }

    greedy_dict = {
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

    if start == "right":
        greedy_dict = {k:v[::-1] for k,v in greedy_dict.items()}

    initial_state = {
        "N":greedy_dict["N"][0],
        "M":greedy_dict["M"][0]
    }

    def to_str_method_factory(brute_params):
        def to_str_method(greedy_params):
            return f"L{brute_params['L']}N{greedy_params['N']}M{greedy_params['M']}"
        return to_str_method


    brute_keys = "".join(list(brute_dict.keys()))
    greedy_keys = "".join(list(greedy_dict.keys()))
    exp_id = f"{x_axis}_vs_{y_axis}_brute_{brute_keys}_greedy_{greedy_keys}_start_{start}"

    rs = []
    tree_file = open(f'glch_results/tree_{exp_id}.txt', 'w')

    tree_fig, ax = plt.subplots(nrows=1, ncols=len(brute_dict["L"]))


    if (x_range is None) and (y_range is None):
        x_range,y_range = get_x_range_y_range(data,x_axis,y_axis)

    for i,L in enumerate(brute_dict["L"]):

        to_str_method = to_str_method_factory({"L":L})

        current_data = data.iloc[[i for i,lbl in enumerate(data.index) if f"L{L}" in lbl],:]
        if start == "right":
            r = build_tree(current_data,greedy_dict,y_axis,x_axis,initial_state,to_str_method,scale_x=scale_y,scale_y=scale_x)
        else:
            r = build_tree(current_data,greedy_dict,x_axis,y_axis,initial_state,to_str_method,scale_x=scale_x,scale_y=scale_y)
        
        rs.append(r)

        print_tree(r,file=tree_file)

        paint_root(data,r,x_axis,y_axis,tree_fig.axes[i])
        # paint_cloud(data,x_axis,y_axis,tree_fig.axes[i],".")
        paint_cloud(current_data,x_axis,y_axis,tree_fig.axes[i],".")
        paint_tree(tree_fig.axes[i],data,r,x_axis,y_axis,x_range,y_range)
        adjust_axes(x_axis,y_axis,x_range,y_range,tree_fig.axes[i])

        if i != 0:
            tree_fig.axes[i].set_yticks([])
            tree_fig.axes[i].set_ylabel('')
    
    true_hull_points,estimated_hull_points,n_trained_networks = compute_hulls(data,rs,x_axis,y_axis)

    print(f"number of trained networks : {n_trained_networks}",file=tree_file)
    tree_file.close()

    for i in range(len(brute_dict["L"])):
        # paint_hull_points(true_hull_points,x_axis,y_axis,tree_fig.axes[i])
        paint_nodes(data,rs[i],x_axis,y_axis,tree_fig.axes[i])

    tree_fig.savefig(f"glch_results/tree_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    
    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    # paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    for i in range(len(brute_dict["L"])):
        paint_nodes(data,rs[i],x_axis,y_axis,ax,is_hulls_fig=True)
    adjust_axes(x_axis,y_axis,None,None,ax)
    hulls_fig.savefig(f"glch_results/hulls_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(f"glch_results/hulls_fig_{exp_id}",true_hull_points,estimated_hull_points)


def glch_model_bits_vs_data_bits(
        csv_path,x_axis,y_axis,
        scale_x=None,scale_y=None,
        x_range=None,y_range=None,
        x_in_log_scale=False,
        x_alias=None,y_alias=None
    ):

    data = pd.read_csv(csv_path)

    data["model_bits"] = data["model_bits/data_samples"] * data["data_samples"]

    data['idx'] = data.apply(lambda x: f"{x.topology}_{x.quantization_bits:02d}b", axis=1)

    data = data.set_index("idx")

    if scale_x is None and scale_y is None:

        scale_x = data.loc[["032_010_010_001_08b","032_640_640_001_32b"],x_axis].max() - data.loc[["032_010_010_001_08b","032_640_640_001_32b"],x_axis].min()
        scale_y = data.loc[["032_010_010_001_08b","032_640_640_001_32b"],y_axis].max() - data.loc[["032_010_010_001_08b","032_640_640_001_32b"],y_axis].min()

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640],
        "qb": [8,16,32]
    }

    # x_axis = "model_bits/data_samples"
    # y_axis = "data_bits/data_samples"

    initial_values = {"h1":10,"h2":10,"qb":8}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths)) + f"_{params['qb']:02d}b"

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias)


if __name__ == "__main__":

    if os.path.isdir("glch_results"):
        import shutil
        shutil.rmtree("glch_results")
    os.mkdir("glch_results")

    if os.path.isdir("debug"):
        import shutil
        shutil.rmtree("debug")
    os.mkdir("debug")

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "micro_joules_per_pixel", # "joules_per_pixel", # "joules",
        "data_bits/data_samples",
        "rate_vs_energy",
        # scale_x=1,scale_y=1,
        # x_range=[135,175],
        # y_range=[0.115,0.145],
        x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
        y_alias="Rate (bits per pixel)"
    )

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "micro_joules_per_pixel", # "joules_per_pixel", # "joules",
        "data_bits/data_samples",
        "rate_vs_energy_noisy",
        # scale_x=1,scale_y=1,
        # x_range=[140,180],
        # y_range=None,
        remove_noise=False,
        x_alias="Complexity ($\SI{}{\mu\joule}$ per pixel)",
        y_alias="Rate (bits per pixel)"
    )

    glch_rate_vs_params(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        # "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_static_rate_x_power_values.csv",
        "params","data_bits/data_samples",
        "rate_vs_params",
        # scale_x=1e6,scale_y=1,
        # x_range=None,
        # y_range=None,
        x_in_log_scale=True,
        x_alias="Complexity (multiply/adds per pixel)",
        y_alias="Rate (bits per pixel)"
    )

    # TODO: the time measurements right now are too comprehensive.
    # They are measuring more than just the network computations.
    # They are also measuring the time taken to load the data, etc.
    # I could try to restrict the time measurements a bit more.
    # In other words, the measurements seem a little biased.
    glch_rate_vs_time(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "time","data_bits/data_samples",
        "rate_vs_time",
        # scale_x=1e6,scale_y=1,
        # x_range=None,
        # y_range=None,
        remove_noise=False,
        x_in_log_scale=False,
        x_alias="Complexity (seconds)",
        y_alias="Rate (bits per pixel)"
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "bpp_loss","mse_loss",
        # scale_x=1,scale_y=1,
        # x_range=[0.1,1.75],
        # y_range=[0.001,0.0045]
    )

    glch_rate_vs_dist_2(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "bpp_loss","mse_loss",#1,1,
        # x_range=[0.1,1.75],
        # y_range=[0.001,0.0045],
        start="left" # start="right"
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "flops","loss",
        # scale_x=1e10,scale_y=1,
        # x_range=[-0.2*1e10,3.75*1e10],
        # y_range=[1.1,3.1]
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "params","loss",
        # scale_x=1e6,scale_y=1,
        # x_range=[-0.1*1e6,4*1e6],
        # y_range=[1.1,3.1]
    )

    glch_model_bits_vs_data_bits(
        "/home/lucas/Documents/perceptronac/results/exp_1676160183/exp_1676160183_model_bits_x_data_bits_values.csv",
        "model_bits","data_bits/data_samples",
        # scale_x=1,scale_y=1,
        # x_range=[-0.1,0.8],
        # y_range=None,
        x_in_log_scale=True,
        x_alias="Complexity (encoded model bits)",
        y_alias="Rate (bits per pixel)"
    )



# %%
