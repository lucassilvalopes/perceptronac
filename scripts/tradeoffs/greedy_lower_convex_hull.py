# %%
import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from perceptronac.convex_hull import convex_hull
from perceptronac.power_consumption import estimate_joules
from perceptronac.power_consumption import group_energy_measurements
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def min_max_convex_hull(data,start="left"):
    """
    start : either "left" or "right"
    """
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    if start=="right":
        data = data[:,::-1]
    return convex_hull(data.tolist())



# %%
class Node:
    """
    https://runestone.academy/ns/books/published/pythonds/Trees/ListofListsRepresentation.html
    https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    """

    def __init__(self,**kwargs):
        self.params = kwargs
        self.children = []
        self.parent = None
        self.chosen_child_index = -1

    def set_parent(self,node):
        self.parent = node

    def set_to_str_method(self,to_str_method):
        self.to_str_method = to_str_method

    def auto_increment(self,param_name,possible_values):
        node = Node(**self.params.copy())
        node.set_to_str_method(self.to_str_method)
        node.set_parent(self)
        param_value = node.params[param_name]
        new_param_value = param_value
        i = possible_values[param_name].index(param_value)
        if i+1 < len(possible_values[param_name]):
            new_param_value = possible_values[param_name][i+1]
        node.params[param_name] = new_param_value
        return node

    def __str__(self):
        return self.to_str_method(self.params)

    

# %%

def open_debug_txt_file(x_axis,y_axis):
    txt_file = open(f"debug/transitions_{x_axis.replace('/','_over_')}_vs_{y_axis.replace('/','_over_')}.txt", 'w')
    print(f"src_x,src_y,dst_x,dst_y,taken",file=txt_file)
    return txt_file

def close_debug_txt_file(txt_file):
    txt_file.close()

def plot_choice_2(x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,txt_file=None,title=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.text(x=node_coord[0],y=node_coord[1],s=str(node))

    print(f"-1,-1,-1,-1,-1",file=txt_file)

    for i,pt in enumerate(candidate_coord):
        ax.plot(
            [node_coord[0],pt[0]],
            [node_coord[1],pt[1]],
            color=("g" if i == chosen_node_index else "r") )
        
        ax.text(x=pt[0],y=pt[1],s=f"{str(candidate_nodes[i])}")

        print(f"{node_coord[0]},{node_coord[1]},{pt[0]},{pt[1]},{(1 if i == chosen_node_index else 0)}",file=txt_file)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if title is not None:
        ax.set_title(title)

    # fig.show()

    if len([n for n in candidate_nodes if (str(n) != str(node))]) > 1:

        fig.savefig(
            f"debug/{x_axis.replace('/','_over_')}_vs_{y_axis.replace('/','_over_')}-{'-'.join([str(c) for c in candidate_nodes])}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


def plot_choice(data,x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,dists=None,txt_file=None,title=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(data[x_axis].values, data[y_axis].values,marker="x",linestyle="")

    ax.text(x=node_coord[0],y=node_coord[1],s=str(node))

    print(f"-1,-1,-1,-1,-1",file=txt_file)

    for i,pt in enumerate(candidate_coord):
        ax.plot(
            [node_coord[0],pt[0]],
            [node_coord[1],pt[1]],
            color=("g" if i == chosen_node_index else "r") )

        ax.text(x=pt[0],y=pt[1],s=f"{str(candidate_nodes[i])}" + \
                (f",d={dists[i]}" if (dists is not None) else ""))

        print(f"{node_coord[0]},{node_coord[1]},{pt[0]},{pt[1]},{(1 if i == chosen_node_index else 0)}",file=txt_file)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if title is not None:
        ax.set_title(title)

    # fig.show()

    if len([n for n in candidate_nodes if (str(n) != str(node))]) > 1:

        fig.savefig(
            f"debug/{x_axis.replace('/','_over_')}_vs_{y_axis.replace('/','_over_')}-{'-'.join([str(c) for c in candidate_nodes])}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


class GLCH:

    def __init__(self,data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",scale_x=1,scale_y=1,debug=True):
        """
        data = 
        |---------------|joules|data_bits/data_samples|  
        |---topology----|------|----------------------|
        |032_010_010_001|420.00|0.99999999999999999999|

        possible_values = {
            "h1": [10,20,40,80,160,320,640],
            "h2": [10,20,40,80,160,320,640]
        }

        x_axis = "joules"
        y_axis = "data_bits/data_samples"

        initial_values = {"h1":10,"h2":10}

        def to_str_method(params):
            widths = [32,params["h1"],params["h2"],1]
            return '_'.join(map(lambda x: f"{x:03d}",widths))
        """
        self.data = data
        self.possible_values = possible_values
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.initial_values = initial_values
        self.to_str_method = to_str_method
        self.start = start
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.debug = debug

    def get_node_coord(self,node):
        if isinstance(node,list):
            return [self.data.loc[str(n),[self.x_axis,self.y_axis]].values.tolist() for n in node]
        else:
            return self.data.loc[str(node),[self.x_axis,self.y_axis]].values.tolist()

    def setup_build_tree(self):

        root = Node(**self.initial_values)
        root.set_to_str_method(self.to_str_method)
        self.nodes = [root]
        return root

        # self.chull = [0]
    
    # def teardown_build_tree(self):

    def build_tree(self):

        root = self.setup_build_tree()

        node = root

        ref_node = root

        prev_candidate_nodes = []

        while True:

            candidate_nodes = []
            for p in sorted(self.possible_values.keys()):
                node_p = node.auto_increment(p,self.possible_values)
                candidate_nodes.append(node_p)
            node.children = candidate_nodes

            if all([str(n) == str(node) for n in candidate_nodes]):
                break

            chosen_node_index,update_ref_node = self.make_choice(ref_node,prev_candidate_nodes,candidate_nodes)

            if chosen_node_index >= len(prev_candidate_nodes):
                chosen_node = candidate_nodes[chosen_node_index - len(prev_candidate_nodes)]
            else:
                chosen_node = prev_candidate_nodes[chosen_node_index]

            self.nodes += candidate_nodes

            chosen_node.parent.chosen_child_index = chosen_node.parent.children.index(chosen_node)
            node = chosen_node

            if update_ref_node:
                ref_node = chosen_node

            prev_candidate_nodes = [n for n in prev_candidate_nodes if str(n) != str(chosen_node)]
            prev_candidate_nodes += [n for n in candidate_nodes if str(n) != str(chosen_node)]

        return root


    def make_choice_2(self,node,prev_candidate_nodes,candidate_nodes):

        candidate_nodes = prev_candidate_nodes + candidate_nodes

        coord = self.get_node_coord(node)

        candidate_coord = self.get_node_coord(candidate_nodes)

        n_candidates = len(candidate_coord)

        deltacs = [(pt[0] - coord[0]) for pt in candidate_coord]
        deltars = [(pt[1] - coord[1]) for pt in candidate_coord]
        
        ne = [i for i in range(n_candidates) if deltacs[i]>=0 and deltars[i]>=0 and str(candidate_nodes[i]) != str(node)]
        nw = [i for i in range(n_candidates) if deltacs[i]<0 and deltars[i]>0 and str(candidate_nodes[i]) != str(node)]
        sw = [i for i in range(n_candidates) if deltacs[i]<0 and deltars[i]<=0 and str(candidate_nodes[i]) != str(node)]
        se = [i for i in range(n_candidates) if deltacs[i]>=0 and deltars[i]<0 and str(candidate_nodes[i]) != str(node)]

        if (len(sw + se)) > 0:

            filtered_idx = self.sorted_deltac_over_minus_deltar(
                self,(sw+se),deltacs,deltars)

            chosen_node_index = filtered_idx[0]

            update_ref_node = True

        elif len(nw) > 0 :

            filtered_idx = self.sorted_deltac_over_minus_deltar(
                nw,deltacs,deltars)

            chosen_node_index = filtered_idx[-1]

            update_ref_node = False

        else:

            filtered_idx = self.sorted_deltac_over_minus_deltar(
                ne,deltacs,deltars)

            chosen_node_index = filtered_idx[0]

            update_ref_node = False             

        return chosen_node_index, update_ref_node


    def sorted_deltac_over_minus_deltar(self,ii,deltacs,deltars):

        dists = []

        for i in (ii):

            if (np.sign(deltacs[i]) == np.sign(deltars[i])) and (deltars[i] == 0):
                dist = np.inf
            elif (np.sign(deltacs[i]) != np.sign(deltars[i])) and (deltars[i] == 0):
                dist = - np.inf
            else:
                dist = (deltacs[i])/(-deltars[i])

            dists.append(dist)
    
        idx = np.argsort(dists)
    
        return idx


    def find_candidates_in_chull(self,candidate_nodes):

        len_nodes = len(self.nodes)

        coord = self.get_node_coord(self.nodes) + self.get_node_coord(candidate_nodes)

        chull = min_max_convex_hull(coord,start=self.start)

        candidates_in_chull = [i-len_nodes for i in chull if i >= len_nodes]

        return candidates_in_chull


    def dist_to_chull(self,pt):

        lmbd = ((self.scale_x/self.scale_y)/6)

        improv = pt[0] + pt[1]*lmbd

        return improv


    def make_choice_tie_break(self,node,candidate_nodes):

        dists = []

        for pt in self.get_node_coord(candidate_nodes):

            dist = self.dist_to_chull(pt)

            dists.append(dist)
        
        idx = np.argsort(dists)

        filtered_idx = [i for i in idx if str(candidate_nodes[i]) != str(node)]

        chosen_node_index = filtered_idx[0]

        return chosen_node_index


    def make_choice(self,node,prev_candidate_nodes,candidate_nodes):
        """
        Params:
            node: current source node
            candidate_nodes: local nodes to choose from
        
        Returns:
            chosen_node_index: index of the chosen local node.
                -1 if all options are equal to the source node
        """

        filtered_nodes = [n for n in candidate_nodes if str(n) != str(node)]

        candidates_in_chull = self.find_candidates_in_chull(filtered_nodes)

        if (len(candidates_in_chull)==1):

            chosen_node_index = candidates_in_chull[0]

            chosen_node_index = candidate_nodes.index(filtered_nodes[chosen_node_index])

        else:

            chosen_node_index = self.make_choice_tie_break(node,candidate_nodes)

        return len(prev_candidate_nodes) + chosen_node_index, True


def build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",scale_x=1,scale_y=1,debug=True):
    return GLCH(data,possible_values,x_axis,y_axis,initial_values,to_str_method,start,scale_x,scale_y,debug).build_tree()



# %%
def print_tree(node,file=None):
    # print("parent", node)

    children_str = ""
    for i,c in enumerate(node.children):
        prefix = "!" if i == node.chosen_child_index else ""
        children_str += f"{prefix}{c} "
    # print("children",children_str,file=file)
    print(node,children_str,file=file)

    print("\n",file=file)
    if node.chosen_child_index != -1:
        print_tree(node.children[node.chosen_child_index],file=file)


# %%

from line_clipping import cohenSutherlandClip


def paint_tree(ax,data,node,x_axis,y_axis,x_range,y_range):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"

    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    """

    for i,c in enumerate(node.children):
        color = "green" if i == node.chosen_child_index else "firebrick"
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

    if node.chosen_child_index != -1:
        paint_tree(ax,data,node.children[node.chosen_child_index],x_axis,y_axis,x_range,y_range)

# %%

def tree_nodes(r, all_nodes = True):

    new_points = [str(r)]

    n = r
    while True:
        for i,c in enumerate(n.children):
            if (all_nodes) or (n.chosen_child_index == i):
                new_points.append(str(c))
        if n.chosen_child_index != -1:
            n = n.children[n.chosen_child_index]
        else:
            break

    return new_points

# %%


def paint_cloud(data,x_axis,y_axis,ax,marker):
    ax.plot(data.loc[:,x_axis].values,data.loc[:,y_axis].values,linestyle="",color="tab:blue",marker=marker)


def paint_root(data,r,x_axis,y_axis,ax):
    ax.plot([data.loc[str(r),x_axis]],[data.loc[str(r),y_axis]],linestyle="",color="yellow",marker="o")


def adjust_axes(x_axis,y_axis,x_range,y_range,ax,x_in_log_scale=False):
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


    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    if x_range:
        ax.set_xlim(x_range[0],x_range[1])
    if y_range:
        ax.set_ylim(y_range[0],y_range[1])


# %%

def compute_hulls(data,rs,x_axis,y_axis):

    true_hull_points = data.iloc[min_max_convex_hull(data.loc[:,[x_axis,y_axis]].values.tolist()),:]
    
    new_points = []
    for r in rs:
        new_points += tree_nodes(r)
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[min_max_convex_hull(probe.loc[:,[x_axis,y_axis]].values.tolist()),:]

    n_trained_networks = len(set(new_points))

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


def paint_nodes(data,r,x_axis,y_axis,ax):

    all_nodes = tree_nodes(r, all_nodes = True)
    selected_nodes = tree_nodes(r, all_nodes = False)
    unselected_nodes = list(set(all_nodes) - set(selected_nodes))

    selected_nodes_xy = data.loc[selected_nodes,:]
    unselected_nodes_xy = data.loc[unselected_nodes,:]
    ax.plot(selected_nodes_xy[x_axis],selected_nodes_xy[y_axis],linestyle="",color="green",marker=".")
    ax.plot(unselected_nodes_xy[x_axis],unselected_nodes_xy[y_axis],linestyle="",color="firebrick",marker=".")


def save_all_data(data,r,x_axis,y_axis,x_range,y_range,data_id,x_in_log_scale=False):

    true_hull_points,estimated_hull_points,n_trained_networks = compute_hulls(data,[r],x_axis,y_axis)

    with open(f'tree_{data_id}.txt', 'w') as f:
        print_tree(r,file=f)
        print(f"number of trained networks : {n_trained_networks}",file=f)

    tree_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,".")
    paint_root(data,r,x_axis,y_axis,ax)
    paint_tree(ax,data,r,x_axis,y_axis,x_range,y_range)
    # paint_hull_points(true_hull_points,x_axis,y_axis,ax)
    paint_nodes(data,r,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,x_range,y_range,ax,x_in_log_scale)
    tree_fig.savefig(f"tree_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,None,None,ax,x_in_log_scale)
    hulls_fig.savefig(f"hulls_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(f"hulls_{data_id}",true_hull_points,estimated_hull_points)



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

def limit_energy_significant_digits(data):

    last_significant_digit_position = fexp(data["joules_std"].max())

    data[["joules","joules_std"]] = data[["joules","joules_std"]].apply(lambda x: pd.Series({
        "joules":limit_significant_digits(x["joules"],last_significant_digit_position),
        "joules_std":limit_significant_digits(x["joules_std"],last_significant_digit_position)
        # "joules":limit_significant_digits(x["joules"],fexp(x["joules_std"])),
        # "joules_std":limit_significant_digits(x["joules_std"],fexp(x["joules_std"]))
    },index=["joules","joules_std"]), axis=1)
    return data

# %%


def save_hull_points(file_name,true_hull_points,estimated_hull_points):

    with open(f'{file_name}.txt', 'w') as f:
        print("true_hull_points",file=f)
        print(true_hull_points,file=f)
        print("estimated_hull_points",file=f)
        print(estimated_hull_points,file=f)


def glch_rate_vs_energy(csv_path,x_axis,y_axis,title,scale_x=None,scale_y=None,x_range=None,y_range=None,x_in_log_scale=False,remove_noise=True):

    data = pd.read_csv(csv_path)

    csv_path_2 = csv_path.replace("raw_values","power_draw")

    power_draw = np.loadtxt(csv_path_2)

    power_draw[:,1] = power_draw[:,1] - 16 # np.min(power_draw[:,1])

    joules = estimate_joules(data,power_draw)

    data["joules"] = joules

    data = group_energy_measurements(data).set_index("topology")

    if remove_noise:

        limit_energy_significant_digits(data)

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

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,title,x_in_log_scale)



def glch_rate_vs_params(csv_path,x_axis,y_axis,title,scale_x=None,scale_y=None,x_range=None,y_range=None,x_in_log_scale=False):

    data = pd.read_csv(csv_path).set_index("topology")

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

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,title,x_in_log_scale)



def glch_rate_vs_dist(csv_path,x_axis,y_axis,scale_x=None,scale_y=None,x_range=None,y_range=None,start="left"):

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
    
    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,f'{x_axis}_vs_{y_axis}_start_{start}')


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
    tree_file = open(f'tree_{exp_id}.txt', 'w')

    tree_fig, ax = plt.subplots(nrows=1, ncols=len(brute_dict["L"]))

    for i,L in enumerate(brute_dict["L"]):

        to_str_method = to_str_method_factory({"L":L})

        current_data = data.iloc[[i for i,lbl in enumerate(data.index) if f"L{L}" in lbl],:]
        
        r = build_tree(current_data,greedy_dict,x_axis,y_axis,initial_state,to_str_method,scale_x=scale_x,scale_y=scale_y)

        rs.append(r)

        print_tree(r,file=tree_file)

        paint_root(data,r,x_axis,y_axis,tree_fig.axes[i])
        paint_cloud(data,x_axis,y_axis,tree_fig.axes[i],".")
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

    tree_fig.savefig(f"tree_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    
    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,None,None,ax)
    hulls_fig.savefig(f"hulls_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(f"hulls_fig_{exp_id}",true_hull_points,estimated_hull_points)


def glch_model_bits_vs_data_bits(csv_path,x_axis,y_axis,scale_x=None,scale_y=None,x_range=None,y_range=None,x_in_log_scale=False):

    data = pd.read_csv(csv_path)

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

    x_axis = "model_bits/data_samples"
    y_axis = "data_bits/data_samples"

    initial_values = {"h1":10,"h2":10,"qb":8}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths)) + f"_{params['qb']:02d}b"

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,scale_x=scale_x,scale_y=scale_y)

    save_all_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',x_in_log_scale)


if __name__ == "__main__":

    if os.path.isdir("debug"):
        import shutil
        shutil.rmtree("debug")
    os.mkdir("debug")

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "joules","data_bits/data_samples",
        "rate_vs_energy",
        # scale_x=1,scale_y=1,
        # x_range=[135,175],
        # y_range=[0.115,0.145]
    )

    glch_rate_vs_energy(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",
        "joules","data_bits/data_samples",
        "rate_vs_energy_noisy",
        # scale_x=1,scale_y=1,
        # x_range=[140,180],
        # y_range=None,
        remove_noise=False
    )

    glch_rate_vs_params(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_static_rate_x_power_values.csv",
        "params","data_bits/data_samples",
        "rate_vs_params",
        # scale_x=1e6,scale_y=1,
        # x_range=None,
        # y_range=None,
        x_in_log_scale=True
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
        start="right"
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
        "model_bits/data_samples","data_bits/data_samples",
        # scale_x=1,scale_y=1,
        # x_range=[-0.1,0.8],
        # y_range=None,
        x_in_log_scale=True
    )



# %%
