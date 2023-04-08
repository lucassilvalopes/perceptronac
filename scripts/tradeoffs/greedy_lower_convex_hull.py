# %%
import pandas as pd
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from perceptronac.convex_hull import convex_hull
import numpy as np
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
        self.chosen_child_index = -1

    def set_to_str_method(self,to_str_method):
        self.to_str_method = to_str_method

    def auto_increment(self,param_name,possible_values):
        node = Node(**self.params.copy())
        node.set_to_str_method(self.to_str_method)
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




def dist_to_chull(chull,coord,pt):

    if len(chull) == 1:
        return pt[0] - coord[chull[0]][0]
        
    dists = []

    for i in range(len(chull)-1):

        line_vec = np.array(coord[chull[i+1]]) - np.array(coord[chull[i]])

        if line_vec[0]<0 and line_vec[1]>0:
            line_vec = -line_vec
        elif line_vec[0]==0 and line_vec[1]>0:
            line_vec = np.array([line_vec[0],-line_vec[1]])
        elif line_vec[0]<0 and line_vec[1]==0:
            line_vec = np.array([-line_vec[0],line_vec[1]])

        pt_vec = np.array(pt) - np.array(coord[chull[i]])

        proj_vec = (line_vec / np.linalg.norm(line_vec)) * (pt_vec.reshape(1,-1) @ line_vec.reshape(-1,1))

        orth_vec = np.array(pt) - (np.array(coord[chull[i]]) + proj_vec)

        dist = np.sign(np.cross(line_vec,orth_vec)) * np.linalg.norm(orth_vec)

        dists.append(dist)
    
    return np.min(dists)


def make_choice(chull,node,coord,candidate_nodes,candidate_coord):

    scaler = MinMaxScaler()
    scaler.fit(coord)
    coord = scaler.transform(coord)
    candidate_coord = scaler.transform(candidate_coord)


    if all([str(n) == str(node) for n in candidate_nodes]):
        return -1

    dists = []

    for pt in candidate_coord:

        dist = dist_to_chull(chull,coord,pt)

        dists.append(dist)
    
    idx = np.argsort(dists)

    filtered_idx = [i for i in idx if str(candidate_nodes[i]) != str(node)]

    return filtered_idx[0]
    



def build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method):

    root = Node(**initial_values)
    root.set_to_str_method(to_str_method)

    node = root

    nodes = [node]
    coord = [data.loc[str(node),[x_axis,y_axis]].values.tolist()]
    chull = [0]

    while True:

        candidate_coord = []
        candidate_nodes = []
        for p in sorted(possible_values.keys()):
            node_p = node.auto_increment(p,possible_values)
            candidate_nodes.append(node_p)
            data_p = data.loc[str(node_p),[x_axis,y_axis]].values.tolist()
            candidate_coord.append(data_p)
        
        chosen_node_index = make_choice(
            chull,node,coord,candidate_nodes,candidate_coord)

        if chosen_node_index == -1:
            break

        chosen_node = candidate_nodes[chosen_node_index]
        chosen_coord = candidate_coord[chosen_node_index]

        nodes = nodes + [chosen_node]
        coord = coord + [chosen_coord]

        chull = min_max_convex_hull(coord)

        node.children = candidate_nodes

        node.chosen_child_index = chosen_node_index
        node = chosen_node

    return root


def build_tree_old(data,possible_values,x_axis,y_axis,initial_values,to_str_method):
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
    root = Node(**initial_values)
    root.set_to_str_method(to_str_method)

    node = root

    live_nodes = [node]
    live_coord = [data.loc[str(node),[x_axis,y_axis]].values.tolist()]
    current_global_hull = [0]

    while True:

        coord = []
        nodes = []
        promising = []
        for p in sorted(possible_values.keys()):
            node_p = node.auto_increment(p,possible_values)
            nodes.append(node_p)
            data_p = data.loc[str(node_p),[x_axis,y_axis]].values.tolist()
            coord.append(data_p)

            live_nodes.append(node_p)
            live_coord.append(data_p)
            new_global_hull = min_max_convex_hull(live_coord,"right")
            if new_global_hull != current_global_hull:
                current_global_hull = new_global_hull
                promising.append(True)
            else:
                promising.append(False)

        if promising.count(True) == 1:
            chosen_node_index = promising.index(True)
            chosen_node = nodes[chosen_node_index]
            node.children = nodes
            node.chosen_child_index = chosen_node_index
            node = chosen_node
            continue
            
        else: # > 1 or == 0
            
            # option 1
            node.children = nodes

            strs = [str(n) for n in nodes]

            duplicated = [s == str(node) for s in strs]

            if all(duplicated):
                break

            chull = min_max_convex_hull([c for c,d in zip(coord,duplicated) if not d],"right")

            chosen_node_index = chull[0]
            chosen_node = [n for n,d in zip(nodes,duplicated) if not d][chosen_node_index]

            node.chosen_child_index = strs.index(str(chosen_node))
            node = chosen_node

            # option 2
            # # nodes = [node] + nodes
            # # coord = [data.loc[str(node),[x_axis,y_axis]].values.tolist()] + coord
            # chull = min_max_convex_hull(coord)

            # # nodes = nodes[1:]
            # # coord = coord[1:]
            # # if 0 in chull:
            # #     chull.pop(chull.index(0))
            # # chull = [e-1 for e in chull]

            # # if len(chull) == 0:
            # #     chull.append(0)

            # chosen_node_index = chull[0]
            # chosen_node = nodes[chosen_node_index]

            # node.children = nodes
            # if str(chosen_node) == str(node):
            #     break
            # node.chosen_child_index = chosen_node_index
            # node = chosen_node

    return root

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
def paint_tree(ax,data,node,x_axis,y_axis):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """

    for i,c in enumerate(node.children):
        color = "green" if i == node.chosen_child_index else "red"
        ax.plot(
            data.loc[[str(node), str(c)],x_axis].values,
            data.loc[[str(node), str(c)],y_axis].values,
            linestyle="solid",color=color,marker=None
        )

    if node.chosen_child_index != -1:
        paint_tree(ax,data,node.children[node.chosen_child_index],x_axis,y_axis)

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

def tree_figure(data,r,x_axis,y_axis):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data.loc[:,x_axis].values,data.loc[:,y_axis].values,linestyle="",marker="x")
    paint_tree(ax,data,r,x_axis,y_axis)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig

# %%
def hulls_figure(data,r,x_axis,y_axis):
    """
    x_axis = "joules"
    y_axis = "data_bits/data_samples"
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data.loc[:,x_axis].values,data.loc[:,y_axis].values,linestyle="",marker="x")
    true_hull_points = data.iloc[min_max_convex_hull(data.loc[:,[x_axis,y_axis]].values.tolist()),:]
    ax.plot(true_hull_points[x_axis],true_hull_points[y_axis],linestyle=(0, (5, 5)),color="red",marker=None)
    new_points = tree_nodes(r)
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[min_max_convex_hull(probe.loc[:,[x_axis,y_axis]].values.tolist()),:]
    ax.plot(
        estimated_hull_points[x_axis],estimated_hull_points[y_axis],linestyle="dotted",color="green",marker=None)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig,true_hull_points,estimated_hull_points

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
    data[["joules","joules_std"]] = data[["joules","joules_std"]].apply(lambda x: pd.Series({
        "joules":limit_significant_digits(x["joules"],fexp(x["joules_std"])),
        "joules_std":limit_significant_digits(x["joules_std"],fexp(x["joules_std"]))
    },index=["joules","joules_std"]), axis=1)
    return data

# %%


def glch_rate_vs_energy(csv_path):

    data = pd.read_csv(csv_path).set_index("topology")

    limit_energy_significant_digits(data)

    # labeled_points_fig = labaled_points_figure(data)
    # labeled_points_fig.savefig('labeled_points_fig.png', dpi=300, facecolor='w', bbox_inches = "tight")

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

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method)

    with open('tree_rate_vs_energy.txt', 'w') as f:
        print_tree(r,file=f)

    tree_fig = tree_figure(data,r,x_axis,y_axis)
    tree_fig.savefig(f"tree_fig_rate_vs_energy.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig,true_hull_points,estimated_hull_points = hulls_figure(data,r,x_axis,y_axis)
    hulls_fig.savefig(f"hulls_fig_rate_vs_energy.png", dpi=300, facecolor='w', bbox_inches = "tight")

    print(true_hull_points)
    print(estimated_hull_points)


def glch_rate_vs_dist(csv_path,x_axis,y_axis):

    data = pd.read_csv(csv_path).set_index("labels")

    possible_values = {
        "L": ["5e-3", "1e-2", "2e-2"],
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

    # x_axis = "bpp_loss"
    # y_axis = "mse_loss"

    initial_values = {"L":"5e-3", "N":32,"M":32}

    def to_str_method(params):
        return f"L{params['L']}N{params['N']}M{params['M']}"
    
    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method)

    with open(f'tree_{x_axis}_vs_{y_axis}.txt', 'w') as f:
        print_tree(r,file=f)

    tree_fig = tree_figure(data,r,x_axis,y_axis)
    tree_fig.savefig(f"tree_fig_{x_axis}_vs_{y_axis}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig,true_hull_points,estimated_hull_points = hulls_figure(data,r,x_axis,y_axis)
    hulls_fig.savefig(f"hulls_fig_{x_axis}_vs_{y_axis}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    print(true_hull_points)
    print(estimated_hull_points)


def glch_model_bits_vs_data_bits(csv_path):

    data = pd.read_csv(csv_path)

    data['idx'] = data.apply(lambda x: f"{x.topology}_{x.quantization_bits:02d}b", axis=1)

    data = data.set_index("idx")

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

    r = build_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method)

    with open('tree_model_bits_vs_data_bits.txt', 'w') as f:
        print_tree(r,file=f)

    tree_fig = tree_figure(data,r,x_axis,y_axis)
    tree_fig.savefig(f"tree_fig_model_bits_vs_data_bits.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig,true_hull_points,estimated_hull_points = hulls_figure(data,r,x_axis,y_axis)
    hulls_fig.savefig(f"hulls_fig_model_bits_vs_data_bits.png", dpi=300, facecolor='w', bbox_inches = "tight")

    print(true_hull_points)
    print(estimated_hull_points)

if __name__ == "__main__":

    glch_rate_vs_energy("/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_static_rate_x_power_values.csv")

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "bpp_loss","mse_loss"
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "flops","loss"
    )

    glch_rate_vs_dist(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/bpp-mse-psnr-loss-flops-params_bmshj2018-factorized_10000-epochs_L-2e-2-1e-2-5e-3_N-32-64-96-128-160-192-224_M-32-64-96-128-160-192-224-256-288-320.csv",
        "params","loss"
    )

    glch_model_bits_vs_data_bits("/home/lucas/Documents/perceptronac/results/exp_1676160183/exp_1676160183_model_bits_x_data_bits_values.csv")



# %%
