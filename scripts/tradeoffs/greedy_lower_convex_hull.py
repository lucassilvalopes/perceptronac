# %%
import pandas as pd
from perceptronac.convex_hull import convex_hull
from perceptronac.loading_and_saving import points_in_convex_hull
import numpy as np
import matplotlib.pyplot as plt


# %%
class Node:
    """
    https://runestone.academy/ns/books/published/pythonds/Trees/ListofListsRepresentation.html
    https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    """

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640]
    }

    def __init__(self,**kwargs):
        self.params = kwargs
        self.children = []
        self.chosen_child_index = -1

    def auto_increment(self,param_name):
        node = Node(**self.params.copy())
        param_value = node.params[param_name]
        new_param_value = param_value
        i = self.possible_values[param_name].index(param_value)
        if i+1 < len(self.possible_values[param_name]):
            new_param_value = self.possible_values[param_name][i+1]
        node.params[param_name] = new_param_value
        return node

    def __str__(self):
        widths = [32,self.params["h1"],self.params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths))
    

# %%
def build_tree(data):
    root = Node(h1=10,h2=10)

    node = root
    while True:

        coord = []
        nodes = []
        for p in ["h1","h2"]:
            node_p = node.auto_increment(p)
            nodes.append(node_p)
            data_p = data.loc[str(node_p),["joules","data_bits/data_samples"]].values.tolist()
            coord.append(data_p)
        nodes = [node] + nodes
        coord = [data.loc[str(node),["joules","data_bits/data_samples"]].values.tolist()] + coord
        chull = convex_hull(coord)

        nodes = nodes[1:]
        coord = coord[1:]
        if 0 in chull:
            chull.pop(chull.index(0))
        chull = [e-1 for e in chull]

        if len(chull) == 0:
            chull.append(0)

        chosen_node_index = chull[0]
        chosen_node = nodes[chosen_node_index]

        node.children = nodes
        if str(chosen_node) == str(node):
            break
        node.chosen_child_index = chosen_node_index
        node = chosen_node

    return root

# %%
def print_tree(node):
    # print("parent", node)

    children_str = ""
    for i,c in enumerate(node.children):
        prefix = "!" if i == node.chosen_child_index else ""
        children_str += f"{prefix}{c} "
    # print("children",children_str)
    print(node,children_str)

    print("\n")
    if node.chosen_child_index != -1:
        print_tree(node.children[node.chosen_child_index])


# %%
def paint_tree(ax,node):

    for i,c in enumerate(node.children):
        color = "green" if i == node.chosen_child_index else "red"
        ax.plot(
            data.loc[[str(node), str(c)],"joules"].values,
            data.loc[[str(node), str(c)],"data_bits/data_samples"].values,
            linestyle="solid",color=color,marker=None
        )

    if node.chosen_child_index != -1:
        paint_tree(ax,node.children[node.chosen_child_index])

# %%

def chosen_nodes(r):

    new_points = [str(r)]

    n = r
    while True:
        for i,c in enumerate(n.children):
            if n.chosen_child_index == i:
                new_points.append(str(c))
        if n.chosen_child_index != -1:
            n = n.children[n.chosen_child_index]
        else:
            break

    return new_points

# %%

def tree_figure(data,r):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data.loc[:,"joules"].values,data.loc[:,"data_bits/data_samples"].values,linestyle="",marker="x")
    paint_tree(ax,r)
    ax.set_xlabel("joules")
    ax.set_ylabel("data_bits/data_samples")
    return fig

# %%
def hulls_figure(data,r):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data.loc[:,"joules"].values,data.loc[:,"data_bits/data_samples"].values,linestyle="",marker="x")
    true_hull_points = data.iloc[convex_hull(data.loc[:,["joules","data_bits/data_samples"]].values.tolist()),:]
    ax.plot(true_hull_points["joules"],true_hull_points["data_bits/data_samples"],linestyle=(0, (5, 5)),color="red",marker=None)
    new_points = chosen_nodes(r)
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[convex_hull(probe.loc[:,["joules","data_bits/data_samples"]].values.tolist()),:]
    ax.plot(
        estimated_hull_points["joules"],estimated_hull_points["data_bits/data_samples"],linestyle="dotted",color="green",marker=None)
    ax.set_xlabel("joules")
    ax.set_ylabel("data_bits/data_samples")
    return fig,true_hull_points,estimated_hull_points

# %%
from scipy.sparse.csgraph import connected_components


def labaled_points_figure(data):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(27.9,21.6)
    ax.plot(data.loc[:,"joules"].values,data.loc[:,"data_bits/data_samples"].values,linestyle="",marker="x")
    ax.set_xlabel("joules")
    ax.set_ylabel("data_bits/data_samples")

    X = data.loc[:,["joules","data_bits/data_samples"]].values
    adj_mtx = np.logical_and(
        np.sum((np.expand_dims(X,1) - np.expand_dims(X,0))**2,axis=2) > 0,
        np.logical_and(
            np.abs(np.expand_dims(X[:,0],1) - np.expand_dims(X[:,0],0)) < 0.5,
            np.abs(np.expand_dims(X[:,1],1) - np.expand_dims(X[:,1],0)) < 0.0003,
        )
    )

    n_conn_comp, conn_comp_mask = connected_components(adj_mtx)

    points_to_merge =[]
    for cc in range(n_conn_comp):
        if np.sum(conn_comp_mask == cc) > 1:
            points_to_merge.append(data.index.values[conn_comp_mask == cc].tolist())

    for top,row in data.loc[:,["joules","data_bits/data_samples"]].iterrows():
        normal_point = True
        for g in points_to_merge:
            if top in g:
                normal_point = False
        if normal_point:
            ax.text(
                x=row["joules"], #+0.5,
                y=row["data_bits/data_samples"]-0.0003,
                s=",".join(list(map(lambda x: str(int(x)),top.split("_")[1:3]))), 
                # fontdict=dict(color='black',size=8),
                # bbox=dict(facecolor='yellow',alpha=0.5)
            )

    for g in points_to_merge:
        sorted_i = np.argsort(data.loc[g,"joules"].values)
        ax.text(
            x=data.loc[g[sorted_i[0]],"joules"], #+0.5,
            y=data.loc[g[sorted_i[0]],"data_bits/data_samples"]-0.0003,
            s="/".join(list(map(lambda y: ",".join(list(map(lambda x: str(int(x)),y.split("_")[1:3]))) , np.array(g)[sorted_i].tolist()))), 
            # fontdict=dict(color='black',size=8),
            # bbox=dict(facecolor='yellow',alpha=0.5)
        )

    # fig.savefig('test2png.png', dpi=300, facecolor='w', bbox_inches = "tight")

    return fig

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
if __name__ == "__main__":

    data = pd.read_csv(
        "/home/lucas/Documents/perceptronac/results/exp_1664366985/exp_1664366985_static_rate_x_power_values.csv"
    ).set_index("topology")

    limit_energy_significant_digits(data)

    labeled_points_fig = labaled_points_figure(data)
    labeled_points_fig.savefig('labeled_points_fig.png', dpi=300, facecolor='w', bbox_inches = "tight")

    r = build_tree(data)

    print_tree(r)

    tree_fig = tree_figure(data,r)
    tree_fig.savefig(f"tree_fig.png", dpi=300, facecolor='w', bbox_inches = "tight")

    hulls_fig,true_hull_points,estimated_hull_points = hulls_figure(data,r)
    hulls_fig.savefig(f"hulls_fig.png", dpi=300, facecolor='w', bbox_inches = "tight")

    print(true_hull_points)
    print(estimated_hull_points)

# %%



# %%


# %%



