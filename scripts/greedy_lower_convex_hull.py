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

        if len(chull) > 1:
            nodes.pop(0)
            coord.pop(0)
            if 0 in chull:
                chull.pop(chull.index(0))
            chull = [e-1 for e in chull]

        chosen_node_index = chull[0]
        chosen_node = nodes[chosen_node_index]
        if str(chosen_node) == str(node):
            break
        node.children = nodes
        node.chosen_child_index = chosen_node_index
        node = chosen_node

    return root

# %%
def print_tree(node):
    print("parent", node)

    children_str = ""
    for i,c in enumerate(node.children):
        prefix = "!" if i == node.chosen_child_index else ""
        children_str += f"{prefix}{c} "
    print("children",children_str)

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
    ax.plot(true_hull_points["joules"],true_hull_points["data_bits/data_samples"],linestyle="solid",color="red",marker=None)
    new_points = chosen_nodes(r)
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[convex_hull(probe.loc[:,["joules","data_bits/data_samples"]].values.tolist()),:]
    ax.plot(
        estimated_hull_points["joules"],estimated_hull_points["data_bits/data_samples"],linestyle="solid",color="blue",marker=None)
    ax.set_xlabel("joules")
    ax.set_ylabel("data_bits/data_samples")
    return fig

# %%
if __name__ == "__main__":

    data = pd.read_csv(
        "/home/lucas/Documents/perceptronac/results/exp_1663073558/exp_1663073558_static_rate_x_power_values.csv"
    ).set_index("topology")

    r = build_tree(data)

    # print_tree(r)

    # tree_fig = tree_figure(data,r)
    # tree_fig.savefig(f"tree_fig.png", dpi=300, facecolor='w', bbox_inches = "tight")

    # hulls_fig = hulls_figure(data,r)
    # hulls_fig.savefig(f"hulls_fig.png", dpi=300, facecolor='w', bbox_inches = "tight")


