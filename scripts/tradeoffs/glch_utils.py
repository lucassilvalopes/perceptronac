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


class Node:
    """
    https://runestone.academy/ns/books/published/pythonds/Trees/ListofListsRepresentation.html
    https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python
    """

    def __init__(self,**kwargs):
        self.params = kwargs
        self.children = []
        self.parent = None
        self.chosen_child_indices = []
        self.color = "red"
        self.lch = False

    def set_parent(self,node):
        self.parent = node

    def set_to_str_method(self,to_str_method):
        self.to_str_method = to_str_method
    
    def set_color(self,color):
        self.color = color

    def set_lch(self,lch):
        self.lch = lch

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


def open_debug_txt_file(title):
    txt_file = open(f"debug/transitions_{title}.txt", 'w')
    print(f"src_x,src_y,dst_x,dst_y,taken",file=txt_file)
    return txt_file

def close_debug_txt_file(txt_file):
    txt_file.close()

def plot_choice_2(x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,txt_file=None,title=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.text(x=node_coord[0],y=node_coord[1],s=str(node),color="black")

    print(f"-1,-1,-1,-1,-1",file=txt_file)

    ax.plot([node_coord[0]],[node_coord[1]],linestyle="",color="black",marker="o")

    npts = len(candidate_coord)

    for i in (sorted(set(range(npts)) - {chosen_node_index}) + [chosen_node_index]):

        if str(candidate_nodes[i]) == str(node):
            continue

        pt = candidate_coord[i]

        clr = ("g" if i == chosen_node_index else "r")

        ax.plot(
            [node_coord[0],pt[0]],
            [node_coord[1],pt[1]],
            color= clr)
        
        ax.text(x=pt[0],y=pt[1],s=f"{str(candidate_nodes[i])}",color=clr)

        print(f"{node_coord[0]},{node_coord[1]},{pt[0]},{pt[1]},{(1 if i == chosen_node_index else 0)}",file=txt_file)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if title is None:
        fig.show()
    else:
        fig.savefig(
            f"debug/{title}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


def plot_choice(data,x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,dists=None,txt_file=None,title=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(data[x_axis].values, data[y_axis].values,marker="x",linestyle="")

    ax.text(x=node_coord[0],y=node_coord[1],s=str(node),color="black")

    print(f"-1,-1,-1,-1,-1",file=txt_file)

    ax.plot([node_coord[0]],[node_coord[1]],linestyle="",color="black",marker="o")

    npts = len(candidate_coord)

    for i in (sorted(set(range(npts)) - {chosen_node_index}) + [chosen_node_index]):

        if str(candidate_nodes[i]) == str(node):
            continue

        pt = candidate_coord[i]

        clr = ("g" if i == chosen_node_index else "r")

        ax.plot(
            [node_coord[0],pt[0]],
            [node_coord[1],pt[1]],
            color= clr)

        ax.text(x=pt[0],y=pt[1],s=f"{str(candidate_nodes[i])}" + \
                (f",d={dists[i]}" if (dists is not None) else ""),color=clr)

        print(f"{node_coord[0]},{node_coord[1]},{pt[0]},{pt[1]},{(1 if i == chosen_node_index else 0)}",file=txt_file)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    ax.set_title(f"{chosen_node_index} {str(candidate_nodes[chosen_node_index])}")

    if title is None:
        fig.show()
    else:
        fig.savefig(
            f"debug/{title}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")

