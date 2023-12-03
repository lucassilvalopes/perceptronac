
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from perceptronac.convex_hull import convex_hull
from sklearn.preprocessing import MinMaxScaler
from line_clipping import cohenSutherlandClip


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pgf.preamble": [r"\usepackage{siunitx}"]})


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


def open_debug_txt_file(title,fldr="debug"):
    txt_file = open(f"{fldr}/transitions_{title}.txt", 'w')
    print(f"src_x,src_y,dst_x,dst_y,taken",file=txt_file)
    return txt_file

def close_debug_txt_file(txt_file):
    txt_file.close()

def plot_choice_2(x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,txt_file=None,title=None,fldr="debug"):

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
            f"{fldr}/{title}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


def plot_arrow(ax,src_coord,dst_coord,color):
    ax.annotate('',
        xytext= (src_coord[0],src_coord[1]),
        xy= (dst_coord[0],dst_coord[1]),
        arrowprops=dict(arrowstyle="->", color=color),
        # size=size
    )

def get_line(lmbda,rate_axis,dist_axis,x_lims,y_lims):
    best_point = np.argmin(np.array(rate_axis) + lmbda * np.array(dist_axis))
    if lmbda >= (np.diff(x_lims)/np.diff(y_lims)):
        line_x = np.array(x_lims)
        line_y = (-1/lmbda)*line_x + (rate_axis[best_point]/lmbda + dist_axis[best_point])
    else:
        line_y = np.array(y_lims)
        line_x = (-1)* lmbda * line_y + (rate_axis[best_point] + lmbda * dist_axis[best_point])
    line = [line_x,line_y]
    return line


def plot_choice(
    data,x_axis,y_axis,node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,dists=None,txt_file=None,title=None,fldr="debug"):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(data[x_axis].values, data[y_axis].values,marker=".",linestyle="")

    x_lims = [0.3,1.5]
    y_lims = [0.001,0.004]

    line = get_line((255**2)*2e-2,data[x_axis].values, data[y_axis].values,x_lims,y_lims)

    ax.plot(line[0],line[1],color="k",linewidth=0.5)

    # ax.text(x=node_coord[0],y=node_coord[1],s=str(node),color="black")

    print(f"-1,-1,-1,-1,-1",file=txt_file)

    ax.plot([node_coord[0]],[node_coord[1]],linestyle="",color="black",marker="o")

    npts = len(candidate_coord)

    for i in (sorted(set(range(npts)) - {chosen_node_index}) + [chosen_node_index]):

        if str(candidate_nodes[i]) == str(node):
            continue

        pt = candidate_coord[i]

        clr = ("g" if i == chosen_node_index else "r")

        # ax.plot(
        #     [node_coord[0],pt[0]],
        #     [node_coord[1],pt[1]],
        #     color= clr)
        if str(node) != "D3L2e-2N160M32":
            plot_arrow(ax,node_coord,pt,clr)

        # ax.text(x=pt[0],y=pt[1],s=f"{str(candidate_nodes[i])}" + \
        #         (f",d={dists[i]}" if (dists is not None) else ""),color=clr)

        print(f"{node_coord[0]},{node_coord[1]},{pt[0]},{pt[1]},{(1 if i == chosen_node_index else 0)}",file=txt_file)

    ax.set_xlabel("Rate (bits per pixel)")
    ax.set_ylabel("MSE")

    # ax.set_title(f"{chosen_node_index} {str(candidate_nodes[chosen_node_index])}")

    if title is None:
        fig.show()
    else:

        ax.set_xlim(x_lims[0],x_lims[1])

        ax.set_ylim(y_lims[0],y_lims[1])

        fig.savefig(
            f"{fldr}/{title}.png", 
            dpi=300, facecolor='w', bbox_inches = "tight")


def one_line_of_tree_str(node,prev_children,children,chosen_child_indices):
    children_str = ""
    for i,c in enumerate(children,len(prev_children)):
        if i in chosen_child_indices:
            prefix = ''.join(["!"] * (chosen_child_indices.index(i)+1) )
        else:
            prefix = ""
        
        children_str += f"{prefix}{c} "

    return "{} {}\n".format(node,children_str)


def print_tree(node,file=None):
    # print("parent", node)

    if len(node.children) == 0:
        return

    one_line = one_line_of_tree_str(node,[],node.children,node.chosen_child_indices)

    print(one_line,file=file)
    
    for i,c in enumerate(node.children):
        print_tree(node.children[i],file=file)


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

    ax.set_xlabel(x_alias if x_alias else x_axis, fontsize=16)
    ax.set_ylabel(y_alias if y_alias else y_axis, fontsize=16)
    if x_range:
        ax.set_xlim(x_range[0],x_range[1])
    if y_range:
        ax.set_ylim(y_range[0],y_range[1])


def compute_hulls(data,rs,x_axis,y_axis):

    true_hull_points = data.iloc[min_max_convex_hull(data.loc[:,[x_axis,y_axis]].values.tolist()),:]
    
    new_points = []
    for r in rs:
        new_points += [str(r)]
        new_points += tree_nodes(r,[],"all")
    probe = data.loc[new_points,:]
    estimated_hull_points = probe.iloc[min_max_convex_hull(probe.loc[:,[x_axis,y_axis]].values.tolist()),:]

    return true_hull_points,estimated_hull_points


def get_n_trained_networks(rs):
    new_points = []
    for r in rs:
        new_points += [str(r)]
        new_points += tree_nodes(r,[],"all")
    n_visited_networks = len(set(new_points))
    n_trained_networks = len(set([re.sub(r'_[\d]{2}b','',pt) for pt in new_points]))
    return n_visited_networks,n_trained_networks


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


def paint_hull_nodes(data,r,x_axis,y_axis,ax):

    lch_nodes = [str(r)] + tree_nodes(r,[], "lch")
    lch_nodes_xy = data.loc[lch_nodes,:]
    ax.plot(lch_nodes_xy[x_axis],lch_nodes_xy[y_axis],linestyle="",color="black",marker="o",
            markerfacecolor='none',markersize=8)


def paint_tree_nodes(data,r,x_axis,y_axis,ax):

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


def save_hull_points(data,rs,x_axis,y_axis,file_name):

    true_hull_points,estimated_hull_points = compute_hulls(data,rs,x_axis,y_axis)

    with open(f'{file_name}.txt', 'w') as f:
        print("\ntrue_hull_points:\n",file=f)
        print(true_hull_points[[x_axis,y_axis]],file=f)
        print("\nestimated_hull_points:\n",file=f)
        print(estimated_hull_points[[x_axis,y_axis]],file=f)

        print("\nreference nodes:\n",file=f)
        for i in range(len(rs)):
            lch_nodes = [str(rs[i])] + tree_nodes(rs[i],[], "lch")
            lch_nodes_xy = data.loc[lch_nodes,:]
            lch_nodes_xy[x_axis],lch_nodes_xy[y_axis]
            lch_nodes_df = lch_nodes_xy[[x_axis,y_axis]]
            print(lch_nodes_df,file=f)


def save_tree_data(
    data,r,x_axis,y_axis,x_range,y_range,data_id,x_in_log_scale=False,x_alias=None,y_alias=None,fldr="glch_results",tree_str=None):

    n_visited_networks,n_trained_networks = get_n_trained_networks([r])

    with open(f'{fldr}/tree_{data_id}.txt', 'w') as f:
        if tree_str:
            print(tree_str,file=f)
        else:
            print_tree(r,file=f)
        print(f"number of visited networks : {n_visited_networks}",file=f)
        print(f"number of trained networks : {n_trained_networks}",file=f)

    tree_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,".")
    paint_root(data,r,x_axis,y_axis,ax)
    paint_tree(ax,data,r,x_axis,y_axis,x_range,y_range)
    # paint_hull_points(true_hull_points,x_axis,y_axis,ax)
    paint_tree_nodes(data,r,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,x_range,y_range,ax,x_in_log_scale,x_alias,y_alias)
    tree_fig.savefig(f"{fldr}/tree_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")


def save_hull_data(data,r,x_axis,y_axis,x_range,y_range,data_id,x_in_log_scale=False,x_alias=None,y_alias=None,fldr="glch_results"):

    # true_hull_points,estimated_hull_points = compute_hulls(data,[r],x_axis,y_axis)

    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    # paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    paint_hull_nodes(data,r,x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,None,None,ax,x_in_log_scale,x_alias,y_alias)
    hulls_fig.savefig(f"{fldr}/hulls_fig_{data_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(data,[r],x_axis,y_axis,f"{fldr}/hulls_{data_id}")


def get_x_range_y_range(data,x_axis,y_axis):

    tmp_fig, tmp_ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,tmp_ax,".")
    x_range,y_range = tmp_ax.get_xlim(),tmp_ax.get_ylim()
    return x_range,y_range


def save_trees_data(data,rs,ls,x_axis,y_axis,x_range,y_range,exp_id,fldr="glch_results",tree_strs=None):

    tree_file = open(f'{fldr}/tree_{exp_id}.txt', 'w')

    tree_fig, ax = plt.subplots(nrows=1, ncols=len(rs))

    if (x_range is None) and (y_range is None):
        x_range,y_range = get_x_range_y_range(data,x_axis,y_axis)

    for i,r,L in zip(range(len(rs)),rs,ls):

        current_data = data.iloc[[i for i,lbl in enumerate(data.index) if f"L{L}" in lbl],:]

        if isinstance(tree_strs,list) and (len(tree_strs) == len(rs)) and all(tree_strs):
            print(f"{tree_strs[i]}\n",file=tree_file)
        else:
            print_tree(r,file=tree_file)

        paint_root(data,r,x_axis,y_axis,tree_fig.axes[i])
        # paint_cloud(data,x_axis,y_axis,tree_fig.axes[i],".")
        paint_cloud(current_data,x_axis,y_axis,tree_fig.axes[i],".")
        paint_tree(tree_fig.axes[i],data,r,x_axis,y_axis,x_range,y_range)
        adjust_axes(x_axis,y_axis,x_range,y_range,tree_fig.axes[i])

        if i != 0:
            tree_fig.axes[i].set_yticks([])
            tree_fig.axes[i].set_ylabel('')
    
    n_visited_networks,n_trained_networks = get_n_trained_networks(rs)

    print(f"number of visited networks : {n_visited_networks}",file=tree_file)
    print(f"number of trained networks : {n_trained_networks}",file=tree_file)
    tree_file.close()

    for i in range(len(rs)):
        # paint_hull_points(true_hull_points,x_axis,y_axis,tree_fig.axes[i])
        paint_tree_nodes(data,rs[i],x_axis,y_axis,tree_fig.axes[i])

    tree_fig.savefig(f"{fldr}/tree_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")


def save_hulls_data(data,rs,ls,x_axis,y_axis,x_range,y_range,exp_id,fldr="glch_results"):
    
    # true_hull_points,estimated_hull_points = compute_hulls(data,rs,x_axis,y_axis)

    hulls_fig, ax = plt.subplots(nrows=1, ncols=1)
    paint_cloud(data,x_axis,y_axis,ax,"x")
    # paint_hulls(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    # paint_hull(true_hull_points,estimated_hull_points,x_axis,y_axis,ax)
    for i in range(len(rs)):
        paint_hull_nodes(data,rs[i],x_axis,y_axis,ax)
    adjust_axes(x_axis,y_axis,None,None,ax)
    hulls_fig.savefig(f"{fldr}/hulls_fig_{exp_id}.png", dpi=300, facecolor='w', bbox_inches = "tight")

    save_hull_points(data,rs,x_axis,y_axis,f"{fldr}/hulls_{exp_id}")


def get_optimal_point_info(data,axes,weights):
    ix = np.argmin([sum([c*w for c,w in zip(xyzetc,weights)]) for xyzetc in data.loc[:,axes].values.tolist()])
    info = data.iloc[[ix],:].reset_index().loc[:,["labels"]+axes]
    return info


def get_trained_networks_up_to_node(tree_str,node_lbl):
    tree_data = [[wd for wd in ln.split()] for ln in re.sub("[!]+","",tree_str).split("\n")] 
    row_idx = [(node_lbl in ln) for ln in tree_data].index(True)
    col_idx = [(wd == node_lbl) for wd in tree_data[row_idx]].index(True)
    if col_idx == 0:
        row_idx = row_idx - 1
    trained_networks = {wd for ln in tree_data[:row_idx+1] for wd in ln}.union({tree_data[0][0]})
    return len(trained_networks)


def save_optimal_point(data,r,axes,weights,tree_str,exp_id,fldr="gho_results"):
    new_points = []
    new_points += [str(r)]
    new_points += tree_nodes(r,[],"all")
    probe = data.loc[new_points,:]

    estimated_best = get_optimal_point_info(probe,axes,weights)
    true_best = get_optimal_point_info(data,axes,weights)

    estimated_best_lbl = estimated_best["labels"].iloc[0]
    true_best_lbl = true_best["labels"].iloc[0]

    estimated_best_coord = estimated_best[axes].iloc[0,:].values.tolist()
    true_best_coord = true_best[axes].iloc[0,:].values.tolist()

    estimated_best_loss = sum([c*w for c,w in zip(estimated_best_coord,weights)])
    true_best_loss = sum([c*w for c,w in zip(true_best_coord,weights)])

    percent_higher = 100 * (estimated_best_loss - true_best_loss) / true_best_loss

    n_trained_networks = get_trained_networks_up_to_node(tree_str,estimated_best_lbl)

    with open(f'{fldr}/optimal_point_{exp_id}.txt', 'w') as f:

        print("\nestimated best:\n",file=f)
        print(estimated_best,file=f)
        print("\ntrue best:\n",file=f)
        print(true_best,file=f)
        print("",file=f)
        print(f"number of trained networks: {n_trained_networks}",file=f)
        print(f"estimated best loss: {estimated_best_loss}",file=f)
        print(f"true best loss: {true_best_loss}",file=f)
        print(f"loss higher by (%): {percent_higher}",file=f)
        
