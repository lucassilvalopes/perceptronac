

import os
import numpy as np
from glch_utils import plot_choice, plot_choice_2, open_debug_txt_file, close_debug_txt_file
from glch_utils import Node, min_max_convex_hull


class GLCH:

    def __init__(
        self,data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",scale_x=1,scale_y=1,debug=True,
        title=None
    ):
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
        if title is None:
            self.title = f"{x_axis.replace('/','_over_')}_vs_{y_axis.replace('/','_over_')}"
        else:
            self.title=title
        self.constrained = True

    def get_node_coord(self,node):
        if isinstance(node,list):
            return [self.data.loc[str(n),[self.x_axis,self.y_axis]].values.tolist() for n in node]
        else:
            return self.data.loc[str(node),[self.x_axis,self.y_axis]].values.tolist()

    def setup_build_tree(self):

        root = Node(**self.initial_values)
        root.set_to_str_method(self.to_str_method)
        # self.nodes = [root]
        return root

        # self.chull = [0]
    
    # def teardown_build_tree(self):

    def print_debug(self,node,prev_candidate_nodes,candidate_nodes,chosen_node_index,iteration):
        if not self.debug:
            return
        if chosen_node_index >= len(prev_candidate_nodes):
            chosen_node_index = chosen_node_index - len(prev_candidate_nodes)
        else:
            candidate_nodes = [prev_candidate_nodes[chosen_node_index]] + candidate_nodes
            chosen_node_index = 0
        node_coord = self.get_node_coord(node)
        candidate_coord = self.get_node_coord(candidate_nodes)
        plot_choice(
            self.data,self.x_axis,self.y_axis,
            node,node_coord,candidate_nodes,candidate_coord,chosen_node_index,txt_file=self.txt_file,
            title=f"{self.title}_{iteration}")

    def begin_debug(self):
        if not self.debug:
            return
        self.txt_file = open_debug_txt_file(self.title)
        if not os.path.isdir("debug"):
            os.mkdir("debug")

    def end_debug(self):
        if not self.debug:
            return
        close_debug_txt_file(self.txt_file)


    def build_tree(self):

        self.begin_debug()

        root = self.setup_build_tree()

        node = root

        ref_node = root

        prev_candidate_nodes = []

        iteration = 0

        while True:

            candidate_nodes = []

            for p in sorted(self.possible_values.keys()):
                node_p = node.auto_increment(p,self.possible_values)
                if str(node_p) == str(node):
                    node_p.color = "green"
                candidate_nodes.append(node_p)
            node.children = candidate_nodes

            if all([str(n) == str(node) for n in candidate_nodes]):
                break

            chosen_node_index,update_ref_node = self.make_choice_2(ref_node,node,prev_candidate_nodes,candidate_nodes)

            self.print_debug(node,prev_candidate_nodes,candidate_nodes,chosen_node_index,iteration)

            if chosen_node_index >= len(prev_candidate_nodes):
                chosen_node = candidate_nodes[chosen_node_index - len(prev_candidate_nodes)]
            else:
                chosen_node = prev_candidate_nodes[chosen_node_index]

            # self.nodes += candidate_nodes # TODO : what happens in case of duplicacy ?

            if chosen_node_index < len(prev_candidate_nodes):
                node.chosen_child_indices.append(None)

            chosen_node.color = "green"
            chosen_node.parent.chosen_child_indices.append( chosen_node.parent.children.index(chosen_node) )
            node = chosen_node

            if update_ref_node:
                ref_node = chosen_node
                ref_node.lch = True

            if not self.constrained:
                prev_candidate_nodes += candidate_nodes

            iteration += 1

        self.end_debug()

        return root


    def make_choice_2(self,ref_node,node,prev_candidate_nodes,candidate_nodes):

        filtered_nodes = [n for n in candidate_nodes if str(n) != str(node)]

        blacklist = [str(n) for n in filtered_nodes]

        filt_prev_candidate_nodes = [n for n in prev_candidate_nodes if (n.color == "red") and (str(n) not in blacklist)]

        all_candidate_nodes = filt_prev_candidate_nodes + filtered_nodes

        coord = self.get_node_coord(ref_node)

        candidate_coord = self.get_node_coord(all_candidate_nodes)

        n_candidates = len(candidate_coord)

        deltacs = [(pt[0] - coord[0]) for pt in candidate_coord]
        deltars = [(pt[1] - coord[1]) for pt in candidate_coord]
        
        ne = [i for i in range(n_candidates) if deltacs[i]>=0 and deltars[i]>=0]
        nw = [i for i in range(n_candidates) if deltacs[i]<0 and deltars[i]>0]
        sw = [i for i in range(n_candidates) if deltacs[i]<0 and deltars[i]<=0]
        se = [i for i in range(n_candidates) if deltacs[i]>=0 and deltars[i]<0]

        ne = [i for i in ne if i >= len(filt_prev_candidate_nodes)]
        nw = [i for i in nw if i >= len(filt_prev_candidate_nodes)]

        if (len(sw + se)) > 0:

            sorted_idx = self.sorted_deltac_over_minus_deltar(
                (sw+se),deltacs,deltars,False)

            chosen_node_index = sorted_idx[0]

            update_ref_node = True

        elif len(nw) > 0 :

            sorted_idx = self.sorted_deltac_over_minus_deltar(
                nw,deltacs,deltars,True)

            chosen_node_index = sorted_idx[-1]

            update_ref_node = False

        else:

            sorted_idx = self.sorted_deltac_over_minus_deltar(
                ne,deltacs,deltars,True)

            chosen_node_index = sorted_idx[0]

            update_ref_node = False

        if chosen_node_index >= len(filt_prev_candidate_nodes):
            chosen_node_index = len(prev_candidate_nodes) + \
                candidate_nodes.index(filtered_nodes[chosen_node_index-len(filt_prev_candidate_nodes)])
        else:
            chosen_node_index = prev_candidate_nodes.index(filt_prev_candidate_nodes[chosen_node_index])

        return chosen_node_index, update_ref_node


    def sorted_deltac_over_minus_deltar(self,ii,deltacs,deltars,top_half):

        dists = []

        for i in (ii):

            if deltacs[i]<0 and (deltars[i] == 0):
                if top_half:
                    dist = np.inf
                else:
                    dist = -np.inf
            elif deltacs[i]>0 and (deltars[i] == 0):
                if top_half:
                    dist = -np.inf
                else:
                    dist = np.inf
            else:
                dist = (deltacs[i])/(-deltars[i])

            dists.append(dist)
    
        # idx = np.argsort(dists)

        idx = [z[0] for z in sorted(list(zip(ii,dists)),key=lambda x: x[1])]
    
        return idx

