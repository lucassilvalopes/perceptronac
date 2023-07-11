

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

            chosen_node_index,update_ref_node = self.make_choice(ref_node,node,prev_candidate_nodes,candidate_nodes)

            self.print_debug(node,prev_candidate_nodes,candidate_nodes,chosen_node_index,iteration)

            if chosen_node_index >= len(prev_candidate_nodes):
                chosen_node = candidate_nodes[chosen_node_index - len(prev_candidate_nodes)]
            else:
                chosen_node = prev_candidate_nodes[chosen_node_index]

            self.nodes += candidate_nodes # TODO : what happens in case of duplicacy ?

            if chosen_node_index < len(prev_candidate_nodes):
                node.chosen_child_indices.append(None)

            chosen_node.color = "green"
            chosen_node.parent.chosen_child_indices.append( chosen_node.parent.children.index(chosen_node) )
            node = chosen_node

            if update_ref_node:
                ref_node = chosen_node

            prev_candidate_nodes += candidate_nodes

            iteration += 1

        self.end_debug()

        return root


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


    def make_choice(self,ref_node,node,prev_candidate_nodes,candidate_nodes):
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

