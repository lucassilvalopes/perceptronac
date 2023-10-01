

import os
import numpy as np
from glch_utils import plot_choice, plot_choice_2, open_debug_txt_file, close_debug_txt_file
from glch_utils import Node, min_max_convex_hull


class GLCH:

    def __init__(
        self,data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",debug=True,
        title=None, constrained=True, select_function="corrected_angle_rule", lmbda = 1, debug_folder="debug"
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
        
        select_function:
            gift_wrapping, 
            angle_rule, 
            corrected_angle_rule, 
            point
        """
        self.data = data
        self.possible_values = possible_values
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.initial_values = initial_values
        self.to_str_method = to_str_method
        self.start = start
        self.debug = debug
        if title is None:
            self.title = f"{x_axis.replace('/','_over_')}_vs_{y_axis.replace('/','_over_')}"
        else:
            self.title=title
        self.constrained = constrained
        self.select_function = select_function
        self.lmbda = lmbda
        self.debug_folder = debug_folder

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
            title=f"{self.title}_{iteration}",fldr=self.debug_folder)

    def begin_debug(self):
        if not self.debug:
            return
        if not os.path.isdir(self.debug_folder):
            os.mkdir(self.debug_folder)
        self.txt_file = open_debug_txt_file(self.title,self.debug_folder)

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

            if self.select_function == "gift_wrapping":
                chosen_node_index,update_ref_node = self.make_choice(ref_node,node,prev_candidate_nodes,candidate_nodes)
            elif self.select_function == "angle_rule":
                chosen_node_index,update_ref_node = self.make_choice_2(ref_node,node,prev_candidate_nodes,candidate_nodes)
            elif self.select_function == "corrected_angle_rule":
                chosen_node_index,update_ref_node = self.make_choice_3(ref_node,node,prev_candidate_nodes,candidate_nodes)
            elif self.select_function == "point":
                chosen_node_index,update_ref_node = self.make_choice_4(ref_node,node,prev_candidate_nodes,candidate_nodes)
            else:
                raise ValueError(f"unknown select function {self.select_function}")

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


    def make_choice(self,ref_node,node,prev_candidate_nodes,candidate_nodes):

        filtered_nodes = [n for n in candidate_nodes if str(n) != str(node)]

        blacklist = [str(n) for n in filtered_nodes]

        filt_prev_candidate_nodes = [n for n in prev_candidate_nodes if (n.color == "red") and (str(n) not in blacklist)]

        all_candidate_nodes = filt_prev_candidate_nodes + filtered_nodes

        coord_chull = self.get_node_coord([ref_node]+all_candidate_nodes)

        candidates_in_chull = min_max_convex_hull(coord_chull,start=self.start)

        if len(candidates_in_chull) == 1 and candidates_in_chull[0] == 0:

            candidates_in_chull = min_max_convex_hull(self.get_node_coord(filtered_nodes),start=self.start)
            chosen_node_index = candidates_in_chull[-1]

            update_ref_node = False

            chosen_node_index = len(prev_candidate_nodes) + candidate_nodes.index(filtered_nodes[chosen_node_index])

        else:

            no_nw = [i for i in candidates_in_chull if ((coord_chull[i][1] <= coord_chull[0][1]) and i != 0)]

            if len(no_nw) == 0:

                candidates_in_chull = min_max_convex_hull(self.get_node_coord(filtered_nodes),start=self.start)
                chosen_node_index = candidates_in_chull[-1]

                update_ref_node = False

                chosen_node_index = len(prev_candidate_nodes) + candidate_nodes.index(filtered_nodes[chosen_node_index])


            else:
                chosen_node_index = no_nw[0]
                update_ref_node = True

                if chosen_node_index >= len(filt_prev_candidate_nodes)+1:
                    chosen_node_index = len(prev_candidate_nodes) + \
                        candidate_nodes.index(filtered_nodes[chosen_node_index-len(filt_prev_candidate_nodes)-1])
                else:
                    chosen_node_index = prev_candidate_nodes.index(filt_prev_candidate_nodes[chosen_node_index-1])

        return chosen_node_index, update_ref_node


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


    def make_choice_3(self,ref_node,node,prev_candidate_nodes,candidate_nodes):

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

        update_ref_node = (len(sw + se) > 0)

        if len(sw) > 0:

            sorted_idx = self.sorted_deltac_deltar(
                sw,deltacs,deltars)

            chosen_node_index = sorted_idx[0]

        elif len(se) > 0:

            sorted_idx = self.sorted_minus_deltar_over_deltac(
                se,deltacs,deltars)

            chosen_node_index = sorted_idx[-1]

        elif len(nw) > 0 :

            sorted_idx = self.sorted_deltac_deltar(
                nw,deltacs,deltars)

            chosen_node_index = sorted_idx[0]

        else:

            sorted_idx = self.sorted_minus_deltar_over_deltac(
                ne,deltacs,deltars)

            chosen_node_index = sorted_idx[-1]

        if chosen_node_index >= len(filt_prev_candidate_nodes):
            chosen_node_index = len(prev_candidate_nodes) + \
                candidate_nodes.index(filtered_nodes[chosen_node_index-len(filt_prev_candidate_nodes)])
        else:
            chosen_node_index = prev_candidate_nodes.index(filt_prev_candidate_nodes[chosen_node_index])

        return chosen_node_index, update_ref_node


    def sorted_minus_deltar_over_deltac(self,ii,deltacs,deltars):

        dists = []

        for i in (ii):

            if deltars[i]<0 and (deltacs[i] == 0):
                dist = -(-np.inf)
            elif deltars[i]>0 and (deltacs[i] == 0):
                dist = -(np.inf)
            else:
                dist = -deltars[i]/deltacs[i]
            
            dists.append(dist)
    
        idx = [z[0] for z in sorted(list(zip(ii,dists)),key=lambda x: x[1])]
    
        return idx
    
    def sorted_deltac_deltar(self,ii,deltacs,deltars):
    
        idx = [z[0] for z in sorted([[i,deltacs[i],deltars[i]] for i in (ii)],key=lambda x: (x[1], x[2]))]
    
        return idx  


    def get_best_point(self,coord):
        rate_axis = [c[0] for c in coord]
        dist_axis = [c[1] for c in coord]
        best_point = np.argmin(np.array(rate_axis) + self.lmbda * np.array(dist_axis))
        return best_point


    def make_choice_4(self,ref_node,node,prev_candidate_nodes,candidate_nodes):

        filtered_nodes = [n for n in candidate_nodes if str(n) != str(node)]

        blacklist = [str(n) for n in filtered_nodes]

        filt_prev_candidate_nodes = [n for n in prev_candidate_nodes if (n.color == "red") and (str(n) not in blacklist)]

        all_candidate_nodes = filt_prev_candidate_nodes + filtered_nodes

        chosen_node_index = self.get_best_point(self.get_node_coord([ref_node]+all_candidate_nodes))

        if chosen_node_index == 0:

            chosen_node_index = self.get_best_point(self.get_node_coord(filtered_nodes))

            update_ref_node = False

            chosen_node_index = len(prev_candidate_nodes) + candidate_nodes.index(filtered_nodes[chosen_node_index])

        else:

            update_ref_node = True

            if chosen_node_index >= len(filt_prev_candidate_nodes)+1:
                chosen_node_index = len(prev_candidate_nodes) + \
                    candidate_nodes.index(filtered_nodes[chosen_node_index-len(filt_prev_candidate_nodes)-1])
            else:
                chosen_node_index = prev_candidate_nodes.index(filt_prev_candidate_nodes[chosen_node_index-1])

        return chosen_node_index, update_ref_node
