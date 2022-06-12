
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_complexity_experiment
from perceptronac.main_experiment import MLPTopologyCalculator

if __name__ == "__main__":

    P_vec = list(map(lambda i : MLPTopologyCalculator.mlp_parameters([10,4**i,1]),range(1,7)))
    topologies = []
    for calc in [MLPTopologyCalculator(10,1,hidden_layers_proportions) for hidden_layers_proportions in [ 1 * [1] , 2 * [1] , 3 * [1] , [2,1] , [1,2] ] ]:
        for P in P_vec:
            top,_ = calc.mlp_closest_to_n_params(P)
            topologies.append(top)

    configs = {
        "id": str(int(time.time())),
        "topologies":topologies,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            os.path.join('/home/lucas/Documents/data/SPL2020',f) for f in os.listdir('/home/lucas/Documents/data/SPL2020')
        ],
        "validation_set": [
            os.path.join('/home/lucas/Documents/data/SPL2020V/pages',f) for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2020V/pages'))[:5]
        ],
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": ["1654532230"],
        "N": 10,
        "phases": ['train'], # ['train', 'valid'],
        "xscale": 'log',
        "reduction": 'min', # min, last
        "data_type": 'image', # image, pointcloud, table
        "percentage_of_uncles": 0.0, # must be specified if the data type is pointcloud
        "last_octree_level": 10, # must be specified if the data type is pointcloud
        "save_dir": "results",
        "dset_pieces": 1, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    rate_vs_complexity_experiment(configs)