
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_rate_experiment
from perceptronac.main_experiment import MLPTopologyCalculator

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "qbits_vec":[32,16,8],
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            os.path.join('/home/lucas/Documents/data/SPL2020',f) for f in os.listdir('/home/lucas/Documents/data/SPL2020')
        ],
        "validation_set": [
            os.path.join('/home/lucas/Documents/data/SPL2020V/pages',f) for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2020V/pages'))[:5]
            # os.path.join('/home/lucas/Documents/data/SPL2020',f) for f in os.listdir('/home/lucas/Documents/data/SPL2020')
        ],
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 8192,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "1659355125",
        "N": 32,
        "phases": ['valid'], # ['train', 'valid'],
        "xscale": "log", # 'linear',
        "reduction": 'last', # min, last
        "data_type": 'image', # image, pointcloud, table
        "geo_or_attr": "attributes", # geometry, attributes
        "n_classes": 2, # 2, 256
        "channels": [1,0,0], # [1,1,1], [1,0,0]
        "color_space": "YCbCr", # RGB, YCbCr
        "percentage_of_uncles": 0.0, # must be specified if the data type is pointcloud
        "last_octree_level": 10, # must be specified if the data type is pointcloud
        "save_dir": "/home/lucas/Documents/perceptronac/results",
        "dset_pieces": 2, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
        "energy_measurement_iteration": 8 # number of repetitions for better energy consumtion estimate
    }

    configs["topologies"] = [
        [configs["N"],h1,h2,1] for h1 in [10,20,40,80,160,320,640] for h2 in [10,20,40,80,160,320,640]
    ]

    rate_vs_rate_experiment(configs)