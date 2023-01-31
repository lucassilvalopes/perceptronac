
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_complexity_experiment
from perceptronac.main_experiment import MLPTopologyCalculator

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            os.path.join('/home/lucaslopes/perceptronac/SPL2020',f) for f in os.listdir('/home/lucaslopes/perceptronac/SPL2020')
        ],
        "validation_set": [
            os.path.join('/home/lucaslopes/perceptronac/SPL2020V',f) for f in sorted(os.listdir('/home/lucaslopes/perceptronac/SPL2020V'))[:5]
        ],
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": ["1673019509","1673281302"],
        "N": 32,
        "phases": ['train','valid'], # ['train', 'valid'],
        "xscale": 'log',
        "reduction": 'min', # min, last
        "data_type": 'image', # image, pointcloud, table
        "geo_or_attr": "attributes", # geometry, attributes
        "n_classes": 2, # 2, 256
        "channels": [1,0,0], # [1,1,1], [1,0,0]
        "color_space": "YCbCr", # RGB, YCbCr
        "percentage_of_uncles": 0.0, # must be specified if the data type is pointcloud
        "last_octree_level": 10, # must be specified if the data type is pointcloud
        "save_dir": "results",
        "dset_pieces": 1, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    configs["topologies"] = [
        [configs["N"],h1,h2,1] for h1 in [10,20,40,80,160,320,640] for h2 in [10,20,40,80,160,320,640]
    ]

    rate_vs_complexity_experiment(configs)