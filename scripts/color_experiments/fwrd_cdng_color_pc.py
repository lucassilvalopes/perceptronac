
import torch
import time
import os
from perceptronac.models import MLP_N_64N_32N_256
from perceptronac.main_experiment import experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_256,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            "/home/lucaslopes/perceptronac/NNOC/training/longdress/longdress_vox10_1051.ply"
        ],
        "validation_set": [
            "/home/lucaslopes/perceptronac/NNOC/validation/longdress/longdress_vox10_1300.ply"
        ],
        "epochs": 100,
        "learning_rate": 0.00001,
        "batch_size": 30000,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [3], # [0,3,81],
        "phases": ['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'pointcloud', # image, pointcloud, table
        "geo_or_attr": "attributes", # geometry, attributes
        "n_classes": 256, # 2, 256
        "channels": [1,0,0], # [1,1,1], [1,0,0]
        "color_space": "YCbCr", # RGB, YCbCr
        "percentage_of_uncles": 0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 3, # cabac
        "dset_pieces": 1, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    experiment(configs)

