
import torch
import time
import os
from perceptronac.models import MLP_N_64N_32N_3x256
from perceptronac.main_experiment import experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_3x256,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            "/home/lucaslopes/perceptronac/color_images/baboon.png"
        ],
        "validation_set": [
            "/home/lucaslopes/perceptronac/color_images/fruits.png"
        ],
        "epochs": 80,
        "learning_rate": 0.00001,
        "batch_size": 30000,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [0,1],
        "phases": ['train','valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'image', # image, pointcloud
        "color_mode": 'rgb', # binary,gray,rgb
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 1, # cabac
        "dset_pieces": 1, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    experiment(configs)

