
import torch
import time
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.main_experiment import experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            "eduardo/contexts_symbols.txt"
        ],
        "validation_set": [
            "eduardo/contexts_symbols.txt"
        ],
        "epochs": 300,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [14], # sorted([0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],reverse=True),
        "phases": ['train'], # ['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'csv', # image, pointcloud, csv
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27 # cabac
    }

    experiment(configs)
