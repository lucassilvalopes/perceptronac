
import torch
import time
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.coders import PC_Coder
from perceptronac.main_experiment import experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "CoderClass":PC_Coder,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            os.path.join('SPL2020',f) for f in os.listdir('SPL2020')[0:1]
        ],
        "validation_set": [
            os.path.join('SPL2021',f) for f in sorted(os.listdir('SPL2021'))[:10]
        ],
        "epochs": 1,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "1649959285",
        "N_vec": sorted([0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],reverse=True),
        "phases": ['valid'], # ['train', 'valid', 'coding'],
        "xscale": 'symlog',
        "reduction": 'min',
        "data_type": 'image', # image, pointcloud
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "save_dir": "results"
    }

    experiment(configs)

