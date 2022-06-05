
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
            os.path.join(r,f) for r,ds,fs in os.walk("/home/lucaslopes/perceptronac/NNOC/training") for f in fs if f.endswith("npz")
        ],
        "validation_set": [
            "/home/lucaslopes/perceptronac/NNOC/testing/redandblack/redandblack_vox10_1450_N100_contexts.npz"
        ],
        "epochs": 100, # 300,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [100],
        "phases": ['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'table', # image, pointcloud, table
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27 # cabac
    }

    experiment(configs)

