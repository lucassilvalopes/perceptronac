
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
            os.path.join(r,f) for r,ds,fs in os.walk("/home/lucaslopes/perceptronac/NNOC/training") for f in fs if f.endswith("N89_M18_contexts.npz")
        ],
        "validation_set": [
            os.path.join(r,f) for r,ds,fs in os.walk("/home/lucaslopes/perceptronac/NNOC/validation") for f in fs if f.endswith("N89_M18_contexts.npz")
        ],
        "epochs": 100, # 300,
        "learning_rate": 0.00001,
        "batch_size": 30000,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [107],
        "phases": ['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'table', # image, pointcloud, table
        "geo_or_attr": "geometry", # geometry, attributes
        "n_classes": None, # 2, 256
        "channels": None, # [1,1,1], [1,0,0]
        "color_space": None, # RGB, YCbCr
        "percentage_of_uncles": 18/107, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27, # cabac
        "dset_pieces": 18, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    experiment(configs)

