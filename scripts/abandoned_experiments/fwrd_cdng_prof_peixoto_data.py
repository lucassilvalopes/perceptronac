
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
            f"/home/lucas/Documents/data/eduardo/andrew_frame000{frm}_contexts_symbols.txt" for frm in range(10)
        ],
        "validation_set": [
            f"/home/lucas/Documents/data/eduardo/david_frame000{frm}_contexts_symbols.txt" for frm in range(10)
        ],
        "epochs": 100,
        "learning_rate": 0.00001,
        "batch_size": 16384,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [14], # sorted([0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],reverse=True),
        "phases": ['train','valid'], # ['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'last', # min, last
        "data_type": 'table', # image, pointcloud, table
        "geo_or_attr": "geometry", # geometry, attributes
        "n_classes": None, # 2, 256
        "channels": None, # [1,1,1], [1,0,0]
        "color_space": None, # RGB, YCbCr
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27, # cabac
        "dset_pieces": 5, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    experiment(configs)

