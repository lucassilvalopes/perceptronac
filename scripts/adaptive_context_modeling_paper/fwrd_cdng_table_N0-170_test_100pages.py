
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
            os.path.join('/home/lucas/Documents/data/SPL2020',f) for f in os.listdir('/home/lucas/Documents/data/SPL2020')
        ],
        "validation_set": [
            os.path.join('/home/lucas/Documents/data/SPL2021/pages',f) for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2021/pages')) # test
        ],
        "epochs": 1, # 300,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "1714238871",
        "unfinished_training": 0, # 0 or 1
        "N_vec": sorted([0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],reverse=False),
        "phases": ['valid'], #['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'image', # image, pointcloud
        "geo_or_attr": "attributes", # geometry, attributes
        "n_classes": 2, # 2, 256
        "channels": [1,0,0], # [1,1,1], [1,0,0]
        "color_space": "YCbCr", # RGB, YCbCr
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27, # cabac
        "dset_pieces": 10, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
        "methods": ["MLP","LUT"],
        "manual_th": None,
        "full_page": True
    }

    experiment(configs)

