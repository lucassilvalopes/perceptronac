
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
            # "/home/lucaslopes/perceptronac/NNOC/validation/longdress_vox10_1300_N100_contexts.npz"
            # "/home/lucaslopes/perceptronac/NNOC/testing/redandblack_vox10_1450.ply"
            os.path.join("/home/lucaslopes/perceptronac/NNOC/testing/redandblack",f) for f in sorted(os.listdir("/home/lucaslopes/perceptronac/NNOC/testing/redandblack"))
        ],
        "epochs": 100, # 300,
        "learning_rate": 0.00001,
        "batch_size": 30000,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "1654484527",
        "N_vec": [100],
        "phases": ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'pointcloud', # image, pointcloud, table
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27, # cabac
        "dset_pieces": 36, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
    }

    experiment(configs)

