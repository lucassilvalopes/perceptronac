
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_complexity_experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            os.path.join('SPL2020',f) for f in os.listdir('SPL2020')
        ],
        "validation_set": [
            os.path.join('SPL2020V',f) for f in sorted(os.listdir('SPL2020V'))[:5]
        ],
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 1024,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N": 10,
        # parameters of a 1 hidden layer mlp with powers of 2 number of hidden units : 2 hidden units -> 25 parameters, 4 hidden units -> 49 parameters ...
        "P_vec": [25, 49, 97, 193, 385], # [25, 49, 97, 193, 385, 769, 1537, 3073, 6145, 12289, 24577, 49153], 
        "phases": ['train'], # ['train', 'valid'],
        "xscale": 'log',
        "reduction": 'min', # min, last
        "save_dir": "results",
    }

    rate_vs_complexity_experiment(configs)