
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_complexity_experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
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
        "N": 14,
        "W_vec": [2,4,8,16,32,64,128,256,512,1024,2048,4096],
        "phases": ['train'], # ['train', 'valid'],
        "xscale": 'log',
        "reduction": 'min', # min, last
        "save_dir": "results",
    }

    rate_vs_complexity_experiment(configs)