
import torch
import time
import os
from perceptronac.main_experiment import rate_vs_complexity_experiment
from perceptronac.main_experiment import FixedWidthMLPTopology

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
        "P_vec": list(map(lambda i : FixedWidthMLPTopology.fixed_width_mlp_parameters(10,4**i,1,1),range(1,7))),
        "phases": ['train'], # ['train', 'valid'],
        "xscale": 'log',
        "reduction": 'min', # min, last
        "save_dir": "results",
    }

    rate_vs_complexity_experiment(configs)