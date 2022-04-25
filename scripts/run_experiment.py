
import torch
import time
import os
from perceptronac.loading_and_saving import save_N_data
from perceptronac.loading_and_saving import save_final_data
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.models import train_loop
from perceptronac.coders import MLP_N_64N_32N_1_PC_Coder

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "CoderClass":MLP_N_64N_32N_1_PC_Coder,
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

    os.makedirs(f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}")

    data = dict()
    for phase in configs["phases"]:
        data[phase] = dict()

    for N in configs["N_vec"]:
        print(f"--------------------- context size : {N} ---------------------")    
        N_data = train_loop(
            configs=configs,
            datatraining=configs["training_set"],
            datacoding=configs["validation_set"],
            N=N
        )
        
        save_N_data(configs,N,N_data)

        for phase in [ph for ph in configs["phases"] if ph != "coding"]:
            for k in N_data[phase].keys():
                v = min(N_data[phase][k]) if (configs['reduction'] == 'min') else N_data[phase][k][-1]
                data[phase][k] = (data[phase][k] + [v]) if (k in data[phase].keys()) else [v]

        
    save_final_data(configs,data)
