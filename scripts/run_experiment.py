
import torch
import time
import os
from perceptronac.utils import read_im2bw
from perceptronac.utils import save_N_data
from perceptronac.utils import save_N_model
from perceptronac.utils import save_final_data
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.models import train_loop
import perceptronac.coding3d as c3d


if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            "/home/lucaslopes/longdress/longdress_vox10_1300.ply"
        ],
        "validation_set": [
            # "/home/lucaslopes/redandblack/redandblack_vox10_1450.ply"
            "/home/lucaslopes/longdress/longdress_vox10_1051.ply"
        ],
        "epochs": 420,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec": [0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],
        "phases": ['train', 'valid'],
        "xscale": 'symlog',
        "reduction": 'last',
        "data_type": 'pointcloud', # image, pointcloud
        "percentage_of_uncles": 0.0 # must be specified if the data types is pointcloud
    }

    if configs["data_type"] == "image":
        datatraining = [read_im2bw(img,0.4) for img in configs["training_set"]]
        datacoding = [read_im2bw(img,0.4) for img in configs["validation_set"]]
    elif configs["data_type"] == "pointcloud":
        datatraining = [c3d.read_PC(pc)[1] for pc in configs["training_set"]]
        datacoding = [c3d.read_PC(pc)[1] for pc in configs["validation_set"]]
    else:
        raise ValueError(f'data type {configs["data_type"]} not supported')

    os.makedirs(f"results/exp_{configs['id']}")

    data = dict()
    for phase in configs["phases"]:
        data[phase] = {
            "mlp": [], 
            "static": [],    
            "cabac": []          
        }

    for N in configs["N_vec"]:
        print(f"--------------------- context size : {N} ---------------------")    
        N_data,mlp_model = train_loop(
            configs=configs,
            datatraining=datatraining,
            datacoding=datacoding,
            N=N
        )
        
        save_N_data(configs,N,N_data)

        for phase in configs["phases"]:
            if configs['reduction'] == 'min':
                data[phase]["mlp"].append(min(N_data[phase]["mlp"]))
            else:
                data[phase]["mlp"].append(N_data[phase]["mlp"][-1])
            data[phase]["static"].append(N_data[phase]["static"][-1])
            data[phase]["cabac"].append(N_data[phase]["cabac"][-1])
        
    save_final_data(configs,data)
