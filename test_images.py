
import torch
import time
import os
from utils import read_im2bw
from utils import save_N_data
from utils import save_N_model
from utils import save_final_data
from models import MLP_N_64N_32N_1
from models import train_loop


if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            '/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_1.png',
            '/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_3.png'],
        "validation_set": ['/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_5.png'],
        "epochs": 420,
        "learning_rate":0.00001,
        "batch_size":2048,
        "num_workers":1,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "N_vec":  [0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],
        "phases": ['train', 'valid'],
        "xscale": 'symlog',
        "reduction": 'last'
    }

    imgtraining = [read_im2bw(img,0.4) for img in configs["training_set"]]
    imgcoding = [read_im2bw(img,0.4) for img in configs["validation_set"]]

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
            imgtraining=imgtraining,
            imgcoding=imgcoding,
            N=N
        )
        
        save_N_data(configs,N,N_data)
        
        save_N_model(configs,N,mlp_model)

        for phase in configs["phases"]:
            if configs['reduction'] == 'min':
                data[phase]["mlp"].append(min(N_data[phase]["mlp"]))
            else:
                data[phase]["mlp"].append(N_data[phase]["mlp"][-1])
            data[phase]["static"].append(N_data[phase]["static"][-1])
            data[phase]["cabac"].append(N_data[phase]["cabac"][-1])
        
    save_final_data(configs,data)