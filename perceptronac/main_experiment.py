import torch
from perceptronac.perfect_AC import perfect_AC
from perceptronac.utils import jbig1_rate
from perceptronac.loading_and_saving import save_model
from perceptronac.loading_and_saving import save_fig
from perceptronac.loading_and_saving import save_values
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_configs
from perceptronac.models import Log2BCELoss
from perceptronac.models import CABAC
from perceptronac.models import StaticAC
from perceptronac.models import CausalContextDataset
import numpy as np
from tqdm import tqdm
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.models import MLP_N_64N_32N_1_Constructor
from perceptronac.coders import PC_Coder
import perceptronac.coding3d as c3d
from perceptronac.coders import get_bpov


def get_prefix(configs, id_key = 'id' ):
    return f"{configs['save_dir'].rstrip('/')}/exp_{configs[id_key]}/exp_{configs[id_key]}"


class RatesStaticAC:
    def __init__(self,configs):
        self.configs = configs

    def get_rates(self,trainset,validset):
        phases=self.configs["phases"]
        epochs=self.configs["epochs"]
        N = trainset.N
        staticac = self.load_model(N)        
        train_loss, valid_loss = [], []
        for phase in sorted(phases):
            if phase == 'train':
                dataset = trainset
                staticac.load_p(y=trainset.y)
            else:
                dataset = validset
            X,y = dataset.X,dataset.y
            static_pred = staticac(X)
            final_loss = perfect_AC(y,static_pred)
            if phase=='train':
                train_loss.append(final_loss)
            else:
                valid_loss.append(final_loss)
        self.save_N_model(N,staticac)
        return epochs*train_loss, epochs*valid_loss

    def load_model(self,N):
        staticac = StaticAC()
        if self.configs.get("parent_id"):
            file_name = f"{get_prefix(self.configs,'parent_id')}_{N:03d}_p.npy"
            with open(file_name, 'rb') as f:
                p = np.load(f)
            staticac.load_p(p=p[0])
        return staticac

    def save_N_model(self,N,staticac):
        if ('train' in self.configs["phases"]) and (N==0):
            p = staticac.p
            with open(f"{get_prefix(self.configs)}_{N:03d}_p.npy", 'wb') as f:
                np.save(f, np.array([p]))

class RatesCABAC:
    def __init__(self,configs):
        self.configs = configs
        
    def get_rates(self,trainset,validset):
        phases=self.configs["phases"]
        max_context = self.configs["max_context"]
        epochs=self.configs["epochs"]
        N = trainset.N
        if (N > max_context):
            return epochs*[-1],epochs*[-1]
        cabac = self.load_model(N)        
        train_loss, valid_loss = [], []
        for phase in sorted(phases): # train first then valid
            if phase == 'train':
                dataset = trainset
                cabac.load_lut(X=trainset.X,y=trainset.y)
            else:
                dataset = validset
            X,y = dataset.X,dataset.y
            cabac_pred = cabac(X)
            final_loss = perfect_AC(y,cabac_pred)
            if phase=='train':
                train_loss.append(final_loss)
            else:
                valid_loss.append(final_loss)
        self.save_N_model(N,cabac)
        return epochs*train_loss, epochs*valid_loss

    def load_model(self,N):
        cabac = CABAC(self.configs["max_context"])
        if self.configs.get("parent_id"):
            file_name = f"{get_prefix(self.configs,'parent_id')}_{N:03d}_lut.npy"
            with open(file_name, 'rb') as f:
                lut = np.load(f)
            cabac.load_lut(lut=lut.reshape(-1,1))
        return cabac

    def save_N_model(self,N,cabac):
        if ('train' in self.configs["phases"]) and (N>0):
            lut = cabac.context_p.reshape(-1)
            with open(f"{get_prefix(self.configs)}_{N:03d}_lut.npy", 'wb') as f:
                np.save(f, lut)


class RatesJBIG1:
    def __init__(self,configs):
        self.configs = configs

    def avg_rate(self,pths):
        """
        make sure all images in pths have the same size
        """
        rate = 0
        for pth in pths:
            rate += jbig1_rate(pth)
        return rate/len(pths)

    def get_rates(self,trainset,validset):
        phases=self.configs["phases"]
        epochs=self.configs["epochs"]
        N = trainset.N
        if (N != 10):
            return epochs*[-1],epochs*[-1]
        train_loss, valid_loss = [], []
        for phase in sorted(phases): # train first then valid
            if phase == 'train':
                dataset = trainset
            else:
                dataset = validset
            final_loss = self.avg_rate(dataset.pths)
            if phase=='train':
                train_loss.append(final_loss)
            else:
                valid_loss.append(final_loss)
        return epochs*train_loss, epochs*valid_loss
     

class RatesMLP:

    def __init__(self,configs):
        self.configs = configs

    def get_rates(self,trainset,validset):

        N = trainset.N

        OptimizerClass=self.configs["OptimizerClass"]
        epochs=self.configs["epochs"]
        learning_rate=self.configs["learning_rate"]
        batch_size=self.configs["batch_size"]
        num_workers=self.configs["num_workers"]
        device=self.configs["device"]
        phases=self.configs["phases"]
        
        device = torch.device(device)
        
        model = self.load_model(N)
        model.to(device)
        
        criterion = Log2BCELoss(reduction='sum')
        optimizer = OptimizerClass(model.parameters(), lr=learning_rate)

        train_loss, valid_loss = [], []
        print("starting training")
        print("len trainset : ", len(trainset),", len validset : ",len(validset))
        for epoch in range(epochs):
            
            for phase in phases:
                
                if phase == 'train':
                    model.train(True)
                    dataloader = torch.utils.data.DataLoader(
                        trainset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
                else:
                    model.train(False)
                    dataloader=torch.utils.data.DataLoader(
                        validset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

                running_loss = 0.0
                for data in tqdm(dataloader):

                    Xt_b,yt_b= data
                    Xt_b = Xt_b.float().to(device)
                    yt_b = yt_b.float().to(device)

                    if phase == 'train':
                        optimizer.zero_grad()
                        outputs = model(Xt_b)
                        loss = criterion(outputs, yt_b)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            outputs = model(Xt_b)
                            loss = criterion(outputs, yt_b)

                    running_loss += loss.item()
                
                final_loss = running_loss / len(dataloader.dataset)
                if phase=='train':
                    train_loss.append(final_loss)
                else:
                    valid_loss.append(final_loss)
                
                print("epoch :" , epoch, ", phase :", phase, ", loss :", final_loss)
                

            self.save_N_min_valid_loss_model(valid_loss,N,model)
 
        # save final model
        self.save_N_model(N,model)

        return train_loss, valid_loss

    def load_model(self,N):
        ModelClass=self.configs["ModelClass"]
        model = ModelClass(N)
        if self.configs.get("parent_id"):
            if ('train' not in self.configs["phases"]) and (self.configs["reduction"] == 'min'):
                file_name = f"{get_prefix(self.configs,'parent_id')}_{N:03d}_min_valid_loss_model.pt"
            else:
                file_name = f"{get_prefix(self.configs,'parent_id')}_{N:03d}_model.pt"
            print(f"loading file {file_name}")
            model.load_state_dict(torch.load(file_name))
        return model

    def save_N_min_valid_loss_model(self,valid_loss,N,mlp_model):
        if len(valid_loss) == 0:
            pass
        elif (min(valid_loss) == valid_loss[-1]) and ('train' in self.configs["phases"]) and (N>0):
            save_model(f"{get_prefix(self.configs)}_{N:03d}_min_valid_loss_model",mlp_model)

    def save_N_model(self,N,mlp_model):
        if ('train' in self.configs["phases"]) and (N>0):
            save_model(f"{get_prefix(self.configs)}_{N:03d}_model",mlp_model)


def train_loop(configs,datatraining,datacoding,N):
    
    trainset = CausalContextDataset(
        datatraining,configs["data_type"],N, configs["percentage_of_uncles"])
    validset = CausalContextDataset(
        datacoding,configs["data_type"],N, configs["percentage_of_uncles"])

    if N == 0:
        rates_static_t,rates_static_c = RatesStaticAC(configs).get_rates(trainset,validset)
    else:
        rates_cabac_t,rates_cabac_c = RatesCABAC(configs).get_rates(trainset,validset)        
        rates_mlp_t,rates_mlp_c = RatesMLP(configs).get_rates(trainset,validset)
        if (configs["data_type"] == "image"):
            rates_jbig1_t,rates_jbig1_c = RatesJBIG1(configs).get_rates(trainset,validset)

    phases=configs["phases"]
    epochs=configs["epochs"]

    data = dict()
    for phase in phases:
        data[phase] = dict()

    if N == 0:
        
        for phase in phases:
            data[phase]["MLP"] = rates_static_t if phase == 'train' else rates_static_c
            data[phase]["LUT"] = rates_static_t if phase == 'train' else rates_static_c
            if configs["data_type"] == "image":
                data[phase]["JBIG1"] = epochs*[-1]

    else:

        for phase in phases:
            data[phase]["MLP"] = rates_mlp_t if phase == 'train' else rates_mlp_c
            data[phase]["LUT"] = rates_cabac_t if phase == 'train' else rates_cabac_c
            if configs["data_type"] == "image":
                data[phase]["JBIG1"] = rates_jbig1_t if phase == 'train' else rates_jbig1_c
    
    return data


def coding_loop(configs,N):

    cond1 = (configs["data_type"] == "pointcloud")
    cond2 = (configs["percentage_of_uncles"] == 0)
    cond3 = (configs['ModelClass'] == MLP_N_64N_32N_1)
    cond4 = (len(configs["validation_set"]) == 1)
    ok = cond1 and cond2 and cond3 and cond4
        
    if not ok:
        m = f"""
            coding currently supported only for the combination
            len(validation_set) : 1
            data_type : pointcloud
            percentage_of_uncles : 0
            ModelClass : MLP_N_64N_32N_1
            """
        raise ValueError(m)

    if (configs["reduction"] == 'min'):
        weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_min_valid_loss_model.pt"
    else:
        weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_model.pt"
    constructor = MLP_N_64N_32N_1_Constructor(N,weights)
    coder = PC_Coder(constructor.construct,N,configs["last_octree_level"])
    pc_in = configs["validation_set"][0]
    pc_len = len(c3d.read_PC(pc_in)[1])
    coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
    mlp_rate = get_bpov("/tmp/encoder_out",pc_len)

    data = {
        "coding": {
            "MLP": mlp_rate
        }   
    }
    return data
    

def save_N_data(configs,N,N_data):
    
    xvalues = np.arange(configs["epochs"])
    xlabel = "epoch"

    for phase in configs["phases"]:
        
        fig = plot_comparison(xvalues,N_data[phase],xlabel)
        save_fig(f"{get_prefix(configs)}_{N:03d}_{phase}_graph",fig)
        save_values(f"{get_prefix(configs)}_{N:03d}_{phase}_values",xvalues,N_data[phase],xlabel)
 

def save_final_data(configs,data):
    
    xvalues = configs["N_vec"]
    xlabel = "context size"
    xscale = configs["xscale"]
    
    save_configs(f"{get_prefix(configs)}_conf",configs)
    
    for phase in configs["phases"]:
        
        fig=plot_comparison(xvalues,data[phase],xlabel,xscale=xscale)
        save_fig(f"{get_prefix(configs)}_{phase}_graph",fig)
        save_values(f"{get_prefix(configs)}_{phase}_values",xvalues,data[phase],xlabel)


def experiment(configs):

    os.makedirs(f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}")

    data = dict()
    for phase in configs["phases"]:
        data[phase] = dict()

    if ("train" in configs["phases"]) or ("valid" in configs["phases"]):
        for N in configs["N_vec"]:
            print(f"--------------------- context size : {N} ---------------------")    
            N_data = train_loop(
                configs={
                    k:([ph for ph in v if ph != "coding"] if v == 'phases' else v) for k,v in configs.items()
                },
                datatraining=configs["training_set"],
                datacoding=configs["validation_set"],
                N=N
            )
            
            save_N_data(configs,N,N_data)

            for phase in [ph for ph in configs["phases"] if ph != "coding"]:
                for k in N_data[phase].keys():
                    v = min(N_data[phase][k]) if (configs['reduction'] == 'min') else N_data[phase][k][-1]
                    data[phase][k] = (data[phase][k] + [v]) if (k in data[phase].keys()) else [v]

    if ("coding" in configs["phases"]):
        for N in configs["N_vec"]:
            N_data = coding_loop(configs,N)
            for k in N_data["coding"].keys():
                data["coding"][k] = N_data["coding"][k]

    save_final_data(configs,data)