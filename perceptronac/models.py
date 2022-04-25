#!/usr/bin/env python
# coding: utf-8

# help:
# 
# https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html
# 
# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705


import torch
from perceptronac.context_training import context_training
from perceptronac.context_coding import context_coding
from perceptronac.perfect_AC import perfect_AC
from perceptronac.utils import causal_context_many_imgs
from perceptronac.utils import causal_context_many_pcs
from perceptronac.utils import jbig1_rate
from perceptronac.loading_and_saving import save_model
from perceptronac.loading_and_saving import save_fig
from perceptronac.loading_and_saving import save_values
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_configs
import numpy as np
from tqdm import tqdm
import os


class Perceptron(torch.nn.Module):
    def __init__(self,N):
        super().__init__()
        self.linear = torch.nn.Linear(N, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class MLP_N_64N_32N_1(torch.nn.Module):
    def __init__(self,N):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, 64*N),
            torch.nn.ReLU(),
            torch.nn.Linear(64*N, 32*N),
            torch.nn.ReLU(),
            torch.nn.Linear(32*N, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)


class MLP_N_1024_512_1(torch.nn.Module):
    def __init__(self,N):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)


class MLP_N_2048_1024_1(torch.nn.Module):
    def __init__(self,N):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)


class Log2BCELoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.bce_loss(pred, target)/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class CausalContextDataset(torch.utils.data.Dataset):
    def __init__(self,pths,data_type,N,percentage_of_uncles=None):
        self.pths = pths
        self.data_type = data_type
        self.N = N
        self.percentage_of_uncles = percentage_of_uncles
        self.getXy()

    def getXy(self):
        if self.data_type == "image":
            self.y,self.X = causal_context_many_imgs(self.pths, self.N)
        elif self.data_type == "pointcloud":
            if self.percentage_of_uncles is None:
                m = f"Input percentage_of_uncles must be specified "+\
                    "for data type pointcloud."
                raise ValueError(m)
            self.y,self.X = causal_context_many_pcs(
                self.pths, self.N, self.percentage_of_uncles)
        else:
            m = f"Data type {self.data_type} not supported.\n"+\
                "Supported data types : image, pointcloud."
            raise ValueError(m)

    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx,:],self.y[idx,:]


class StaticAC:        
    def __call__(self,X):
        return self.forward(X)
    def load_p(self,y=None,p=None):
        if y is not None:
            self.p = float(np.sum(y==1)/len(y))
        elif p is not None:
            self.p = p
        else:
            raise ValueError("Specify either p or y")
    def forward(self,X):
        if isinstance(X,torch.Tensor):
            return self.p * torch.ones((X.shape[0],1),device=X.device)
        else:
            return self.p * np.ones((X.shape[0],1))


class CABAC:
    def __init__(self,max_context):
        self.max_context = max_context
    def load_lut(self,X=None,y=None,lut=None):
        if lut is not None:
            self.context_p = lut
        elif (X is not None) and (y is not None):
            self.context_p = context_training(X,y,self.max_context)
        else:
            raise ValueError("Specify either lut or both X and y")
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        if isinstance(X,torch.Tensor):
            device = X.device
            X = X.cpu().detach().numpy()
            pp = context_coding(X,self.context_p)
            return torch.tensor(pp,device=device)
        else:
            return context_coding(X,self.context_p)



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
                staticac.load_p(trainset.y)
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
            staticac.load_p(p[0])
        return staticac

    def save_N_model(self,N,staticac):
        p = staticac.p
        if ('train' in self.configs["phases"]) and (N==0):
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
        cabac = self.load_model(N)        
        train_loss, valid_loss = [], []
        for phase in sorted(phases): # train first then valid
            if phase == 'train':
                dataset = trainset
                cabac.load_lut(trainset.X,trainset.y)
            else:
                dataset = validset
            if (dataset.N > max_context):
                final_loss = -1
            else:
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
            cabac.load_lut(lut.reshape(-1,1))
        return cabac

    def save_N_model(self,N,cabac):
        lut = cabac.context_p.reshape(-1)
        if ('train' in self.configs["phases"]) and (N>0):
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
        train_loss, valid_loss = [], []
        for phase in sorted(phases): # train first then valid
            if phase == 'train':
                dataset = trainset
            else:
                dataset = validset
            if (dataset.N != 10):
                final_loss = -1
            else:
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


def train_loop(configs,datatraining,datacoding,N):
    
    trainset = CausalContextDataset(
        datatraining,configs["data_type"],N, configs["percentage_of_uncles"])
    validset = CausalContextDataset(
        datacoding,configs["data_type"],N, configs["percentage_of_uncles"])

    if N == 0:
        rate_static_t,rate_static_c = RatesStaticAC(configs).get_rates(trainset,validset)
    else:
        rate_cabac_t,rate_cabac_c = RatesCABAC(configs).get_rates(trainset,validset)        
        train_loss, valid_loss = RatesMLP(configs).get_rates(trainset,validset)
        if (configs["data_type"] == "image"):
            rate_jbig1_t,rate_jbig1_c = RatesJBIG1(configs).get_rates(trainset,validset)

    phases=configs["phases"]
    epochs=configs["epochs"]

    data = dict()
    for phase in phases:
        data[phase] = dict()

    if N == 0:
        
        for phase in phases:
            data[phase]["MLP"] = epochs*[rate_static_t] if phase == 'train' else epochs*[rate_static_c]
            data[phase]["LUT"] = epochs*[rate_static_t] if phase == 'train' else epochs*[rate_static_c]
            if configs["data_type"] == "image":
                data[phase]["JBIG1"] = epochs*[-1]

    else:

        for phase in phases:
            data[phase]["MLP"] = train_loss if phase == 'train' else valid_loss
            data[phase]["LUT"] = epochs*[rate_cabac_t] if phase == 'train' else epochs*[rate_cabac_c]
            if configs["data_type"] == "image":
                data[phase]["JBIG1"] = epochs*[rate_jbig1_t] if phase == 'train' else epochs*[rate_jbig1_c]
    
    return data


def experiment(configs):

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