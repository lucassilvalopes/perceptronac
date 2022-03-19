#!/usr/bin/env python
# coding: utf-8

# help:
# 
# https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html
# 
# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705


import torch
from causal_context import causal_context
from context_training import context_training
from context_coding import context_coding
from perfect_AC import perfect_AC
from utils import read_im2bw
from utils import load_model
from utils import causal_context_many_imgs
from utils import save_N_min_valid_loss_model
import numpy as np
from tqdm import tqdm


class StaticAC:
    def fit(self,yt):
        self.p = np.sum(yt==1)/len(yt)
    def predict(self,yc):
        return self.p * np.ones(yc.shape)


class CABAC:
    def __init__(self,max_context = 27):
        self.max_context = max_context
    def fit(self,Xt,yt):
        self.context_p = context_training(Xt,yt,self.max_context)
    def predict(self,Xc,yc):
        return context_coding(Xc,yc,self.context_p)


class CausalContextDataset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx,:],self.y[idx,:]


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


def staticAC_rates(yt,yc):
    staticAC = StaticAC()
    staticAC.fit(yt)
    static_pred_t = staticAC.predict(yt)
    static_pred_c = staticAC.predict(yc)
    rate_static_t = perfect_AC(yt,static_pred_t)
    rate_static_c = perfect_AC(yc,static_pred_c)
    return rate_static_t,rate_static_c


def cabac_rates(Xt,yt,Xc,yc):
    cabac = CABAC()
    if Xt.shape[1] > cabac.max_context:
        return -1, -1
    cabac.fit(Xt,yt)
    cabac_pred_t = cabac.predict(Xt,yt)
    cabac_pred_c = cabac.predict(Xc,yc)
    rate_cabac_t = perfect_AC(yt,cabac_pred_t)
    rate_cabac_c = perfect_AC(yc,cabac_pred_c)
    return rate_cabac_t,rate_cabac_c


def train_loop(configs,imgtraining,imgcoding,N):
    
    OptimizerClass=configs["OptimizerClass"]
    epochs=configs["epochs"]
    learning_rate=configs["learning_rate"]
    batch_size=configs["batch_size"]
    num_workers=configs["num_workers"]
    device=configs["device"]
    phases=configs["phases"]
    
    yt,Xt = causal_context_many_imgs(imgtraining, N)
    yc,Xc = causal_context_many_imgs(imgcoding, N)
    
    rate_static_t,rate_static_c = staticAC_rates(yt,yc)
    if N == 0:
        data = dict()
        for phase in phases:
            data[phase] = {
                "mlp": [rate_static_t] if phase == 'train' else [rate_static_c],
                "cabac": [rate_static_t] if phase == 'train' else [rate_static_c],
                "static": [rate_static_t] if phase == 'train' else [rate_static_c],
            }
        return data,None
    
    rate_cabac_t,rate_cabac_c = cabac_rates(Xt=Xt,yt=yt,Xc=Xc,yc=yc)
    
    trainset = CausalContextDataset(Xt,yt)
    validset = CausalContextDataset(Xc,yc)
    
    device = torch.device(device)
    
    model = load_model(configs,N)
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
            

        save_N_min_valid_loss_model(valid_loss,configs,N,model)
        
    data = dict()
    for phase in phases:
        data[phase] = {
            "mlp": train_loss if phase == 'train' else valid_loss,
            "cabac": [rate_cabac_t] if phase == 'train' else [rate_cabac_c],
            "static": [rate_static_t] if phase == 'train' else [rate_static_c],
        }
    
    return data, model

