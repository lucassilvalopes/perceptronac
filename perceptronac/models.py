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
from perceptronac.utils import causal_context_many_imgs
from perceptronac.utils import causal_context_many_pcs
import numpy as np


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
            X = X.detach().cpu().numpy()
            pp = context_coding(X,self.context_p)
            return torch.tensor(pp,device=device)
        else:
            return context_coding(X,self.context_p)


class MLP_N_64N_32N_1_Constructor:

    def __init__(self,N,weights):
        self.N = N
        self.weights = weights

    def construct(self):
        model = MLP_N_64N_32N_1(self.N)
        model.load_state_dict(torch.load(self.weights))
        model.train(False)
        return model


class StaticAC_Constructor:

    def __init__(self,weights):
        self.weights = weights

    def construct(self):
        staticac = StaticAC()
        with open(self.weights, 'rb') as f:
            p = np.load(f)
        staticac.load_p(p=p[0])
        return staticac


class CABAC_Constructor:

    def __init__(self,weights,max_context):
        self.weights = weights
        self.max_context = max_context

    def construct(self):
        cabac = CABAC(self.max_context)
        with open(self.weights, 'rb') as f:
            lut = np.load(f)
        cabac.load_lut(lut=lut.reshape(-1,1))
        return cabac