#!/usr/bin/env python
# coding: utf-8

# help:
# 
# https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html
# 
# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705


import torch
from perceptronac.context_training import context_training,context_training_nonbinary
from perceptronac.context_coding import context_coding,context_coding_nonbinary
from perceptronac.utils import causal_context_many_imgs
from perceptronac.utils import causal_context_many_pcs
import numpy as np
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


class ArbitraryMLP(torch.nn.Module):
    def __init__(self,widths,intended_loss = "BCELoss", head_x3 = False):
        """
        Examples: 
            For a binary classification, the following are equivalent:
                - widths=[N,64*N,32*N,1] with intended_loss="BCELoss" or "BCEWithLogitsLoss"
                - widths=[N,64*N,32*N,2] with intended_loss="CrossEntropyLoss" oe "NLLLoss"
        """
        super().__init__()
        modules = []
        n_layers = len(widths[:-1])
        for i in range(n_layers):
            
            if (i == n_layers-1) and head_x3: 
                modules.append(ColorHead(widths[i],widths[i+1]))
            else:
                modules.append(torch.nn.Linear(widths[i],widths[i+1]))

            if i < n_layers-1: 
                modules.append(torch.nn.ReLU())
            elif intended_loss == "BCELoss":
                modules.append(torch.nn.Sigmoid())
            elif intended_loss == "BCEWithLogitsLoss":
                # Sigmoid is included in the nn.BCEWithLogitsLoss
                pass                
            elif intended_loss == "CrossEntropyLoss":
                # Softmax is included in the nn.CrossEntropyLoss
                pass
            elif intended_loss == "NLLLoss":
                modules.append(torch.nn.LogSoftmax(dim=1))
        self.layers = torch.nn.Sequential(*modules)
    def forward(self, x):
        return self.layers(x)


class ColorHead(torch.nn.Module):
    """
    inspirations:
    https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
    https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
    https://pytorch.org/docs/stable/generated/torch.cat.html
    """
    def __init__(self,in_dim,C):
        super().__init__()
        self.ch1 = torch.nn.Linear(in_dim,C)
        self.ch2 = torch.nn.Linear(in_dim,C)
        self.ch3 = torch.nn.Linear(in_dim,C)

    def forward(self, x):
        x = torch.cat([self.ch1(x).unsqueeze(2),self.ch2(x).unsqueeze(2),self.ch3(x).unsqueeze(2)], dim=2)
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


# class MLP_N_64N_32N_1(torch.nn.Module):
#     """binary image or point cloud geometry"""
#     def __init__(self,N):
#         super().__init__()
#         self.mlp = ArbitraryMLP([N,64*N,32*N,1],intended_loss="BCELoss",head_x3=False)
#     def forward(self, x):
#         return self.mlp(x)


class MLP_N_64N_32N_256(torch.nn.Module):
    """grayscale image"""
    def __init__(self,N):
        super().__init__()
        self.mlp = ArbitraryMLP([N,64*N,32*N,256],intended_loss="CrossEntropyLoss",head_x3=False)
    def forward(self, x):
        return self.mlp(x)


class MLP_N_64N_32N_3x256(torch.nn.Module):
    """color image"""
    def __init__(self,N):
        super().__init__()
        self.mlp = ArbitraryMLP([3*N,64*N,32*N,256],intended_loss="CrossEntropyLoss",head_x3=True)
    def forward(self, x):
        return self.mlp(x)


class Pointnet_3_64N_32N_16N_1(torch.nn.Module):
    def __init__(self,N):
        super().__init__()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(3, 64*N),
            torch.nn.ReLU(),
            torch.nn.Linear(64*N, 32*N),
            torch.nn.ReLU()
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(32*N, 16*N),
            torch.nn.ReLU(),
            torch.nn.Linear(16*N, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = self.mlp1(x)
        x = torch.max(x,dim=1,keepdim=False)
        x = self.mlp2(x)
        return x


class Log2BCELoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.bce_loss(pred, target)/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class Log2CrossEntropyLoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.cross_entropy_loss(pred, target.long())/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class Log2NLLLoss(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(*args,**kwargs)

    def forward(self, pred, target):
        return self.nll_loss(pred, target.long())/torch.log(torch.tensor(2,dtype=target.dtype,device=target.device))


class CausalContextDataset(torch.utils.data.Dataset):
    def __init__(
        self,pths,data_type,N,percentage_of_uncles=None,getXy_later=False,
        geo_or_attr = None, n_classes=None, channels=None, color_space=None
    ):
        self.pths = pths
        self.data_type = data_type
        self.N = N

        if data_type == "pointcloud":
            self.percentage_of_uncles = percentage_of_uncles
            self.geo_or_attr = geo_or_attr if (geo_or_attr is not None) else "geometry"
            if self.geo_or_attr == "attributes":
                self.n_classes = n_classes if (n_classes is not None) else 256
                self.channels = channels if (channels is not None) else [1,0,0]
                self.color_space = color_space if (color_space is not None) else "YCbCr"
        elif data_type == "image":
            self.n_classes = n_classes if (n_classes is not None) else 2
            self.channels = channels if (channels is not None) else [1,0,0]
            self.color_space = color_space if (color_space is not None) else "YCbCr"

        if getXy_later:
            self.y,self.X = [],[]
        else:
            self.getXy()

    def getXy(self):
        if self.data_type == "image":
            self.y,self.X = causal_context_many_imgs(
                self.pths, self.N, n_classes=self.n_classes,
                channels=self.channels, color_space=self.color_space)
        elif self.data_type == "pointcloud":
            if self.percentage_of_uncles is None:
                m = f"Input percentage_of_uncles must be specified "+\
                    "for data type pointcloud."
                raise ValueError(m)
            self.y,self.X = causal_context_many_pcs(
                self.pths, self.N, self.percentage_of_uncles, geo_or_attr = self.geo_or_attr,
                n_classes=self.n_classes, channels=self.channels, color_space=self.color_space)
        elif self.data_type == "table":
            Xy = np.vstack([self.load_table(pth) for pth in self.pths])
            if (self.channels is None) or np.count_nonzero(self.channels) == 1:
                assert Xy.shape[1] - 1 == self.N
                self.X = Xy[:,:self.N]
                self.y = Xy[:,self.N:]
            elif np.count_nonzero(self.channels) == 3:
                assert Xy.shape[1] - 3 == 3*self.N
                self.X = Xy[:,:3*self.N]
                self.y = Xy[:,3*self.N:]
            else:
                m = f"Unsupported number of channels {np.count_nonzero(self.channels)}."
                raise ValueError(m)
        else:
            # https://stackoverflow.com/questions/12262463/print-out-the-class-parameters-on-python
            m = "Unsupported combination "+ ' '.join([f"{k}={v}" for k,v in self.__dict__.items()])
            raise ValueError(m)

    @staticmethod
    def load_table(pth):
        if pth.endswith("txt") or pth.endswith("csv"):
            return np.genfromtxt(pth)
        elif pth.endswith("npz"):
            return np.load(pth)["arr_0"]
        else:
            raise ValueError(f"Unknown table format {os.path.splitext(pth)[1]}")

    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        if (
            (False if (self.n_classes is None) else (self.n_classes == 256)) and 
            (False if (self.channels is None) else (np.count_nonzero(self.channels) == 1) )
        ):
            return self.X[idx,:],self.y[idx,0] # because of torch.nn.CrossEntropyLoss
        else:
            return self.X[idx,:],self.y[idx,:]


#static binary arithmetic coder
class StaticAC:
    def __call__(self,X):
        return self.forward(X)
    def load(self,y=None,p=None):
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


#static 256 arithmetic coder
class S256AC:
    def __call__(self,X):
        return self.forward(X)
    def load(self,y=None,ps=None):
        if y is not None:
            n_channels = y.shape[1]
            n_symbols = 256
            n_samples = y.shape[0]
            self.ps = np.zeros((1,n_symbols,n_channels))
            for ch in range(n_channels):
                self.ps[0,:,ch] = np.histogram(y[:,ch], bins=list(range(n_symbols+1)))[0]/n_samples
        elif ps is not None:
            self.ps = ps
        else:
            raise ValueError("Specify either ps or y")
    def forward(self,X):
        n_channels = self.ps.shape[2]
        n_symbols = self.ps.shape[1]
        n_predictions = X.shape[0]
        if isinstance(X,torch.Tensor):
            device = X.device
            return torch.ones((n_predictions,n_symbols,n_channels),device=device) * torch.tensor(self.ps,device=device)
        else:
            return np.ones((n_predictions,n_symbols,n_channels)) * self.ps


#context adaptive binary arithmetic coder
class CABAC:
    def __init__(self,max_context):
        self.max_context = max_context
    def load(self,X=None,y=None,lut=None):
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


#context adaptive 256 arithmetic coder
class CA256AC:
    def load(self,X=None,y=None,lut=None):
        if (lut is not None):
            self.lut = lut
        elif (X is not None) and (y is not None):
            n_channels = y.shape[1]
            self.lut = [context_training_nonbinary(X,y[:,ch:ch+1]) for ch in range(n_channels)]
        else:
            raise ValueError("Specify either lut or both X and y")
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        if isinstance(X,torch.Tensor):
            device = X.device
            X = X.detach().cpu().numpy()
            pp = np.concatenate([np.expand_dims(context_coding_nonbinary(X,context_c),2) for context_c in self.lut],axis=2)
            return torch.tensor(pp,device=device)
        else:
            return np.concatenate([np.expand_dims(context_coding_nonbinary(X,context_c),2) for context_c in self.lut],axis=2)


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
        staticac.load(p=p[0])
        return staticac


class CABAC_Constructor:

    def __init__(self,weights,max_context):
        self.weights = weights
        self.max_context = max_context

    def construct(self):
        cabac = CABAC(self.max_context)
        with open(self.weights, 'rb') as f:
            lut = np.load(f)
        cabac.load(lut=lut.reshape(-1,1))
        return cabac