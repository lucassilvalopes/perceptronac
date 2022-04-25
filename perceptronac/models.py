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
from perceptronac.loading_and_saving import load_model
from perceptronac.loading_and_saving import save_N_min_valid_loss_model
from perceptronac.loading_and_saving import save_N_model
import numpy as np
from tqdm import tqdm


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
    def __init__(self,y):
        self.p = float(np.sum(y==1)/len(y))
    def __call__(self,X):
        return self.forward(X)
    def forward(self,X):
        if isinstance(X,torch.Tensor):
            return self.p * torch.ones((X.shape[0],1),device=X.device)
        else:
            return self.p * np.ones((X.shape[0],1))


class CABAC:
    def __init__(self,X,y,max_context):
        self.context_p = context_training(X,y,max_context)
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


class RatesStaticAC:
    def get_rates(self,trainset,validset):
        Xt,yt = trainset.X,trainset.y
        Xc,yc = validset.X,validset.y
        staticac = StaticAC(yt)
        static_pred_t = staticac(Xt)
        static_pred_c = staticac(Xc)
        rate_static_t = perfect_AC(yt,static_pred_t)
        rate_static_c = perfect_AC(yc,static_pred_c)
        return rate_static_t,rate_static_c


class RatesCABAC:
    def __init__(self,max_context = 27):
        self.max_context = max_context
    def get_rates(self,trainset,validset):
        if (trainset.N > self.max_context):
            return -1, -1
        Xt,yt = trainset.X,trainset.y
        Xc,yc = validset.X,validset.y
        cabac = CABAC(Xt,yt,self.max_context)
        cabac_pred_t = cabac(Xt)
        cabac_pred_c = cabac(Xc)
        rate_cabac_t = perfect_AC(yt,cabac_pred_t)
        rate_cabac_c = perfect_AC(yc,cabac_pred_c)
        return rate_cabac_t,rate_cabac_c


class RatesJBIG1:
    def avg_rate(self,pths):
        """
        make sure all images in pths have the same size
        """
        rate = 0
        for pth in pths:
            rate += jbig1_rate(pth)
        return rate/len(pths)

    def get_rates(self,trainset,validset):
        if (trainset.N != 10) or (validset.N != 10):
            return -1, -1
        datatraining = trainset.pths
        datacoding = validset.pths
        rate_jbig1_t = self.avg_rate(datatraining)
        rate_jbig1_c = self.avg_rate(datacoding)
        return rate_jbig1_t,rate_jbig1_c        


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
        
        model = load_model(self.configs,N)
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
                

            save_N_min_valid_loss_model(valid_loss,self.configs,N,model)
 
        # save final model
        save_N_model(self.configs,N,model)

        return train_loss, valid_loss


def train_loop(configs,datatraining,datacoding,N):
    
    trainset = CausalContextDataset(
        datatraining,configs["data_type"],N, configs["percentage_of_uncles"])
    validset = CausalContextDataset(
        datacoding,configs["data_type"],N, configs["percentage_of_uncles"])

    if N == 0:
        rate_static_t,rate_static_c = RatesStaticAC().get_rates(trainset,validset)
    else:
        rate_cabac_t,rate_cabac_c = RatesCABAC().get_rates(trainset,validset)        
        train_loss, valid_loss = RatesMLP(configs).get_rates(trainset,validset)
        if (configs["data_type"] == "image"):
            rate_jbig1_t,rate_jbig1_c = RatesJBIG1().get_rates(trainset,validset)

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

