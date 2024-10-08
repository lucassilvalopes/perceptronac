
import torch
import warnings
from collections import namedtuple
from perceptronac.losses import perfect_AC
from perceptronac.utils import jbig1_rate
from perceptronac.data_exportation import save_model
from perceptronac.data_exportation import save_fig
from perceptronac.data_exportation import save_values
from perceptronac.data_exportation import load_values
from perceptronac.data_exportation import plot_comparison
from perceptronac.data_exportation import save_configs
from perceptronac.data_exportation import save_data
from perceptronac.losses import Log2BCELoss
from perceptronac.losses import Log2CrossEntropyLoss
from perceptronac.models import CABAC,CA256AC
from perceptronac.models import StaticAC,S256AC
from perceptronac.models import CausalContextDataset
import numpy as np
from tqdm import tqdm
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.models import MLP_N_64N_32N_1_Constructor
from perceptronac.models import ArbitraryMLP
from perceptronac.models import CABAC_Constructor
from perceptronac.models import StaticAC_Constructor
from perceptronac.coders import PC_Coder
import perceptronac.coding3d as c3d
from perceptronac.coders import get_bpov
import random
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import time
import subprocess
import pandas as pd

JBIG1Dataset = namedtuple('JBIG1Dataset', ['pths'])


def get_prefix(configs, id_key = 'id', parent_id_index = None ):
    """
    Prefix string used in file names when loading or saving data or models.
    Can return the prefix string for the current experiment (id_key = 'id')
    or a parent experiment (id_key = 'parent_id')

    Args:
        configs : dictionary containing configuration data.
            Has the keys 'id','parent_id','save_dir'.
        id_key : either 'id' or 'parent_id'
        parent_id_index : the value of the parent_id key in the 
            configs dictionary can be "" or [] or "XXXXXXXXX" or
            ["XXXXXXXXX","YYYYYYYYY",...] or ["XXXXXXXXX"], where 
            XXXXXXXXX and YYYYYYYYY are timestamps. In case it is
            a list, and it has length greater than 1, a parent_id_index
            must be specified.

    Returns:
        Prefix string.

    Exceptions:
        ValueError : if the value for the id_key in the configs dictionary is empty.
        ValueError : if there are multiple parents and parent_id_index is not specified
    """
    if not configs[id_key]:
        raise ValueError("Attempted to get prefix for empty id")

    if id_key == "parent_id":
        if isinstance(configs["parent_id"],list):
            if len(configs["parent_id"])==1:
                identifier = configs["parent_id"][0]
            else:
                if parent_id_index is None:
                    raise ValueError("parent_id_index must be specified in case of multiple parents")
                else:
                    identifier = configs["parent_id"][parent_id_index]
        else:
            identifier = configs[id_key]
    else:
        identifier = configs[id_key]
    return f"{configs['save_dir'].rstrip('/')}/exp_{identifier}/exp_{identifier}"


class RatesStaticAC:
    def __init__(self,configs,N):
        self.configs = configs
        self.N = N
        self.phases=self.configs["phases"]
        self.epochs=self.configs["epochs"]
        self.geo_or_attr=self.configs["geo_or_attr"]
        self.n_classes=self.configs["n_classes"]
        self.channels=self.configs["channels"]
        self.color_space=self.configs["color_space"]
        self.data_type = self.configs["data_type"]
        self.percentage_of_uncles = self.configs["percentage_of_uncles"]
        self.manual_th=self.configs["manual_th"]
        self.full_page=self.configs["full_page"]
        self.binary = (self.geo_or_attr=="geometry" or (self.n_classes==2 and np.count_nonzero(self.channels)==1) )
        
    def get_rates(self,datatraining,datacoding):

        staticac = self.load_model()
        train_loss, valid_loss = [], []
        for phase in sorted(self.phases):
            if phase == 'train':
                dataset = CausalContextDataset(datatraining, self.data_type, self.N, self.percentage_of_uncles,
                    geo_or_attr=self.geo_or_attr,n_classes=self.n_classes,channels=self.channels,color_space=self.color_space,
                    manual_th=self.manual_th,full_page=self.full_page)
                staticac.load(y=dataset.y)
            else:
                dataset = CausalContextDataset(datacoding, self.data_type, self.N, self.percentage_of_uncles,
                    geo_or_attr=self.geo_or_attr,n_classes=self.n_classes,channels=self.channels,color_space=self.color_space,
                    manual_th=self.manual_th,full_page=self.full_page)
            X,y = dataset.X,dataset.y
            static_pred = staticac(X)
            final_loss = perfect_AC(y,static_pred,binary=self.binary)
            if phase=='train':
                train_loss.append(final_loss)
            else:
                valid_loss.append(final_loss)
        self.save_N_model(staticac)
        return self.epochs*train_loss, self.epochs*valid_loss

    def load_model(self):
        if self.binary:
            staticac = StaticAC()
            if self.configs.get("parent_id"):
                with open(f"{get_prefix(self.configs,'parent_id')}_{self.N:03d}_p.npy", 'rb') as f:
                    p = np.load(f)
                staticac.load(p=p[0])
        elif self.n_classes==256:
            staticac = S256AC()
            if self.configs.get("parent_id"):
                n_channels = np.count_nonzero(self.channels)
                n_symbols = 256
                with open(f"{get_prefix(self.configs,'parent_id')}_{self.N:03d}_ps.npy", 'rb') as f:
                    ps = np.load(f)
                staticac.load(ps=ps.reshape(1,n_symbols,n_channels))
        else:
            raise ValueError(f"We only support 1-channel binary data and N-channel 256-ary data")
        return staticac

    def save_N_model(self,staticac):
        if ('train' in self.configs["phases"]) and (self.N==0):
            if isinstance(staticac,StaticAC):
                p = staticac.p
                with open(f"{get_prefix(self.configs)}_{self.N:03d}_p.npy", 'wb') as f:
                    np.save(f, np.array([p]))
            elif isinstance(staticac,S256AC):
                ps = staticac.ps.reshape(-1)
                with open(f"{get_prefix(self.configs)}_{self.N:03d}_ps.npy", 'wb') as f:
                    np.save(f, ps)
            else:
                ValueError("Unknown object")



class RatesCAAC:
    def __init__(self,configs,N):
        self.configs = configs
        self.N = N
        self.phases=self.configs["phases"]
        self.max_context = self.configs["max_context"]
        self.epochs=self.configs["epochs"]
        self.geo_or_attr=self.configs["geo_or_attr"]
        self.n_classes=self.configs["n_classes"]
        self.channels=self.configs["channels"]
        self.color_space=self.configs["color_space"]
        self.data_type = self.configs["data_type"]
        self.percentage_of_uncles = self.configs["percentage_of_uncles"]
        self.manual_th=self.configs["manual_th"]
        self.full_page=self.configs["full_page"]
        self.binary = (self.geo_or_attr=="geometry" or (self.n_classes==2 and np.count_nonzero(self.channels)==1) )
        
    def get_rates(self,datatraining,datacoding):
        if (self.N > self.max_context):
            return self.epochs*[-1],self.epochs*[-1]
        cabac = self.load_model()        
        train_loss, valid_loss = [], []
        for phase in sorted(self.phases): # train first then valid
            if phase == 'train':
                dataset = CausalContextDataset(datatraining, self.data_type, self.N, self.percentage_of_uncles,
                    geo_or_attr=self.geo_or_attr,n_classes=self.n_classes,channels=self.channels,color_space=self.color_space,
                    manual_th=self.manual_th,full_page=self.full_page)
                cabac.load(X=dataset.X,y=dataset.y)

                X,y = dataset.X,dataset.y
                cabac_pred = cabac(X)
                final_loss = perfect_AC(y,cabac_pred,binary=self.binary)
                train_loss.append(final_loss)
            else:
                # dataset = CausalContextDataset(datacoding, self.data_type, self.N, self.percentage_of_uncles,
                #     geo_or_attr=self.geo_or_attr,n_classes=self.n_classes,channels=self.channels,color_space=self.color_space,
                #     manual_th=self.manual_th,full_page=self.full_page)
                
                pths = datacoding  
                
                pths_per_dset = max(1,len(pths)//self.configs["dset_pieces"])

                running_loss = 0.0
                n_samples = 0.0
                for pths_i in range(0,len(pths),pths_per_dset):

                    dataset = CausalContextDataset(pths[pths_i:(pths_i+pths_per_dset)], self.data_type, self.N, self.percentage_of_uncles,
                        geo_or_attr=self.geo_or_attr,n_classes=self.n_classes,channels=self.channels,color_space=self.color_space,
                        manual_th=self.manual_th,full_page=self.full_page)

                    X,y = dataset.X,dataset.y
                    cabac_pred = cabac(X)
                    running_loss += perfect_AC(y,cabac_pred,binary=self.binary) * y.size
                    n_samples += y.size
                
                final_loss = running_loss / n_samples
                valid_loss.append(final_loss)

        self.save_N_model(cabac)
        return self.epochs*train_loss, self.epochs*valid_loss

    def load_model(self):
        if self.binary:
            cabac = CABAC(self.max_context)
            if self.configs.get("parent_id"):
                file_name = f"{get_prefix(self.configs,'parent_id')}_{self.N:03d}_lut.npy"
                with open(file_name, 'rb') as f:
                    lut = np.load(f)
                cabac.load(lut=lut.reshape(-1,1))
        elif self.n_classes==256:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.load_npz.html#scipy.sparse.load_npz
            cabac = CA256AC()
            if self.configs.get("parent_id"):
                n_channels = np.count_nonzero(self.channels)
                lut = []
                for i in range(n_channels):
                    with open(f"{get_prefix(self.configs,'parent_id')}_{self.N:03d}_lut_ch{i}.npy", 'rb') as f:
                        lut.append( load_npz(f).tolil() )
                cabac.load(lut=lut)
        else:
            raise ValueError(f"We only support 1-channel binary data and N-channel 256-ary data")
        return cabac

    def save_N_model(self,cabac):
        if ('train' in self.configs["phases"]) and (self.N>0):
            if isinstance(cabac,CABAC):
                lut = cabac.context_p.reshape(-1)
                with open(f"{get_prefix(self.configs)}_{self.N:03d}_lut.npy", 'wb') as f:
                    np.save(f, lut)
            elif isinstance(cabac,CA256AC):
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html
                for i,context_c in enumerate(cabac.lut):
                    with open(f"{get_prefix(self.configs)}_{self.N:03d}_lut_ch{i}.npy", 'wb') as f:
                        save_npz(f, context_c.tocsr())
            else:
                ValueError("Unknown object")


class RatesJBIG1:
    def __init__(self,configs,N):
        if not (configs["n_classes"] == 2 and np.count_nonzero(configs["channels"])==1 and configs["data_type"] == "image"):
            raise ValueError("RatesJBIG1 currently supports only binary images")
        self.configs = configs
        self.N = N

    def avg_rate(self,pths):
        """
        make sure all images in pths have the same size
        """
        rate = 0
        for pth in pths:
            rate += jbig1_rate(pth)
        return rate/len(pths)

    def get_rates(self,datatraining,datacoding):
        phases=self.configs["phases"]
        epochs=self.configs["epochs"]
        if (self.N != 10):
            return epochs*[-1],epochs*[-1]
        train_loss, valid_loss = [], []
        for phase in sorted(phases): # train first then valid
            if phase == 'train':
                dataset = JBIG1Dataset(datatraining)
                # dataset = CausalContextDataset(datatraining, self.configs["data_type"], self.N, self.configs["percentage_of_uncles"],
                #                                manual_th=self.configs["manual_th"],full_page=self.configs["full_page"])
            else:
                dataset = JBIG1Dataset(datacoding)
                # dataset = CausalContextDataset(datacoding, self.configs["data_type"], self.N, self.configs["percentage_of_uncles"],
                #                                manual_th=self.configs["manual_th"],full_page=self.configs["full_page"])
            final_loss = self.avg_rate(dataset.pths)
            if phase=='train':
                train_loss.append(final_loss)
            else:
                valid_loss.append(final_loss)
        return epochs*train_loss, epochs*valid_loss
     

class RatesMLP:

    def __init__(self,configs,N):
        self.configs = configs
        self.N = N

    def get_rates(self,datatraining,datacoding,output_n_samples=False):
        
        device = torch.device(self.configs["device"])
        
        model = self.load_model()
        model.to(device)
        
        criterion = self.load_criterion()
        OptimizerClass=self.configs["OptimizerClass"]
        optimizer = OptimizerClass(model.parameters(), lr=self.configs["learning_rate"])

        print("starting training")
        print(f"len trainset : {len(datatraining)} {self.configs['data_type']}s, len validset : {len(datacoding)} {self.configs['data_type']}s")

        phases = self.configs["phases"]

        train_loss, valid_loss, train_samples, valid_samples, epochs =\
            self.initialize_data(model,criterion,optimizer,datatraining,datacoding)

        for epoch in epochs:
            
            for phase in phases:
                final_loss, n_samples = self.train_one_epoch(model,criterion,optimizer,datatraining,datacoding,phase)

                if phase=='train':
                    train_loss.append(final_loss)
                    train_samples.append(n_samples)
                else:
                    valid_loss.append(final_loss)
                    valid_samples.append(n_samples)
                
                print("epoch :" , epoch, ", phase :", phase, ", loss :", final_loss)
            
                save_values(f"{get_prefix(self.configs)}_{self.N:03d}_{phase}_values",np.arange(epoch+1),
                    {"MLP": train_loss if phase=='train' else valid_loss},"epoch")
            self.save_N_model(model)
            self.save_N_min_valid_loss_model(valid_loss,model)

        # save final model
        self.save_N_model(model)
        if output_n_samples:
            return train_loss, valid_loss, train_samples, valid_samples
        else:
            return train_loss, valid_loss

    def train_one_epoch(self,model,criterion,optimizer,datatraining,datacoding,phase):

        device = torch.device(self.configs["device"])

        if phase == 'train':
            model.train(True)
            pths = datatraining
        else:
            model.train(False)
            pths = datacoding 

        running_loss = 0.0
        n_samples = 0.0

        shuffled_pths = random.sample(pths, len(pths))

        pths_per_dset = max(1,len(shuffled_pths)//self.configs["dset_pieces"])

        for shuffled_pths_i in range(0,len(shuffled_pths),pths_per_dset):

            dset = CausalContextDataset(shuffled_pths[shuffled_pths_i:(shuffled_pths_i+pths_per_dset)], 
                self.configs["data_type"], self.N, self.configs["percentage_of_uncles"],geo_or_attr=self.configs["geo_or_attr"],
                n_classes=self.configs["n_classes"],channels=self.configs["channels"],color_space=self.configs["color_space"],
                manual_th=self.configs["manual_th"],full_page=self.configs["full_page"])

            dataloader=torch.utils.data.DataLoader(
                dset,batch_size=self.configs["batch_size"],shuffle=True,num_workers=self.configs["num_workers"])
            pbar = tqdm(total=np.ceil(len(dset)/self.configs["batch_size"]))
            for data in dataloader:

                Xt_b,yt_b= data
                Xt_b = Xt_b.float().to(device)
                yt_b = yt_b.float().to(device)

                Xt_b = Xt_b/(self.configs["n_classes"]-1) # input values in the range [0,1]

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
                n_samples += yt_b.numel()
                pbar.update(1)
            pbar.close()
            
        final_loss = running_loss / n_samples

        return final_loss, n_samples

    def min_valid_loss_model_name(self, id_key = "id", parent_id_index = None):
        prefix = get_prefix(self.configs,id_key=id_key,parent_id_index=parent_id_index)
        return f"{prefix}_{self.N:03d}_min_valid_loss_model.pt"

    def last_train_loss_model_name(self, id_key = "id", parent_id_index = None):
        prefix = get_prefix(self.configs,id_key=id_key,parent_id_index=parent_id_index)
        return f"{prefix}_{self.N:03d}_model.pt"

    def instantiate_model(self):
        ModelClass=self.configs["ModelClass"]
        return ModelClass(self.N)

    def load_criterion(self):
        if self.configs["n_classes"] == 2:
            criterion = Log2BCELoss(reduction='sum')
        else: # "gray" or "rgb"
            criterion = Log2CrossEntropyLoss(reduction='sum')
        return criterion

    def get_available_models(self):
        return self.__get_available_models()

    def __get_available_models(self, use_min=None):

        if use_min is None:
            use_min = ('train' not in self.configs["phases"]) and (self.configs["reduction"] == 'min')

        parent_ids = []
        if self.configs.get("parent_id"):
            parent_ids = self.configs["parent_id"]
            parent_ids = parent_ids if isinstance(parent_ids,list) else [parent_ids]
        
        available_models = []
        for i,_ in enumerate(parent_ids):
            file_name = self.min_valid_loss_model_name(id_key='parent_id',parent_id_index=i) if use_min else \
                self.last_train_loss_model_name(id_key='parent_id',parent_id_index=i)
            if os.path.isfile(file_name):
                available_models.append(file_name)
        return available_models

    def load_model(self):
        return self.__load_model()


    def __load_model(self, use_min=None):
        """
        Loads the initial model. 
        If no parent_id is specified in the configs dict, the model is randomly initialized.
        If a parent_id is specified and configs=={...,"phases":["valid"],"reduction":"min",...}
        then the initial model is the previous model with minimum validation loss.
        If a parent_id is specified and configs=={...,"phases":["train",...],...}
        then the initial model is the previous model trained for the most number of epochs.
        If there are multiple models available for the same N (context size),
        an exception is raised.
        """
        model = self.instantiate_model()

        available_models = self.__get_available_models(use_min=use_min)

        if len(available_models) >= 1:
            file_name = available_models[0]
            print(f"loading file {file_name}")
            model.load_state_dict(torch.load(file_name))
        return model


    def get_available_data(self,phase):

        parent_ids = []
        if self.configs.get("parent_id"):
            parent_ids = self.configs["parent_id"]
            parent_ids = parent_ids if isinstance(parent_ids,list) else [parent_ids]
        
        available_data = []
        for i,_ in enumerate(parent_ids):
            prefix = get_prefix(self.configs,id_key='parent_id',parent_id_index=i)
            file_name = f"{prefix}_{self.N:03d}_{phase}_values.csv"
            if os.path.isfile(file_name):
                available_data.append(file_name)
        return available_data


    def initialize_data(self,model,criterion,optimizer,datatraining,datacoding):

        train_loss, valid_loss = [], []
        train_samples, valid_samples = [], []

        epochs = range(self.configs["epochs"])

        if self.configs.get("unfinished_training"): # 1 or 0
            for phase in ["train","valid"]:

                device = torch.device(self.configs["device"])
                model = self.__load_model(use_min=(phase=="valid"))
                model.to(device)

                final_loss, n_samples = self.train_one_epoch(
                    model,criterion,optimizer,datatraining,(datacoding if phase == "valid" else datatraining),"valid")
                if phase == "valid":
                    self.save_N_min_valid_loss_model([final_loss],model)
                else:
                    self.save_N_model(model)
                
                available_data = self.get_available_data(phase)
                if len(available_data) >= 1:
                    file_name = available_data[0]
                    if phase == "train":
                        train_loss = load_values(file_name)["MLP"]
                        if np.allclose(final_loss,train_loss[-1],atol=1e-04):
                            pass
                        elif np.allclose(final_loss,train_loss[-2],atol=1e-04):
                            train_loss = train_loss[:-1]
                        else:
                            # raise Exception(
                            warnings.warn(
                                f"""Recalculated train loss and it did not match the last two values provided.
                                {final_loss} != {train_loss[-2]} and {train_loss[-1]}.
                                This might be because the previous values were calculated during SGD.""")
                        train_samples = len(train_loss) * [n_samples]
                    else:
                        valid_loss = load_values(file_name)["MLP"]
                        if not any([np.allclose(final_loss,v,atol=1e-04) for v in valid_loss]):
                            # raise Exception(
                            warnings.warn(
                                f"""Recalculated valid loss and it did not match any previous value provided.
                                {final_loss} != {np.min(valid_loss)}.
                                This might be because the previous values were calculated during SGD.""")
                        valid_samples = len(valid_loss) * [n_samples]

            assert abs(len(train_loss)-len(valid_loss)) <= 1
            if len(valid_loss) < len(train_loss):
                valid_loss.append(valid_loss[-1])
            elif len(valid_loss) > len(train_loss):
                valid_loss = valid_loss[:-1]
            epochs = range(len(train_loss),self.configs["epochs"])

        return train_loss, valid_loss, train_samples, valid_samples, epochs
        


    def save_N_min_valid_loss_model(self,valid_loss,mlp_model):
        if len(valid_loss) == 0:
            pass
        elif (min(valid_loss) == valid_loss[-1]) and ('train' in self.configs["phases"]) and (self.N>0):
            save_model(self.min_valid_loss_model_name(),mlp_model)

    def save_N_model(self,mlp_model):
        if ('train' in self.configs["phases"]) and (self.N>0):
            save_model(self.last_train_loss_model_name(),mlp_model)


class RatesArbitraryMLP(RatesMLP):

    def __init__(self,configs,widths):
        super().__init__(configs,widths[0])
        self.widths = widths

    def min_valid_loss_model_name(self, id_key = "id", parent_id_index = None):
        prefix = get_prefix(self.configs,id_key=id_key,parent_id_index=parent_id_index)
        return f"{prefix}_{'_'.join(map(str,self.widths))}_min_valid_loss_model.pt"

    def last_train_loss_model_name(self, id_key = "id", parent_id_index = None):
        prefix = get_prefix(self.configs,id_key=id_key,parent_id_index=parent_id_index)
        return f"{prefix}_{'_'.join(map(str,self.widths))}_model.pt"

    def instantiate_model(self):
        return ArbitraryMLP(self.widths)


def train_loop(configs,datatraining,datacoding,N):
    
    phases=configs["phases"]
    epochs=configs["epochs"]

    methods=configs["methods"]

    data = dict()
    for phase in phases:
        data[phase] = dict()

    static_condition = (N == 0) and (("LUT" in methods) or ("MLP" in methods))
    cabac_condition = (N > 0) and ("LUT" in methods)
    mlp_condition = (N > 0) and ("MLP" in methods)
    jbig1_condition = (configs["data_type"] == "image") and (configs["n_classes"] == 2) and \
        (np.count_nonzero(configs["channels"])==1) and ("JBIG1" in methods)

    rates_empty_t,rates_empty_c= epochs*[-1],epochs*[-1]
    if static_condition:
        rates_static_t,rates_static_c = RatesStaticAC(configs,N).get_rates(datatraining,datacoding)
    if mlp_condition:
        rates_mlp_t,rates_mlp_c = RatesMLP(configs,N).get_rates(datatraining,datacoding)
    if cabac_condition:
        rates_cabac_t,rates_cabac_c = RatesCAAC(configs,N).get_rates(datatraining,datacoding)
    if jbig1_condition:
        rates_jbig1_t,rates_jbig1_c = RatesJBIG1(configs,N).get_rates(datatraining,datacoding)

    for phase in phases:
        data[phase]["MLP"] = (rates_static_t if static_condition else (rates_mlp_t if mlp_condition else rates_empty_t)) \
            if phase == 'train' else (rates_static_c if static_condition else (rates_mlp_c if mlp_condition else rates_empty_c))
        data[phase]["LUT"] = (rates_static_t if static_condition else (rates_cabac_t if cabac_condition else rates_empty_t)) \
            if phase == 'train' else (rates_static_c if static_condition else (rates_cabac_c if cabac_condition else rates_empty_c))
        data[phase]["JBIG1"] = (rates_jbig1_t if jbig1_condition else rates_empty_t) \
            if phase == 'train' else (rates_jbig1_c if jbig1_condition else rates_empty_c)

    return data


def coding_loop(configs,N):

    cond1 = (configs["data_type"] == "pointcloud")
    cond2 = (configs['ModelClass'] == MLP_N_64N_32N_1)
    cond3 = (configs["geo_or_attr"] == "geometry")
    ok = cond1 and cond2 and cond3
        
    if not ok:
        m = f"""
            coding currently supported only for the combination
            data_type : pointcloud
            geo_or_attr : geometry
            ModelClass : MLP_N_64N_32N_1
            """
        raise ValueError(m)

    if N == 0:

        staticac_rates = []
        for pc_in in configs["validation_set"]:
            pc_len = len(c3d.read_PC(pc_in)[1])
            weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_p.npy"
            constructor = StaticAC_Constructor(weights)
            coder = PC_Coder(constructor.construct,N,configs["last_octree_level"],percentage_of_uncles=configs["percentage_of_uncles"])
            coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
            staticac_rate = get_bpov("/tmp/encoder_out",pc_len)
            staticac_rates.append(staticac_rate)

            save_values(f"{get_prefix(configs)}_{N:03d}_coding_values",np.arange(len(staticac_rates)),
                {"MLP": staticac_rates,"LUT": staticac_rates},"frame",None)

    else:

        mlp_rates = []
        cabac_rates = []
        for pc_in in configs["validation_set"]:
            pc_len = len(c3d.read_PC(pc_in)[1])
            if (configs["reduction"] == 'min'):
                weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_min_valid_loss_model.pt"
            else:
                weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_model.pt"
            constructor = MLP_N_64N_32N_1_Constructor(N,weights)
            coder = PC_Coder(constructor.construct,N,configs["last_octree_level"],
                percentage_of_uncles=configs["percentage_of_uncles"], device=configs["device"])
            coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
            mlp_rate = get_bpov("/tmp/encoder_out",pc_len)
            mlp_rates.append(mlp_rate)

            if N <= configs["max_context"] :

                pc_len = len(c3d.read_PC(pc_in)[1])
                weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_lut.npy"
                constructor = CABAC_Constructor(weights,configs["max_context"])
                coder = PC_Coder(constructor.construct,N,configs["last_octree_level"],percentage_of_uncles=configs["percentage_of_uncles"])
                coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
                cabac_rate = get_bpov("/tmp/encoder_out",pc_len)
                cabac_rates.append(cabac_rate)

            else:

                cabac_rate = -1
                cabac_rates.append(cabac_rate)

            save_values(f"{get_prefix(configs)}_{N:03d}_coding_values",np.arange(len(mlp_rates)),
                {"MLP": mlp_rates,"LUT": cabac_rates},"frame",None)

    if N == 0:
        data = {
            "coding": {
                "MLP": staticac_rates,
                "LUT": staticac_rates
            }   
        }
    else:
        data = {
            "coding": {
                "MLP": mlp_rates,
                "LUT": cabac_rates
            }   
        }

    return data
    

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
                    k:([ph for ph in v if ph != "coding"] if k == 'phases' else v) for k,v in configs.items()
                },
                datatraining=configs["training_set"],
                datacoding=configs["validation_set"],
                N=N
            )
            
            for phase in [ph for ph in configs["phases"] if ph != "coding"]:
                save_data(f"{get_prefix(configs)}_{N:03d}_{phase}",np.arange(configs["epochs"]),N_data[phase],"epoch")

            for phase in [ph for ph in configs["phases"] if ph != "coding"]:
                for k in N_data[phase].keys():
                    v = min(N_data[phase][k]) if (configs['reduction'] == 'min') else N_data[phase][k][-1]
                    data[phase][k] = (data[phase][k] + [v]) if (k in data[phase].keys()) else [v]

    if ("coding" in configs["phases"]):
        for N in configs["N_vec"]:
            N_data = coding_loop(configs,N)

            save_data(f"{get_prefix(configs)}_{N:03d}_coding",np.arange(len(configs["validation_set"])),N_data["coding"],"frame")        

            for k in N_data["coding"].keys():
                v = np.mean(N_data["coding"][k])
                data["coding"][k] = (data["coding"][k] + [v]) if (k in data["coding"].keys()) else [v]

    save_configs(f"{get_prefix(configs)}_conf",configs)

    for phase in configs["phases"]:
        ylabel = 'bpov' if ((phase == "coding") and (configs["data_type"] == "pointcloud")) else 'bits/sample'
        save_data(f"{get_prefix(configs)}_{phase}",configs["N_vec"],data[phase],"context size",ylabel,configs["xscale"])


class MLPTopologyCalculator:
    def __init__(self,inputs,outputs,hidden_layers_proportions):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers_proportions = hidden_layers_proportions

    def mlp_closest_to_n_params(self,n_parameters):
        base_width,params = self.find_best_width_for_this_number_of_parameters(n_parameters)
        widths = self.mlp_for_base_width(base_width)
        return widths,params

    def find_best_width_for_this_number_of_parameters(self,number_of_parameters):
        minimum_number_of_parameters = self.mlp_parameters_for_base_width(1)
        assert number_of_parameters > minimum_number_of_parameters
        chosen_number_of_parameters = minimum_number_of_parameters
        chosen_width = 1
        while chosen_number_of_parameters < number_of_parameters:
            previous_chosen_width = chosen_width
            previous_chosen_number_of_parameters = chosen_number_of_parameters
            chosen_width += 1
            chosen_number_of_parameters = self.mlp_parameters_for_base_width(chosen_width)
        if number_of_parameters - previous_chosen_number_of_parameters < chosen_number_of_parameters - number_of_parameters:
            chosen_width = previous_chosen_width
            chosen_number_of_parameters = previous_chosen_number_of_parameters
        return chosen_width, chosen_number_of_parameters
    
    def mlp_for_base_width(self,base_width):
        widths = [self.inputs] + list(map(lambda mf : base_width * mf , self.hidden_layers_proportions)) + [self.outputs]
        return widths

    def mlp_parameters_for_base_width(self,base_width):
        widths = self.mlp_for_base_width(base_width)
        return self.mlp_parameters(widths)

    @staticmethod
    def mlp_parameters(widths):
        p=0
        for i in range(1,len(widths)):
            p += widths[i] * ( widths[i-1] + 1 )
        return p


def rate_vs_complexity_experiment(configs):

    os.makedirs(f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}")

    # data = dict()
    # for phase in configs["phases"]:
    #     data[phase] = dict()

    mlp_data = dict()
    actual_params = []
    for widths in configs["topologies"]:

        params = MLPTopologyCalculator.mlp_parameters(widths)
        actual_params.append(params)

        rates_mlp_t,rates_mlp_c = RatesArbitraryMLP(configs,widths).get_rates(configs["training_set"],configs["validation_set"]) # .get_rates(trainset,validset)

        for phase in configs["phases"]:
            rates_mlp = rates_mlp_t if phase == 'train' else rates_mlp_c
            save_data(
                f"{get_prefix(configs)}_{'_'.join(map(str,widths))}_{phase}",np.arange(configs["epochs"]),{"MLP":rates_mlp},"epoch"
                # ,linestyles={"MLP":"None"}, colors={"MLP":"k"}, markers={"MLP":"x"}
            )
            v = min(rates_mlp) if (configs['reduction'] == 'min') else rates_mlp[-1]
            # data[phase]["MLP"] = (data[phase]["MLP"] + [v]) if ("MLP" in data[phase].keys()) else [v]
            mlp_data[phase] = (mlp_data[phase] + [v]) if (phase in mlp_data.keys()) else [v]

    save_configs(f"{get_prefix(configs)}_conf",configs)

    # for phase in configs["phases"]:
    #     save_data(f"{get_prefix(configs)}_{phase}",actual_params,data[phase],"complexity",xscale=configs["xscale"],
    #         extra={"topology": ['_'.join(map(str,widths)) for widths in configs["topologies"]] },
    #         linestyles={"MLP":"None"}, colors={"MLP":"k"}, markers={"MLP":"x"})

    for phase in configs["phases"]:
        data0=pd.DataFrame({
            "complexity": actual_params,
            "topology": ['_'.join(map(str,widths)) for widths in configs["topologies"]],
            "bits/sample": mlp_data[phase]
        })

        save_dataframe(f"{get_prefix(configs)}_{phase}_values",data0,"complexity","bits/sample")



# take pre-calculated weights, load, quantize with different numbers of bits, recalculate the train and validation rates
# calculate the rate to compress the network, calculate the total number of bits needed to compress the network and the 
# training or the validation data. Create a class that when loading also quantize the network with a given number of bits.

from perceptronac.mlp_quantization import estimate_midtread_uniform_quantization_delta
from perceptronac.mlp_quantization import midtread_uniform_quantization
from perceptronac.mlp_quantization import encode_network_integer_symbols

class RatesQuantizedArbitraryMLP(RatesArbitraryMLP):

    def __init__(self,configs,widths,quantization_bits):
        super().__init__(configs,widths)
        self.params = MLPTopologyCalculator.mlp_parameters(widths)
        self.quantization_bits = quantization_bits
        assert configs["phases"] == ["valid"]
        assert len(self.get_available_models()) > 0
        self.pre_load_model()

    def pre_load_model(self):
        self.model,self.bits,self.samples = self.quantize_model()

    def quantize_model(self):
        model = super().load_model()
        # if f"torch.float{int(self.quantization_bits)}" == str(model.layers[0].weight.data.dtype):
        if self.quantization_bits == 32:
            bits = self.params * self.quantization_bits
            samples = self.params
        else:
            Delta,shift = estimate_midtread_uniform_quantization_delta(model,self.quantization_bits)
            model,all_parameters = midtread_uniform_quantization(model,Delta,shift)
            bits,samples = encode_network_integer_symbols(all_parameters)
        return model,bits,samples

    def load_model(self):
        # model = self.quantize_model()[0]
        model = self.model
        return model 

    def quantization_info(self):
        # _,bits,samples = self.quantize_model()
        bits = self.bits
        samples = self.samples
        return bits,samples

# next step : create an experiment similar to the rate_vs_complexity_experiment , but instead of plotting rate vs complexity,
# it plots number of bits to encode the data and the model, or (data_bits + model_bits)/data_samples, AND maybe "distortion",
# measured as the decrease in the rate to encode the data, or data_bits / data_samples. Then it is rate-including-the-model
# vs rate-for-just-the-data. Or rate-data-model vs rate-data. And I can include the points for the original networks to compare.

# When the script begins, start a background process using os. This background process should use nvidia-smi to store the 
# power consumption and time of measurement. For each network, store the metadata : start time, end time . Then, in a 
# post processing script, check the power consumption measurements during the time when the network was running. Given
# the power consumption values, and the duration of the network processing, calculate the energy consumption in joules.

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
# https://stackoverflow.com/questions/45429831/valueerror-a-value-in-x-new-is-above-the-interpolation-range-what-other-re


from perceptronac.data_exportation import save_dataframe
from perceptronac.power_consumption import estimate_joules


def quantization_and_power_experiment(configs):

    p = subprocess.Popen([
        os.path.join(os.path.dirname(os.path.abspath(__file__)),"power_consumption.sh"),
        f"{get_prefix(configs)}_power_draw.csv"
    ])

    try:

        os.makedirs(f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}")

        assert configs["phases"] == ["valid"]
        assert configs["epochs"] == 1

        data_bits_plus_model_bits_per_sample_vec = []
        model_bits_per_sample_vec = []
        data_bits_per_sample_vec = []
        params_vec = []
        qbits_vec = []
        topology_vec = []
        start_time_vec = []
        end_time_vec = []
        data_samples_vec = []
        energy_measurement_iteration = []
        for e_iter in range(configs["energy_measurement_iteration"]):

            wq_comb = [(w,q) for w in configs["topologies"] for q in configs["qbits_vec"]]

            for widths,qbits in random.sample(wq_comb, len(wq_comb)):

                params = MLPTopologyCalculator.mlp_parameters(widths)
                params_vec.append(params)
                qbits_vec.append(qbits)
                topology_vec.append('_'.join(map(lambda x: f"{x:03d}",widths)))

                quantizedMLP = RatesQuantizedArbitraryMLP(configs,widths,qbits)
                start_time_vec.append(time.time())
                quantized_mlp_results = quantizedMLP.get_rates(configs["training_set"],configs["validation_set"],output_n_samples=True)
                end_time_vec.append(time.time())
                data_rate = quantized_mlp_results[1][0]
                data_samples = quantized_mlp_results[3][0]
                model_bits,model_samples = quantizedMLP.quantization_info()
                data_bits = data_rate * data_samples

                data_bits_plus_model_bits_per_sample = (data_bits + model_bits)/data_samples

                model_bits_per_sample = model_bits/data_samples

                model_bits_per_sample_vec.append(model_bits_per_sample)
                data_bits_plus_model_bits_per_sample_vec.append(data_bits_plus_model_bits_per_sample)
                data_bits_per_sample_vec.append(data_rate)

                data_samples_vec.append(data_samples)

                energy_measurement_iteration.append(e_iter)


        save_configs(f"{get_prefix(configs)}_conf",configs)


        data=pd.DataFrame({
            "data_bits/data_samples":data_bits_per_sample_vec,
            "(data_bits+model_bits)/data_samples":data_bits_plus_model_bits_per_sample_vec,
            "model_bits/data_samples":model_bits_per_sample_vec,
            "data_samples":data_samples_vec,
            "topology": topology_vec, "params":params_vec,"quantization_bits":qbits_vec,
            "start_time":start_time_vec,"end_time":end_time_vec,
            "energy_measurement_iteration": energy_measurement_iteration
        }).sort_values(["energy_measurement_iteration","topology","quantization_bits"])

        save_dataframe(f"{get_prefix(configs)}_raw_values",data,
            "topology","data_bits/data_samples")

        model_bits_x_data_bits_data = data.loc[data["energy_measurement_iteration"]==0,:].drop(
            ["start_time", "end_time", "energy_measurement_iteration"], axis=1)

        save_dataframe(f"{get_prefix(configs)}_model_bits_x_data_bits_values",model_bits_x_data_bits_data,
            "model_bits/data_samples","data_bits/data_samples",sort_by_x_col=True)

        params_x_rate_data = data.loc[
            np.logical_and(data["energy_measurement_iteration"]==0,data["quantization_bits"]==32)].drop(
            ["start_time", "end_time", "energy_measurement_iteration","model_bits/data_samples",
            "(data_bits+model_bits)/data_samples", "quantization_bits"], axis=1)

        save_dataframe(f"{get_prefix(configs)}_params_x_rate_values",params_x_rate_data,
            "params","data_bits/data_samples",sort_by_x_col=True)

        p.kill()

        power_draw = np.loadtxt(f"{get_prefix(configs)}_power_draw.csv")

        joules = estimate_joules(data,power_draw)

        data["joules"] = joules

        data = data.groupby(["topology","quantization_bits"]).apply(
            lambda x: pd.Series({
                "data_bits/data_samples":x["data_bits/data_samples"].iloc[0],
                "(data_bits+model_bits)/data_samples":x["(data_bits+model_bits)/data_samples"].iloc[0],
                "model_bits/data_samples":x["model_bits/data_samples"].iloc[0],
                "data_samples":x["data_samples"].iloc[0],
                "topology": x["topology"].iloc[0], 
                "params": x["params"].iloc[0],
                "quantization_bits": x["quantization_bits"].iloc[0],
                "joules": x["joules"].mean(),
                "joules_std": x["joules"].std()
            },index=[
                "data_bits/data_samples",
                "(data_bits+model_bits)/data_samples",
                "model_bits/data_samples",
                "data_samples",
                "topology",
                "params",
                "quantization_bits",
                "joules",
                "joules_std"
            ]))

        static_data = data.loc[data["quantization_bits"]==32,:].drop(
            ["model_bits/data_samples","(data_bits+model_bits)/data_samples", "quantization_bits"], axis=1)

        save_dataframe(f"{get_prefix(configs)}_static_rate_x_power_values",static_data,
            "data_bits/data_samples","joules",sort_by_x_col=True)

        save_dataframe(f"{get_prefix(configs)}_semi_adaptive_rate_x_power_values",data,
            "(data_bits+model_bits)/data_samples","joules",sort_by_x_col=True)

    finally:
        # https://stackoverflow.com/questions/43274476/
        # https://github.com/python/cpython/blob/main/Lib/subprocess.py
        if p.poll() is None:
            p.kill()