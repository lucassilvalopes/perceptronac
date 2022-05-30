import torch
from perceptronac.perfect_AC import perfect_AC
from perceptronac.utils import jbig1_rate
from perceptronac.loading_and_saving import save_model
from perceptronac.loading_and_saving import save_fig
from perceptronac.loading_and_saving import save_values
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_configs
from perceptronac.loading_and_saving import save_data
from perceptronac.models import Log2BCELoss
from perceptronac.models import CABAC
from perceptronac.models import StaticAC
from perceptronac.models import CausalContextDataset
import numpy as np
from tqdm import tqdm
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.models import MLP_N_64N_32N_1_Constructor
from perceptronac.models import ArbitraryWidthMLP
from perceptronac.models import ArbitraryMLP
from perceptronac.models import CABAC_Constructor
from perceptronac.models import StaticAC_Constructor
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

    def __init__(self,configs,N):
        self.configs = configs
        self.N = N

    def get_rates(self,trainset,validset):

        OptimizerClass=self.configs["OptimizerClass"]
        epochs=self.configs["epochs"]
        learning_rate=self.configs["learning_rate"]
        batch_size=self.configs["batch_size"]
        num_workers=self.configs["num_workers"]
        device=self.configs["device"]
        phases=self.configs["phases"]
        
        device = torch.device(device)
        
        model = self.load_model()
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
                

            self.save_N_min_valid_loss_model(valid_loss,model)
 
        # save final model
        self.save_N_model(model)

        return train_loss, valid_loss


    def min_valid_loss_model_name(self, id_key = "id"):
        return f"{get_prefix(self.configs,id_key=id_key)}_{self.N:03d}_min_valid_loss_model.pt"

    def last_train_loss_model_name(self, id_key = "id"):
        return f"{get_prefix(self.configs,id_key=id_key)}_{self.N:03d}_model.pt"

    def instantiate_model(self):
        ModelClass=self.configs["ModelClass"]
        return ModelClass(self.N)

    def load_model(self):
        model = self.instantiate_model()
        if self.configs.get("parent_id"):
            if ('train' not in self.configs["phases"]) and (self.configs["reduction"] == 'min'):
                file_name = self.min_valid_loss_model_name(id_key='parent_id')
            else:
                file_name = self.last_train_loss_model_name(id_key='parent_id')
            print(f"loading file {file_name}")
            model.load_state_dict(torch.load(file_name))
        return model

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

    def min_valid_loss_model_name(self, id_key = "id"):
        return f"{get_prefix(self.configs,id_key=id_key)}_{'_'.join(map(str,self.widths))}_min_valid_loss_model.pt"

    def last_train_loss_model_name(self, id_key = "id"):
        return f"{get_prefix(self.configs,id_key=id_key)}_{'_'.join(map(str,self.widths))}_model.pt"

    def instantiate_model(self):
        return ArbitraryMLP(self.widths)


def train_loop(configs,datatraining,datacoding,N):
    
    trainset = CausalContextDataset(
        datatraining,configs["data_type"],N, configs["percentage_of_uncles"],getXy_later=('train' not in configs["phases"]))
    validset = CausalContextDataset(
        datacoding,configs["data_type"],N, configs["percentage_of_uncles"],getXy_later=('valid' not in configs["phases"]))

    if N == 0:
        rates_static_t,rates_static_c = RatesStaticAC(configs).get_rates(trainset,validset)
    else:
        rates_cabac_t,rates_cabac_c = RatesCABAC(configs).get_rates(trainset,validset)        
        rates_mlp_t,rates_mlp_c = RatesMLP(configs,N).get_rates(trainset,validset)
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

    pc_in = configs["validation_set"][0]
    pc_len = len(c3d.read_PC(pc_in)[1])

    if N == 0:
        weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_p.npy"
        constructor = StaticAC_Constructor(weights)
        coder = PC_Coder(constructor.construct,N,configs["last_octree_level"])
        coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
        staticac_rate = get_bpov("/tmp/encoder_out",pc_len)

    else:
        if (configs["reduction"] == 'min'):
            weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_min_valid_loss_model.pt"
        else:
            weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_model.pt"
        constructor = MLP_N_64N_32N_1_Constructor(N,weights)
        coder = PC_Coder(constructor.construct,N,configs["last_octree_level"])
        coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
        mlp_rate = get_bpov("/tmp/encoder_out",pc_len)

        if N <= configs["max_context"] :
            weights = f"{get_prefix(configs,'parent_id')}_{N:03d}_lut.npy"
            constructor = CABAC_Constructor(weights,configs["max_context"])
            coder = PC_Coder(constructor.construct,N,configs["last_octree_level"])
            coder.encode(pc_in,encoder_in="/tmp/encoder_in",encoder_out="/tmp/encoder_out")
            cabac_rate = get_bpov("/tmp/encoder_out",pc_len)
        else:
            cabac_rate = -1

    if N == 0:
        data = {
            "coding": {
                "MLP": staticac_rate,
                "LUT": staticac_rate
            }   
        }
    else:
        data = {
            "coding": {
                "MLP": mlp_rate,
                "LUT": cabac_rate
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
            for k in N_data["coding"].keys():
                v = N_data["coding"][k]
                data["coding"][k] = (data["coding"][k] + [v]) if (k in data["coding"].keys()) else [v]

    save_configs(f"{get_prefix(configs)}_conf",configs)

    for phase in configs["phases"]:
        ylabel = 'bpov' if ((phase == "coding") and (configs["data_type"] == "pointcloud")) else 'bits/sample'
        save_data(f"{get_prefix(configs)}_{phase}",configs["N_vec"],data[phase],"context size",ylabel,configs["xscale"])


class FixedWidthMLPTopology:
    def __init__(self,name,linestyle,color,marker,inputs,outputs,n_hidden_layers):
        self.name = name # "OneHiddenLayerMLP"
        self.linestyle = linestyle # "None"
        self.color = color #"b"
        self.marker = marker #"o"
        self.inputs = inputs
        self.outputs = outputs
        self.n_hidden_layers = n_hidden_layers # 1

    def mlp_closest_to_n_params(self,n_parameters):
        parameters_for_width_function = lambda inputs,width,outputs: self.fixed_width_mlp_parameters(inputs,width,outputs,self.n_hidden_layers)
        width,params = self.find_best_width_for_this_number_of_parameters(self.inputs,self.outputs,n_parameters,parameters_for_width_function)
        widths = [self.inputs] + self.n_hidden_layers * [width] + [self.outputs]
        return widths,params

    @staticmethod
    def find_best_width_for_this_number_of_parameters(inputs,outputs,number_of_parameters,parameters_for_width_function):
        minimum_number_of_parameters = parameters_for_width_function(inputs,1,outputs)
        assert number_of_parameters > minimum_number_of_parameters
        chosen_number_of_parameters = minimum_number_of_parameters
        chosen_width = 1
        while chosen_number_of_parameters < number_of_parameters:
            previous_chosen_width = chosen_width
            previous_chosen_number_of_parameters = chosen_number_of_parameters
            chosen_width += 1
            chosen_number_of_parameters = parameters_for_width_function(inputs,chosen_width,outputs)
        if number_of_parameters - previous_chosen_number_of_parameters < chosen_number_of_parameters - number_of_parameters:
            chosen_width = previous_chosen_width
            chosen_number_of_parameters = previous_chosen_number_of_parameters
        return chosen_width, chosen_number_of_parameters
    
    @staticmethod
    def fixed_width_mlp_parameters(inputs,width,outputs,n_hidden_layers):
        return width * (inputs + 1) + (n_hidden_layers - 1) * width * (width + 1) + outputs * (width + 1)


class FW1HLMLPTopology(FixedWidthMLPTopology):
    """fixed width 1 hidden layer multi layer perceptron topology"""
    def __init__(self,inputs,outputs):
        super().__init__("FW1HLMLP","None","b","o",inputs,outputs,1)

class FW2HLMLPTopology(FixedWidthMLPTopology):
    """fixed width 2 hidden layer multi layer perceptron topology"""
    def __init__(self,inputs,outputs):
        super().__init__("FW2HLMLP","None","c","s",inputs,outputs,2)

class FW3HLMLPTopology(FixedWidthMLPTopology):
    """fixed width 3 hidden layer multi layer perceptron topology"""
    def __init__(self,inputs,outputs):
        super().__init__("FW3HLMLP","None","m","*",inputs,outputs,3)


def rate_vs_complexity_experiment(configs):

    os.makedirs(f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}")

    trainset = CausalContextDataset(
        configs["training_set"],"image",configs["N"], percentage_of_uncles=None,getXy_later=('train' not in configs["phases"]))
    validset = CausalContextDataset(
        configs["validation_set"],"image",configs["N"], percentage_of_uncles=None,getXy_later=('valid' not in configs["phases"]))

    data = dict()
    for phase in configs["phases"]:
        data[phase] = dict()

    topologies = [ t(configs["N"],1) for t in configs["topologies"] ]

    linestyles={t.name:t.linestyle for t in topologies}
    colors={t.name:t.color for t in topologies}
    markers={t.name:t.marker for t in topologies}

    actual_params = dict()
    for topology in topologies:
        actual_params[topology.name] = []
        for P in configs["P_vec"]:
            widths,params = topology.mlp_closest_to_n_params(P)
            actual_params[topology.name].append(params)
            rates_mlp_t,rates_mlp_c = RatesArbitraryMLP(configs,widths).get_rates(trainset,validset)

            for phase in configs["phases"]:
                rates_mlp = rates_mlp_t if phase == 'train' else rates_mlp_c
                save_data(f"{get_prefix(configs)}_{'_'.join(map(str,widths))}_{phase}",np.arange(configs["epochs"]),{topology.name:rates_mlp},"epoch",
                    linestyles=linestyles, colors=colors, markers=markers)
                v = min(rates_mlp) if (configs['reduction'] == 'min') else rates_mlp[-1]
                data[phase][topology.name] = (data[phase][topology.name] + [v]) if (topology.name in data[phase].keys()) else [v]

    save_configs(f"{get_prefix(configs)}_conf",configs)

    for phase in configs["phases"]:
        save_data(f"{get_prefix(configs)}_{phase}",actual_params,data[phase],"complexity",xscale=configs["xscale"],
            linestyles=linestyles, colors=colors, markers=markers)
