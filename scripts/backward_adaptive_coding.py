
# Lucas, do you have any plots of convergence (loss as a function of iteration number)?  
# If we make the batch size = 1, and plot the cross-entropy loss after each step k 
# (i.e., the code length for batch k = - y[k]log(q[k]) - (1-y[k])log(1-q[k]) where 
# q[k] is the output of the perceptron at step k), then this is exactly the code length 
# for batch k if we did backward-adaptive coding, like in CABAC.  We can also plot the 
# code length for CABAC after each step k in the same way.  (Or better, plot the average 
# code lengths up to step k, for both methods.  We should see this converge as O((log n)/n).)  
# I think this would greatly strengthen our story, as we could claim a drop-in replacement 
# for CABAC, with no pre-training required.  A related plot is Fig. 3 in your March 30 
# write-up, which shows the average bitrate after longer and longer training.

# The model gives you a prediction q, which gives you an error C
# analytically you have dC/dq and you evaluate at the current q
# backpropagation gives you dC/dw and dC/db for all w and b
# you walk one step in the direction opposite to dC/d* to minimize C
# if you use batch size of 1 you are updating in real time
# the learning rate will affect the convergence

import os
import math
import torch
import numpy as np
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.utils import causal_context_many_imgs
from perceptronac.perfect_AC import perfect_AC
from perceptronac.models import Log2BCELoss, CausalContextDataset
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_values
from tqdm import tqdm


class RNG:
    """
    16 bit taps at [16,15,13,4] Fibonacci LFSR

    https://stackoverflow.com/questions/7602919/how-do-i-generate-random-numbers-without-rand-function
    https://en.wikipedia.org/wiki/Linear-feedback_shift_register
    """

    def __init__(self):
        self.start_state = 1 << 15 | 1
        self.lfsr = self.start_state

    def rand(self):

        #taps: 16 15 13 4; feedback polynomial: x^16 + x^15 + x^13 + x^4 + 1
        bit = (self.lfsr ^ (self.lfsr >> 1) ^ (self.lfsr >> 3) ^ (self.lfsr >> 12)) & 1
        self.lfsr = (self.lfsr >> 1) | (bit << 15)
        output = self.lfsr / 65535.0
        if (self.lfsr == self.start_state):
            self.start_state = (self.start_state + 1) & 65535
            self.lfsr = self.start_state
        return output


def rand_vec(rng,lb,ub,length):
    vec = []
    for _ in range(length):
        vec.append((ub - lb) * rng.rand() + lb)
    return vec


def randperm(rng,n):
    """
    Sattolo's algorithm.

    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    """
    items = list(range(n))
    i = n
    while i > 1:
        i = i - 1
        j = round((i-1) * rng.rand())  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]
    return items


def initialize_MLP_N_64N_32N_1(model):
    """
    https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156/2
    https://discuss.pytorch.org/t/access-weights-of-a-specific-module-in-nn-sequential/3627
    https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
    https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    https://pytorch.org/docs/stable/nn.init.html
    """
    rng = RNG()
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            fan_in = model.layers[i].weight.size(1)
            fan_out = model.layers[i].weight.size(0)
            stdv = 1. / math.sqrt(fan_in)
            with torch.no_grad():
                model.layers[i].weight.data = \
                    torch.linspace(-stdv,stdv,fan_in*fan_out)[randperm(rng,fan_in*fan_out)].reshape(fan_out,fan_in)
                    # torch.tensor(rand_vec(rng,-stdv,stdv,fan_in*fan_out)).reshape(fan_out,fan_in)

                model.layers[i].bias.data = \
                    torch.linspace(-stdv,stdv,fan_out)[randperm(rng,fan_out)]
                    # torch.tensor(rand_vec(rng,-stdv,stdv,fan_out))


class RealTimeCABAC:

    def __init__(self,N):
        self.po2 = 2 ** np.arange(0,N).reshape(-1,1)
        self.c1 = np.zeros((2**N,1))
        self.c0 = np.zeros((2**N,1))
    def update(self,Xi,yi):
        context = Xi @ self.po2
        if (yi[0,0] == 1):
            self.c1[context[0,0],0] = self.c1[context[0,0],0] + 1
        else:
            self.c0[context[0,0],0] = self.c0[context[0,0],0] + 1
    def predict(self,Xi):
        context = Xi @ self.po2
        c1 = self.c1[context[0,0],0]
        c0 = self.c0[context[0,0],0]

        pp = np.zeros((1,1)) 
        pp[0,0] = np.max([1,c1]) / (np.max([1,c1]) + np.max([1,c0]))
        if c0 != 0 and c1 == 0:
            pp[0,0]=(0 + np.finfo(pp.dtype).eps)
        elif c0 == 0 and c1 != 0:
            pp[0,0]=(1 - np.finfo(pp.dtype).eps)

        return pp

class RealTimeAC:
    def __init__(self):
        self.c1 = 0
        self.c0 = 0
    def update(self,yi):
        if (yi[0,0] == 1):
            self.c1 = self.c1 + 1
        else:
            self.c0 = self.c0 + 1
    def predict(self):
        pp = np.zeros((1,1)) 
        pp[0,0] = np.max([1,self.c1]) / (np.max([1,self.c1]) + np.max([1,self.c0]))
        if self.c0 != 0 and self.c1 == 0:
            pp[0,0]=(0 + np.finfo(pp.dtype).eps)
        elif self.c0 == 0 and self.c1 != 0:
            pp[0,0]=(1 - np.finfo(pp.dtype).eps)
        return pp

def backward_adaptive_coding(pths,N,lr,with_cabac=False,with_mlp=True):
    

    trainset = CausalContextDataset(pths,"image",N)
    y,X = trainset.y,trainset.X
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=6)

    #with_mlp = True
    with_static = False
    if N == 0:
        with_mlp = False
        with_cabac = False
        with_static = True

    if with_mlp:

        device=torch.device("cuda:0")

        model = MLP_N_64N_32N_1(N)
        initialize_MLP_N_64N_32N_1(model)

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], torch.nn.Linear):
                weightvalues = model.layers[i].weight.detach().numpy().reshape(-1).tolist()
                w_avg = np.mean(weightvalues)
                w_mn = np.min(weightvalues)
                w_mx = np.max(weightvalues)
                print(f"layer {i} weights mean {w_avg} min {w_mn} max {w_mx}")
                biasvalues = model.layers[i].bias.detach().numpy().reshape(-1).tolist()
                b_avg = np.mean(biasvalues)
                b_mn = np.min(biasvalues)
                b_mx = np.max(biasvalues)
                print(f"layer {i} biases mean {b_avg} min {b_mn} max {b_mx}")
                

        model.to(device)
        model.train(True)

        criterion = Log2BCELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        mlp_avg_code_length_history = []
        
        mlp_running_loss = 0.0
    
    if with_cabac:
        cabac = RealTimeCABAC(N)
        cabac_avg_code_length_history = []
        cabac_running_loss = 0.0

    if with_static:
        static = RealTimeAC()
        static_avg_code_length_history = []
        static_running_loss = 0.0

    iteration = 0

    for data in tqdm(dataloader):

        X_b,y_b= data

        if with_mlp:

            X_b = X_b.float().to(device)
            y_b = y_b.float().to(device)

            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            mlp_running_loss += loss.item()
            mlp_avg_code_length_history.append( mlp_running_loss / (iteration + 1) )

        assert y_b.item() == y[iteration,0]
        assert np.allclose(X_b.cpu().numpy().reshape(-1) , X[iteration,:].reshape(-1))

        if with_cabac:
            cabac_pred_t = cabac.predict(X[iteration:iteration+1,:])
            cabac.update(X[iteration:iteration+1,:],y[iteration:iteration+1,:])

            cabac_loss = perfect_AC(y[iteration:iteration+1,:],cabac_pred_t)

            cabac_running_loss += cabac_loss
            cabac_avg_code_length_history.append( cabac_running_loss / (iteration + 1) )
        
        if with_static:
            static_pred_t = static.predict()
            static.update(y[iteration:iteration+1,:])

            static_loss = perfect_AC(y[iteration:iteration+1,:],static_pred_t)

            static_running_loss += static_loss
            static_avg_code_length_history.append( static_running_loss / (iteration + 1) )


        iteration += 1
            

    data = dict()
    if with_mlp:
        data["MLPlr={:.0e}".format(lr)] = mlp_avg_code_length_history
    if with_cabac:
        data["LUT"] = cabac_avg_code_length_history
    if with_static:
        data["STATIC"] = static_avg_code_length_history
    
    return data


def backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates):

    max_N = 26

    for N in Ns:

        len_data = len(docs[0]) * 1024*768

        data = dict()
        if N > 0:
            for lr in learning_rates:
                data["MLPlr={:.0e}".format(lr)] = np.zeros((len_data))
            if (N<=max_N):
                data["LUT"] = np.zeros((len_data))
        else:
            data["STATIC"] = np.zeros((len_data))

        for doc in docs:
            if N > 0:
                for i_lr,lr in enumerate(learning_rates):
                    partial_data = backward_adaptive_coding(doc,N,lr)
                    k = "MLPlr={:.0e}".format(lr)
                    data[k] = data[k] + np.array(partial_data[k])
                if (N<=max_N):
                    partial_data = backward_adaptive_coding(doc,N,0,with_cabac=True,with_mlp=False)
                    data["LUT"] = data["LUT"] + np.array(partial_data["LUT"])
            else:
                partial_data = backward_adaptive_coding(doc,0,0)
                data["STATIC"] = data["STATIC"] + np.array(partial_data["STATIC"])

        for k in data.keys():
            data[k] = data[k]/len(docs)
            
        xvalues = np.arange( len_data )
            
        fig = plot_comparison(xvalues,data,"iteration",
            linestyles={k:"solid" for k in data.keys()},
            colors={k:c for k,c in zip(data.keys(),["g","b","r","c","m","y","k","0.75"])},
            markers={k:"" for k in data.keys()})

        xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

        fig.axes[0].set_xticks( xticks)
        fig.axes[0].set_xticklabels( xticks)

        fname = f"backward_adaptive_coding_{exp_name}_N{N}"

        fig.savefig(fname+".png", dpi=300)

        # save_values(fname,[xvalues[-1]],{k:[v[-1]] for k,v in data.items()},"iteration")

        save_values(fname,xvalues,data,"iteration")


if __name__ == "__main__":

    # exp_name = "SPL2021_last_10_sorted_pages_lr1e-2"

    exp_name = "Adaptive_Detection_of_Dim_corrected_lut"

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            #"/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            #"/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            "SPL2021/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            #"/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            #"/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png"
        ]
    ]

    # docs = [[os.path.join('/home/lucas/Documents/data/SPL2021/pages',f)] for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2021/pages'))[-10:]]

    Ns = [26] # [0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = [] # (3.162277659**np.array([-2,-4,-8]))

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates)