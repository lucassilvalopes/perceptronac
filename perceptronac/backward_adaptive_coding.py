
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
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.loading_and_saving import change_aspect
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


class RealTimeLUT:

    def __init__(self,N,central_tendency="mode"):
        self.po2 = 2 ** np.arange(0,N).reshape(-1,1)
        self.central_tendency = central_tendency
        if self.central_tendency=="mode":
            self.c1 = np.zeros((2**N,1))
            self.c0 = np.zeros((2**N,1))
        elif self.central_tendency=="mean":
            self.c1 = np.ones((2**N,1))
            self.c0 = np.ones((2**N,1))
        else:
            self.bad_central_tendency()

    def bad_central_tendency(self):
        raise ValueError(f"{self.central_tendency} not an option, only \"mode\" or \"mean\"")

    def update(self,Xi,yi):
        context = Xi @ self.po2 # np.array([[]]) @ (2 ** np.arange(0,0).reshape(-1,1)) == 0
        if (yi[0,0] == 1):
            self.c1[context[0,0],0] = self.c1[context[0,0],0] + 1
        else:
            self.c0[context[0,0],0] = self.c0[context[0,0],0] + 1
    def predict(self,Xi):
        context = Xi @ self.po2 # np.array([[]]) @ (2 ** np.arange(0,0).reshape(-1,1)) == 0
        c1 = self.c1[context[0,0],0]
        c0 = self.c0[context[0,0],0]

        pp = np.zeros((1,1)) 
        if self.central_tendency == "mode":
            pp[0,0] = np.max([1,c1]) / (np.max([1,c1]) + np.max([1,c0]))
            if c0 != 0 and c1 == 0:
                pp[0,0]=(0 + np.finfo(pp.dtype).eps)
            elif c0 == 0 and c1 != 0:
                pp[0,0]=(1 - np.finfo(pp.dtype).eps)
        elif self.central_tendency=="mean":
            pp[0,0] = c1 / (c1 + c0)
        else:
            self.bad_central_tendency()
        return pp


def backward_adaptive_coding(pths,N,lr,central_tendencies,with_lut=False,with_mlp=True):
    

    trainset = CausalContextDataset(pths,"image",N)
    y,X = trainset.y,trainset.X
    dataloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle=False,num_workers=6)

    if N == 0:
        with_mlp = False

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
    
    if with_lut:
        lut_avg_code_length_histories = dict()
        lut_running_losses = dict()
        luts = dict()
        for central_tendency in central_tendencies:
            luts[central_tendency] = RealTimeLUT(N,central_tendency=central_tendency)
            lut_avg_code_length_histories[central_tendency] = []
            lut_running_losses[central_tendency] = 0.0 

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

        if with_lut:
            for central_tendency in central_tendencies:

                lut_pred_t = luts[central_tendency].predict(X[iteration:iteration+1,:])
                luts[central_tendency].update(X[iteration:iteration+1,:],y[iteration:iteration+1,:])

                lut_loss = perfect_AC(y[iteration:iteration+1,:],lut_pred_t)

                lut_running_losses[central_tendency] += lut_loss
                lut_avg_code_length_histories[central_tendency].append( lut_running_losses[central_tendency] / (iteration + 1) )


        iteration += 1
            

    data = dict()
    if with_mlp:
        data["MLPlr={:.0e}".format(lr)] = mlp_avg_code_length_history
    if with_lut:
        for central_tendency in central_tendencies:
            data[f"LUT{central_tendency}"] = lut_avg_code_length_histories[central_tendency]

    return data


def backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,
    labels,legend_ncol,ylim):

    max_N = 26

    for N in Ns:

        len_data = len(docs[0]) * 1024*768

        data = dict()
        if N > 0:
            for lr in learning_rates:
                data["MLPlr={:.0e}".format(lr)] = np.zeros((len_data))
        if (N<=max_N):
            for central_tendency in central_tendencies:
                data[f"LUT{central_tendency}"] = np.zeros((len_data))


        for doc in docs:
            partial_data = dict()
            if N > 0:
                for i_lr,lr in enumerate(learning_rates):
                    with_lut = ((i_lr == len(learning_rates)-1) and (N<=max_N))
                    partial_data = backward_adaptive_coding(doc,N,lr,central_tendencies,with_lut=with_lut)
                    k = "MLPlr={:.0e}".format(lr)
                    data[k] = data[k] + np.array(partial_data[k])
            if (N<=max_N):
                if not all([f"LUT{central_tendency}" in partial_data.keys() for central_tendency in central_tendencies]):
                    partial_data = backward_adaptive_coding(doc,N,0,central_tendencies,with_lut=True,with_mlp=False)
                for central_tendency in central_tendencies:
                    k = f"LUT{central_tendency}"
                    data[k] = data[k] + np.array(partial_data[k])

        for k in data.keys():
            data[k] = data[k]/len(docs)
            
        xvalues = np.arange( len_data )
            
        fig = plot_comparison(xvalues,data,"iteration",
            linestyles={k:ls for k,ls in zip(sorted(data.keys()),linestyles)},
            colors={k:c for k,c in zip(sorted(data.keys()),colors)},
            markers={k:"" for k in sorted(data.keys())},
            labels={k:lb for k,lb in zip(sorted(data.keys()),labels)},
            legend_ncol=legend_ncol)

        xticks = np.round(np.linspace(0,len_data-1,5)).astype(int)

        fig.axes[0].set_xticks( xticks)
        fig.axes[0].set_xticklabels( xticks)

        ax, = fig.axes
        ax.set_ylim(ylim)

        change_aspect(ax)

        fname = f"backward_adaptive_coding_{exp_name}_N{N}"

        fig.savefig(fname+".png", dpi=300)

        # save_values(fname,[xvalues[-1]],{k:[v[-1]] for k,v in data.items()},"iteration")

        save_values(fname,xvalues,data,"iteration")


