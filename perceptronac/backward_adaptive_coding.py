
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
import time
import math
import torch
import numpy as np
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.utils import causal_context_many_imgs
from perceptronac.losses import perfect_AC
from perceptronac.losses import Log2BCELoss
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
        L,N = Xi.shape
        context = Xi @ self.po2 # np.array([[]]) @ (2 ** np.arange(0,0).reshape(-1,1)) == 0
        for k in range(L):
            if (yi[k,0] == 1):
                self.c1[context[k,0],0] = self.c1[context[k,0],0] + 1
            else:
                self.c0[context[k,0],0] = self.c0[context[k,0],0] + 1

    def predict(self,Xi):
        L,N = Xi.shape
        context = Xi @ self.po2 # np.array([[]]) @ (2 ** np.arange(0,0).reshape(-1,1)) == 0
        pp = np.zeros((L,1))
        for k in range(L):
            c1 = self.c1[context[k,0],0]
            c0 = self.c0[context[k,0],0]

            if self.central_tendency == "mode":
                pp[k,0] = np.max([1,c1]) / (np.max([1,c1]) + np.max([1,c0]))
                if c0 != 0 and c1 == 0:
                    pp[k,0]=(0 + np.finfo(pp.dtype).eps)
                elif c0 == 0 and c1 != 0:
                    pp[k,0]=(1 - np.finfo(pp.dtype).eps)
            elif self.central_tendency=="mean":
                pp[k,0] = c1 / (c1 + c0)
            else:
                self.bad_central_tendency()
        
        return pp



from perceptronac.loading_and_saving import save_configs
from perceptronac.loading_and_saving import save_model


def save_nn_model(file_name,model):
    save_model(file_name,model)

def load_nn_model(model,file_name):
    # model = MLP_N_64N_32N_1(N)
    model.load_state_dict(torch.load(file_name))

def save_lut_model(file_name,model):
    file_name = os.path.splitext(file_name)[0]
    np.savez(f"{file_name}.npz", c0=model.c0, c1=model.c1)

def load_lut_model(model,file_name):
    # model = RealTimeLUT(N,central_tendency=central_tendency)
    npz_kw = np.load(file_name)
    model.c0 = npz_kw["c0"]
    model.c1 = npz_kw["c1"]



def backward_adaptive_coding(exp_id,
                             pths,N,lr,central_tendencies,with_lut=False,with_mlp=True,parallel=False,samples_per_time=1,n_pieces=1,
                             manual_th=None,full_page=True,page_len = (1024*768),parent_id=None):
    """
    
    Assumptions:
    --> page_len must be divisible by n_pieces (parallel=True)
    --> page_len * len(pths) must be divisible by n_pieces
    --> (page_len * len(pths)) // n_pieces must be divisible by samples_per_time
    --> Or equivalently, page_len * len(pths) must be divisible by n_pieces*samples_per_time

    Observations:
    - start_page and end_page may have the same value, for example, when there is only one page
    - the actual "piece length", when parallel = False, is upper_lim - lower_lim
    - "piece" is so that arbitrarily large sequences can be coded
    - samples_per_time is the batch_size when parallel = False
    - currently only samples_per_time=1 is supported for parallel = True, in which case the batch_size is len(pths)
    
    """
    
    
    
    if N == 0:
        with_mlp = False

    if with_mlp:

        device=torch.device("cuda:0")

        model = MLP_N_64N_32N_1(N)

        if parent_id:
            load_nn_model(model,"results/exp_{}/exp_{}_mlp_lr{:.0e}.pt".format(parent_id,parent_id,lr))
        else:
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
            if parent_id:
                load_lut_model(luts[central_tendency],f"results/exp_{parent_id}/exp_{parent_id}_lut_{central_tendency}.npz")
            lut_avg_code_length_histories[central_tendency] = []
            lut_running_losses[central_tendency] = 0.0 

    iteration = 0
    for piece in range(n_pieces):

        if parallel:
            if samples_per_time != 1:
                raise ValueError("parallel processing with more than one sample per page at a time is not supported yet")

            batch_size=len(pths)
            y = []
            X = []
            for pth in pths:
                partial_y,partial_X = causal_context_many_imgs([pth], N,
                                                               manual_th=manual_th,full_page=full_page)
                y.append(partial_y[(piece*(page_len//n_pieces)):((piece+1)*(page_len//n_pieces)),:].copy())
                X.append(partial_X[(piece*(page_len//n_pieces)):((piece+1)*(page_len//n_pieces)),:].copy())
                del partial_y,partial_X

            y = np.concatenate(y,axis=1).reshape(-1,1) # interleaves the samples from different pages
            if N > 0:
                X = np.concatenate(X,axis=1).reshape(-1,N)
            else:
                X = np.zeros((y.shape[0],0),dtype=int)
        else:
            batch_size=samples_per_time

            lower_lim = (piece*((page_len*len(pths))//n_pieces))
            upper_lim = ((piece+1)*((page_len*len(pths))//n_pieces))

            start_page_upper_lim = page_len
            start_page_lower_lim = 0
            start_page = 0
            while lower_lim >= start_page_upper_lim:
                start_page += 1
                start_page_lower_lim = start_page*page_len
                start_page_upper_lim = (start_page+1)*page_len

            end_page_upper_lim = page_len
            end_page_lower_lim = 0
            end_page = 0
            while upper_lim > end_page_upper_lim:
                end_page += 1
                end_page_lower_lim = end_page*page_len
                end_page_upper_lim = (end_page+1)*page_len

            y,X = causal_context_many_imgs(np.array(pths)[start_page:end_page+1].tolist(), N,
                                           manual_th=manual_th,full_page=full_page)
            y = y[lower_lim-start_page_lower_lim:upper_lim-start_page_lower_lim,:]
            X = X[lower_lim-start_page_lower_lim:upper_lim-start_page_lower_lim,:]


        trainset = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        dataloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False)


        n_batches = len(y)//batch_size
        pbar = tqdm(total=n_batches)

        # for iteration in range(n_batches):
        #     ...
        #     X_b,y_b= torch.tensor(X[start:stop,:]),torch.tensor(y[start:stop,:])

        
        for data in dataloader:
            X_b,y_b=data

            start = iteration * batch_size - (piece*((page_len*len(pths))//n_pieces))
            stop = (iteration+1)* batch_size - (piece*((page_len*len(pths))//n_pieces))

            if with_mlp:

                X_b = X_b.float().to(device)
                y_b = y_b.float().to(device)

                optimizer.zero_grad()
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()

                mlp_running_loss += loss.item()
                mlp_avg_code_length_history.append( mlp_running_loss / ((iteration + 1) * batch_size) )

            assert np.allclose(y_b.detach().cpu().numpy().astype(int) , y[start:stop,:].astype(int))
            assert np.allclose(X_b.detach().cpu().numpy().astype(int) , X[start:stop,:].astype(int))

            if with_lut:
                for central_tendency in central_tendencies:

                    lut_pred_t = luts[central_tendency].predict(X[start:stop,:])
                    luts[central_tendency].update(X[start:stop,:],y[start:stop,:])

                    lut_loss = batch_size * perfect_AC(y[start:stop,:],lut_pred_t)

                    lut_running_losses[central_tendency] += lut_loss
                    lut_avg_code_length_histories[central_tendency].append(lut_running_losses[central_tendency]/((iteration+1)*batch_size))

            iteration+=1
            pbar.update(1)
        pbar.close()            

    data = dict()
    if with_mlp:
        data["MLPlr={:.0e}".format(lr)] = mlp_avg_code_length_history
    if with_lut:
        for central_tendency in central_tendencies:
            data[f"LUT{central_tendency}"] = lut_avg_code_length_histories[central_tendency]


    save_nn_model("results/exp_{}/exp_{}_mlp_lr{:.0e}.pt".format(exp_id,exp_id,lr),model)

    for central_tendency in central_tendencies:
        save_lut_model(f"results/exp_{exp_id}/exp_{exp_id}_lut_{central_tendency}.npz",luts[central_tendency])


    return data


def backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,
    labels,legend_ncol,ylim,parallel=False,samples_per_time=1,n_pieces=1,
    manual_th=None,full_page=True,page_shape = (1024,768), parent_id=None):

    max_N = 26

    for N in Ns:

        exp_id = str(int(time.time()))

        os.makedirs(f"results/exp_{exp_id}")

        configs = {
            "id": exp_id,
            "docs": docs,
            "N": N,
            "learning_rates": learning_rates,
            "central_tendencies": central_tendencies,
            "parallel": parallel,
            "samples_per_time": samples_per_time,
            "n_pieces": n_pieces,
            "manual_th": manual_th,
            "full_page": full_page,
            "page_shape": page_shape,
            "parent_id": parent_id
        }

        save_configs(f"results/exp_{exp_id}/exp_{exp_id}_conf",configs)


        nr,nc = page_shape
        if full_page is False:
            ns = int(np.ceil(np.sqrt(N)))
            page_len = (nr-ns) * (nc-2*ns)
        else:
            page_len = nr * nc


        if parallel:
            if samples_per_time != 1:
                raise ValueError("parallel processing with more than one sample per page at a time is not supported yet")
            len_data = page_len
        else:
            len_data = (len(docs[0]) * page_len) // samples_per_time

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
                    partial_data = backward_adaptive_coding(exp_id,doc,N,lr,central_tendencies,with_lut=with_lut,
                        parallel=parallel,samples_per_time=samples_per_time,n_pieces=n_pieces,
                        manual_th=manual_th,full_page=full_page,page_len = page_len,parent_id=parent_id)
                    k = "MLPlr={:.0e}".format(lr)
                    data[k] = data[k] + np.array(partial_data[k])
            if (N<=max_N):
                if not all([f"LUT{central_tendency}" in partial_data.keys() for central_tendency in central_tendencies]):
                    partial_data = backward_adaptive_coding(exp_id,doc,N,0,central_tendencies,with_lut=True,with_mlp=False,
                        parallel=parallel,samples_per_time=samples_per_time,n_pieces=n_pieces,
                        manual_th=manual_th,full_page=full_page,page_len = page_len,parent_id=parent_id)
                for central_tendency in central_tendencies:
                    k = f"LUT{central_tendency}"
                    data[k] = data[k] + np.array(partial_data[k])

        for k in data.keys():
            data[k] = data[k]/len(docs)
            
        xvalues = np.arange( len_data )
            

        ## Figure/Values

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

        fig.savefig(f"results/exp_{exp_id}/exp_{exp_id}_graph.png", dpi=300)

        save_values(f"results/exp_{exp_id}/exp_{exp_id}_values",xvalues,data,"iteration")


