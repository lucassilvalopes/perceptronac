"""
RNN online coding

based on pytorch's char rnn classification tutorial:

https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import time 
import os
from perceptronac.backward_adaptive_coding import RNG
from perceptronac.backward_adaptive_coding import randperm
from perceptronac.models import Log2NLLLoss
from perceptronac.utils import causal_context_many_imgs
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_values
from perceptronac.loading_and_saving import change_aspect


def onehot(y):
    return torch.cat([torch.logical_not(y,out=torch.empty(y.size(), dtype=y.dtype, device=y.device)),y],axis=1).unsqueeze(1)



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def initialize_rnn(model):
    rng = RNG()
    for layer in [model.i2h,model.i2o]:
        fan_in = layer.weight.size(1)
        fan_out = layer.weight.size(0)
        stdv = 1. / math.sqrt(fan_in)
        with torch.no_grad():
            layer.weight.data = \
                torch.linspace(-stdv,stdv,fan_in*fan_out)[randperm(rng,fan_in*fan_out)].reshape(fan_out,fan_in)
            layer.bias.data = \
                torch.linspace(-stdv,stdv,fan_out)[randperm(rng,fan_out)]


def train(rnn,hidden,criterion,learning_rate,category_tensor, line_tensor):
    """
    line_tensor: [sequence_len x samples_per_time x alphabet_size]
    category_tensor: [samples_per_time]
    input: [samples_per_time x alphabet_size]
    hidden: [samples_per_time x hidden_size]
    output: [samples_per_time x n_categories]

    https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    """
    rnn.zero_grad()

    losses = []

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        losses.append(loss.item())

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return hidden.detach().clone(),sum(losses)



def rnn_online_coding(pths,lr,samples_per_time=1,n_pieces=1):

    device=torch.device("cuda:0")

    model = RNN(2, 128, 2)
    initialize_rnn(model)

    model.to(device)
    model.train(True)

    criterion = Log2NLLLoss(reduction='sum')

    avg_code_length_history = []
    
    running_loss = 0.0
    

    # (1024*768) * len(pths) must be divisible by n_pieces*samples_per_time
    # (1024*768) must be divisible by n_pieces
    # ((1024*768) * len(pths)) // n_pieces must be divisible by samples_per_time
    piece_len = (1024*768) // n_pieces

    iteration = 0

    hidden = model.initHidden()

    for piece in range(n_pieces):

        lower_lim = (piece*piece_len*len(pths))
        upper_lim = ((piece+1)*piece_len*len(pths))

        page_len = (piece_len * n_pieces)

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

        y,X = causal_context_many_imgs(np.array(pths)[start_page:end_page+1].tolist(), 0)
        y = y[lower_lim-start_page_lower_lim:upper_lim-start_page_lower_lim,:]


        n_iterations = len(y)//samples_per_time
        pbar = tqdm(total=n_iterations)

        
        for iteration in range(n_iterations):

            start = iteration * samples_per_time - (piece*piece_len*len(pths))
            stop = (iteration+1)* samples_per_time - (piece*piece_len*len(pths))

            y_b= torch.tensor(y[start:stop,:])

            y_b = y_b.float().to(device)

            hidden,loss = train(model,hidden,criterion,lr,y_b, onehot(y_b))

            running_loss += loss
            avg_code_length_history.append( running_loss / ((iteration + 1) * samples_per_time) )


            iteration+=1
            pbar.update(1)
        pbar.close()            

    data = dict()

    data["RNNlr={:.0e}".format(lr)] = avg_code_length_history


    return data


def rnn_online_coding_experiment(exp_name,docs,learning_rates,colors,linestyles,
    labels,legend_ncol,ylim,samples_per_time=1,n_pieces=1):


    len_data = (len(docs[0]) * 1024*768) // samples_per_time

    data = dict()

    for lr in learning_rates:
        data["RNNlr={:.0e}".format(lr)] = np.zeros((len_data))

    for doc in docs:
        partial_data = dict()

        for lr in learning_rates:
            partial_data = rnn_online_coding(doc,lr,samples_per_time=samples_per_time,n_pieces=n_pieces)
            k = "RNNlr={:.0e}".format(lr)
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

    exp_id = str(int(time.time()))
    save_dir = f"results/exp_{exp_id}"

    os.makedirs(save_dir)

    fname = f"{save_dir.rstrip('/')}/rnn_online_coding_{exp_name}"

    fig.savefig(fname+".png", dpi=300)

    # save_values(fname,[xvalues[-1]],{k:[v[-1]] for k,v in data.items()},"iteration")

    save_values(fname,xvalues,data,"iteration")