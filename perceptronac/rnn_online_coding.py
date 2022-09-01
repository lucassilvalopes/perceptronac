"""
RNN online coding

based on pytorch's char rnn classification tutorial:

https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch
import numpy as np
from tqdm import tqdm
import time 
import os
from perceptronac.models import Log2NLLLoss
from perceptronac.utils import causal_context_many_imgs
from perceptronac.loading_and_saving import plot_comparison
from perceptronac.loading_and_saving import save_values
from perceptronac.loading_and_saving import change_aspect
from perceptronac.rnn_models import create_rnn


def onehot(y):
    return torch.cat([torch.logical_not(y,out=torch.empty(y.size(), dtype=y.dtype, device=y.device)),y],axis=1).unsqueeze(1)


def train(rnn,hidden,criterion,learning_rate,category_tensor, line_tensor):
    """
    line_tensor: [sequence_len x batch_size x alphabet_size]
    category_tensor: [sequence_len x batch_size]
    input: [batch_size x alphabet_size]
    hidden: [batch_size x hidden_size]
    output: [batch_size x n_categories]

    https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
    """
    rnn.zero_grad()

    losses = []

    output, hidden = rnn(line_tensor[0], hidden)

    for i in range(category_tensor.size()[0]):

        output, hidden = rnn(line_tensor[i+1], hidden)

        loss = criterion(output, category_tensor[i])

        losses.append( loss )

    losses = torch.sum(torch.stack(losses))
    losses.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        # print(p.data.shape,p.requires_grad,p.name,p.grad)
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return hidden.detach().clone() ,losses.item()



def rnn_online_coding(pths,lr,which_model,hidden_units,n_layers,samples_per_time=1,n_pieces=1):

    device=torch.device("cuda:0")

    model = create_rnn(which_model,hidden_units,n_layers)

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

    hidden = model.initHidden(device)
    previous_targets = torch.ones(2,1,device=device)

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

            y_b= torch.tensor(y[start:stop,:],dtype=torch.float32,device=device)

            # y_b = y_b.float().to(device)

            inpt = onehot(torch.cat([previous_targets,y_b[:-1,:]],axis=0) )

            hidden,loss = train(model,hidden,criterion,lr,y_b, inpt)

            previous_targets = torch.cat([previous_targets,y_b],axis=0)[-2:,:]

            running_loss += loss
            avg_code_length_history.append( running_loss / ((iteration + 1) * samples_per_time) )


            iteration+=1
            pbar.update(1)
        pbar.close()            

    data = dict()

    data["RNNlr={:.0e}".format(lr)] = avg_code_length_history


    return data


def rnn_online_coding_experiment(exp_name,docs,learning_rates,colors,linestyles,
    labels,legend_ncol,ylim,which_model,hidden_units,n_layers=1,samples_per_time=1,n_pieces=1):

    exp_id = str(int(time.time()))
    save_dir = f"results/exp_{exp_id}"

    os.makedirs(save_dir)

    fname = f"{save_dir.rstrip('/')}/rnn_online_coding_{exp_name}"

    len_data = (len(docs[0]) * 1024*768) // samples_per_time

    data = dict()

    for lr in learning_rates:
        data["RNNlr={:.0e}".format(lr)] = np.zeros((len_data))

    for doc in docs:
        partial_data = dict()

        for lr in learning_rates:
            partial_data = rnn_online_coding(doc,lr,which_model,hidden_units,n_layers,
                samples_per_time=samples_per_time,n_pieces=n_pieces)
            k = "RNNlr={:.0e}".format(lr)
            data[k] = data[k] + np.array(partial_data[k])


    for k in data.keys():
        data[k] = data[k]/len(docs)
        
    xvalues = np.arange( len_data )

    save_values(fname,xvalues,data,"iteration")
        
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

    fig.savefig(fname+".png", dpi=300)
