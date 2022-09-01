import torch
import torch.nn as nn
import numpy as np
import random
import math
from perceptronac.backward_adaptive_coding import RNG
from perceptronac.backward_adaptive_coding import randperm


class LinearRNN(nn.Module):
    """https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearRNN, self).__init__()

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

    def initHidden(self,device):
        return torch.zeros(1, self.hidden_size, device=device)


class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self,device):
        return torch.zeros(1, self.hidden_size, device=device)


def initialize_rnn(model):
    """Works for linear RNN and Elman RNN"""
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


class GRURNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers = 1):
        super(GRURNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.i2h = nn.GRU(input_size, hidden_size, n_layers)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        
        gru_output,hidden = self.i2h(input.unsqueeze(0), hidden)
        output = self.i2o(gru_output.squeeze(0))
        output = self.softmax(output)
        return output, hidden

    def initHidden(self,device):
        return torch.zeros(self.n_layers, 1, self.hidden_size,device=device)


def create_rnn(which,hidden_units):
    """https://pytorch.org/docs/stable/notes/randomness.html"""

    if which == "LinearRNN":
        model = LinearRNN(2, hidden_units, 2)
        initialize_rnn(model)
    elif which == "ElmanRNN":
        model = ElmanRNN(2, hidden_units, 2)
        initialize_rnn(model)
    elif which == "GRURNN":
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        model = GRURNN(2, hidden_units, 2)
    else:
        raise ValueError(f"{which}")
    
    return model