import torch
import torch.nn as nn
import numpy as np
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

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


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

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


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