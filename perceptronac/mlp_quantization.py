# %%
import torch
from perceptronac.models import ArbitraryMLP
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from perceptronac.perfect_AC import perfect_AC_binary
from perceptronac.models import StaticAC
from perceptronac.perfect_AC import perfect_AC_generic
from perceptronac.models import S256AC

# %%
def get_model_parameters_values(model):
    all_parameters = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            with torch.no_grad():
                all_parameters.append(model.layers[i].weight.data.detach().numpy().reshape(-1))
                all_parameters.append(model.layers[i].bias.data.detach().numpy().reshape(-1))
    all_parameters = np.concatenate(all_parameters,axis=0)
    return all_parameters

def parameters_histogram(model):

    all_parameters = get_model_parameters_values(model)

    max_abs_value = np.max(np.abs(all_parameters))
    
    y,x = np.histogram(all_parameters,np.linspace(-max_abs_value,max_abs_value,100))
    plt.plot(x[:-1],y)

def estimate_midtread_uniform_quantization_delta(model,n_bits):
    """
    https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
    https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
    https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5

    The np.round function follows the round-half-to-even rule.
    For midtread, the last positive quantization value is 2**(n_bits-1)
    which is an even number. The two integers closest to the positive extreme
    are this number and the next odd number. Therefore, the most extreme positive value
    will be rounded to this even number due to the round-half-to-even rule.
    """

    all_parameters = get_model_parameters_values(model)

    max_abs_value = np.max(np.abs(all_parameters))

    # Delta = (max_abs_value - (-max_abs_value))/(2**n_bits) # midrise

    Delta = max_abs_value /(2**(n_bits-1)+(1/2)) # midtread

    return Delta

def midtread_uniform_quantization(model,Delta):
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            with torch.no_grad():
                model.layers[i].weight.data = Delta * torch.round(model.layers[i].weight.data/Delta)
                model.layers[i].bias.data = Delta * torch.round(model.layers[i].bias.data/Delta)
    return model

def visualize_parameters_quantization_process(all_parameters,Delta):

    print("Delta")
    print(Delta)

    print("original")
    print(all_parameters)
    
    print("quantized")
    print(np.round(all_parameters/Delta))

    print("dequantized")
    print(Delta * np.round(all_parameters/Delta))

def test_midtread_uniform_quantization(model,n_bits=8):

    parameters_histogram(model)

    Delta = estimate_midtread_uniform_quantization_delta(model,n_bits)

    all_parameters = get_model_parameters_values(model)

    visualize_parameters_quantization_process(all_parameters,Delta)

    model2 = midtread_uniform_quantization(model,Delta)

    all_parameters2 = get_model_parameters_values(model2)

    assert np.allclose( all_parameters2, Delta * np.round(all_parameters/Delta) )

# %%

def int8_to_bits(v):
    """
    Sign-Magnitude representation

    This function works for 8-bit integers
    
    https://www3.ntu.edu.sg/home/ehchua/programming/java/datarepresentation.html
    https://en.wikipedia.org/wiki/Two%27s_complement
    https://stackoverflow.com/questions/699866/python-int-to-binary-string

    """
    sign = 1 if v < 0 else 0
    bitseq = [int(c) for c in "{:08b}".format(np.abs(v))]
    bitseq[0] = sign
    return bitseq

def entropy(seq):
    p = np.zeros(len(np.unique(seq)))
    for i,v in enumerate(np.unique(seq)):
        p[i] = np.sum(seq == v)/len(seq)
    # p = np.histogram(quantized.flatten(), bins=np.arange(quantized.min(), quantized.max()+1))[0]
    # p = np.delete(p, p==0)
    # p = p / len(seq)
    return -(p * np.log2(p)).sum()


def encode_network_int8_to_binary_symbols(values):

    bitstream = np.array([int8_to_bits(v) for v in values]).reshape(-1)

    staticac = StaticAC()

    staticac.load(bitstream.reshape(-1,1))

    rate = perfect_AC_binary(bitstream.reshape(-1,1),staticac(np.zeros((bitstream.shape[0], 0), dtype=np.int64)))

    return rate


def encode_network_integer_symbols(quantized):

    rate = entropy(quantized)

    return rate

def encode_network_integer_symbols_2(values):

    intstream = values - np.min(values)

    staticac = S256AC()

    staticac.load(intstream.reshape(-1,1))

    rate = perfect_AC_generic(intstream.reshape(-1,1),staticac(np.zeros((intstream.shape[0], 0), dtype=np.int64)))

    return rate

# %%

if __name__ == "__main__":

    configs = pd.read_csv("/home/lucas/Documents/perceptronac/results/exp_1656174158/exp_1656174158_conf.csv",index_col="key").to_dict()["value"]
    configs['save_dir'] = "/home/lucas/Documents/perceptronac/results/"
    topologies = ast.literal_eval(configs["topologies"])

    widths = topologies[0]
    file_name = f"{configs['save_dir'].rstrip('/')}/exp_{configs['id']}/exp_{configs['id']}_{'_'.join(map(str,widths))}_min_valid_loss_model.pt"

    model = ArbitraryMLP(widths)
    model.load_state_dict(torch.load(file_name))

    # test_midtread_uniform_quantization(model)

    Delta = estimate_midtread_uniform_quantization_delta(model,8)

    model2 = midtread_uniform_quantization(model,Delta)

    print(encode_network_int8_to_binary_symbols(np.round(get_model_parameters_values(model)/Delta).astype(int)))

    print(encode_network_integer_symbols(np.round(get_model_parameters_values(model)/Delta).astype(int)))
    
    print(encode_network_integer_symbols_2(np.round(get_model_parameters_values(model)/Delta).astype(int)))


