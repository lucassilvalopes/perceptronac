
import torch
import numpy as np
import pandas as pd


def get_model_parameters_values(model):
    all_parameters = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            with torch.no_grad():
                all_parameters.append(model.layers[i].weight.data.detach().numpy().reshape(-1))
                all_parameters.append(model.layers[i].bias.data.detach().numpy().reshape(-1))
    all_parameters = np.concatenate(all_parameters,axis=0)
    return all_parameters


def estimate_midtread_uniform_quantization_delta(model,n_bits):
    """
    https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
    https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
    https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5

    The np.round function follows the round-half-to-even rule.
    """

    all_parameters = get_model_parameters_values(model)

    mx = np.max(all_parameters) + np.finfo(all_parameters.dtype).eps

    mn = np.min(all_parameters) - np.finfo(all_parameters.dtype).eps

    values_range = mx - mn

    Delta = values_range / (2**n_bits)

    shift = mn + (Delta/2)

    return Delta,shift


def midtread_uniform_quantization(model,Delta,shift):
    all_parameters = []
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], torch.nn.Linear):
            with torch.no_grad():
                quantized_weight_data = torch.round((model.layers[i].weight.data-shift)/Delta)
                all_parameters.append( quantized_weight_data.detach().numpy().reshape(-1) )
                model.layers[i].weight.data = (Delta * quantized_weight_data) + shift
                
                quantized_bias_data = torch.round((model.layers[i].bias.data-shift)/Delta)
                all_parameters.append( quantized_bias_data.detach().numpy().reshape(-1) )
                model.layers[i].bias.data = (Delta * quantized_bias_data) + shift
                
    all_parameters = np.concatenate(all_parameters,axis=0)
    return model,all_parameters


def entropy(seq):
    p = np.zeros(len(np.unique(seq)))
    for i,v in enumerate(np.unique(seq)):
        p[i] = np.sum(seq == v)/len(seq)
    # p = np.histogram(quantized.flatten(), bins=np.arange(quantized.min(), quantized.max()+1))[0]
    # p = np.delete(p, p==0)
    # p = p / len(seq)
    return -(p * np.log2(p)).sum()


def encode_network_integer_symbols(quantized):

    rate = entropy(quantized)

    n_samples = len(quantized)

    return rate * n_samples, n_samples



def get_quantization_data(csv_path):

    data = pd.read_csv(csv_path)

    data["model_bits"] = data["model_bits/data_samples"] * data["data_samples"]

    data['idx'] = data.apply(lambda x: f"{x.topology}_{x.quantization_bits:02d}b", axis=1)

    data = data.set_index("idx")

    return data



if __name__ == "__main__":

    get_quantization_data(
        "/home/lucas/Documents/perceptronac/results/exp_1676160183/exp_1676160183_model_bits_x_data_bits_values.csv"
    ).to_csv(
        "/home/lucas/Documents/perceptronac/complexity/data/rate-model-bits_hx-10-20-40-80-160-320-640_b-8-16-32.csv"
    )

