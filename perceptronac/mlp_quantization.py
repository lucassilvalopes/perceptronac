
import torch
import numpy as np


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


if __name__ == "__main__":

    from perceptronac.models import ArbitraryMLP

    model = ArbitraryMLP([10, 10, 10, 1])

    n_bits = 8

    Delta,shift = estimate_midtread_uniform_quantization_delta(model,n_bits)

    model2,quantized_values = midtread_uniform_quantization(model,Delta,shift)

    assert np.min(quantized_values) == 0
    assert np.max(quantized_values) == (2**n_bits) - 1

