

import numpy as np
from perceptronac.models import ArbitraryMLP
from perceptronac.mlp_quantization import estimate_midtread_uniform_quantization_delta
from perceptronac.mlp_quantization import midtread_uniform_quantization

def test_mlp_quantization():

    model = ArbitraryMLP([10, 10, 10, 1])

    n_bits = 8

    Delta,shift = estimate_midtread_uniform_quantization_delta(model,n_bits)

    model2,quantized_values = midtread_uniform_quantization(model,Delta,shift)

    assert np.min(quantized_values) == 0
    assert np.max(quantized_values) == (2**n_bits) - 1