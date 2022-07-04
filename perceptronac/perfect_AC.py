# function rate = perfect_AC(b, p)
# rate = mean(- log2(p) .* (b == 1) - log2(1 - p)  .* (b == 0));

import numpy as np

def perfect_AC(b, p, binary=True):
    if binary:
        return perfect_AC_binary(b, p)
    else:
        return perfect_AC_generic(b, p)


def perfect_AC_binary(b, p):
    """
    Args:
        b (-1 by 1): bitstream
        p (-1 by 1): probability of bit 1 as predicted by some method
    """
    rate = np.mean(
        - np.log2(p) * (b == 1).astype(int) - np.log2(1 - p)  * (b == 0).astype(int))
    return rate


def one_hot_encode(y, n_classes):
    one_hot = np.zeros((np.prod(y.shape),n_classes))
    for i in range(n_classes):
        vec = np.zeros((n_classes))
        vec[i] = 1
        one_hot[y.reshape(-1)==i] = vec
    new_shape = list(y.shape)
    new_shape.append(n_classes)
    return one_hot.reshape(*new_shape)


def perfect_AC_generic(b, p):
    """
    Args:
        b (-1 by n_channels): symbols to encode for each of the n_channels of each sample
        p (-1 by n_symbols by n_channels): probabilities for each of the n_symbols for each of the n_channels
    """
    rate = np.sum( - np.log2(p) * np.transpose(one_hot_encode(b,p.shape[1]), (0, 2, 1)) ) / np.prod(b.shape)
    return rate
