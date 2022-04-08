# function rate = perfect_AC(b, p)
# rate = mean(- log2(p) .* (b == 1) - log2(1 - p)  .* (b == 0));

import numpy as np

def perfect_AC(b, p):
    rate = np.mean(
        - np.log2(p) * (b == 1).astype(int) - np.log2(1 - p)  * (b == 0).astype(int))
    return rate