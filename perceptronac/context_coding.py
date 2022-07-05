# function r = context_coding(X, y, context_p) 

# [L,N] = size(X); 
# po2 = (2.^(0:N-1))';
# X = (X > 0);
# context = X * po2; 
# pp = zeros(size(y)); 
# for k = 1:L
#     pp(k) = context_p(context(k) + 1) ;
# end
# r = perfect_AC(y, pp);

import numpy as np
from tqdm import tqdm

def context_coding(X, context_p):
    L,N = X.shape
    X = (X > 0).astype(int)
    po2 = 2 ** np.arange(0,N).reshape(-1,1)
    context = X @ po2

    pp = np.zeros((L,1)) 
    for k in range(L):
        pp[k,0] = context_p[context[k,0],0]
    # r = perfect_AC(y, pp)
    return pp


def context_coding_nonbinary(X,table,contexts):

    n_channels = table.shape[2]
    n_symbols = 256
    n_predictions = X.shape[0]

    predictions = np.ones((n_predictions,n_symbols,n_channels))/256
    for i in tqdm(range(n_predictions),desc="coding with lut"):
        mask = np.all(contexts - X[i:i+1,:] == 0,axis=1)
        if np.any(mask):
            predictions[i:i+1,:,:] = table[mask,:,:]
    return predictions