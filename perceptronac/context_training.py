# function p = context_training(X, y) 
# [L,N] = size(X); 
# X = (X > 0);
# po2 = (2.^(0:N-1))';
# context = X * po2; 
# p1 = zeros(2^N,1);
# p0 = zeros(2^N,1);
# for k=1:L
#     if (y(k) == 1) 
#         p1(context(k)+1) = p1(context(k)+1) + 1; 
#     else
#         p0(context(k)+1) = p0(context(k)+1) + 1; 
#     end
# end
# p1 = p1 + (p1 == 0);
# p0 = p0 + (p0 == 0);
# p = p1 ./ (p1 + p0); 

import numpy as np

def context_training(X,y,max_context=20):
    L,N = X.shape
    if N > max_context:
        m=f"max_context is {max_context} but X.shape[1] is {N}"
        raise ValueError(m)
    X = (X > 0).astype(int)
    po2 = 2 ** np.arange(0,N).reshape(-1,1)
    context = X @ po2
    p1 = np.zeros((2**N,1))
    p0 = np.zeros((2**N,1))
    for k in range(L):
        if (y[k,0] == 1):
            p1[context[k,0],0] = p1[context[k,0],0] + 1
        else:
            p0[context[k,0],0] = p0[context[k,0],0] + 1

    # p1 = p1 + (p1 == 0).astype(int)
    # p0 = p0 + (p0 == 0).astype(int)
    # p = p1 / (p1 + p0)

    p = np.clip(p1,1,None) / (np.clip(p1,1,None) + np.clip(p0,1,None))
    p[np.logical_and(p0 != 0,p1 == 0)]=(0 + np.finfo(p.dtype).eps)
    p[np.logical_and(p0 == 0,p1 != 0)]=(1 - np.finfo(p.dtype).eps)

    return p