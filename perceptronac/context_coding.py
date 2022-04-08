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

def context_coding(X, y, context_p):
    L,N = X.shape
    X = (X > 0).astype(int)
    po2 = 2 ** np.arange(0,N).reshape(-1,1)
    context = X @ po2

    pp = np.zeros(y.shape) 
    for k in range(L):
        pp[k,0] = context_p[context[k,0],0]
    # r = perfect_AC(y, pp)
    return pp

