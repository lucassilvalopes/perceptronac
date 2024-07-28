
import numpy as np

def causal_context(img,N):
    """
    Causal contexts and samples for 1 channel images.
    The context is composed of the N closest pixels in the causal neighborhood.
    Works with grayscale image, binary image or 1 channel from a color image.
    For binary image, pass in a matrix of 0s and 1s and type int.
    For grayscale image or 1 channel from a color image, 
    pass in a matrix with int values in the range [0,255].
    Samples for which a complete context cannot be obtained are removed.
    Those are the pixels within a distance of ns from the top,left or right borders, 
    where ns is the radius of the circle needed to contain the N causal samples, 
    which is (approximately,overestimated) calculated as int(np.ceil(np.sqrt(N))).
    
    Args:
        img (nr by nc) : 1 channel image passed as a matrix (2D tensor)
        N : context size

    Returns:
        X (nr-ns by nc-2*ns,N) : matrix with one causal context in each line
        y (nr-ns by nc-2*ns,1) : vector with samples
    """

    ns = int(np.ceil(np.sqrt(N)))
    Im = np.arange(-ns,1).reshape((-1,1),order='F') @ np.ones((1,2*ns+1))
    Jm = np.ones((ns+1,1)) * np.arange(-ns,ns+1).reshape((1,-1),order='F')
    Jm[ns,ns:2*ns+1] = 0
    iv = Im.astype(int).reshape((-1,1),order='F')
    jv = Jm.astype(int).reshape((-1,1),order='F')
    d = np.sqrt(0.99*jv*jv+iv*iv)
    ind = np.argsort(d, axis=0, kind='mergesort')

    # this is nice for debugging 
    cord = np.zeros((ns+1,2*ns+1))
    for k in range(N):
        ki = ns + 1 + k
        cord[ns+iv[ind[ki,0],0], ns+jv[ind[ki,0],0]] = k+1
    print(cord)

    S = np.zeros((2,N),dtype=int)
    for k in range(N):
        S[0,k] = iv[ind[ns+1+k,0],0]
        S[1,k] = jv[ind[ns+1+k,0],0]
    # img = (img > 0).astype(int)
    nr,nc = img.shape 
    imgy = img[ns:nr,ns:nc-ns]
    y = imgy.reshape((-1,1),order='F')
    Npixels = len(y)
    X = np.zeros((Npixels, N),dtype=int)
    for k in range(N):
        imgx = img[ns+S[0,k]:nr+S[0,k],ns+S[1,k]:nc-ns+S[1,k]]
        X[:,k] = imgx.reshape((-1),order='F')
    return y,X


# function [y, X] = causal_context(img, N)
# ns  =  ceil(sqrt(N));
# Im = (-ns:0)' * ones(1,2*ns+1);
# Jm = ones(ns+1,1) * (-ns:ns);
# Jm(ns+1,ns+1:2*ns+1) = 0;
# iv = Im(:);
# jv = Jm(:);
# d = sqrt(0.99*jv.*jv+iv.*iv);
# [~,ind] = sort(d);
# S = zeros(2,N);
# for k = 1:N
#     S(1,k) = iv(ind(ns+1+k));
#     S(2,k) = jv(ind(ns+1+k));
# end
# img = (img > 0);
# [nr, nc] = size(img); 
# imgy = img(ns+1:nr,ns+1:nc-ns);
# y = imgy(:);
# Npixels = length(y);
# X = zeros(Npixels, N);
# for k = 1:N
#     imgx = img(ns+1+S(1,k):nr+S(1,k),ns+1+S(2,k):nc-ns+S(2,k));
#     X(:,k) = imgx(:);
# end