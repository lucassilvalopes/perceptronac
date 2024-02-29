
from PIL import Image
import numpy as np
import sys

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


def add_border(img,N):
    mx = np.max(img)
    ns = int(np.ceil(np.sqrt(N)))
    nr,nc = img.shape[:2]
    new_img = mx*np.ones((nr+ns,nc+2*ns),dtype=img.dtype)
    new_img[ns:nr+ns,ns:nc+2*ns-ns] = img.copy()
    return new_img


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

    # p = np.clip(p1,1,None) / (np.clip(p1,1,None) + np.clip(p0,1,None))
    # p[np.logical_and(p0 != 0,p1 == 0)]=(0 + np.finfo(p.dtype).eps)
    # p[np.logical_and(p0 == 0,p1 != 0)]=(1 - np.finfo(p.dtype).eps)

    return np.hstack([p0,p1])


if __name__ == "__main__":

    N = 2

    if len(sys.argv)>1:
        pth = sys.argv[1] # /path/to/file.png

        img = np.array(Image.open(pth))
    else:
        img = np.array([
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,1,0,0,0,0,1,1,1],
            [1,1,0,1,1,1,1,0,1,1],
            [1,0,1,0,1,1,0,1,0,1],
            [1,0,1,1,1,1,1,1,0,1],
            [1,0,1,0,1,1,0,1,0,1],
            [1,0,1,1,0,0,1,1,0,1],
            [1,1,0,1,1,1,1,0,1,1],
            [1,1,1,0,0,0,0,1,1,1],
            [1,1,1,1,1,1,1,1,1,1],
        ])

    img = add_border(img,N)
    y,X = causal_context((img > 0).astype(int), N)

    p = context_training(X,y)

    print(p)