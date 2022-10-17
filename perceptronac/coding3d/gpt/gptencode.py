import numpy as np
from tqdm import tqdm

# https://stackoverflow.com/questions/18982650/differences-between-matlab-and-numpy-and-pythons-round-function
matlab_round = np.vectorize(round)


def ac_lapl_rate(xq, sd):
    """
    In the paper there was a Q in the exponent of the exponentials, 
    because the standard deviation was of dequantized coefficients.
    Here the standard deviation is of quantized coefficients. 
    If the standard deviation of quantized coefficients is sd, 
    then the standard deviation of dequantized coefficients is Q*sd. 
    Replacing the standard deviation of dequantized coefficients by Q*sd
    in the paper, gives the formulas used here.

    The mean of the coefficients of any bin is 0, regardless of the standard deviation.
    If the standard deviation is 0, this means the coefficients in the bin are 0.
    If they are zero, they do not need to be sent, they can simply be discarded, ignored.
    """

    nz = sd > 1e-6 
    
    xqa = np.abs(xq[nz])
    
    sdnz = sd[nz]
    
    rgt0 = (1/np.log(2)) * ( (np.sqrt(2) * xqa) / sdnz) - np.log2( np.sinh(1/(np.sqrt(2) * sdnz) ) )
    
    r0 = -np.log2(1-np.exp(-1/(np.sqrt(2) * sdnz)))
    
    r = rgt0 * (xqa > 0).astype(int) + r0 * (xqa == 0).astype(int)
    
    rateac = np.sum(r)
    
    return rateac


def gptencode(V,C,Q=40,block_side=8,rho=0.95):

    # see how many blocks are there
    cubes = np.unique(np.floor(V/block_side),axis=0)
    ncubes = cubes.shape[0]

    # loop and encode blocks
    Nvox = C.shape[0]
    Crec = np.zeros((Nvox,3))
    p = 0
    mse = 0
    S = np.zeros((Nvox,4))
    
    pbar = tqdm(total=ncubes)
    for n in range(ncubes):

        # get the voxels in the cube
        supx = (cubes[n,0] + 1) * block_side
        supy = (cubes[n,1] + 1) * block_side
        supz = (cubes[n,2] + 1) * block_side
        infx = supx-block_side
        infy = supy-block_side
        infz = supz-block_side
        vi = (V[:,0]<supx)*(V[:,1]<supy)*(V[:,2]<supz)*(V[:,0]>=infx)*(V[:,1]>=infy)*(V[:,2]>=infz)
        
        Vb = V[vi,:]
        Cb = C[vi,:]-128

        # calculate distances among all voxels
        N = Vb.shape[0]

        dij = np.sqrt(np.sum((np.expand_dims(Vb,1) - np.expand_dims(Vb,0))**2,axis=2))
        Rxx = rho**dij
        _, s, vh = np.linalg.svd(Rxx, full_matrices=True)
        W = -vh.real
        lambdas = np.expand_dims(s.real,1)

        # transform and quantize
        # Q is a quantizer step (10 or 40 for example)
        yb = W @ Cb
        yq = matlab_round(yb / Q); 

        # output data for encoding
        # yq is Nx3, lambdas is Nx1
        S[p:p+N,:] = np.concatenate([yq , np.sqrt(lambdas)],axis=1) 

        # inverse quantize and inverse transform
        Cbr = W.T @ (yq * Q)
        e = Cb[:,0:1]-Cbr[:,0:1] # Y channel
        mse = mse + e.T @ e
        Crec[p:p+N,:] = Cbr+128
        p = p + N

        pbar.update(1)
    pbar.close()

    NBINS = 70
    BITSPERMAXLAMBDA = 5
    BITSPERBIN = 60

    # create eigenvalue bins 
    lambdas = S[:,3]
    maxlambda = np.ceil(np.max(lambdas)) # we convey this to the decoder
    lambdastep = maxlambda / NBINS
    lambdaq = matlab_round(lambdas / lambdastep)
    lambdaq[lambdaq == 0] = 1 # the occasional zeros are moved to the 1st bin


    # calculate the standard deviation of each eigenvalue bin 
    sdsum = np.zeros((NBINS,3))
    sdcount = np.zeros((NBINS,1))
    for n in range(Nvox):
        sdsum[lambdaq[n]-1,:] = sdsum[lambdaq[n]-1,:] + S[n,:3]*S[n,:3]
        sdcount[lambdaq[n]-1] = sdcount[lambdaq[n]-1] + 1

    sdcount[sdcount==0] = 1
    sdsum[sdsum==0] = 1
    sdbin = np.sqrt(sdsum / np.tile(sdcount,(1,3)) )


    # quantize the std deviations
    # sdstep is Qs in the paper
    sdstep = 1
    sdquant = matlab_round(sdbin / sdstep)
    sdrec = sdquant * sdstep

    # find rates for bin variances
    ratet = BITSPERMAXLAMBDA + BITSPERBIN * NBINS


    # create vectors with estimated stddev for each color
    sv = np.zeros((Nvox,3))
    for n in range(Nvox):
        sv[n,0] = sdbin[lambdaq[n]-1,0]
        sv[n,1] = sdbin[lambdaq[n]-1,1]
        sv[n,2] = sdbin[lambdaq[n]-1,2]


    # rate for encoding coefficients 
    rateY = ac_lapl_rate(S[:,0], sv[:,0])
    rateU = ac_lapl_rate(S[:,1], sv[:,1])
    rateV = ac_lapl_rate(S[:,2], sv[:,2])
    ratet = ratet + rateY + rateU + rateV



    # final Rate Distortion numbers
    mse = mse / Nvox; 
    dist = 10 * np.log10(255*255/mse)
    rate = ratet / Nvox

    return rate,dist



def rgb2yuv(rgb):
    """
    https://github.com/python-pillow/Pillow/issues/4668
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
    """
    yuv = np.zeros(rgb.shape)
    yuv[:,0] = rgb[:,0] *  0.29900 + rgb[:,1] *  0.58700 + rgb[:,2] *  0.11400 
    yuv[:,1] = rgb[:,0] * -0.16874 + rgb[:,1] * -0.33126 + rgb[:,2] *  0.50000 + 128 
    yuv[:,2] = rgb[:,0] *  0.50000 + rgb[:,1] * -0.41869 + rgb[:,2] * -0.08131 + 128 
    return yuv



if __name__ == "__main__":

    from perceptronac.coding3d import read_PC

    _,V,C = read_PC("/home/lucas/Documents/data/ricardo9_frame0039.ply")

    C = rgb2yuv(C)

    rate,dist = gptencode(V,C)

    print(rate,dist)

    # import scipy.io
    # scipy.io.savemat('ricardo9_frame0039_yuv.mat', dict(V=V+1, C=C))