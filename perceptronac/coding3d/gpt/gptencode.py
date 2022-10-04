import numpy as np


# https://stackoverflow.com/questions/18982650/differences-between-matlab-and-numpy-and-pythons-round-function
matlab_round = np.vectorize(round)



def ac_lapl_rate(y, sd, Q=40):

    sinh = lambda x : ( 1/2 * (np.exp(x) - np.exp(-x)) )

    gt0 = np.exp( - (np.abs(y) * Q * np.sqrt(2)) / sd ) * sinh( Q / (np.sqrt(2) * sd ) )

    eq0 = 1 - np.exp( - Q / (np.sqrt(2) * sd ) )

    p = (y == 0).astype(int) * eq0 + (y != 0).astype(int) * gt0

    return -np.mean(np.log2(p))



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
    
    for n in range(ncubes):

        # get the voxels in the cube
        supx = (cubes[n,0] + 1) * block_side
        supy = (cubes[n,1] + 1) * block_side
        supz = (cubes[n,2] + 1) * block_side
        infx = supx-block_side
        infy = supy-block_side
        infz = supz-block_side
        vi = (V[:,0]<=supx)*(V[:,1]<=supy)*(V[:,2]<=supz)*(V[:,0]>=infx)*(V[:,1]>=infy)*(V[:,2]>=infz)
        
        Vb = V[vi,:]
        Cb = C[vi,:]-128

        # calculate distances among all voxels
        N = Vb.shape[0]

        dij = np.sqrt(np.sum((np.expand_dims(Vb,1) - np.expand_dims(Vb,0))**2,axis=2))
        Rxx = rho**dij
        L, W = np.linalg.eig(Rxx)
        W = -W.real.T
        lambdas = np.expand_dims(L.real,1)

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



    NBINS = 70
    BITSPERMAXLAMBDA = 5
    BITSPERBIN = 60

    # create eigenvalue bins 
    lambdas = S[:,4]
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
    rateY = ac_lapl_rate(S[:,1], sv[:,1])
    rateU = ac_lapl_rate(S[:,2], sv[:,2])
    rateV = ac_lapl_rate(S[:,3], sv[:,3])
    ratet = ratet + rateY + rateU + rateV



    # final Rate Distortion numbers
    mse = mse / Nvox; 
    dist = 10 * np.log10(255*255/mse)
    rate = ratet / Nvox

    return rate,dist



if __name__ == "__main__":

    from perceptronac.coding3d import read_PC

    _,V,C = read_PC("/home/lucas/Documents/data/frame0039.ply")

    rate,dist = gptencode(V,C)

    print(rate,dist)