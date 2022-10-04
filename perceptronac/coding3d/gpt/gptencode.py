import numpy as np


# https://stackoverflow.com/questions/18982650/differences-between-matlab-and-numpy-and-pythons-round-function
matlab_round = np.vectorize(round)


def segment_pc(V, block_side):

    max_octree_level = np.ceil(np.log2(np.max(V.reshape(-1))+1))

    side = 2**max_octree_level

    secn = np.zeros(V.shape[0],1)

    i = 0
    for x in range(0,side,block_side):
        for y in range(0,side,block_side):
            for z in range(0,side,block_side):

                block_mask = np.logical_and(
                    np.logical_and(
                        np.logical_and(V[:,0]>=x, V[:,0]<x+block_side),
                        np.logical_and(V[:,1]>=y, V[:,1]<y+block_side)
                    ),
                    np.logical_and(V[:,2]>=z, V[:,2]<z+block_side)
                )

                if not np.all(np.logical_not(block_mask)):

                    secn[block_mask,:] = i

                    i=i+1

    return secn



def ac_lapl_rate(y, sd, Q=40):

    sinh = lambda x : ( 1/2 * (np.exp(x) - np.exp(-x)) )

    gt0 = np.exp( - (np.abs(y) * Q * np.sqrt(2)) / sd ) * sinh( Q / (np.sqrt(2) * sd ) )

    eq0 = 1 - np.exp( - Q / (np.sqrt(2) * sd ) )

    p = (y == 0).astype(int) * eq0 + (y != 0).astype(int) * gt0

    return -np.mean(np.log2(p))



def gptencode(V,C,Q=40,block_side=8,rho=0.95):

    # loop and encode blocks
    Nvox = C.shape[0]
    Crec = np.zeros((Nvox,3))
    p = 0
    mse = 0
    S = np.zeros((Nvox,4))


    secn = segment_pc(V, block_side)
    ncubes = max(secn)+1
    
    for n in range(ncubes):

        # get the voxels in the section
        Vb = V[secn == n, :]
        Cb = C[secn == n, :] - 128

        # calculate distances among all voxels
        N = Vb.shape[0]

        dij = np.sqrt(np.sum((np.expand_dims(Vb,1) - np.expand_dims(Vb,0))**2,axis=2))
        Rxx = rho**dij
        lambdas, W = np.linalg.eig(Rxx)
        W = -W.T

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



