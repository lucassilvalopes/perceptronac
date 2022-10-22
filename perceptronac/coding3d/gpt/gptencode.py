import numpy as np
from tqdm import tqdm
import torch


class Model(torch.nn.Module):
    def __init__(self):
        N=1
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, 64*N),
            torch.nn.ReLU(),
            torch.nn.Linear(64*N, 32*N),
            torch.nn.ReLU(),
            torch.nn.Linear(32*N, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)


class LaplacianRate(torch.nn.Module):
    def forward(self, pred, target):
        """
        target xq, S[:,0]
        pred sdnz

        obs: negative standard deviation makes no sense
        """

        two = torch.tensor(2,dtype=target.dtype,device=target.device)
        
        rgt0 = (1/torch.log(two)) * ( (torch.sqrt(two) * torch.abs(target)) / pred) - torch.log2( torch.sinh(1/(torch.sqrt(two) * pred) ) )
        
        r0 = -torch.log2(1-torch.exp(-1/(torch.sqrt(two) * pred)))

        rgt0_mask = torch.abs(target) > 0

        r0_mask = torch.abs(target) == 0
        
        rateac = torch.sum(rgt0[rgt0_mask]) + torch.sum(r0[r0_mask])
        
        return rateac


class NNModel:

    def __init__(self):
        self.model = Model()

    def train(self,S):
        return self._apply(S,"train")

    def validate(self,S):
        return self._apply(S,"valid")

    def _apply(self,S, phase):

        device = torch.device("cuda:0")

        model = self.model
        model.to(device)

        criterion = LaplacianRate()
        OptimizerClass=torch.optim.SGD
        optimizer = OptimizerClass(model.parameters(), lr=0.0001)

        batch_size = 1024

        dset = torch.utils.data.TensorDataset(torch.tensor(S[:,3:4]),torch.tensor(S[:,0:1]))
        dataloader = torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=False)

        if phase == 'train':
            model.train(True)
        else:
            model.train(False) 

        running_loss = 0.0
        n_samples = 0.0

        pbar = tqdm(total=np.ceil(len(dset)/batch_size))
        for data in dataloader:

            Xt_b,yt_b= data
            Xt_b = Xt_b.float().to(device)
            yt_b = yt_b.float().to(device)

            if phase == 'train':
                optimizer.zero_grad()
                outputs = model(Xt_b)
                loss = criterion(outputs, yt_b)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(Xt_b)
                    loss = criterion(outputs, yt_b)

            running_loss += loss.item()
            n_samples += yt_b.numel()
            pbar.update(1)
            pbar.set_description(f"loss: {running_loss / n_samples}")
        pbar.close()

        return running_loss / n_samples


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



def permutation_selection_matrices(Vb,block_side,infx,infy,infz):
    
    Nvox = Vb.shape[0]
    v_sorting_indices = np.lexsort((Vb[:, 2]-infz, Vb[:, 1]-infy, Vb[:, 0]-infx))
    Pb1 = np.eye(Nvox)
    Pb1 = Pb1[v_sorting_indices]

    ny,nx,nz = block_side,block_side,block_side
    infxyz = np.array([[infx,infy,infz]])
    mask = np.zeros((ny*nx*nz),dtype=int)
    mask[[int(y*nx*nz+x*nz+z) for x,y,z in Pb1 @ (Vb-infxyz)]] = 1
    Pb2 = np.eye(ny*nx*nz)
    Pb2 = Pb2[mask.astype(bool)]

    return Pb1,Pb2


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

        Pb1,Pb2 = permutation_selection_matrices(Vb,block_side,infx,infy,infz)

        # Vb = Pb1.T @ (Pb2 @ (Pb2.T @ (Pb1 @ Vb)))
        # Cb = Pb1.T @ (Pb2 @ (Pb2.T @ (Pb1 @ Cb)))
        Vb = Pb1 @ Vb
        Cb = Pb1 @ Cb


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

    # final Rate Distortion numbers
    mse = mse / Nvox; 
    dist = 10 * np.log10(255*255/mse)

    return S,dist


def lut(S):
    """
    S: Nvox-by-4, with the YUV coefficients in first 3 columns and lambdas in the last column 
    """

    Nvox = S.shape[0]

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
    rate = ratet / Nvox

    return rate


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

    S,dist = gptencode(V,C)
    # rate = lut(S)

    nnmodel = NNModel()
    epochs = 10
    for epoch in range(epochs):
        rate = nnmodel.train(S)

    print(rate,dist)

    # import scipy.io
    # scipy.io.savemat('ricardo9_frame0039_yuv.mat', dict(V=V+1, C=C))