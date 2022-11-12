import numpy as np
from tqdm import tqdm
import torch
import random
import sys
import os
import scipy.io
from perceptronac.coding3d import read_PC
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self,N): 
        super().__init__()
        self.max_sd = 0

        a1_i = N
        a1_o = a2_i = min(8192,64*N) # min(2048,64*N)
        a2_o = a3_i = min(4096,32*N) # min(1024,32*N)
        a3_o = 1

        self.a1 = torch.nn.Linear( a1_i, a1_o )
        self.a1_act = torch.nn.ReLU()
        self.a2 = torch.nn.Linear( a2_i , a2_o )
        self.a2_act = torch.nn.ReLU()
        self.a3 = torch.nn.Linear( a3_i , a3_o )
        self.a3_act = torch.nn.Sigmoid()

        # self.b1 = torch.nn.Linear(N, min(1024,32*N) )
        # self.b1_act = torch.nn.ReLU()
        # self.b2 = torch.nn.Linear( min(1024,32*N), 1)
        # self.b2_act = torch.nn.ReLU()

    def forward(self, x):
        xa = self.a1(x)
        xa = self.a1_act(xa)
        xa = self.a2(xa)
        xa = self.a2_act(xa)
        xa = self.a3(xa)
        xa = self.a3_act(xa)

        # xb = self.b1(x)
        # xb = self.b1_act(xb)
        # xb = self.b2(xb)
        # xb = self.b2_act(xb)

        # return 0.01 + xa * (1 + xb )

        return 0.01 + self.max_sd * xa


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

    def __init__(self,configs,N):
        # seed = 7
        # torch.manual_seed(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        self.model = Model(N)
        self.lr = configs["learning_rate"]
        self.batch_size = configs["batch_size"]

    def train(self,S):
        self.model.train()
        return self._apply(S,"train")

    def validate(self,S):
        self.model.eval()
        return self._apply(S,"valid")

    def _apply(self,S, phase):

        device = torch.device("cuda:0")

        model = self.model
        model.to(device)

        criterion = LaplacianRate()
        OptimizerClass=torch.optim.SGD
        optimizer = OptimizerClass(model.parameters(), lr=self.lr)

        dset = torch.utils.data.TensorDataset(torch.tensor(S[:,3:]),torch.tensor(S[:,0:1]))
        dataloader = torch.utils.data.DataLoader(dset,batch_size=self.batch_size,shuffle=True)

        if phase == 'train':
            model.train(True)
        else:
            model.train(False) 

        running_loss = 0.0
        n_samples = 0.0

        pbar = tqdm(total=np.ceil(len(dset)/self.batch_size))
        for data in dataloader:

            Xt_b,yt_b= data
            Xt_b = Xt_b.float().to(device)
            yt_b = yt_b.float().to(device)

            if phase == 'train':
                optimizer.zero_grad()
                model.max_sd = max([model.max_sd,torch.max(torch.abs(yt_b.detach())).item()])
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
            pbar.set_description(f"loss: {running_loss / n_samples} max_sd: {model.max_sd}")
        pbar.close()

        return running_loss / n_samples , n_samples


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


def gpt(pth,Q=40,block_side=8,rho=0.95):
    """
    Applies gaussian process transform to point cloud blocks,
    using the Ornstein-Uhlenbeck model to estimate the covariance matrix,
    then uniformly quantize the coefficients.

    Args:
        pth : path to .ply file containing a voxelized point cloud with rgb attributes 
        Q : step-size for the uniform quantization of the coefficients
        block_side : block side
        rho : parameter of the Ornstein-Uhlenbeck model

    Returns:
        S : Nvox-by-4 matrix where the first 3 columns contain the coefficients of the gpt applied to the Y,U,V 
            channels, and the last column contains the eigenvalues
        dist : peak signal to noise ratio in dB
        Evec : Nvox-by-block_side**3 matrix of eigenvectors zero-filled at unoccupied positions to size block_side**3
    """

    _,V,C = read_PC(pth)

    C = rgb2yuv(C)

    # see how many blocks are there
    cubes = np.unique(np.floor(V/block_side),axis=0)
    ncubes = cubes.shape[0]

    # loop and encode blocks
    Nvox = C.shape[0]
    Crec = np.zeros((Nvox,3))
    p = 0
    mse = 0
    S = np.zeros((Nvox,4))
    Evec = np.zeros((Nvox,block_side**3))
    pos = np.zeros((Nvox,1))

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

        Evec[p:p+N,:] = W @ Pb2 # (Pb2.T @ W.T).T

        pos[p:p+N,:] = np.arange(0,N).reshape(-1,1)

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

    return {
        "S":S,
        "dist":dist.item(),
        "Evec": Evec,
        "pos": pos
    }


def lut(gpt_return):
    """
    S: Nvox-by-4, with the YUV coefficients in first 3 columns and lambdas in the last column 
    """

    S = gpt_return["S"]

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
    # ratet = ratet + rateY + rateU + rateV
    ratet = rateY

    # final Rate Distortion numbers
    rate = ratet / Nvox


    pos = gpt_return["pos"]
    bits_y_per_coef_idx = []
    bits_u_per_coef_idx = []
    bits_v_per_coef_idx = []
    samples_per_coef_idx = []
    for i in range(int(np.max(pos))+1):
        mask_i = (pos == i).reshape(-1)
        bits_y_per_coef_idx.append( ac_lapl_rate(S[mask_i,0],sv[mask_i,0]) )
        bits_u_per_coef_idx.append( ac_lapl_rate(S[mask_i,1],sv[mask_i,1]) )
        bits_v_per_coef_idx.append( ac_lapl_rate(S[mask_i,2],sv[mask_i,2]) )
        samples_per_coef_idx.append( np.sum(mask_i) )



    return {
        "rate":rate,
        "sv":sv,
        "bits_y_per_coef_idx":bits_y_per_coef_idx,
        "bits_u_per_coef_idx":bits_u_per_coef_idx,
        "bits_v_per_coef_idx":bits_v_per_coef_idx,
        "samples_per_coef_idx":samples_per_coef_idx

    }


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



def rd_curve(rates_lut,rates_nn,distortions):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    handle1,= ax.plot(rates_lut,distortions,linestyle="dotted",marker="^",color="green",label="LUT")
    handle2,= ax.plot(rates_nn,distortions,linestyle="solid",marker="o",color="blue",label="NN")

    ax.set_xlabel("Rate Y (bpv)")
    ax.set_ylabel("PSNR Y (db)")

    ax.legend(handles=[handle1,handle2])

    fig.savefig(f"gpt_nn.png", dpi=300, facecolor='w', bbox_inches = "tight")




if __name__ == "__main__":

    ################ I used these lines of code for some tests (remove later) ################

    if len(sys.argv) > 1 and sys.argv[1] == "0":

        _,V,C = read_PC("/home/lucas/Documents/data/ricardo9_frame0039.ply")
        C = rgb2yuv(C)
        scipy.io.savemat('ricardo9_frame0039_yuv.mat', dict(V=V+1, C=C))
        sys.exit()

    elif len(sys.argv) > 1 and sys.argv[1] == "1":

        gpt_return = gpt("/home/lucas/Documents/data/ricardo9_frame0039.ply")
        S,dist,Evec = gpt_return["S"],gpt_return["dist"],gpt_return["Evec"]
        lut_return = lut(gpt_return)
        rate,sv = gpt_return["rate"],gpt_return["sv"] 
        print(np.min(S[:,:3]),np.max(S[:,:3]))
        print(np.min(sv),np.max(sv))

        criterion = LaplacianRate()
        x_axis = np.linspace(0,100,10)
        x_axis = np.concatenate([np.array([0.000001,0.00001,0.0001,0.001,0.01,0.1,1]),x_axis],axis=0)
        y_axis = []
        for pred in x_axis:
            y_axis.append( criterion( torch.tensor(pred),torch.tensor(1) ).item() )
        print(y_axis)
        sys.exit()

    elif len(sys.argv) > 1 and sys.argv[1] == "2":

        Q = 40
        
        for r,ds,fs in os.walk("/home/lucas/Documents/data/GPT/training"):
            for f in fs: 
                if f.endswith(".ply"):
                    pth = os.path.join(r,f) 

                    gpt_return = gpt(pth,Q=Q)
                    S,dist,Evec = gpt_return["S"],gpt_return["dist"],gpt_return["Evec"]

                    npz_pth = f"{pth.rstrip('.ply')}_Q{Q}_blocksize8_rho95e-2_contexts.npz"

                    np.savez_compressed(npz_pth,np.concatenate([S,Evec],axis=1))

        sys.exit()

    elif len(sys.argv) > 1 and sys.argv[1] == "3":

        gpt_return = gpt("/home/lucas/Documents/data/ricardo9_frame0039.ply")
        lut_return = lut(gpt_return)

        bits_y_per_coef_idx = np.array(lut_return["bits_y_per_coef_idx"])
        bits_u_per_coef_idx = np.array(lut_return["bits_u_per_coef_idx"])
        bits_v_per_coef_idx = np.array(lut_return["bits_v_per_coef_idx"])
        samples_per_coef_idx = np.array(lut_return["samples_per_coef_idx"])

        fig, ax = plt.subplots(nrows=1, ncols=3)

        x_axis = np.arange(samples_per_coef_idx.shape[0])


        ax[0].plot(x_axis,bits_y_per_coef_idx/np.sum(bits_y_per_coef_idx))
        ax[1].plot(x_axis,bits_u_per_coef_idx/np.sum(bits_u_per_coef_idx))
        ax[2].plot(x_axis,bits_v_per_coef_idx/np.sum(bits_v_per_coef_idx))

        # ax[0].plot(x_axis,bits_y_per_coef_idx/samples_per_coef_idx)
        # ax[1].plot(x_axis,bits_u_per_coef_idx/samples_per_coef_idx)
        # ax[2].plot(x_axis,bits_v_per_coef_idx/samples_per_coef_idx)

        ax[0].set_xlabel("Coefficient index")
        ax[0].set_ylabel("Rate (bpv)")
        ax[0].set_title(f"Y (DC: {bits_y_per_coef_idx[0]/np.sum(bits_y_per_coef_idx):.2f})")
        ax[1].set_xlabel("Coefficient index")
        ax[1].set_ylabel("Rate (bpv)")
        ax[1].set_title(f"U (DC: {bits_u_per_coef_idx[0]/np.sum(bits_u_per_coef_idx):.2f})")
        ax[2].set_xlabel("Coefficient index")
        ax[2].set_ylabel("Rate (bpv)")
        ax[2].set_title(f"V (DC: {bits_v_per_coef_idx[0]/np.sum(bits_v_per_coef_idx):.2f})")

        fig.savefig(f"rate_per_coef_idx.png", dpi=300, facecolor='w', bbox_inches = "tight")

        print(gpt_return["dist"])
        print(lut_return["rate"])

        sys.exit()

    ##########################################################################################


    configs = {
        "training_set": [
            # "/home/lucas/Documents/data/david10_frame0115.ply"
            # "/home/lucas/Documents/data/david9_frame0115.ply"
            "/home/lucas/Documents/data/GPT/training/sarah9/frame0180_Q40_blocksize8_rho95e-2_contexts.npz",
            "/home/lucas/Documents/data/GPT/training/phil9/frame0050_Q40_blocksize8_rho95e-2_contexts.npz",
            "/home/lucas/Documents/data/GPT/training/andrew9/frame0240_Q40_blocksize8_rho95e-2_contexts.npz",
            "/home/lucas/Documents/data/GPT/training/david9/frame0120_Q40_blocksize8_rho95e-2_contexts.npz",
            # os.path.join(r,f) for r,ds,fs in os.walk("/home/lucas/Documents/data/GPT/training") for f in fs if f.endswith("Q40_blocksize8_rho95e-2_contexts.npz")
            
        ],
        "validation_set": [
            # "/home/lucas/Documents/data/ricardo10_frame0000.ply"
            # "/home/lucas/Documents/data/ricardo9_frame0000.ply"
            "/home/lucas/Documents/data/GPT/validation/ricardo9/ricardo9_frame0000_Q40_blocksize8_rho95e-2_contexts.npz"
        ],
        "outer_loop_epochs": 80,
        "inner_loop_epochs": 10,
        "learning_rate": 1e-5,
        "batch_size": 1024,
        "phases": ['train', 'valid'],
        "dset_pieces": 1,
        "N": 513
    }

    if configs["N"] not in [1,513]:
        raise Exception(f'Option N={configs["N"]} not available')


    for Q in [40]: # [10,20,30,40]:

        nnmodel = NNModel(configs,configs["N"])
        
        for outer_loop_epoch in range(configs["outer_loop_epochs"]):

            valid_rates = []
            valid_samples = []
            train_rates = []
            train_samples = []

            for phase in sorted(configs["phases"]):

                pths = configs["training_set"] if phase == "train" else configs["validation_set"]

                shuffled_pths = random.sample(pths, len(pths))

                pths_per_dset = max(1,len(shuffled_pths)//configs["dset_pieces"])

                for shuffled_pths_i in range(0,len(shuffled_pths),pths_per_dset):

                    piece_pths = shuffled_pths[shuffled_pths_i:(shuffled_pths_i+pths_per_dset)]

                    full_S = []
                    for pth in piece_pths:

                        if pth.endswith(".npz"):
                            dist = np.nan
                            if configs["N"] == 1:
                                full_S.append(np.load(pth)["arr_0"][:,:4])
                            elif configs["N"] == 513:
                                full_S.append(np.load(pth)["arr_0"])
                        else:
                            gpt_return = gpt(pth,Q=Q)
                            S,dist,Evec = gpt_return["S"],gpt_return["dist"],gpt_return["Evec"]
                            if configs["N"] == 1:
                                full_S.append( S )
                            elif configs["N"] == 513:
                                full_S.append( np.concatenate([S,Evec],axis=1) )

                    full_S = np.concatenate(full_S,axis=0)

                    if phase == "train":
                        for _ in range(configs["inner_loop_epochs"]):
                            t_rate,t_samples = nnmodel.train(full_S)
                        train_rates.append(t_rate)
                        train_samples.append(t_samples)
                    else:

                        v_rate,v_samples = nnmodel.validate(full_S)

                        valid_rates.append(v_rate)
                        valid_samples.append(v_samples)

                if phase == "train":
                    final_loss = np.sum( np.array(train_rates) * np.array(train_samples) ) / np.sum(train_samples) 
                else:
                    final_loss = np.sum( np.array(valid_rates) * np.array(valid_samples) ) / np.sum(valid_samples) 
                print("epoch :" , outer_loop_epoch, ", phase :", phase, ", loss :", final_loss)


        torch.save(nnmodel.model.eval().state_dict(), f"checkpoint_Q{Q}.pt")




