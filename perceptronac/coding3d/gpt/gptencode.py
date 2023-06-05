import numpy as np
from tqdm import tqdm
import torch
import random
import sys
import os
import scipy.io
from perceptronac.coding3d import read_PC, write_PC
from perceptronac.losses import LaplacianRate
from perceptronac.losses import ac_lapl_rate
import matplotlib.pyplot as plt
from pprint import pprint


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


@np.vectorize
def matlab_round(x):
    """
    Numpy's round tie breaking rule is to round x to the even integer nearest to x 
    if the fractional part of x is 0.5 (round half to even).

    Python's round tie breaking rule is the more common round half away from zero.
    That is, if the fractional part of x is 0.5 and x is positive, it is rounded up.
    If the fractional part of x is 0.5 and x is negative, it is rounded down.
    
    Python's round tie breaking rule is the one that agrees with Matlab's tie breaking rule.

    https://en.wikipedia.org/wiki/Rounding#Rounding_to_the_nearest_integer
    https://stackoverflow.com/questions/18982650/differences-between-matlab-and-numpy-and-pythons-round-function
    """
    return round(x)


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


def gpt(pth,Q=40,block_side=8,rho=0.95,dcs=None):
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

    cubes = cubes[np.lexsort((cubes[:, 2], cubes[:, 1], cubes[:, 0]))]

    if dcs is not None:
        for k,v in dcs.items():
            dcs[k] = v[np.lexsort((v[:, 2], v[:, 1], v[:, 0]))]

        mses = {k:0 for k,v in dcs.items()}

    ncubes = cubes.shape[0]

    # loop and encode blocks
    Nvox = C.shape[0]
    Crec = np.zeros((Nvox,3))
    p = 0
    mse = 0
    S = np.zeros((Nvox,4))
    Evec = np.zeros((Nvox,block_side**3))
    pos = np.zeros((Nvox,1))
    points = np.zeros((Nvox,3))
    colors = np.zeros((Nvox,3))

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

        points[p:p+N,:] = Vb # cubes[n:n+1,:]
        colors[p:p+N,:] = (W.T[:,0:1] @ yb[0:1,:])+128 # (W.T[:,0:1] @ (yq[0:1,:] * Q))+128

        # inverse quantize and inverse transform
        if dcs is not None:
            for k,v in dcs.items():
                Cbrk = W.T @ np.concatenate([(v[n:(n+1),3:]-128) / W[0,0],yq[1:,:]*Q],axis=0)
                ek = Cb[:,0:1]-Cbrk[:,0:1] # Y channel
                mses[k] = mses[k] + ek.T @ ek
        
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

    out = {
        "S":S,
        "dist":dist.item(),
        "Evec": Evec,
        "pos": pos,
        "points": points,
        "colors": colors
    }

    if dcs is not None:
        for k,v in mses.items():
            mses[k] = mses[k] / Nvox
            out[f"dist_{k}"] = (10 * np.log10(255*255/mses[k])).item()

    return out


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
    ratet_yuv = ratet + rateY + rateU + rateV
    ratet = rateY

    # final Rate Distortion numbers
    rate = ratet / Nvox
    rate_yuv = ratet_yuv / Nvox


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
        "rate_yuv": rate_yuv,
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


def yuv2rgb(yuv):
    """
    https://stackoverflow.com/questions/7041172/pils-colour-space-conversion-ycbcr-rgb
    """

    rgb = np.zeros(yuv.shape)
    rgb[:,0] = yuv[:,0] +                             + (yuv[:,2] - 128) *  1.40200
    rgb[:,1] = yuv[:,0] + (yuv[:,1] - 128) * -0.34414 + (yuv[:,2] - 128) * -0.71414
    rgb[:,2] = yuv[:,0] + (yuv[:,1] - 128) *  1.77200
    return rgb


def rd_curve(rates_lut,rates_nn,distortions):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    handle1,= ax.plot(rates_lut,distortions,linestyle="dotted",marker="^",color="green",label="LUT")
    handle2,= ax.plot(rates_nn,distortions,linestyle="solid",marker="o",color="blue",label="NN")

    ax.set_xlabel("Rate Y (bpv)")
    ax.set_ylabel("PSNR Y (db)")

    ax.legend(handles=[handle1,handle2])

    fig.savefig(f"gpt_nn.png", dpi=300, facecolor='w', bbox_inches = "tight")


def clip_colors(colors):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html
    https://stackoverflow.com/questions/16963956/difference-between-python-float-and-numpy-float32
    """
    colors = np.clip(colors, 0, 255)
    colors = (colors/255).astype(np.float32)
    return colors


def normalize_colors(colors,min_value,max_value):

    colors = colors - min_value
    colors = (colors/(max_value-min_value)).astype(np.float32)

    return colors


def denormalize_colors(colors,min_value,max_value):

    colors = (colors*(max_value-min_value))
    colors = colors + min_value

    return colors


def read_dcs_dec_info(path,sheet_name,pcs_column_name,pc_name,n_rates):

    df = pd.read_excel(path,engine='openpyxl',sheet_name=None)

    start_i = df[sheet_name][pcs_column_name].tolist().index(pc_name)

    df = df[sheet_name].iloc[start_i:(start_i+n_rates),:]

    return df


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

        import pandas as pd

        filepath = "/home/lucas/Documents/data/NNOC/validation/longdress/longdress_vox10_1300.ply"
        # filepath = "/home/lucas/Documents/data/ricardo10_frame0000.ply"

        filename = os.path.splitext(os.path.basename(filepath))[0]

        block_side = 8
        gpt_Q = 40

        dcs_identification = f"GPT_blocksize{block_side}_rho95e-2"
        acs_identification = f"GPT_Q{gpt_Q}_blocksize{block_side}_rho95e-2"

        dcs_dict = None
        dcs_dict = {
            "dcs_dec_dir" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/regptdcs/longdress",
            "dcs_enc_info" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/results/longdress_vox10_1300_GPT_blocksize8_rho95e-2_DC_YUV2RGB.csv",
            "dcs_dec_info" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/regptdcs/Resultados_PCs.xlsx",
            "dcs_dec_info_sheet_name" : "Longdress",
            "dcs_dec_info_pcs_column_name" : "Point Cloud",
            "dcs_dec_info_original_pc_name" : 'Longdress_1300 vox 10',
            "dcs_dec_info_pc_name" : 'Longdress_1300 vox 7',
            "dcs_dec_info_rate_col" : "Rate normalized vox 10 [bpov]",
            "dcs_dec_info_dist_col" : "PSNR_y [dB]",
            "dcs_dec_info_n_rates" : 7
        }

        # dcs_dict = {
        #     "dcs_dec_dir" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/regptdcs/ricardo",
        #     "dcs_enc_info" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/results/ricardo10_frame0000_GPT_blocksize8_rho95e-2_DC_YUV2RGB.csv",
        #     "dcs_dec_info" : "/home/lucas/Documents/perceptronac/perceptronac/coding3d/gpt/regptdcs/Resultados_PCs.xlsx",
        #     "dcs_dec_info_sheet_name" : "Ricardo",
        #     "dcs_dec_info_pcs_column_name" : "Point Cloud",
        #     "dcs_dec_info_original_pc_name" : 'Ricardo vox 10',
        #     "dcs_dec_info_pc_name" : 'Ricardo vox 7',
        #     "dcs_dec_info_rate_col" : "Rate normalized vox 10 [bpov]",
        #     "dcs_dec_info_dist_col" : "PSNR_y [dB]",
        #     "dcs_dec_info_n_rates" : 7
        # }

        if (dcs_dict is not None):
            dcs_enc_info = pd.read_csv(dcs_dict["dcs_enc_info"])
            dcs = dict()
            for fn in os.listdir(dcs_dict["dcs_dec_dir"]):
                _,V,C = read_PC( os.path.join(dcs_dict["dcs_dec_dir"],fn) )
                C = denormalize_colors(C/255,dcs_enc_info.loc[0,"min"],dcs_enc_info.loc[0,"max"])
                C = rgb2yuv(C)
                dcs[os.path.splitext(fn)[0]] = np.concatenate([V,C],axis=1)
        else:
            dcs = None

        gpt_return = gpt(filepath,Q=gpt_Q,block_side=block_side,dcs=dcs)        
        lut_return = lut(gpt_return)

        bits_y_per_coef_idx = np.array(lut_return["bits_y_per_coef_idx"])
        bits_u_per_coef_idx = np.array(lut_return["bits_u_per_coef_idx"])
        bits_v_per_coef_idx = np.array(lut_return["bits_v_per_coef_idx"])
        samples_per_coef_idx = np.array(lut_return["samples_per_coef_idx"])

        x_axis = np.arange(samples_per_coef_idx.shape[0])

        pd.DataFrame({
            "x_axis":x_axis,"fraction of bits":bits_y_per_coef_idx/np.sum(bits_y_per_coef_idx),
            "bits_y_per_coef_idx":bits_y_per_coef_idx,"samples_per_coef_idx":samples_per_coef_idx}).to_csv(f"results/{filename}_{acs_identification}_y.csv")
        pd.DataFrame({
            "x_axis":x_axis,"fraction of bits":bits_u_per_coef_idx/np.sum(bits_u_per_coef_idx),
            "bits_u_per_coef_idx":bits_u_per_coef_idx,"samples_per_coef_idx":samples_per_coef_idx}).to_csv(f"results/{filename}_{acs_identification}_u.csv")
        pd.DataFrame({
            "x_axis":x_axis,"fraction of bits":bits_v_per_coef_idx/np.sum(bits_v_per_coef_idx),
            "bits_v_per_coef_idx":bits_v_per_coef_idx,"samples_per_coef_idx":samples_per_coef_idx}).to_csv(f"results/{filename}_{acs_identification}_v.csv")
        
        dc_rate = (bits_y_per_coef_idx[0] + bits_u_per_coef_idx[0] + bits_v_per_coef_idx[0]) / gpt_return["points"].shape[0]
        ac_rate = (np.sum(bits_y_per_coef_idx[1:]) + np.sum(bits_u_per_coef_idx[1:]) + np.sum(bits_v_per_coef_idx[1:])) / gpt_return["points"].shape[0]
        pd.DataFrame({
            "component": ["DC","AC","total_rate","distortion"],
            "bpp": [
                dc_rate,
                ac_rate,
                lut_return["rate_yuv"],
                gpt_return["dist"]
            ]}
        ).to_csv(f"results/{filename}_{acs_identification}_yuv.csv")



        if (dcs_dict is not None):

            dcs_dec_info = read_dcs_dec_info(
                dcs_dict["dcs_dec_info"],
                dcs_dict["dcs_dec_info_sheet_name"],
                dcs_dict["dcs_dec_info_pcs_column_name"],
                dcs_dict["dcs_dec_info_pc_name"],
                dcs_dict["dcs_dec_info_n_rates"])

            original_pc_dec_info = read_dcs_dec_info(
                dcs_dict["dcs_dec_info"],
                dcs_dict["dcs_dec_info_sheet_name"],
                dcs_dict["dcs_dec_info_pcs_column_name"],
                dcs_dict["dcs_dec_info_original_pc_name"],
                dcs_dict["dcs_dec_info_n_rates"])

            gpcc_rates = original_pc_dec_info[dcs_dict["dcs_dec_info_rate_col"]].values
            gpcc_dists = original_pc_dec_info[dcs_dict["dcs_dec_info_dist_col"]].values

            gptgpcc_rates = (dcs_dec_info[dcs_dict["dcs_dec_info_rate_col"]].values + ac_rate).tolist()
            gptgpcc_dists = [gpt_return[f"dist_{k}"] for k in sorted(dcs.keys())]

            gpt_rates = dcs_dict["dcs_dec_info_n_rates"] * [lut_return["rate_yuv"]]
            gpt_dists = dcs_dict["dcs_dec_info_n_rates"] * [gpt_return["dist"]]

            pd.DataFrame({
                "gpcc_rate":gpcc_rates,
                "gpcc_dist":gpcc_dists,
                "hybrid_rate":gptgpcc_rates,
                "hybrid_dist":gptgpcc_dists,
                "gpt_rate":gpt_rates,
                "gpt_dist":gpt_dists
            }).to_csv(f"results/{filename}_{acs_identification}_gptgpcc_results.csv")

            fig, ax = plt.subplots(nrows=1, ncols=1)

            h1,= ax.plot(gpcc_rates,gpcc_dists,
                linestyle="solid",label="gpcc",color="r",marker="o")
            h2,= ax.plot(gptgpcc_rates,gptgpcc_dists,
                linestyle="dashed",label="hybrid",color="b",marker="^")
            h3,= ax.plot(gpt_rates,gpt_dists,
                linestyle="dotted",label="gpt",color="g",marker="s")
            ax.legend(handles=[h1,h2,h3],loc="upper right")
            ax.set_xlabel("bpov yuv")
            ax.set_ylabel("psnr y")
            fig.savefig(f"results/{filename}_{acs_identification}_gptgpcc_results.png", dpi=300, facecolor='w')

        fig, ax = plt.subplots(nrows=2, ncols=3)

        ax[0,0].plot(x_axis,bits_y_per_coef_idx/np.sum(bits_y_per_coef_idx))
        ax[0,1].plot(x_axis,bits_u_per_coef_idx/np.sum(bits_u_per_coef_idx))
        ax[0,2].plot(x_axis,bits_v_per_coef_idx/np.sum(bits_v_per_coef_idx))

        ax[1,0].plot(x_axis,bits_y_per_coef_idx/samples_per_coef_idx)
        ax[1,1].plot(x_axis,bits_u_per_coef_idx/samples_per_coef_idx)
        ax[1,2].plot(x_axis,bits_v_per_coef_idx/samples_per_coef_idx)

        ax[0,0].set_xlabel("Coefficient index")
        ax[0,0].set_ylabel("fraction of bits (%)")
        ax[0,0].set_title(f"Y (DC: {bits_y_per_coef_idx[0]/np.sum(bits_y_per_coef_idx):.2f})")
        ax[0,1].set_xlabel("Coefficient index")
        ax[0,1].set_ylabel("fraction of bits (%)")
        ax[0,1].set_title(f"U (DC: {bits_u_per_coef_idx[0]/np.sum(bits_u_per_coef_idx):.2f})")
        ax[0,2].set_xlabel("Coefficient index")
        ax[0,2].set_ylabel("fraction of bits (%)")
        ax[0,2].set_title(f"V (DC: {bits_v_per_coef_idx[0]/np.sum(bits_v_per_coef_idx):.2f})")

        ax[1,0].set_xlabel("Coefficient index")
        ax[1,0].set_ylabel("Rate (bpv)")
        ax[1,0].set_title(f"Y (DC: {bits_y_per_coef_idx[0]/samples_per_coef_idx[0]:.2f})")
        ax[1,1].set_xlabel("Coefficient index")
        ax[1,1].set_ylabel("Rate (bpv)")
        ax[1,1].set_title(f"U (DC: {bits_u_per_coef_idx[0]/samples_per_coef_idx[0]:.2f})")
        ax[1,2].set_xlabel("Coefficient index")
        ax[1,2].set_ylabel("Rate (bpv)")
        ax[1,2].set_title(f"V (DC: {bits_v_per_coef_idx[0]/samples_per_coef_idx[0]:.2f})")

        fig.tight_layout()
        fig.savefig(f"results/{filename}_{acs_identification}_rate_per_coef_idx.png", dpi=300, facecolor='w')

        print(gpt_return["dist"])
        print(lut_return["rate"])

        pos = gpt_return["pos"]
        mask_0 = (pos == 0).reshape(-1)
        points = np.floor(gpt_return["points"][mask_0,:]/block_side)

        colors = yuv2rgb(gpt_return["colors"][mask_0,:])
        # print(np.min(colors),np.max(colors))
        dcs_filename= f"{filename}_{dcs_identification}_DC_YUV2RGB"
        pd.DataFrame({"min":[np.min(colors)],"max":[np.max(colors)]}).to_csv(f"results/{dcs_filename}.csv")
        write_PC(f"results/{dcs_filename}.ply",xyz=points,colors=normalize_colors(colors,np.min(colors),np.max(colors)))

        colors = np.tile(gpt_return["colors"][mask_0,0:1],(1,3))
        # print(np.min(colors),np.max(colors))
        dcs_filename= f"{filename}_{dcs_identification}_DC_Y2RGB"
        pd.DataFrame({"min":[np.min(colors)],"max":[np.max(colors)]}).to_csv(f"results/{dcs_filename}.csv")
        write_PC(f"results/{dcs_filename}.ply",xyz=points,colors=normalize_colors(colors,np.min(colors),np.max(colors)))

        points = gpt_return["points"]
        colors = yuv2rgb(gpt_return["colors"])
        # print(np.min(colors),np.max(colors))
        dcs_filename= f"{filename}_{dcs_identification}_DC_YUV2RGB_orig_geo"
        pd.DataFrame({"min":[np.min(colors)],"max":[np.max(colors)]}).to_csv(f"results/{dcs_filename}.csv")
        write_PC(f"results/{dcs_filename}.ply",xyz=points,colors=normalize_colors(colors,np.min(colors),np.max(colors)))


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




