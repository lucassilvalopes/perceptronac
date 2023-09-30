

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from compressai.models import FactorizedPrior
from vae_model import CustomFactorizedPrior
import torch
from ptflops import get_model_complexity_info


def get_factorized_prior_complexity(N,M,D):
    INPUT_SHAPE = (3, 256, 256)
    with torch.cuda.device(0):
        net = CustomFactorizedPrior(N,M,D)
        macs, params = get_model_complexity_info(net, (INPUT_SHAPE),as_strings=False)
    return 2 * macs, params


# # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
# # https://stackoverflow.com/questions/61025847/light-font-weight-in-matplotlib
# # https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts
# # https://stackoverflow.com/questions/68750760/matplotlib-font-size-and-line-width-inconsistent
params = {
    'axes.labelsize': 20,
    'axes.titlesize':20,
    'xtick.labelsize':20,
    'ytick.labelsize':20,
    # 'font.size': 12
}

plt.rcParams.update(params)


def mse2psnr(mse):
    return 10 * np.log10( 1/ mse  )


def save_curve(x_axis,y_axis,labels,fig_name,x_lbl,y_lbl,line_axes=None):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(21.6,16.72)
    ax.plot(x_axis,y_axis,marker="*",linestyle="None")
    for r,d,lbl in zip(x_axis,y_axis,labels):
        ax.text(x=r,y=d,s=lbl)

    if line_axes:
        for line_x,line_y in line_axes:
            ax.plot(line_x,line_y)

    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    fig.savefig(f'{fig_name}.png', dpi=300, facecolor='w', bbox_inches = "tight")


if __name__ == "__main__":

    srcdirs = []
    for i in range(len(sys.argv)):
        if i == 0:
            continue
        srcdirs.append(sys.argv[i])
        

    model = "bmshj2018-factorized"

    epochs = 10000

    all_L = ["1e-2", "5e-3", "2e-2"]
    all_N = [32, 64, 96, 128, 160, 192, 224]
    all_M = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    all_D = [3,4]

    dist_axis = {L:[] for L in all_L}
    rate_axis = {L:[] for L in all_L}
    loss_axis = {L:[] for L in all_L}
    flops_axis = {L:[] for L in all_L}
    params_axis = {L:[] for L in all_L}
    labels = {L:[] for L in all_L}
    partial_L = []
    partial_N = []
    partial_M = []
    partial_D = []
    for D in all_D:
        for L in all_L:
            for N in all_N:
                for M in all_M:
                    history = None
                    for srcdir in srcdirs:
                        try:
                            history = pd.read_csv(os.path.join(srcdir,f"D{D}_L{L}_N{N}_M{M}_test_history.csv"))
                        except FileNotFoundError:
                            warnings.warn(f"could not find data for D={D} L={L} N={N} M={M}")
                            continue
                        
                    if history is None:
                        continue
                    
                    for c in history.columns:
                        history[c] = history[c].apply(float)
                    idx = history["loss"].argmin()
                    mse_loss = history["mse_loss"].iloc[idx]
                    bpp_loss = history["bpp_loss"].iloc[idx]

                    overall_loss = history["loss"].iloc[idx]
                    flops,params = get_factorized_prior_complexity(N,M,D)

                    dist_axis[L].append( mse_loss )
                    rate_axis[L].append( bpp_loss )
                    loss_axis[L].append( overall_loss )
                    flops_axis[L].append( flops )
                    params_axis[L].append( params )
                    labels[L].append(f"D{D}L{L}N{N}M{M}")
                    partial_L.append(L)
                    partial_N.append(N)
                    partial_M.append(M)
                    partial_D.append(D)
    

    line_axes = []
    for L in list(set(partial_L)):
        best_point=np.argmin(loss_axis[L])
        lmbda = (255**2) * float(L)
        line_x = np.array([np.min(rate_axis[L]),np.max(rate_axis[L])])
        line_y = (-1/lmbda)*line_x + (rate_axis[L][best_point]/lmbda + dist_axis[L][best_point])
        line_axes.append([line_x,line_y])


    def join_Ls_data(data):
        return [v for k in sorted(data.keys()) for v in data[k]]


    dist_axis = join_Ls_data(dist_axis)
    rate_axis = join_Ls_data(rate_axis)
    loss_axis = join_Ls_data(loss_axis)
    flops_axis = join_Ls_data(flops_axis)
    params_axis = join_Ls_data(params_axis)
    labels = join_Ls_data(labels)



    fig_name = \
        f"{model}_" + \
        f"{str(epochs)}-epochs_" +\
        f"D-{'-'.join(list(map(str,list(set(partial_D)))))}_" +\
        f"L-{'-'.join(list(set(partial_L)))}_" +\
        f"N-{'-'.join(list(map(str,list(set(partial_N)))))}_" +\
        f"M-{'-'.join(list(map(str,list(set(partial_M)))))}"

    save_curve(rate_axis,dist_axis,labels,f"rate-dist_{fig_name}","rate (bpp)","mse",line_axes = line_axes)
    save_curve(rate_axis,list(map(mse2psnr,dist_axis)),labels,f"rate-psnr_{fig_name}","rate (bpp)","psnr (db)")

    save_curve(params_axis,loss_axis,labels,f"params-loss_{fig_name}","params","R+lambda*D")
    save_curve(flops_axis,loss_axis,labels,f"flops-loss_{fig_name}","FLOPs","R+lambda*D")

    pd.DataFrame({
        "bpp_loss":rate_axis,
        "mse_loss":dist_axis,
        "psnr":list(map(mse2psnr,dist_axis)),
        "loss": loss_axis,
        "flops": flops_axis,
        "params": params_axis,
        "labels":labels
    }).to_csv(f"bpp-mse-psnr-loss-flops-params_{fig_name}.csv")