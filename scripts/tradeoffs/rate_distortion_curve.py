

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from compressai.models import FactorizedPrior
import torch
from ptflops import get_model_complexity_info


def get_factorized_prior_complexity(N,M):
    INPUT_SHAPE = (3, 256, 256)
    with torch.cuda.device(0):
        net = FactorizedPrior(N,M)
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


def save_curve(x_axis,y_axis,labels,fig_name,x_lbl,y_lbl):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(21.6,16.72)
    ax.plot(x_axis,y_axis,marker="*",linestyle="None")
    for r,d,lbl in zip(x_axis,y_axis,labels):
        ax.text(x=r,y=d,s=lbl)
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

    all_N = [32, 64, 96, 128, 160, 192, 224]
    all_M = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]

    dist_axis = []
    rate_axis = []
    loss_axis = []
    flops_axis = []
    params_axis = []
    labels = []
    partial_N = []
    partial_M = []
    for N in all_N:
        for M in all_M:
            history = None
            for srcdir in srcdirs:
                try:
                    history = pd.read_csv(os.path.join(srcdir,f"N{N}_M{M}_test_history.csv"))
                except FileNotFoundError:
                    continue
                
            if history is None:
                continue
            
            for c in history.columns:
                history[c] = history[c].apply(float)
            idx = history["loss"].argmin()
            mse_loss = history["mse_loss"].iloc[idx]
            bpp_loss = history["bpp_loss"].iloc[idx]

            overall_loss = history["loss"].iloc[idx]
            flops,params = get_factorized_prior_complexity(N,M)

            dist_axis.append( mse_loss )
            rate_axis.append( bpp_loss )
            loss_axis.append( overall_loss )
            flops_axis.append( flops )
            params_axis.append( params )
            labels.append(f"N{N}M{M}")
            partial_N.append(N)
            partial_M.append(M)

    fig_name = \
        f"{model}_" + \
        f"{str(epochs)}-epochs_" +\
        f"N-{'-'.join(list(map(str,list(set(partial_N)))))}_" +\
        f"M-{'-'.join(list(map(str,list(set(partial_M)))))}"

    save_curve(rate_axis,dist_axis,labels,f"rate-dist_{fig_name}","rate (bpp)","mse")
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