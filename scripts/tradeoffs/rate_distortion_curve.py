

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def mse2psnr(mse):
    return 10 * np.log10( 1/ mse  )


def save_rate_dist_curve(rate,dist,labels,fig_name,to_psnr = False):
    if to_psnr:
        dist = list(map(mse2psnr,dist))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(rate,dist,marker="*",linestyle="None")
    for r,d,lbl in zip(rate,dist,labels):
        ax.text(x=r,y=d,s=lbl)
    ax.set_xlabel("rate (bpp)")
    ax.set_ylabel("psnr (db)" if to_psnr else "mse")
    fig.savefig(f'{fig_name}.png')


if __name__ == "__main__":

    srcdir = sys.argv[1]

    model = "bmshj2018-factorized"

    epochs = 10000

    all_N = [96, 128, 192]
    all_M = [128, 192, 320]

    dist_axis = []
    rate_axis = []
    labels = []
    for N in all_N:
        for M in all_M:
            if N==192 and M==320:
                continue
            history = pd.read_csv(os.path.join(srcdir,f"N{N}_M{M}_test_history.csv"))
            
            for c in history.columns:
                history[c] = history[c].apply(float)
            idx = history["loss"].argmin()
            mse_loss = history["mse_loss"].iloc[idx]
            bpp_loss = history["bpp_loss"].iloc[idx]

            dist_axis.append( mse_loss )
            rate_axis.append( bpp_loss )
            labels.append(f"N{N}M{M}")
    

    fig_name = \
        f"{model}_" + \
        f"{str(epochs)}-epochs_" +\
        f"N-{'-'.join(list(map(str,all_N)))}_" +\
        f"M-{'-'.join(list(map(str,all_M)))}"

    save_rate_dist_curve(rate_axis,dist_axis,labels,os.path.join(srcdir,f"rate-dist_{fig_name}"),to_psnr=False)
    save_rate_dist_curve(rate_axis,dist_axis,labels,os.path.join(srcdir,f"rate-psnr_{fig_name}"),to_psnr=True)