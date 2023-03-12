

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def save_rate_dist_curve(rate,dist,labels,fig_name,to_psnr = False):
    if to_psnr:
        dist = list(map(mse2psnr,dist))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(21.6,16.72)
    ax.plot(rate,dist,marker="*",linestyle="None")
    for r,d,lbl in zip(rate,dist,labels):
        ax.text(x=r,y=d,s=lbl)
    ax.set_xlabel("rate (bpp)")
    ax.set_ylabel("psnr (db)" if to_psnr else "mse")
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
    all_M = [128, 160, 192, 224, 256, 288, 320]

    dist_axis = []
    rate_axis = []
    labels = []
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

            dist_axis.append( mse_loss )
            rate_axis.append( bpp_loss )
            labels.append(f"N{N}M{M}")
    

    fig_name = \
        f"{model}_" + \
        f"{str(epochs)}-epochs_" +\
        f"N-{'-'.join(list(map(str,all_N)))}_" +\
        f"M-{'-'.join(list(map(str,all_M)))}"

    save_rate_dist_curve(rate_axis,dist_axis,labels,f"rate-dist_{fig_name}",to_psnr=False)
    save_rate_dist_curve(rate_axis,dist_axis,labels,f"rate-psnr_{fig_name}",to_psnr=True)

    pd.DataFrame({
        "bpp_loss":rate_axis,
        "mse_loss":dist_axis,
        "psnr":list(map(mse2psnr,dist_axis)),
        "labels":labels
    }).to_csv(f"bpp-mse-psnr_{fig_name}.csv")