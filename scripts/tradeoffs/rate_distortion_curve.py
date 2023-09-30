

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


def get_factorized_prior_complexity(N,M,D):
    INPUT_SHAPE = (3, 256, 256)
    with torch.cuda.device(0):
        net = CustomFactorizedPrior(N,M,D)
        macs, params = get_model_complexity_info(net, (INPUT_SHAPE),as_strings=False)
    return 2 * macs, params


def mse2psnr(mse):
    return 10 * np.log10( 1/ mse  )


def join_Ls_data(data):
    return [v for k in sorted(data.keys()) for v in data[k]]


def get_line(lmbda,rate_axis,dist_axis,x_lims,y_lims):
    best_point = np.argmin(np.array(rate_axis) + lmbda * np.array(dist_axis))
    if lmbda >= (np.diff(x_lims)/np.diff(y_lims)):
        line_x = np.array(x_lims)
        line_y = (-1/lmbda)*line_x + (rate_axis[best_point]/lmbda + dist_axis[best_point])
    else:
        line_y = np.array(y_lims)
        line_x = (-1)* lmbda * line_y + (rate_axis[best_point] + lmbda * dist_axis[best_point])
    line = [line_x,line_y]
    return line


def save_L_curve(x_axis,y_axis,labels,fig_name,x_lbl,y_lbl,L):

    lmbda = (255**2) * float(L)
    best_point = np.argmin(np.array(x_axis) + lmbda * np.array(y_axis))

    fig, ax = plt.subplots(nrows=1, ncols=1)

    x_center = x_axis[best_point]
    y_center = y_axis[best_point]
    x_range = (np.max(x_axis) - np.min(x_axis))/5
    y_range = (np.max(y_axis) - np.min(y_axis))/5

    x_lims = [x_center-x_range/2,x_center+x_range/2]
    y_lims = [y_center-y_range/2,y_center+y_range/2]

    line = get_line(lmbda,x_axis,y_axis,x_lims,y_lims)

    idx = [i for i,x,y in zip(range(len(labels)),x_axis,y_axis) 
           if ((x_lims[0]<=x<=x_lims[1]) and (y_lims[0]<=y<=y_lims[1]))]

    x_axis = [x_axis[i] for i in idx ]
    y_axis = [y_axis[i] for i in idx ]
    labels = [labels[i] for i in idx ]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(21.6,16.72)
    ax.plot(x_axis,y_axis,marker="*",linestyle="None")
    for r,d,lbl in zip(x_axis,y_axis,labels):
        ax.text(x=r,y=d,s=lbl)

    ax.plot(line[0],line[1])

    ax.set_xlim(x_lims[0] - 0.01 * x_range,x_lims[1] + 0.01 * x_range)
    ax.set_ylim(y_lims[0] - 0.01 * y_range,y_lims[1] + 0.01 * y_range)

    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    fig.savefig(f'{fig_name}.png', dpi=300, facecolor='w', bbox_inches = "tight")


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

    figs_folder = "rate_distortion_curve_debug"
    log_file_name = "rate_distortion_curve.log"

    if not os.path.isdir(figs_folder):
        os.mkdir(figs_folder)

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
    all_D = [3] #[3,4]

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
    log_file = open(log_file_name, 'w')
    for D in all_D:
        for L in all_L:
            for N in all_N:
                for M in all_M:
                    history = None
                    for srcdir in srcdirs:
                        try:
                            history = pd.read_csv(os.path.join(srcdir,f"D{D}_L{L}_N{N}_M{M}_test_history.csv"))
                        except FileNotFoundError:
                            continue
                        
                    if history is None:
                        print(f"could not find data for D={D} L={L} N={N} M={M}",file=log_file)
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

    log_file.close()

    line_axes = []
    for L in list(set(partial_L)):
        best_point=np.argmin(loss_axis[L])
        lmbda = (255**2) * float(L)
        line_x = np.array([np.min(rate_axis[L]),np.max(rate_axis[L])])
        line_y = (-1/lmbda)*line_x + (rate_axis[L][best_point]/lmbda + dist_axis[L][best_point])
        line_axes.append([line_x,line_y])

    joint_dist_axis = join_Ls_data(dist_axis)
    joint_rate_axis = join_Ls_data(rate_axis)
    joint_loss_axis = join_Ls_data(loss_axis)
    joint_flops_axis = join_Ls_data(flops_axis)
    joint_params_axis = join_Ls_data(params_axis)
    joint_labels = join_Ls_data(labels)

    fig_name_template = "{}_{}-epochs_D-{}_L-{}_N-{}_M-{}"
    formatted_Ds = '-'.join(list(map(str,list(set(partial_D)))))
    formatted_Ls = '-'.join(list(set(partial_L)))
    formatted_Ns = '-'.join(list(map(str,list(set(partial_N)))))
    formatted_Ms = '-'.join(list(map(str,list(set(partial_M)))))

    for L in all_L:
        fig_name = fig_name_template.format(model,str(epochs),formatted_Ds,L,formatted_Ns,formatted_Ms)
        save_L_curve(rate_axis[L],dist_axis[L],labels[L],f"{figs_folder}/rate-dist-lmbd_{fig_name}","rate (bpp)","mse",L)

    fig_name = fig_name_template.format(model,str(epochs),formatted_Ds,formatted_Ls,formatted_Ns,formatted_Ms)
    save_curve(joint_rate_axis,joint_dist_axis,joint_labels,f"{figs_folder}/rate-dist_{fig_name}","rate (bpp)","mse",line_axes=line_axes)
    save_curve(joint_rate_axis,list(map(mse2psnr,joint_dist_axis)),joint_labels,f"{figs_folder}/rate-psnr_{fig_name}","rate (bpp)","psnr (db)")

    save_curve(joint_params_axis,joint_loss_axis,joint_labels,f"{figs_folder}/params-loss_{fig_name}","params","R+lambda*D")
    save_curve(joint_flops_axis,joint_loss_axis,joint_labels,f"{figs_folder}/flops-loss_{fig_name}","FLOPs","R+lambda*D")

    pd.DataFrame({
        "bpp_loss":joint_rate_axis,
        "mse_loss":joint_dist_axis,
        "psnr":list(map(mse2psnr,joint_dist_axis)),
        "loss": joint_loss_axis,
        "flops": joint_flops_axis,
        "params": joint_params_axis,
        "labels":joint_labels
    }).to_csv(f"bpp-mse-psnr-loss-flops-params_{fig_name}.csv")