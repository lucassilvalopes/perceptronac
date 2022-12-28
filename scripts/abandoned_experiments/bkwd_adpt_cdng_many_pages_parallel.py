import numpy as np
import os
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    lr = 0.00001

    n_pages = 320

    n_pieces = 32

    exp_name = f"SPL2021allpages_last_{n_pages}_sorted_pages_parallel_lut_mean_lr1e{str(int(np.log10(lr)))}"

    docs = [[os.path.join('/home/lucas/Documents/data/SPL2021/all_pages',f) for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2021/all_pages'))[-n_pages:]]]

    Ns = [26] #[0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = [lr] # (3.162277659**np.array([-2,-4,-6,-8]))

    central_tendencies = ["mean"]#["mean","mode"]

    labels = [
        'ALUT', # LUTmean
        'APC $\lambda=10^{'+ str(int(np.log10(lr))) + '}$', # MLPlr=1e-0?
    ]

    linestyles = [
        "solid", # "LUTmean"
        "dashed", # MLPlr=1e-0?
    ]

    colors = [
        "g", # "LUTmean"
        "b", # MLPlr=1e-0?
    ]

    legend_ncol = 1

    ylim = [0.0, 0.5]

    backward_adaptive_coding_experiment(
        exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim,
        parallel=True,n_pieces=n_pieces)
