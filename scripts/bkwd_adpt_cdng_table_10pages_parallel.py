import numpy as np
import os
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    exp_name = "SPL2021_last_10_sorted_pages_parallel_lut_mean_lr1e-3"

    docs = [[os.path.join('/home/lucas/Documents/data/SPL2021/pages',f) for f in sorted(os.listdir('/home/lucas/Documents/data/SPL2021/pages'))[-10:]]]

    Ns = [0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = [0.001] # (3.162277659**np.array([-2,-4,-6,-8]))

    central_tendencies = ["mean"]#["mean","mode"]

    labels = [
        'ALUT', # LUTmean
        r'APC $\lambda=10^{-3}$', # MLPlr=1e-03
    ]

    linestyles = [
        "solid", # "LUTmean"
        "dashed", # MLPlr=1e-03
    ]

    colors = [
        "g", # "LUTmean"
        "b", # MLPlr=1e-03
    ]

    legend_ncol = 1

    ylim = [0.0, 0.5]

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim,parallel=True)
