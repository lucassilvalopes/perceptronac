import numpy as np
import os
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    exp_name = "SPL2021_last_10_sorted_pages_lut_mean"

    docs = [[os.path.join('SPL2021',f)] for f in sorted(os.listdir('SPL2021'))[-10:]]

    Ns = [0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = [0.01] # (3.162277659**np.array([-2,-4,-6,-8]))

    central_tendencies = ["mean"]#["mean","mode"]

    labels = [
        'B LUT', # LUTmean
        r'APC $\lambda=10^{-2}$', # MLPlr=1e-02
    ]

    linestyles = [
        "solid", # "LUTmean"
        "dashed", # MLPlr=1e-02
    ]

    colors = [
        "g", # "LUTmean"
        "b", # MLPlr=1e-02 
    ]

    legend_ncol = 1

    ylim = [0.0, 1.0]

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim)
