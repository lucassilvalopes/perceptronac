import numpy as np
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment
from perceptronac.loading_and_saving import linestyle_tuple


if __name__ == "__main__":

    exp_name = "A_PDE_Model_for_Smooth_page2_lut_mean_mode_lr1e-1_lr1e-2_lr1e-4"

    docs = [
        [

            "/home/lucas/Documents/data/SPL2020V/pages/A_PDE_Model_for_Smooth_Surface_Reconstruction_from_2D_Parallel_Slices_2.png"
        ]
    ]

    Ns = [10,26] # [0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = (3.162277659**np.array([-2,-4,-8]))

    central_tendencies = ["mean"] # ["mean","mode"]

    labels = [
        'ALUT', # LUTmean
        # 'U LUT', # LUTmode
        r'APC $\lambda=10^{-1}$', # MLPlr=1e-01
        r'APC $\lambda=10^{-2}$', # MLPlr=1e-02
        r'APC $\lambda=10^{-4}$' # MLPlr=1e-04
    ]

    linestyles = [
        "solid", # "LUTmean"
        # linestyle_tuple['densely dotted'], # "LUTmode"
        "dashdot", # MLPlr=1e-01
        "dashed", # MLPlr=1e-02
        "dotted", # MLPlr=1e-04
    ]

    colors = [
        "g", # "LUTmean"
        # "0.5", # "LUTmode"
        "r", # MLPlr=1e-01
        "b", # MLPlr=1e-02
        "c", # MLPlr=1e-04  
    ]

    legend_ncol = 1 #2

    ylim = [0.0, 0.5] # [0.0, 0.5]

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim)