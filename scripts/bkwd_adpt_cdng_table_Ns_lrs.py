import numpy as np
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    # exp_name = "Adaptive_Detection_of_Dim_page1_lut_mean_mode_lr1e-1_lr1e-2_lr1e-4"

    # docs = [ # docs[i,j] = the path to the j'th page from the i'th document
    #     [
    #         "SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
    #         "SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
    #         "SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
    #         "SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
    #         "SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
    #     ]
    # ]

    exp_name = "A_PDE_Model_for_Smooth_lr1e-1_lr1e-2_lr1e-3_lr1e-4"

    docs = [
        [
            "/home/lucas/Documents/data/SPL2020V/pages/A_PDE_Model_for_Smooth_Surface_Reconstruction_from_2D_Parallel_Slices_2.png"
        ]
    ]

    # exp_name = "SPL2021_last_10_sorted_pages"

    # docs = [[os.path.join('SPL2021',f)] for f in sorted(os.listdir('SPL2021'))[-10:]]

    Ns = [67,26,10,4] # [0,2,4,10,26,67,170] # [26,33,42,53,67,84,107,135,170]
    
    learning_rates = (3.162277659**np.array([-2,-4,-6,-8]))

    central_tendencies = []#["mean","mode"]

    labels = [
        # 'B LUT', # LUTmean
        # 'U LUT', # LUTmode
        r'APC $\lambda=10^{-1}$', # MLPlr=1e-01
        r'APC $\lambda=10^{-2}$', # MLPlr=1e-02
        r'APC $\lambda=10^{-3}$', # MLPlr=1e-03
        r'APC $\lambda=10^{-4}$' # MLPlr=1e-04
    ]

    linestyles = [
        # "solid", # "LUTmean"
        # linestyle_tuple['densely dotted'], # "LUTmode"
        "dashdot", # MLPlr=1e-01
        "dashed", # MLPlr=1e-02
        linestyle_tuple['densely dashdotted'], # MLPlr=1e-03
        "dotted", # MLPlr=1e-04
    ]

    colors = [
        # "g", # "LUTmean"
        # "0.5", # "LUTmode"
        "r", # MLPlr=1e-01
        "b", # MLPlr=1e-02
        "m", # MLPlr=1e-03
        "c", # MLPlr=1e-04  
    ]

    legend_ncol = 1

    ylim = [0.0, 1.0]

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim)
