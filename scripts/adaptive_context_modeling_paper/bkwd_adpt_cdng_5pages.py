import numpy as np
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    exp_name = "Adaptive_Detection_of_Dim_5pages_lut_mean_lr1e-2"

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            "/home/lucaslopes/perceptronac/SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            "/home/lucaslopes/perceptronac/SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            "/home/lucaslopes/perceptronac/SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            "/home/lucaslopes/perceptronac/SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            "/home/lucaslopes/perceptronac/SPL2021all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
        ]
    ]

    Ns = [
        10,
        26,
        67
    ]
    
    learning_rates = [0.01] #(3.162277659**np.array([-2,-4,-6,-8]))

    central_tendencies = ["mean"] #["mean","mode"]

    labels = [
        'ALUT', # LUTmean
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

    ylim = [0.0, 0.5]

    backward_adaptive_coding_experiment(exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim)
