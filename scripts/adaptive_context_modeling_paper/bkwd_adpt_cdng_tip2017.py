import numpy as np
import os
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    exp_name = "TIP2017_page1_lut_mean"

    docs = [
        ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_1.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_2.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_3.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_4.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_5.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_6.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_7.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_8.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_9.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_10.png"],
        # ["/home/lucas/Documents/data/TIP2017/ieee_tip2017_klt1024_11.png"],
    ]

    Ns = [26]
    
    learning_rates = [0.01]

    central_tendencies = ["mean"]

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

    ylim = [0.0, 1.0]

    backward_adaptive_coding_experiment(
        exp_name,docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim,
        manual_th=0.4,full_page=False,page_shape = (1024,791)
    )
