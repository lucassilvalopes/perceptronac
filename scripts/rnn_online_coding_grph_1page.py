import numpy as np

from perceptronac.rnn_online_coding import rnn_online_coding_experiment

if __name__ == "__main__":

    lr = 0.001

    exp_name = f"Adaptive_Detection_of_Dim_page1_mikolov500_lr1e{str(int(np.log10(lr)))}"

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            # "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            # "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            # "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            # "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
        ]
    ]

    learning_rates = [lr]

    labels = [
        'ARNN $\lambda=10^{'+ str(int(np.log10(lr))) + '}$',
    ]

    linestyles = [
        "solid",
    ]

    colors = [
        "r", 
    ]

    legend_ncol = 1

    ylim = [0.0, 1.0]

    rnn_online_coding_experiment(exp_name,docs,learning_rates,colors,linestyles,
        labels,legend_ncol,ylim,samples_per_time=1,n_pieces=1)

