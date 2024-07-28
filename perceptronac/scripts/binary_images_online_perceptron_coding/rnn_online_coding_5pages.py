import numpy as np
import re
from perceptronac.rnn_online_coding import rnn_online_coding_experiment

if __name__ == "__main__":

    lr = 0.01

    which_model = "GRURNN"

    hidden_units = 650

    samples_per_time = 64

    n_layers = 2

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
        ]
    ]

    learning_rates = [lr]

    rnn_online_coding_experiment(docs,learning_rates,
        which_model,hidden_units,n_layers=n_layers,samples_per_time=samples_per_time,n_pieces=1)

