import numpy as np
import re
from perceptronac.rnn_online_coding import rnn_online_coding_experiment

if __name__ == "__main__":

    lr_list = [0.001,0.005,0.01,0.05]

    samples_per_time_list = [1, 64, 96]

    for lr in lr_list:

        for samples_per_time in samples_per_time_list:

            which_model = "GRURNN" # "ElmanRNN"

            hidden_units = 650

            n_layers = 2

            docs = [ # docs[i,j] = the path to the j'th page from the i'th document
                [
                    "/home/lucaslopes/perceptronac/SPL2020V/A_PDE_Model_for_Smooth_Surface_Reconstruction_from_2D_Parallel_Slices_2.png"
                ]
            ]

            learning_rates = [lr]

            rnn_online_coding_experiment(docs,learning_rates,
                which_model,hidden_units,n_layers=n_layers,samples_per_time=samples_per_time,n_pieces=1)

