
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [
        [
            "/home/lucaslopes/perceptronac/SPL2020V/A_PDE_Model_for_Smooth_Surface_Reconstruction_from_2D_Parallel_Slices_2.png" # "/home/lucaslopes/perceptronac/SPL2021/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png"
        ]
    ]

    Ns = [67,26,10,4] 
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    central_tendencies = []

    backward_adaptive_coding_experiment(docs,Ns,learning_rates,central_tendencies)
