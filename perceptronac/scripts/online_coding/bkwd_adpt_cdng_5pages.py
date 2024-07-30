
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            "/home/lucas/Documents/data/SPL2021/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
        ]
    ]

    Ns = [26]
    
    learning_rates = [0.01] 

    central_tendencies = ["mean"] 

    init_method = "/home/lucas/Documents/perceptronac/results/exp_1714238871/exp_1714238871_026_min_valid_loss_model.pt"

    backward_adaptive_coding_experiment(docs,Ns,learning_rates,central_tendencies,init_method=init_method)
