import numpy as np
import re
from perceptronac.rnn_online_coding import rnn_online_coding_experiment

if __name__ == "__main__":

    lr = 0.01

    which_model = "GRURNN" # "ElmanRNN"

    hidden_units = 650

    samples_per_time = 64 # 1

    n_layers = 2

    exp_name = f"last_10_sorted_pages_{which_model}{hidden_units}_lr{lr:.0e}_batchsize{samples_per_time}"

    docs = [
        ['/home/lucas/Documents/data/SPL2021/pages/TRICE_A_Channel_Estimation_Framework_for_RIS-Aided_Millimeter-Wave_MIMO_Systems_1.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/The_Hilbert_Transform_of_B-Spline_Wavelets_5.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/The_NLMS_Is_Steady-State_Schur-Convex_4.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Total_Variation_Constrained_Graph-Regularized_Convex_Non-Negative_Matrix_Factorization_for_Data_Representation_1.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Towards_NIR-VIS_Masked_Face_Recognition_2.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Unsupervised_Discriminative_Deep_Hashing_With_Locality_and_Globality_Preservation_3.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Utterance_Verification-Based_Dysarthric_Speech_Intelligibility_Assessment_Using_Phonetic_Posterior_Features_5.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Variational_Measurement_Update_for_Extended_Object_Tracking_Using_Gaussian_Processes_1.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/Wavelet_Frame-Based_Image_Restoration_via_ell__2-Relaxed_Truncated_ell__0_Regularization_and_Nonlocal_Estimation_5.png'], 
        ['/home/lucas/Documents/data/SPL2021/pages/ell__1-Norm_Minimization_With_Regula_Falsi_Type_Root_Finding_Methods_5.png']
    ]




    learning_rates = [lr]

    labels = [
        'ARNN $\lambda='+str(re.search(r'(?<=^).*(?=e-)',f"{lr:.0e}").group())+\
            '\cdot10^{-'+ str(int(re.search(r'(?<=e-).*(?=$)',f"{lr:.0e}").group())) + '}$',
    ]

    linestyles = [
        "solid",
    ]

    colors = [
        "r", 
    ]

    legend_ncol = 1

    ylim = [0.0, 1.0]

    rnn_online_coding_experiment(exp_name,docs,learning_rates,colors,linestyles,labels,legend_ncol,ylim,
        which_model,hidden_units,n_layers=n_layers,samples_per_time=samples_per_time,n_pieces=1)

