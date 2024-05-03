import numpy as np
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [
        [
            '/home/lucas/Documents/data/SPL2021/pages/TRICE_A_Channel_Estimation_Framework_for_RIS-Aided_Millimeter-Wave_MIMO_Systems_1.png', 
            '/home/lucas/Documents/data/SPL2021/pages/The_Hilbert_Transform_of_B-Spline_Wavelets_5.png', 
            '/home/lucas/Documents/data/SPL2021/pages/The_NLMS_Is_Steady-State_Schur-Convex_4.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Total_Variation_Constrained_Graph-Regularized_Convex_Non-Negative_Matrix_Factorization_for_Data_Representation_1.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Towards_NIR-VIS_Masked_Face_Recognition_2.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Unsupervised_Discriminative_Deep_Hashing_With_Locality_and_Globality_Preservation_3.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Utterance_Verification-Based_Dysarthric_Speech_Intelligibility_Assessment_Using_Phonetic_Posterior_Features_5.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Variational_Measurement_Update_for_Extended_Object_Tracking_Using_Gaussian_Processes_1.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Wavelet_Frame-Based_Image_Restoration_via_ell__2-Relaxed_Truncated_ell__0_Regularization_and_Nonlocal_Estimation_5.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/ell__1-Norm_Minimization_With_Regula_Falsi_Type_Root_Finding_Methods_5.png'
        ]
    ]

    Ns = [170]
    learning_rates = [0.01] 
    central_tendencies = ["mean"] 
    labels = ['ALUT', r'APC $\eta=10^{-2}$']
    linestyles = ["solid", "dashed"]
    colors = ["g", "b"]
    legend_ncol = 1
    ylim = [0.0, 1.0]

    backward_adaptive_coding_experiment("",docs,Ns,learning_rates,central_tendencies,colors,linestyles,labels,legend_ncol,ylim,
                                        parallel=False,samples_per_time=1,n_pieces=3, parent_id="")
