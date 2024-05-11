import numpy as np
from perceptronac.loading_and_saving import linestyle_tuple
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [
        [
            # '/home/lucas/Documents/data/SPL2021/pages/TRICE_A_Channel_Estimation_Framework_for_RIS-Aided_Millimeter-Wave_MIMO_Systems_1.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/The_Hilbert_Transform_of_B-Spline_Wavelets_5.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/The_NLMS_Is_Steady-State_Schur-Convex_4.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Total_Variation_Constrained_Graph-Regularized_Convex_Non-Negative_Matrix_Factorization_for_Data_Representation_1.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Towards_NIR-VIS_Masked_Face_Recognition_2.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Unsupervised_Discriminative_Deep_Hashing_With_Locality_and_Globality_Preservation_3.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Utterance_Verification-Based_Dysarthric_Speech_Intelligibility_Assessment_Using_Phonetic_Posterior_Features_5.png', 
            # '/home/lucas/Documents/data/SPL2021/pages/Variational_Measurement_Update_for_Extended_Object_Tracking_Using_Gaussian_Processes_1.png', 
            '/home/lucas/Documents/data/SPL2021/pages/Wavelet_Frame-Based_Image_Restoration_via_ell__2-Relaxed_Truncated_ell__0_Regularization_and_Nonlocal_Estimation_5.png', 
            '/home/lucas/Documents/data/SPL2021/pages/ell__1-Norm_Minimization_With_Regula_Falsi_Type_Root_Finding_Methods_5.png',

            '/home/lucas/Documents/data/SPL2021/pages/ATA_Attentional_Non-Linear_Activation_Function_Approximation_for_VLSI-Based_Neural_Networks_1.png',
            '/home/lucas/Documents/data/SPL2021/pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_4.png',
            # '/home/lucas/Documents/data/SPL2021/pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_2.png',
            # '/home/lucas/Documents/data/SPL2021/pages/A_Novel_H_2_Approach_to_FIR_Prediction_Under_Disturbances_and_Measurement_Errors_1.png',
            # '/home/lucas/Documents/data/SPL2021/pages/A_Novel_Nested_Array_for_Real-Valued_Sources_Exploiting_Array_Motion_2.png',
            # '/home/lucas/Documents/data/SPL2021/pages/A_Tibetan_Language_Model_That_Considers_the_Relationship_Between_Suffixes_and_Functional_Words_5.png',
            # '/home/lucas/Documents/data/SPL2021/pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png',
            # '/home/lucas/Documents/data/SPL2021/pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_4.png',
            # '/home/lucas/Documents/data/SPL2021/pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_5.png',
            # '/home/lucas/Documents/data/SPL2021/pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_4.png',

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
                                        parallel=False,samples_per_time=1,n_pieces=4, parent_id="1714924088")
