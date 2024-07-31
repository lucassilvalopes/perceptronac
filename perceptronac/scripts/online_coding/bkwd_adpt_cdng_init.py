
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [ # docs[i,j] = the path to the j'th page from the i'th document
        [
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_1.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_2.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_4.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_5.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_1.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_2.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_3.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_4.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_5.png",
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_1.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_2.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_3.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_4.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_5.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_1.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_2.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_3.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_4.png", 
            # "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/Adversarially-Trained_Nonnegative_Matrix_Factorization_5.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_1.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_2.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_3.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_4.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/A_Multi-Task_CNN_for_Maritime_Target_Detection_5.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_1.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_2.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_3.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_4.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_5.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_1.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_2.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_3.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_4.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_5.png",
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_1.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_2.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_3.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_4.png", 
            "/home/lucas/Documents/data/SPL2021_no_margins/all_pages/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_5.png",

        ]
    ]

    Ns = [10]
    
    learning_rates = [0.0001] 

    central_tendencies = []

    # for init_method in [
    #     "/home/lucas/Documents/perceptronac/results/exp_1714238871/exp_1714238871_010_min_valid_loss_model.pt",
    #     "custom"  
    # ]:
    #     backward_adaptive_coding_experiment(
    #         docs,Ns,learning_rates,central_tendencies,init_method=init_method,page_shape = (895,670))
        
    for parent_id in [
        # "1722367905",
        # "1722370587"
        "1722424344",
        "1722432024"
    ]:
        

        backward_adaptive_coding_experiment(
            docs,Ns,learning_rates,central_tendencies,
            parallel=False,samples_per_time=1,n_pieces=4,
            page_shape = (895,670), parent_id=parent_id)
