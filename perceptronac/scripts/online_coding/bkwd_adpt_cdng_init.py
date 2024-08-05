
from perceptronac.backward_adaptive_coding import backward_adaptive_coding_experiment

if __name__ == "__main__":

    docs = [
        [
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/TRICE_A_Channel_Estimation_Framework_for_RIS-Aided_Millimeter-Wave_MIMO_Systems_1.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/The_Hilbert_Transform_of_B-Spline_Wavelets_5.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/The_NLMS_Is_Steady-State_Schur-Convex_4.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Total_Variation_Constrained_Graph-Regularized_Convex_Non-Negative_Matrix_Factorization_for_Data_Representation_1.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Towards_NIR-VIS_Masked_Face_Recognition_2.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Unsupervised_Discriminative_Deep_Hashing_With_Locality_and_Globality_Preservation_3.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Utterance_Verification-Based_Dysarthric_Speech_Intelligibility_Assessment_Using_Phonetic_Posterior_Features_5.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Variational_Measurement_Update_for_Extended_Object_Tracking_Using_Gaussian_Processes_1.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Wavelet_Frame-Based_Image_Restoration_via_ell__2-Relaxed_Truncated_ell__0_Regularization_and_Nonlocal_Estimation_5.png', 
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/ell__1-Norm_Minimization_With_Regula_Falsi_Type_Root_Finding_Methods_5.png',

            # '/home/lucaslopes/perceptronac/SPL2021no_margins/ATA_Attentional_Non-Linear_Activation_Function_Approximation_for_VLSI-Based_Neural_Networks_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/A_Multi-Task_CNN_for_Maritime_Target_Detection_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/A_Novel_H_2_Approach_to_FIR_Prediction_Under_Disturbances_and_Measurement_Errors_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/A_Novel_Nested_Array_for_Real-Valued_Sources_Exploiting_Array_Motion_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/A_Tibetan_Language_Model_That_Considers_the_Relationship_Between_Suffixes_and_Functional_Words_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Adversarially-Trained_Nonnegative_Matrix_Factorization_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_4.png',

            # '/home/lucaslopes/perceptronac/SPL2021no_margins/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_3.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/An_Uncertainty-Aware_Performance_Measure_for_Multi-Object_Tracking_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Automatic_Classification_of_Glaucoma_Stages_Using_Two-Dimensional_Tensor_Empirical_Wavelet_Transform_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/BRAFT_Recurrent_All-Pairs_Field_Transforms_for_Optical_Flow_Based_on_Correlation_Blocks_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Channel_Estimation_for_Underwater_Acoustic_Communications_Based_on_Orthogonal_Chirp_Division_Multiplexing_3.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Cross-Epoch_Learning_for_Weakly_Supervised_Anomaly_Detection_in_Surveillance_Videos_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/DESA_Disparity_Estimation_With_Surface_Awareness_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/DOA_Estimation_With_Nonuniform_Moving_Sampling_Scheme_Based_on_a_Moving_Platform_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Data-Driven_Parameter_Choice_for_Illumination_Artifact_Correction_of_Digital_Images_1.png',
            
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Deep_Bilateral_Learning_for_Stereo_Image_Super-Resolution_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/DiFNet_Densely_High-Frequency_Convolutional_Neural_Networks_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Difference_Value_Network_for_Image_Super-Resolution_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Dilated-Scale-Aware_Category-Attention_ConvNet_for_Multi-Class_Object_Counting_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Discriminative_Ensemble_Loss_for_Deep_Neural_Network_on_Classification_of_Ship-Radiated_Noise_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Distributed_Matrix_Multiplication_Using_Group_Algebra_for_On-Device_Edge_Computing_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Dynamic_Center_Aggregation_Loss_With_Mixed_Modality_for_Visible-Infrared_Person_Re-Identification_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Efficient_Image-Warping_Framework_for_Content-Adaptive_Superpixels_Generation_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Efficiently_Fusing_Pretrained_Acoustic_and_Linguistic_Encoders_for_Low-Resource_Speech_Recognition_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/End-to-End_Image_Stitching_Network_via_Multi-Homography_Estimation_4.png',

            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Enhanced_Separable_Convolution_Network_for_Lightweight_JPEG_Compression_Artifacts_Reduction_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Enriched_Music_Representations_With_Multiple_Cross-Modal_Contrastive_Learning_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Expectation-Maximization-Aided_Hybrid_Generalized_Expectation_Consistent_for_Sparse_Signal_Reconstruction_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Exponential_Hyperbolic_Cosine_Robust_Adaptive_Filters_for_Audio_Signal_Processing_4.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Extending_Ordinary-Label_Learning_Losses_to_Complementary-Label_Learning_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Fast_Adaptive_Active_Noise_Control_Based_on_Modified_Model-Agnostic_Meta-Learning_Algorithm_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Finite-Length_Bounds_on_Hypothesis_Testing_Subject_to_Vanishing_Type_I_Error_Restrictions_2.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/GA-NET_Global_Attention_Network_for_Point_Cloud_Semantic_Segmentation_5.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/Graph-Theoretic_Properties_of_Sub-Graph_Entropy_1.png',
            # '/home/lucaslopes/perceptronac/SPL2021no_margins/HLFNet_High-low_Frequency_Network_for_Person_Re-Identification_4.png',

            '/home/lucaslopes/perceptronac/SPL2021no_margins/Hierarchical_Factorization_Strategy_for_High-Order_Tensor_and_Application_to_Data_Completion_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/High_Precision_Error_Prediction_Algorithm_Based_on_Ridge_Regression_Predictor_for_Reversible_Data_Hiding_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Improving_End-to-End_Contextual_Speech_Recognition_via_a_Word-Matching_Algorithm_With_Backward_Search_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Improving_the_Visual_Quality_of_Video_Frame_Prediction_Models_Using_the_Perceptual_Straightening_Hypothesis_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Infrared_Image_Super-Resolution_via_Transfer_Learning_and_PSRGAN_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Interactive_Multimodal_Attention_Network_for_Emotion_Recognition_in_Conversation_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Joint_Cramr-Rao_Lower_Bound_for_Nonlinear_Parametric_Systems_With_Cross-Correlated_Noises_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Joint_Radar_Scheduling_and_Beampattern_Design_for_Multitarget_Tracking_in_Netted_Colocated_MIMO_Radar_Systems_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Knowledge_Distillation_With_Multi-Objective_Divergence_Learning_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Lightweight_2D_Imaging_for_Integrated_Imaging_and_Communication_Applications_4.png',
            
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Linear_Prediction-Based_Covariance_Matrix_Reconstruction_for_Robust_Adaptive_Beamforming_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Linear_Transformations_and_Signal_Estimation_in_the_Joint_Spatial-Slepian_Domain_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Linguistic_Steganography_From_Symbolic_Space_to_Semantic_Space_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Maximum_Likelihood_Sensor_Array_Calibration_Using_Non-Approximate_Hession_Matrix_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Memory-Free_Stochastic_Weight_Averaging_by_One-Way_Variational_Pruning_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Minimum_Gaussian_Entropy_Based_Distortionless_Adaptive_Beamforming_Algorithms_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Multi-Modal_Emotion_Recognition_by_Fusing_Correlation_Features_of_Speech-Visual_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Multi-Modal_Visual_Place_Recognition_in_Dynamics-Invariant_Perception_Space_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Multi-Target_Tracking_on_Riemannian_Manifolds_via_Probabilistic_Data_Association_1.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Non-Degraded_Adaptive_HEVC_Steganography_by_Advanced_Motion_Vector_Prediction_2.png',

            '/home/lucaslopes/perceptronac/SPL2021no_margins/Object_Detection_in_Hyperspectral_Images_1.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/On_the_Compensation_Between_Magnitude_and_Phase_in_Speech_Separation_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/On_the_Identifiability_of_Sparse_Vectors_From_Modulo_Compressed_Sensing_Measurements_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/On_the_Performance_of_the_SPICE_Method_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Online_Censoring_Based_Weighted-Frequency_Fourier_Linear_Combiner_for_Estimation_of_Pathological_Hand_Tremors_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Optimizing_Multi-Taper_Features_for_Deep_Speaker_Verification_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Orthogonal_Subspace_Based_Fast_Iterative_Thresholding_Algorithms_for_Joint_Sparsity_Recovery_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/PSF_Estimation_in_Crowded_Astronomical_Imagery_as_a_Convolutional_Dictionary_Learning_Problem_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Parallax-Estimation-Enhanced_Network_With_Interweave_Consistency_Feature_Fusion_for_Binocular_Salient_Object_Detection_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Parameter_Estimation_for_Sinusoidal_Frequency-Modulated_Signals_Using_Phase_Modulation_3.png',

            '/home/lucaslopes/perceptronac/SPL2021no_margins/Performance_Ranking_of_Kalman_Filter_With_Pre-Determined_Initial_State_Prior_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Pitch_Estimation_by_Multiple_Octave_Decoders_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Pseudo-Boolean_Functions_for_Optimal_Z-Complementary_Code_Sets_With_Flexible_Lengths_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Quickest_Detection_of_COVID-19_Pandemic_Onset_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Ray-Space-Based_Multichannel_Nonnegative_Matrix_Factorization_for_Audio_Source_Separation_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Reconfigurable_Intelligent_Surface_Aided_Sparse_DOA_Estimation_Method_With_Non-ULA_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Recurrent_Context_Aggregation_Network_for_Single_Image_Dehazing_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Reduced_Biquaternion_Stacked_Denoising_Convolutional_AutoEncoder_for_RGB-D_Image_Classification_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Reversible_Data_Hiding_for_JPEG_Images_Based_on_Multiple_Two-Dimensional_Histograms_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Risk-Aware_Multi-Armed_Bandits_With_Refined_Upper_Confidence_Bounds_1.png',
            
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Robust_Localization_Employing_Weighted_Least_Squares_Method_Based_on_MM_Estimator_and_Kalman_Filter_With_Maximum_Versoria_Criterion_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/SMoA_Searching_a_Modality-Oriented_Architecture_for_Infrared_and_Visible_Image_Fusion_4.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Semidefinite_Programming_Two-Way_TOA_Localization_for_User_Devices_With_Motion_and_Clock_Drift_2.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Signal_Lag_Measurements_Based_on_Temporal_Correlations_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Simultaneous_Graph_Learning_and_Blind_Separation_of_Graph_Signal_Sources_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Single_Haze_Image_Restoration_Under_Non-Uniform_Dense_Scattering_Media_1.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Single_Image_Deraining_Integrating_Physics_Model_and_Density-Oriented_Conditional_GAN_Refinement_5.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Split_Bregman_Approach_to_Linear_Prediction_Based_Dereverberation_With_Enforced_Speech_Sparsity_1.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Staged-Learning_Assessing_the_Quality_of_Screen_Content_Images_from_Distortion_Information_3.png',
            '/home/lucaslopes/perceptronac/SPL2021no_margins/Stereo_Feature_Enhancement_and_Temporal_Information_Extraction_Network_for_Automatic_Music_Transcription_3.png',

        ]
    ]

    Ns = [10]
    
    learning_rates = [0.0001] 

    central_tendencies = []

    # for init_method in [
    #     # "/home/lucas/Documents/perceptronac/results/exp_1714238871/exp_1714238871_010_min_valid_loss_model.pt",
    #     "/home/lucaslopes/perceptronac/results/exp_1672069178/exp_1672069178_010_min_valid_loss_model.pt",
    #     "custom"  
    # ]:
    #     backward_adaptive_coding_experiment(
    #         docs,Ns,learning_rates,central_tendencies,
    #         parallel=False,samples_per_time=1,n_pieces=50,
    #         init_method=init_method,page_shape = (895,670))
        
    for parent_id in [
        # # "1722367905",
        # # "1722370587"
        # "1722424344",
        # "1722432024"
        "1722455997",
        "1722486955"
    ]:
        

        backward_adaptive_coding_experiment(
            docs,Ns,learning_rates,central_tendencies,
            parallel=False,samples_per_time=1,n_pieces=50,
            page_shape = (895,670), parent_id=parent_id)
