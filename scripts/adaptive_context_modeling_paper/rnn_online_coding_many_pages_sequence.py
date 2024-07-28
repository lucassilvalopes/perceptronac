import numpy as np
import re
from perceptronac.rnn_online_coding import rnn_online_coding_experiment

if __name__ == "__main__":

    lr = 0.01

    which_model = "GRURNN" # "ElmanRNN"

    hidden_units = 650

    samples_per_time = 64 # 1

    n_layers = 2

    exp_name = f"one_hundred_pages_{which_model}{hidden_units}_lr{lr:.0e}_batchsize{samples_per_time}"

    docs = [
        [
            '/home/lucaslopes/perceptronac/SPL2021/TRICE_A_Channel_Estimation_Framework_for_RIS-Aided_Millimeter-Wave_MIMO_Systems_1.png', 
            '/home/lucaslopes/perceptronac/SPL2021/The_Hilbert_Transform_of_B-Spline_Wavelets_5.png', 
            '/home/lucaslopes/perceptronac/SPL2021/The_NLMS_Is_Steady-State_Schur-Convex_4.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Total_Variation_Constrained_Graph-Regularized_Convex_Non-Negative_Matrix_Factorization_for_Data_Representation_1.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Towards_NIR-VIS_Masked_Face_Recognition_2.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Unsupervised_Discriminative_Deep_Hashing_With_Locality_and_Globality_Preservation_3.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Utterance_Verification-Based_Dysarthric_Speech_Intelligibility_Assessment_Using_Phonetic_Posterior_Features_5.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Variational_Measurement_Update_for_Extended_Object_Tracking_Using_Gaussian_Processes_1.png', 
            '/home/lucaslopes/perceptronac/SPL2021/Wavelet_Frame-Based_Image_Restoration_via_ell__2-Relaxed_Truncated_ell__0_Regularization_and_Nonlocal_Estimation_5.png', 
            '/home/lucaslopes/perceptronac/SPL2021/ell__1-Norm_Minimization_With_Regula_Falsi_Type_Root_Finding_Methods_5.png',

            '/home/lucaslopes/perceptronac/SPL2021/ATA_Attentional_Non-Linear_Activation_Function_Approximation_for_VLSI-Based_Neural_Networks_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/A_Disparity_Feature_Alignment_Module_for_Stereo_Image_Super-Resolution_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/A_Multi-Task_CNN_for_Maritime_Target_Detection_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/A_Novel_H_2_Approach_to_FIR_Prediction_Under_Disturbances_and_Measurement_Errors_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/A_Novel_Nested_Array_for_Real-Valued_Sources_Exploiting_Array_Motion_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/A_Tibetan_Language_Model_That_Considers_the_Relationship_Between_Suffixes_and_Functional_Words_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Adaptive_Detection_of_Dim_Maneuvering_Targets_in_Adjacent_Range_Cells_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Adaptive_Flexible_Optimal_Graph_for_Unsupervised_Dimensionality_Reduction_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Adversarially-Trained_Nonnegative_Matrix_Factorization_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/An_Adaptive_Formulation_of_the_Sliding_Innovation_Filter_4.png',

            '/home/lucaslopes/perceptronac/SPL2021/An_Efficient_Quasi-Newton_Method_for_Nonlinear_Inverse_Problems_via_Learned_Singular_Values_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/An_Efficient_Sampling_Scheme_for_the_Eigenvalues_of_Dual_Wishart_Matrices_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/An_Uncertainty-Aware_Performance_Measure_for_Multi-Object_Tracking_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Automatic_Classification_of_Glaucoma_Stages_Using_Two-Dimensional_Tensor_Empirical_Wavelet_Transform_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/BRAFT_Recurrent_All-Pairs_Field_Transforms_for_Optical_Flow_Based_on_Correlation_Blocks_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Channel_Estimation_for_Underwater_Acoustic_Communications_Based_on_Orthogonal_Chirp_Division_Multiplexing_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Cross-Epoch_Learning_for_Weakly_Supervised_Anomaly_Detection_in_Surveillance_Videos_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/DESA_Disparity_Estimation_With_Surface_Awareness_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/DOA_Estimation_With_Nonuniform_Moving_Sampling_Scheme_Based_on_a_Moving_Platform_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Data-Driven_Parameter_Choice_for_Illumination_Artifact_Correction_of_Digital_Images_1.png',
            
            '/home/lucaslopes/perceptronac/SPL2021/Deep_Bilateral_Learning_for_Stereo_Image_Super-Resolution_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/DiFNet_Densely_High-Frequency_Convolutional_Neural_Networks_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Difference_Value_Network_for_Image_Super-Resolution_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Dilated-Scale-Aware_Category-Attention_ConvNet_for_Multi-Class_Object_Counting_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Discriminative_Ensemble_Loss_for_Deep_Neural_Network_on_Classification_of_Ship-Radiated_Noise_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Distributed_Matrix_Multiplication_Using_Group_Algebra_for_On-Device_Edge_Computing_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Dynamic_Center_Aggregation_Loss_With_Mixed_Modality_for_Visible-Infrared_Person_Re-Identification_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Efficient_Image-Warping_Framework_for_Content-Adaptive_Superpixels_Generation_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Efficiently_Fusing_Pretrained_Acoustic_and_Linguistic_Encoders_for_Low-Resource_Speech_Recognition_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/End-to-End_Image_Stitching_Network_via_Multi-Homography_Estimation_4.png',

            '/home/lucaslopes/perceptronac/SPL2021/Enhanced_Separable_Convolution_Network_for_Lightweight_JPEG_Compression_Artifacts_Reduction_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Enriched_Music_Representations_With_Multiple_Cross-Modal_Contrastive_Learning_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Expectation-Maximization-Aided_Hybrid_Generalized_Expectation_Consistent_for_Sparse_Signal_Reconstruction_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Exponential_Hyperbolic_Cosine_Robust_Adaptive_Filters_for_Audio_Signal_Processing_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Extending_Ordinary-Label_Learning_Losses_to_Complementary-Label_Learning_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Fast_Adaptive_Active_Noise_Control_Based_on_Modified_Model-Agnostic_Meta-Learning_Algorithm_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Finite-Length_Bounds_on_Hypothesis_Testing_Subject_to_Vanishing_Type_I_Error_Restrictions_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/GA-NET_Global_Attention_Network_for_Point_Cloud_Semantic_Segmentation_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Graph-Theoretic_Properties_of_Sub-Graph_Entropy_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/HLFNet_High-low_Frequency_Network_for_Person_Re-Identification_4.png',

            '/home/lucaslopes/perceptronac/SPL2021/Hierarchical_Factorization_Strategy_for_High-Order_Tensor_and_Application_to_Data_Completion_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/High_Precision_Error_Prediction_Algorithm_Based_on_Ridge_Regression_Predictor_for_Reversible_Data_Hiding_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Improving_End-to-End_Contextual_Speech_Recognition_via_a_Word-Matching_Algorithm_With_Backward_Search_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Improving_the_Visual_Quality_of_Video_Frame_Prediction_Models_Using_the_Perceptual_Straightening_Hypothesis_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Infrared_Image_Super-Resolution_via_Transfer_Learning_and_PSRGAN_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Interactive_Multimodal_Attention_Network_for_Emotion_Recognition_in_Conversation_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Joint_Cramr-Rao_Lower_Bound_for_Nonlinear_Parametric_Systems_With_Cross-Correlated_Noises_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Joint_Radar_Scheduling_and_Beampattern_Design_for_Multitarget_Tracking_in_Netted_Colocated_MIMO_Radar_Systems_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Knowledge_Distillation_With_Multi-Objective_Divergence_Learning_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Lightweight_2D_Imaging_for_Integrated_Imaging_and_Communication_Applications_4.png',
            
            '/home/lucaslopes/perceptronac/SPL2021/Linear_Prediction-Based_Covariance_Matrix_Reconstruction_for_Robust_Adaptive_Beamforming_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Linear_Transformations_and_Signal_Estimation_in_the_Joint_Spatial-Slepian_Domain_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Linguistic_Steganography_From_Symbolic_Space_to_Semantic_Space_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Maximum_Likelihood_Sensor_Array_Calibration_Using_Non-Approximate_Hession_Matrix_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Memory-Free_Stochastic_Weight_Averaging_by_One-Way_Variational_Pruning_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Minimum_Gaussian_Entropy_Based_Distortionless_Adaptive_Beamforming_Algorithms_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Multi-Modal_Emotion_Recognition_by_Fusing_Correlation_Features_of_Speech-Visual_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Multi-Modal_Visual_Place_Recognition_in_Dynamics-Invariant_Perception_Space_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Multi-Target_Tracking_on_Riemannian_Manifolds_via_Probabilistic_Data_Association_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Non-Degraded_Adaptive_HEVC_Steganography_by_Advanced_Motion_Vector_Prediction_2.png',

            '/home/lucaslopes/perceptronac/SPL2021/Object_Detection_in_Hyperspectral_Images_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/On_the_Compensation_Between_Magnitude_and_Phase_in_Speech_Separation_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/On_the_Identifiability_of_Sparse_Vectors_From_Modulo_Compressed_Sensing_Measurements_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/On_the_Performance_of_the_SPICE_Method_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Online_Censoring_Based_Weighted-Frequency_Fourier_Linear_Combiner_for_Estimation_of_Pathological_Hand_Tremors_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Optimizing_Multi-Taper_Features_for_Deep_Speaker_Verification_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Orthogonal_Subspace_Based_Fast_Iterative_Thresholding_Algorithms_for_Joint_Sparsity_Recovery_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/PSF_Estimation_in_Crowded_Astronomical_Imagery_as_a_Convolutional_Dictionary_Learning_Problem_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Parallax-Estimation-Enhanced_Network_With_Interweave_Consistency_Feature_Fusion_for_Binocular_Salient_Object_Detection_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Parameter_Estimation_for_Sinusoidal_Frequency-Modulated_Signals_Using_Phase_Modulation_3.png',

            '/home/lucaslopes/perceptronac/SPL2021/Performance_Ranking_of_Kalman_Filter_With_Pre-Determined_Initial_State_Prior_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Pitch_Estimation_by_Multiple_Octave_Decoders_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Pseudo-Boolean_Functions_for_Optimal_Z-Complementary_Code_Sets_With_Flexible_Lengths_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Quickest_Detection_of_COVID-19_Pandemic_Onset_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Ray-Space-Based_Multichannel_Nonnegative_Matrix_Factorization_for_Audio_Source_Separation_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Reconfigurable_Intelligent_Surface_Aided_Sparse_DOA_Estimation_Method_With_Non-ULA_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Recurrent_Context_Aggregation_Network_for_Single_Image_Dehazing_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Reduced_Biquaternion_Stacked_Denoising_Convolutional_AutoEncoder_for_RGB-D_Image_Classification_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Reversible_Data_Hiding_for_JPEG_Images_Based_on_Multiple_Two-Dimensional_Histograms_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Risk-Aware_Multi-Armed_Bandits_With_Refined_Upper_Confidence_Bounds_1.png',
            
            '/home/lucaslopes/perceptronac/SPL2021/Robust_Localization_Employing_Weighted_Least_Squares_Method_Based_on_MM_Estimator_and_Kalman_Filter_With_Maximum_Versoria_Criterion_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/SMoA_Searching_a_Modality-Oriented_Architecture_for_Infrared_and_Visible_Image_Fusion_4.png',
            '/home/lucaslopes/perceptronac/SPL2021/Semidefinite_Programming_Two-Way_TOA_Localization_for_User_Devices_With_Motion_and_Clock_Drift_2.png',
            '/home/lucaslopes/perceptronac/SPL2021/Signal_Lag_Measurements_Based_on_Temporal_Correlations_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Simultaneous_Graph_Learning_and_Blind_Separation_of_Graph_Signal_Sources_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Single_Haze_Image_Restoration_Under_Non-Uniform_Dense_Scattering_Media_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Single_Image_Deraining_Integrating_Physics_Model_and_Density-Oriented_Conditional_GAN_Refinement_5.png',
            '/home/lucaslopes/perceptronac/SPL2021/Split_Bregman_Approach_to_Linear_Prediction_Based_Dereverberation_With_Enforced_Speech_Sparsity_1.png',
            '/home/lucaslopes/perceptronac/SPL2021/Staged-Learning_Assessing_the_Quality_of_Screen_Content_Images_from_Distortion_Information_3.png',
            '/home/lucaslopes/perceptronac/SPL2021/Stereo_Feature_Enhancement_and_Temporal_Information_Extraction_Network_for_Automatic_Music_Transcription_3.png',

        ]
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
        which_model,hidden_units,n_layers=n_layers,samples_per_time=samples_per_time,n_pieces=100)

