
import torch
import time
import os
from perceptronac.models import MLP_N_64N_32N_1
from perceptronac.main_experiment import experiment

if __name__ == "__main__":

    configs = {
        "id": str(int(time.time())),
        "ModelClass":MLP_N_64N_32N_1,
        "OptimizerClass":torch.optim.SGD,
        "training_set": [
            '/home/lucas/Documents/data/TIP2020/2D_Quaternion_Sparse_Discriminant_Analysis_14.png',
            '/home/lucas/Documents/data/TIP2020/Blind_Deblurring_of_Text_Images_Using_a_Text-Specific_Hybrid_Dictionary_10.png',
            '/home/lucas/Documents/data/TIP2020/Deep_Heterogeneous_Hashing_for_Face_Video_Retrieval_13.png',
            '/home/lucas/Documents/data/TIP2020/Deep_Learning-Based_Picture-Wise_Just_Noticeable_Distortion_Prediction_Model_for_Image_Compression_15.png',
            '/home/lucas/Documents/data/TIP2020/MV-GNN_Multi-View_Graph_Neural_Network_for_Compression_Artifacts_Reduction_5.png',
            '/home/lucas/Documents/data/TIP2020/Needles_in_a_Haystack_Tracking_City-Scale_Moving_Vehicles_From_Continuously_Moving_Satellite_1.png',
            '/home/lucas/Documents/data/TIP2020/Perceptual_Temporal_Incoherence-Guided_Stereo_Video_Retargeting_8.png',
            '/home/lucas/Documents/data/TIP2020/Personalized_Image_Enhancement_Using_Neural_Spline_Color_Transforms_13.png',
            '/home/lucas/Documents/data/TIP2020/The_Fourier-Argand_Representation_An_Optimal_Basis_of_Steerable_Patterns_11.png',
            '/home/lucas/Documents/data/TIP2020/Visual_Saliency_via_Embedding_Hierarchical_Knowledge_in_a_Deep_Neural_Network_10.png'
        ],
        "validation_set": [
            '/home/lucas/Documents/data/TIP2021/3DCD_Scene_Independent_End-to-End_Spatiotemporal_Feature_Learning_Framework_for_Change_Detection_in_Unseen_Videos_6.png'
        ],
        "epochs": 1, # 300,
        "learning_rate": 0.00001,
        "batch_size": 2048,
        "num_workers":4,
        "device":"cuda:0", #"cpu"
        "parent_id": "",
        "unfinished_training": 0, # 0 or 1
        "N_vec": sorted([0] + [round(pow(1.595, i)) for i in range(12) if (i+1)%2==0],reverse=False),
        "phases": ['train','valid'], #['train', 'valid'], # ['coding'],
        "xscale": 'symlog',
        "reduction": 'min', # min, last
        "data_type": 'image', # image, pointcloud
        "geo_or_attr": "attributes", # geometry, attributes
        "n_classes": 2, # 2, 256
        "channels": [1,0,0], # [1,1,1], [1,0,0]
        "color_space": "YCbCr", # RGB, YCbCr
        "percentage_of_uncles": 0.0, # must be specified if the data types is pointcloud
        "last_octree_level": 10, # must be specified if the data types is pointcloud
        "save_dir": "results",
        "max_context": 27, # cabac
        "dset_pieces": 1, # if not enough memory to hold all data at once, specify into how many pieces the data should be divided
        "methods": ["LUT"], # ["MLP","LUT","JBIG1"],
        "manual_th": 0.4, # 0,
        "full_page": False, # True
    }

    experiment(configs)

