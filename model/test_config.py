from os.path import join, exists
from os import makedirs
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:
    experiment_root = '/data1/yinzihao/bone_age'
    img_root = join(experiment_root, 'keypoint_hard_examples', 'raw_img')
#     keypoint_checkpoint_path = join(experiment_root, 'keypoint_model', 'c0c1_randomdrop_0115_1542', 'best.pth.tar')
    keypoint_checkpoint_path = join(experiment_root, 'keypoint_model', 'c_all_randomdrop_0115_2119', 'best.pth.tar')
    index_path = '/data1/yinzihao/bone_age/keypoint_hard_examples/single_hand.txt'
#     index_path = None
    result_root = join(experiment_root, 'result')
    img_tag, model_tag = img_root.split('/')[-1], keypoint_checkpoint_path.split('/')[-2]
    save_dir = join(result_root, img_tag + '_' + model_tag) 
    
    model = 'CPN101' # option 'CPN50', 'CPN101'
    paf_flag = False
    paf_version = 2
    num_class = 15
    norm_type = 'instance'
    
    data_shape = (384, 384) # (h, w) 
#     scale_search = [1.0, 0.5, 1.5, 2.0]
    scale_search = [1.0]
    
    vec_pair = [[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 4, 12, 13],  [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 14]] 
    num_vec = len(vec_pair[0])
    stride = 4
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)
    theta = 1.0
     

    ##################################
    # attributes about to be deprecated
    ##################################
    output_shape = (96, 96)
    

cfg = Config()
