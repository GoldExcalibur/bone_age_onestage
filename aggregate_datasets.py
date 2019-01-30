#coding=utf-8
import glob
from os.path import join
import pickle


__all__ = ['c1b1_datasets', 'c3b1_datasets', 'c2b2_datasets', 'c3b1old_datasets']

class hdatasets(object):
    def __init__(self, crop_root, deg_ann_list, kp_label_root, index_root, index_path_dict, area):
        self.crop_root = crop_root
        self.deg_ann_list = deg_ann_list
        self.kp_root = kp_label_root
        self.index_root = index_root
        self.index_path_dict = index_path_dict
        self.area = area

c1b1_ann_list = glob.glob('/data1/bone_age/ann/degree/骨骺等级标注WX_0_*.txt')
c2b2_ann_list =  ['/data1/bone_age/ann/degree_seg/task_136_c2_b2_db1st_0115.json']
c3b1_old_ann_list = glob.glob('/data1/bone_age/ann/degree/骨龄等级标注C3_B1_*.json')
c3b1_new_ann_list =  glob.glob('/data1/bone_age/ann/degree/task_*_20181229.json')


c1b1_root = '/data1/bone_age/data/c1/b1'
c1b1_settings = {
    'rus':{'input_root': join(c1b1_root, 'crop')},
}
c1b1_datasets = hdatasets(
    join(c1b1_root, 'hand'),
    c1b1_ann_list,
    join(c1b1_root, 'label'),
    join(c1b1_root, 'sub_dir'),
    {'train': 'wx1426_train.txt', 'val': 'wx1426_val.txt', 'test': 'wx1426_test.txt', 'drop': None},
    'rus'
)

c3b1_root = '/data1/bone_age/data/c3/b1'
c3b1_settings = {
    'rus': {'input_root': join(c3b1_root, 'crop')}
}
c3b1_datasets = hdatasets(
    join(c3b1_root, 'hand'),
    c3b1_new_ann_list,
    join(c3b1_root, 'label'),
    join(c3b1_root, 'sub_dir'),
    {'train': 'train_sub_dir_2486.txt', 'val': 'valid_sub_dir_99.txt', 'test': 'test_sub_dir_99.txt', 'drop':  'keypoints_fail.txt'},
    'rus'
)

c3b1old_datasets = hdatasets(
    join(c3b1_root, 'hand'),
    c3b1_old_ann_list,
    join(c3b1_root, 'label'),
    join(c3b1_root, 'sub_dir'),
    {'train': 'train_sub_dir_2486.txt', 'val': 'valid_sub_dir_99.txt', 'test': 'test_sub_dir_99.txt', 'drop': 'keypoints_fail.txt'},
    'rus'
)

c2b2_root = '/data1/bone_age/data/c2/b2'
c2b2_settings = {
    'rus': {'input_root': join(c2b2_root, 'crop')}
}
c2b2_datasets = hdatasets(
    join(c2b2_root, 'hand'),
    c2b2_ann_list,
    join(c2b2_root, 'label'),
    join(c2b2_root, 'sub_dir'),
    {'train': 'train_pri_sub_dir_1380.txt', 'val':'valid_sub_dir_99.txt', 'test':'test_sub_dir_99.txt', 'drop': None},
    'rus'
)        

