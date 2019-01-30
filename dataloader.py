import sys
import os
from os.path import join, exists, isdir, isfile, realpath
import pickle
import cv2
import numpy as np
import torch
import glob
from PIL import Image
from utils.Mytransforms import instance_normalize, to_tensor
from data_preprocess.JsonLabelProcesser import JsonLabelProcesser 
from keypoint_epiphysis.utils import imutils

def read_file_name(file):
    path_list = []
    with open(file, 'rb') as f:
        line = f.readline()
        while line:
            path = line.strip()
            path_list.append(path)
            line = f.readline()
    return path_list

def process_label(ann_list):
    yes_docids = None
    no_docids = None
    min_valid_ann = 1
    jlp = JsonLabelProcesser()
    
    rus_dict_mean, carpal_dict_mean = jlp.process_label_files(ann_list, yes_docids, no_docids, min_valid_ann, False)
    assert len(rus_dict_mean) == len(carpal_dict_mean)
    label_dict = {'rus': rus_dict_mean, 'carpal': carpal_dict_mean}
    return label_dict
    
def get_surround_gray_val(img):
    h, w = img.shape
    h_width = int(round(h/10.0))
    w_width = int(round(w/10.0))
    m1 = np.mean(img[h_width:2*h_width]) 
    m2 = np.mean(img[-2*h_width:-h_width])
    m3 = np.mean(img[:, w_width:2*w_width])
    m4 = np.mean(img[:, -2*w_width:-w_width])
    m = (m1 + m2 + m3 + m4)/4.0
    return m

class AggDataLoader(object):
    def __init__(self, dataset, config, area, mode, transformer = None):
        self.index_root = dataset.index_root
        index_path_dict = dataset.index_path_dict
        if mode in index_path_dict.keys():
            self.index_path = join(self.index_root, index_path_dict[mode])
        else:
            raise Exception('mode not supported yet')

        self.drop_index_path =  index_path_dict['drop']

        self.deg_ann_list = dataset.deg_ann_list
        self.kp_root = dataset.kp_root
        self.area = area
        self.deg_label= process_label(self.deg_ann_list)[self.area]
        self.img_crop_root = dataset.crop_root
        self.index_list = self.select_subdir()

        self.transformer = transformer
        
        self.num_class = config.num_class
        self.vec_pair = config.vec_pair
        self.stride = config.stride
        self.theta = config.theta
        self.gk15, self.gk11, self.gk9, self.gk7 = config.gk15, config.gk11, config.gk9, config.gk7
      
    def select_subdir(self):
        subdir = read_file_name(self.index_path)
        subdir_json = self.deg_label.keys()
        subdir_drop = []
        if self.drop_index_path is not None:
            subdir_drop += read_file_name(join(self.index_root, self.drop_index_path))
         
        index_list = list(set(subdir) & set(subdir_json) - set(subdir_drop))
        print '#' * 80
        print 'subdir read from {} is {}'.format(self.index_path, len(subdir))
        print 'subdir read from json files is {}'.format(len(subdir_json))
        print 'subdir drop is {}'.format(len(subdir_drop))
        print 'their intersection is {}'.format(len(index_list))
        index_list = [t for t in index_list if exists(join(self.img_crop_root, t.replace('/', '|')+'.png'))]
        print 'real number after examine whether crop img exists {}'.format(len(index_list))
        print '#' * 80
        return index_list
        
    def __getitem__(self, index, transformer = None):
        threeid = self.index_list[index]
        input_path = join(self.img_crop_root, threeid.replace('/', '|')+ '.png')
        input_png = cv2.imread(input_path)
        
        degrees = self.deg_label[threeid]
        deg_targets = torch.Tensor(degrees)
        
        kp_path = join(self.kp_root, threeid, 'label.pickle')
        kp_list = pickle.load(open(kp_path))['keypoint']
        kp_original = np.array([np.array(k) for k in kp_list])
        
        if self.transformer is not None:
            input_tensor, kp_transformed = self.transformer(input_png, kp_original)
        input_tensor = instance_normalize(to_tensor(input_tensor)) 
       
       
        c, h, w = input_tensor.size()
        '''
        if c == 3:
            input_tensor = input_tensor[0:1, :, :]
        '''   
        target_height = int(h // self.stride)
        target_width = int(w // self.stride)
        kp = kp_transformed / self.stride
       
        target15 = imutils.generate_heatmap(target_height, target_width, self.num_class, kp, self.gk15)
        target11 = imutils.generate_heatmap(target_height, target_width, self.num_class, kp, self.gk11)
        target9 = imutils.generate_heatmap(target_height, target_width, self.num_class, kp, self.gk9)
        target7 = imutils.generate_heatmap(target_height, target_width, self.num_class, kp, self.gk7)
        kp_targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
        # vecmap = torch.Tensor(imutils.generate_vector(target_height, target_width, pts, self.vec_pair, self.theta))
      
       
        return input_tensor, deg_targets, kp_targets
        
    def __len__(self):
        return len(self.index_list)
   
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.dataset_list = datasets
        self.lengths = [d.__len__() for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)
        
    def __getitem__(self, index):
        real_index = index
        for i, offset in enumerate(self.offsets):
            if real_index < offset:
                if i > 0:
                    real_index -= self.offsets[i - 1]
                return self.dataset_list[i][real_index]
        raise IndexError('index %d exceed length' % (index))
        
    def __len__(self):
        return self.length
        
