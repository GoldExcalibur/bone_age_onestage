#coding:utf-8

import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform
import pickle
import torch
import torch.utils.data as data
from utils import Mytransforms
from utils import imutils
from utils.prep_pipeline import PrepPipe
from os.path import join, isfile, isdir, abspath, normpath
from utils.sub_dir_util import load_sub_dir

def read_file_name(file_name):
    return load_sub_dir(file_name, verbose=False)

def get_pts(data, allow_out_of_order=False, verbose=False):
    tags = [u'第五中节指骨远端', u'第五近节指骨远端', u'第五近节指骨近端', u'第五掌骨远端', u'第五掌骨近端', 
           u'第三中节指骨远端', u'第三近节指骨远端', u'第三近节指骨近端', u'第三掌骨远端', u'第三掌骨近端',
           u'第一近节指骨远端', u'第一掌骨远端', u'第一掌骨近端', u'尺骨', u'桡骨'
           ]
    pts_list = []
    pts_dict = {}
    if len(data) != 15:
        raise Exception('Expecting 15 RUS keypoint, got %d' % len(data))
    for pos_idx in range(15):
        descText = data[pos_idx]['descText']
        if len(descText) >= 2:
            chn_str = descText[1][0][0]['select']
            if chn_str not in [tags[pos_idx], u"%d-%s" % (pos_idx+1, tags[pos_idx])]:
                if not allow_out_of_order:
                    if verbose:
                        print u'annotated tag: %s, expected tag: %s' % (chn_str, tags[pos_idx])
                    raise Exception('tag not in correct order!')
                elif chn_str in pts_dict:
                    if verbose:
                        print u'annotated tag: %s, annotated more than once!' % (chn_str)
                    raise Exception('tag annotated more than once')
        pts = data[pos_idx]['rois'][0]['edge'][0]
        pts_list.append(pts)
        if allow_out_of_order and len(descText) >= 2:
            pts_dict[chn_str] = pts
    if len(pts_dict) == 15:
        return [pts_dict[tag] for tag in tags]
    else:
        return pts_list

def process_label(ann_list, verbose=False):
    result_dict = {}
    fail_list = []
    skip_dict = {}
    total_ann_cnt = 0
    for ann_path in ann_list:
        json_data_list = json.load(open(ann_path))
        total_ann_cnt += len(json_data_list)
        for idx, json_data in enumerate(json_data_list):
            pid = json_data['patientID'] 
            study_id = json_data['studyUID']
            series_id = json_data['seriesUID']
            threeid = os.path.join(pid, study_id, series_id)
            doctor_id = json_data['other_info']['doctorId']
            if verbose:
                print '[parse keypoint]', threeid
            try:
                nodes = json_data['nodes']
                if len(nodes) == 0:
                    skip_dict[threeid] = json_data[u'other_info'][u'passReason']
                    if verbose:
                        print '\t[keypoint json: skip] %s: %s' % (threeid, skip_dict[threeid])
                    continue
                else:
                    # if 'c0_b1' in ann_path:
                    #     pts_list = get_pts(nodes, allow_out_of_order=True, verbose=verbose)
                    pts_list = get_pts(nodes, allow_out_of_order=False, verbose=verbose)
            except Exception as e:
                fail_list.append(threeid)
                if verbose:
                    print '\t[keypoint json: fail] %s: %s' % (threeid, str(e))
                continue
                
            if threeid not in result_dict.keys():
                result_dict[threeid] = [pts_list]
            else:
                result_dict[threeid].append(pts_list)

    print '-' * 80
    print 'Loading keypoint from json: Total %d annotated' % total_ann_cnt
    print 'Success {} Skip {} Fail {}'.format(len(result_dict), len(skip_dict), len(fail_list))

    return result_dict, fail_list


def cal_mean_keypoint(kp_dict):
    """
    Input: 
      sub_dir -> 3dim list: nDoc * 15  * 2
    Output:
      sub_dir -> 2dim list 15 * 2
    """
    print 'calculatign mean RUS keypoints ...',
    mean_kp_dict = {}
    for sub_dir, kp_list in kp_dict.items():
        kp_np = np.array(kp_list)
        assert kp_np.ndim == 3
        kp_np = np.mean(kp_list, 0)
        assert kp_np.shape == (15, 2)
        mean_kp_dict[sub_dir] = kp_np
    print '[DONE]'
    return mean_kp_dict

def prep_keypoint(kp_dict, prep_pipe, label_root):
    """
    Input:
      sub_dir -> 2dim list (15 * 2)

    Output:
      sub_dir -> 2dim list (15 * 2)
    """
    print 'preprocessing pipeline ...',
    pp_kp_dict = {}
    for sub_dir, kp_list in kp_dict.items():
        key_info_fname = join(label_root, sub_dir, 'label.pickle')
        key_info = pickle.load(open(key_info_fname))
        prep_pipe.set_params(key_info)
        pp_kp_list = prep_pipe.forward(kp_list)
        pp_kp_dict[sub_dir] = pp_kp_list
    print '[DONE]'
    return pp_kp_dict
        
class boneage_loader(data.Dataset):
    def __init__(self, dataset, config, mode, transformer = None):
    
#         self.index_root = '/home/xingzijian/workspace/Hand_Keypoint_Estimation/indexlist/'
        if mode not in ['train', 'val', 'debug', 'test']:
            raise Exception('mode not in [train, val, debug, test]')
            
        self.mode = mode
        self.index_root = dataset.index_root
        index_path_dict = dataset.index_path_dict
        self.index_path = os.path.join(self.index_root, index_path_dict[self.mode])
        self.drop_index_path = index_path_dict['drop']
        self.ann_list = dataset.ann_list
        self.label_root = dataset.label_root
        kp_ann_dict, fail_list = process_label(self.ann_list)
        kp_ann_dict = cal_mean_keypoint(kp_ann_dict)
        self.prep_pipe = PrepPipe()
        self.kp_pp_dict = prep_keypoint(kp_ann_dict, self.prep_pipe, self.label_root)
        # insert keypoint GT label directly from dataset.prep_keypoint_dict
        direct_prep_label_cnt = 0
        for sub_dir, kp in dataset.prep_keypoint_dict.items():
            if sub_dir not in self.kp_pp_dict:
                self.kp_pp_dict[sub_dir] = dataset.prep_keypoint_dict[sub_dir]
                direct_prep_label_cnt += 1
        if dataset.prep_keypoint_pkl_fname is not None:
            print 'loading %d preprocessed keypoints from %s' % (direct_prep_label_cnt, dataset.prep_keypoint_pkl_fname)
        self.hand_root = dataset.hand_root
        self.index_list = self.select_subdir()

        self.transformer = transformer
        self.num_class = config.num_class
        self.vec_pair = config.vec_pair
        self.stride = config.stride
        self.theta = config.theta
        self.gk15, self.gk11, self.gk9, self.gk7 = config.gk15, config.gk11, config.gk9, config.gk7
        
    def select_subdir(self):
        subdir = read_file_name(self.index_path)
        subdir_json = self.kp_pp_dict.keys()
        subdir_drop = []
        if self.drop_index_path is not None:
            subdir_drop += read_file_name(os.path.join(self.index_root, self.drop_index_path))
         
        index_list = list(set(subdir) & set(subdir_json) - set(subdir_drop))
        print '-' * 80
        print 'subdir read from {} is {}'.format(self.index_path, len(subdir))
        print 'subdir read from json files is {}'.format(len(subdir_json))
        print 'subdir drop is {}'.format(len(subdir_drop))
        print 'their intersection is {}'.format(len(index_list))
        index_list = [t for t in index_list if os.path.exists(os.path.join(self.hand_root, t.replace('/', '|') + '.png'))]
        print 'real number after examine whether hand exists {}'.format(len(index_list))
        print '-' * 80
        return index_list
    
    def __getitem__(self, index):
        threeid = self.index_list[index]
        kp_pp = self.kp_pp_dict[threeid] # keypoints after preprocess pipeline

        img_path = os.path.join(self.hand_root, threeid.replace('/', '|') + '.png')
        image = cv2.imread(img_path)
        h, w, c= image.shape
        for x, y in kp_pp:
            if not(x >= 0 and x < w and y >= 0 and y < h):
                print "kp_pp: ", kp_pp
                print "h, w: ", h, w
                print "x, y: ", x, y
                print "sub_dir: ", threeid
                assert 0
        
        if self.transformer != None:
            img, kp = self.transformer(image, kp_pp)
                        
        points = np.zeros((len(kp), 3))
        for i in range(len(kp)):
            points[i, 2] = 2
            points[i, 0] = kp[i][0]
            points[i, 1] = kp[i][1]
            
        pts = torch.Tensor(points) 
      
        inputs = imutils.im_to_torch(img)
        inputs = Mytransforms.instance_normalize(inputs)

        c, h, w = inputs.size()
        if self.mode != 'test':
            pts[:, :2] //= self.stride
            target_width  =  w// self.stride
            target_height = h//self.stride
           
          
            # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
            target15 = imutils.generate_heatmap(target_height, target_width, self.num_class, pts, self.gk15)
            target11 = imutils.generate_heatmap(target_height, target_width, self.num_class, pts, self.gk11)
            target9 = imutils.generate_heatmap(target_height, target_width, self.num_class, pts, self.gk9)
            target7 = imutils.generate_heatmap(target_height, target_width, self.num_class, pts, self.gk7)
            targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
            vecmap = torch.Tensor(imutils.generate_vector(target_height, target_width, pts, self.vec_pair, self.theta))
            valid = pts[:, 2]
          
            return inputs, targets, vecmap, valid
        else:
            meta = {'subdir' : threeid, 'img_path' : img_path, 'size': (h, w)}
            keypoint = torch.Tensor(kp_pp)
            valid = pts[:, 2]
            return inputs, keypoint, valid, meta
    

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
        

