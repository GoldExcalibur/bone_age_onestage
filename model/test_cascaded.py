#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import argparse
import time
import matplotlib.pyplot as plt
import pickle
import copy 
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import cv2
import glob
import numpy as np
from os.path import dirname, exists, join, realpath
from os import makedirs
import tqdm

import sys

bone_age_path = dirname(dirname(dirname(realpath(__file__))))
if bone_age_path not in sys.path:
    sys.path.insert(0, bone_age_path)

from keypoint_model.utils.osutils import mkdir_p, isfile, isdir, join
from keypoint_model.utils.transforms import fliplr, flip_back
from keypoint_model.utils.imutils import im_to_numpy, im_to_torch
from keypoint_model.utils import Mytransforms 
# from keypoint_model.cpn_network import network
from keypoint_model.paf_network import network

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],[170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

from packaging import version
if version.parse(torch.__version__) < version.parse('0.4.0'):
    PYTORCH_VERSION_LESS_THAN_040 = True
else:
    PYTORCH_VERSION_LESS_THAN_040 = False
    
def construct_cpn_model(test_cfg, gpu_id, logger=None):
    paf_version = 1
    paf_flag = False
    model = network.__dict__[test_cfg.model](test_cfg.num_class, test_cfg.num_vec, paf_version, paf_flag, pretrained = False,)

    model = torch.nn.DataParallel(model, device_ids=[int(gpu_id)])
    checkpoint_file_path = test_cfg.keypoint_checkpoint_path
    checkpoint = torch.load(checkpoint_file_path, map_location = lambda storage, loc:storage.cuda(int(gpu_id)))
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k in model.state_dict().keys():
        if k in state_dict:
            new_state_dict[k] = state_dict[k]

    if PYTORCH_VERSION_LESS_THAN_040:
        model.load_state_dict(new_state_dict, strict=False) # strict = False for 0.3.1
    else:
        model.load_state_dict(new_state_dict) # pytorch 0.4.1
    msg = "=> [BoneAge-CPN] loaded checkpoint '{}' (epoch {})".format(checkpoint_file_path, checkpoint['epoch'])
    if logger is not None:
        logger.info(msg)
    print(msg)    
    # change to evaluation mode
    model.eval()
    return model

def get_test_imglist(img_root, vallist=None):
    print 'loading test pics from', img_root
    if vallist is not None:
        with open(vallist) as f:
            dir_list = f.readlines()
        
        img_path_list = [join(img_root, (threeid.strip()).replace('/', '|') + '.png') for threeid in dir_list] 
    else:
        img_path_list = glob.glob(join(img_root,  '*.png'))
    return img_path_list 

def draw(img, result):
    m = copy.deepcopy(img)
    for j, pts in enumerate(result):
        x = int(round(pts[0]))
        y = int(round(pts[1]))
        cv2.circle(m, (x, y), 10, colors[j], -1)
    return m

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
    
def preprocess(origin_img, shape, gpu_id):
   
    img = cv2.resize(origin_img, shape, interpolation = cv2.INTER_CUBIC)
    input_totest = Mytransforms.instance_normalize(im_to_torch(img)).unsqueeze(0)
    input_totest = input_totest.cuda(gpu_id)
    return input_totest

def process(model, input_path, gpu_id, kp_in_size, verbose = False):
    """
    kp_in_size:  (H, W),  not shape_list is in (W, H) order
    """
    gpu_id = int(gpu_id)
    torch.cuda.set_device(gpu_id)
    if isinstance(input_path, np.ndarray):
        origin_img = input_path
    else:
        print 'Loading image for CPN ...'
        if input_path.endswith('.npz'):
            origin_img = np.load(open(input_path))['arr_0']
        elif input_path.endswith('.png'):
            origin_img = cv2.imread(input_path)[:, :, 0]
        else:
            raise Exception('Invalid Hand Image type format!')
            
    if get_surround_gray_val(origin_img) > 120.0:
        origin_img = 255.0 - origin_img
    origin_img = np.tile(origin_img[:,:, np.newaxis], (1, 1, 3))
    
    shape_list =  [(kp_in_size[1], kp_in_size[0])]

    input_totest_list = [preprocess(origin_img, shape, gpu_id) for shape in shape_list]
    
    score_map_list = []
    for input_totest in input_totest_list:
        if PYTORCH_VERSION_LESS_THAN_040:
            global_outputs, refine_output = model(Variable(input_totest, volatile=True))  # pytorch 0.3.1
        else:
            with torch.no_grad(): # requires_grad=True
                global_outputs, refine_output = model(input_totest)
        # print input_totest.size()
        score_map = refine_output.data.cpu()
        score_map = score_map.numpy()
        score_map_list.append(score_map)
        
    single_map_avg = None
    for score_map in score_map_list:
        single_map = np.transpose(score_map[0], (1, 2, 0))
        single_map_4 = cv2.resize(single_map, (0, 0), fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        single_map_original = cv2.resize(single_map_4, (origin_img.shape[1], origin_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        if single_map_avg is None:
            single_map_avg = single_map_original
        else:
            single_map_avg += single_map_original
            
    single_map_avg /= len(score_map_list)
    pts = get_x_y_from_heatmap(single_map_avg)
    
    if verbose:
        return  origin_img, pts
    else:
        return pts

def get_x_y_from_heatmap(heat_map):
    single_map = copy.deepcopy(heat_map)
    h, w, c= single_map.shape
#     print 'input heatmap size:', (h, w, c)
    single_result = []
    for p in range(15): 
        single_map[:, :, p] /= np.amax(single_map[:, :, p])
        border = 10
        dr = np.zeros((h + 2*border, w + 2*border))
#                     print dr.shape, single_map.shape
        dr[border:-border, border:-border] = single_map[:, :, p].copy()
        dr = cv2.GaussianBlur(dr, (21, 21), 0)
        lb = dr.argmax()
        y, x = np.unravel_index(lb, dr.shape)
        
        dr[max(0, y-30) : min(y + 30, h+2*border), max(0, x-30): min(x+30, w+2*border)] = 0
        lb = dr.argmax()
        py, px = np.unravel_index(lb, dr.shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 10:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        single_result.append([x, y])
    return single_result

def criterion_kp(predict_pts, truth_pts):
    type1, type2 = type(predict_pts).__name__, type(truth_pts).__name__
    assert type1  == 'ndarray' and type2 == 'ndarray'
    l2_dis = np.sqrt(np.sum((predict_pts - truth_pts) ** 2, 1))
    
    mean_l2_dis = np.mean(l2_dis)
    return mean_l2_dis

def main(args):
    gpu_id = args.g
    from test_config import cfg
    from keypoint_model.keypoint_datasets import c3b1_datasets, c4b1_datasets
    from keypoint_model.boneageMulti import boneage_loader, ConcatDataset
    
    kp_shape = cfg.data_shape
    kp_scale_search = cfg.scale_search
    
    
    strong_transform = Mytransforms.Compose([
                Mytransforms.RandomRotate(15, prob=0.5),
                Mytransforms.RandomHorizontalFlip(prob=0.5),
                Mytransforms.RandomVerticalFlip(prob=0.5),
                Mytransforms.RandomDrop(prob_list = [0.4, 0.3, 0.3]),
                Mytransforms.Resized(target_size = kp_shape),
            ])
    
    weak_transform = Mytransforms.Compose([
               Mytransforms.Resized(target_size = kp_shape)
          ])
    
    test_transform = {'strong': strong_transform, 'weak': weak_transform}
    
    c3b1_test = boneage_loader(c3b1_datasets, cfg, 'test', test_transform['weak'])
    c4b1_test = boneage_loader(c4b1_datasets, cfg, 'test', test_transform['weak'])
    
    test_dataset = ConcatDataset([c3b1_test, c4b1_test])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = args.w)
    print test_loader.__len__()
   
    model = construct_cpn_model(cfg, gpu_id)

    img_root = cfg.img_root
    val_list = cfg.index_path
    save_dir  = cfg.save_dir
    if exists(save_dir) == False:
        makedirs(save_dir)
        
    img_path_list= get_test_imglist(img_root, val_list)
    print len(img_path_list)
    
    '''
    mean_l2_dis = []
    for i, (inputs, targets, valid, meta) in enumerate(test_loader):

        img, pts = inputs.squeeze(0).numpy()[0], targets.squeeze(0).numpy()
        origin_img = cv2.imread(meta['img_path'][0])
        predict_pts =  process(model, origin_img[:, :, 0], kp_shape, gpu_id, verbose = False)
        l2_dis = criterion_kp(np.array(predict_pts), pts)
        mean_l2_dis.append(l2_dis)
        

        img_with_pts = draw(origin_img, predict_pts)
        img_with_label = draw(img_with_pts, pts)
        save_path = join(save_dir, meta['subdir'][0].replace('/', '|') + '.png')
        cv2.imwrite(save_path, img_with_label)
        
    print np.mean(np.array(mean_l2_dis))
    assert 0
    
    '''
    for cnt, img_path in enumerate(tqdm.tqdm(img_path_list)):
        origin_img = cv2.imread(img_path)[:, :, 0]
        img, pts = process(model, origin_img, kp_shape, gpu_id, verbose = True)
        img_with_pts = draw(img, pts)
        save_path = img_path.replace(img_root, save_dir)
        cv2.imwrite(save_path, img_with_pts)
        
    return 
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('--w', '-workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--g',  default='6', type=str, metavar='N',
                        help='id of GPU to use (default: 6)')  
    args = parser.parse_args()
    main(args)
