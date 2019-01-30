#!/usr/bin/python  
# -*- coding: utf-8 -*-  
import os
import argparse
import time
import matplotlib.pyplot as plt
import pickle
import copy 
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import glob
import numpy as np

import sys
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from utils import Mytransforms 
from keypoint_model.cpn_network import network

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],[170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def construct_cpn_model(test_cfg, gpu_id, logger=None):
    model = network.__dict__[test_cfg.model](test_cfg.num_class, pretrained = False)
    model = torch.nn.DataParallel(model, device_ids=[int(gpu_id)])
    checkpoint_file_path = test_cfg.keypoint_checkpoint_path
    checkpoint = torch.load(checkpoint_file_path, map_location = lambda storage, loc:storage.cuda(int(gpu_id)))
    model.load_state_dict(checkpoint['state_dict'])
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
        
        img_path_list = [os.path.join(img_root, os.path.dirname(x), 'png_data_rotated.npz') for x in dir_list] 
    else:
        img_path_list = glob.glob(os.path.join('/data1/yinzihao/boneage_data/wuxi_201807', '*/*/*/png_data_rotated.npz'))
    return img_path_list 

def draw(img, result):
    m = copy.deepcopy(img)
    for j, pts in enumerate(result):
        x = int(round(pts[0]))
        y = int(round(pts[1]))
        cv2.circle(m, (x, y), 4, colors[j], -1)
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

def process(model, input_path, gpu_id, verbose = False):
    gpu_id = int(gpu_id)
    torch.cuda.set_device(gpu_id)
    if isinstance(input_path, np.ndarray):
        origin_img = input_path
    else:
        print 'Loading image for CPN ...'
        origin_img = np.load(open(input_path))['arr_0']
    h, w = origin_img.shape
        
    if get_surround_gray_val(origin_img) > 120.0:
        origin_img = 255.0 - origin_img
    origin_img = np.tile(origin_img[:,:, np.newaxis], (1, 1, 3))
    
    shape_list =  [(288+i*32, 384+i*32) for i in [0]]
    
    input_totest_list = [preprocess(origin_img, shape, gpu_id) for shape in shape_list]
    
    score_map_list = []
    for input_totest in input_totest_list:
        with torch.no_grad():
            global_outputs, refine_output = model(input_totest)
#         print input_totest.size()
        score_map = refine_output.data.cpu()
        score_map = score_map.numpy()
        score_map_list.append(score_map)
        
    single_map_avg = None
    for score_map in score_map_list:
        single_map = np.transpose(score_map[0], (1, 2, 0))
        single_map_4 = cv2.resize(single_map, (0, 0), fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        '''
        single_map_original = cv2.resize(single_map_4, (origin_img.shape[1], origin_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        '''
        
        if single_map_avg is None:
            single_map_avg = single_map_4
        else:
            single_map_avg += single_map_4
            
    single_map_avg /= len(score_map_list)
    pts = get_x_y_from_heatmap(single_map_avg, (w/288.0, h/384.0))
    
    if verbose:
        result_img = draw(origin_img, pts)
        return  result_img
    else:
        return pts

def get_x_y_from_heatmap(heat_map, stride):
    single_map = copy.deepcopy(heat_map)
    h, w, c= single_map.shape
    single_result = []
    
    for p in range(15):
#         single_map[:, :, p] /= np.amax(single_map[:, :, p])
        border = 3
        kernel_size = 2*border + 1
        dr = np.zeros((h + 2*border, w + 2*border))
        dr[border:-border, border:-border] = single_map[:, :, p].copy()
        dr = cv2.GaussianBlur(dr, (kernel_size, kernel_size), 0)

        lb = dr.argmax()
        y0, x0 = np.unravel_index(lb, dr.shape)
        
#         dr[y0, x0] = 0 
        dr[max(0, y0 - 3) : min(y0 + 3, h + 2*border), max(0, x0 - 3): min(x0 + 3, w + 2*border)] = 0
        lb = dr.argmax()
        y1, x1 = np.unravel_index(lb, dr.shape)
        
        py, px = y1 - y0, x1 - x0
  
        ln = (px ** 2 + py ** 2) ** 0.5
        
        x, y = x0 - border, y0 - border
        delta = 0.0
        if ln > 0.0:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
      
        single_result.append([x * stride[0], y * stride[1] ])
        
    return single_result

def main(args):
    # create model
    model = construct_model(cfg)

    img_root = cfg.img_root
    val_list = cfg.index_path
    save_dir  = cfg.save_dir
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    print('start testing...')
    img_path_list= get_test_imglist(img_root, val_list)
    print len(img_path_list)
    
   
    for cnt, img_path in enumerate(np.array(img_path_list)):
       
        pts = process(model, img_path, verbose = False)
        
        '''
#         以下代码常用来做inference 为新图片生成关键点的label
        dir_names =  img_path.split('/')[-4:]
        three_level = os.path.join(dir_names[0], dir_names[1], dir_names[2])
        save_path = os.path.join(save_dir, three_level)
        result_dict = pickle.load(open(os.path.join(save_path, 'label.pickle')))
        result_dict['keypoint'] = pts
        result_dict['path'] = os.path.join(three_level, 'png_data_rotated.npz')
        
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, 'label.pickle'), 'wb') as f:
            pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print cnt
        '''
        dir_list = img_path.split('/')
        output_path = os.path.join(save_dir, dir_list[-4] + '.png')
        print('writing an immage at', output_path)
        cv2.imwrite(output_path, result_img)
        
        
    return 
      


if __name__ == '__main__':
    from test_config import cfg
    # scale_research = cfg.scale_research
    box_size = cfg.data_shape
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('--g',  default='6', type=str, metavar='N',
                        help='id of GPU to use (default: 6)')  
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    main(parser.parse_args())
