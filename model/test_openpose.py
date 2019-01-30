import argparse
import os
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.append('..')

import torch
import cv2
import pickle
import glob

from utils.imutils import im_to_numpy, im_to_torch
from new_networks import network

import keypoint_estimation
import Mytransforms

boxsize = 368
scale_search = [0.5, 1.0, 1.5, 2.0]
stride = 8
padValue = 0.
thre_point = 0.015 - 0.015
thre_line = 0.05
stickwidth = 4

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],[170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def get_surround_gray_val(image):
    img = image.copy()
    h, w = img.shape
    h_width = int(round(h/10.0))
    w_width = int(round(w/10.0))
    m1 = np.mean(img[h_width:2*h_width]) 
    m2 = np.mean(img[-2*h_width:-h_width])
    m3 = np.mean(img[:, w_width:2*w_width])
    m4 = np.mean(img[:, -2*w_width:-w_width])
    m = (m1 + m2 + m3 + m4)/4.0
    return m
    
def construct_model(args):

    model = keypoint_estimation.KeypointModel(num_point=16, num_vector=10)
    print args.model
    state_dict = torch.load(args.model)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    state_dict = model.state_dict()
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    return model

def get_test_imglist(img_root, vallist=None):
    print 'loading test pics from', img_root
    if vallist is not None:
        with open(vallist) as f:
            dir_list = f.readlines()
        img_path_list = [os.path.join(img_root, os.path.dirname(x), 'png_data_rotated.npz') for x in dir_list] 
    else:
        img_path_list = glob.glob(os.path.join(img_root, '*/*/*/png_data_rotated.npz'))
    return img_path_list 



def process(model, input_path):

    origin_img = np.load(open(input_path))['arr_0']
    if get_surround_gray_val(origin_img) > 127.0:
        origin_img = 255.0 - origin_img
    origin_img = np.tile(origin_img[:,:, np.newaxis], (1, 1, 3))
    

    height, width, _ = origin_img.shape

    multiplier = [(x * boxsize / width, x * boxsize / height) for x in scale_search]

    heatmap_avg = np.zeros((height, width, 16)) # num_point

    for m in range(len(multiplier)):
        scale = multiplier[m] # a tuple, [0] is x-axis scale, [1] is y-axis scale

        # preprocess
        imgToTest = cv2.resize(origin_img, (0, 0), fx=scale[0], fy=scale[1], interpolation=cv2.INTER_CUBIC)
#         input_img = 255.0 - imgToTest
#         input_img = Mytransforms.normalize(Mytransforms.to_tensor(imgToTest),  [0, 0, 0], [255.0, 255.0, 255.0])
        input_img = Mytransforms.instance_normalize(Mytransforms.to_tensor(imgToTest))
        input_img = input_img.unsqueeze(0)
        
        '''
        input_img = np.transpose(imgToTest[:,:,:,np.newaxis], (3, 2, 0, 1)) # required shape (1, c, h, w)
        input_img = torch.from_numpy(input_img)
        input_img = input_img.float()
        input_img.div_(255.0)
        '''
        
        input_img = input_img.cuda()
        # get the features
        with torch.no_grad(): 
            _, _, _, _, _, _, _, _, _, _, vec6, heat6 = model(input_img)

        # get the heatmap
        print heat6.shape
        heatmap = heat6.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

    
    
    peak_counter = 0
    max_peaks = []
    for part in range(1, 16):
        map_ori = heatmap_avg[:, :, part]
        map_blur = gaussian_filter(map_ori, sigma=3)

        map_blur_left = np.zeros(map_blur.shape)
        map_blur_left[:, 1:] = map_blur[:, :-1]
        map_blur_right = np.zeros(map_blur.shape)
        map_blur_right[:, :-1] = map_blur[:, 1:]
        map_blur_up = np.zeros(map_blur.shape)
        map_blur_up[1:, :] = map_blur[:-1, :]
        map_blur_down = np.zeros(map_blur.shape)
        map_blur_down[:-1, :] = map_blur[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
                (map_blur >= map_blur_left, map_blur >= map_blur_right, map_blur >= map_blur_up, map_blur >= map_blur_down, map_blur > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # (w, h)

        # a point format: (w, h, score, number), number is the seq of keypoint response, the number is only
        # in our case, the each channel, there should only one response
        max_idx = 0
        max_score = 0
        for idx in xrange(len(peaks)):
            cur_peak = peaks[idx]
            cur_score = map_ori[cur_peak[1], cur_peak[0]]
            if cur_score > max_score:
                max_idx = idx
                max_score = cur_score
     
        max_peaks.append([peaks[max_idx][0], peaks[max_idx][1]])
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        idx = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (idx[i], ) for i in range(len(idx))]
       
        peak_counter += len(peaks)
   
    # draw points
    canvas = np.load(open(input_path))['arr_0']
    canvas = np.tile(canvas[:,:, np.newaxis], (1, 1, 3))
    canvas_point = canvas.copy()
    for i in range(15):
        if len(max_peaks[i]) > 0:
            cv2.circle(canvas_point, (int(max_peaks[i][0]), int(max_peaks[i][1])), 4, colors[i], thickness=-1)

    return canvas_point, max_peaks


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_list', type=str, default = None, help='input image list')
    parser.add_argument('--save_dir', type=str, default='save_img', help='the directory to save output img')
    parser.add_argument('--model', type=str, default='../training/hand_keypoint_best.pth.tar', help='path to the weights file')
    parser.add_argument('--img_root', type=str, default = '/data1/boneage_uint8_data', help = 'path to load test pics')
    
    args = parser.parse_args()
    args.img_root = '/data1/yinzihao/boneage_data/wuxi_201807'
    args.val_list = None
    args.model = '/home/xingzijian/workspace/Hand_Keypoint_Estimation/training/data_0820_model/hand_keypoint_best.pth.tar'
    args.save_dir = '/data1/yinzihao/inf_wuxi_201807_openpose_new'
    img_root = args.img_root
    val_list = args.val_list
    save_dir = args.save_dir
    
    
#     args.model = args.model[:26] + pre_net + str(stages) + '_' + args.model[26:]
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        
    # load model
    model = construct_model(args)

    tic = time.time()
    print('start processing...')
    # generate image with body parts
    
    img_path_list = get_test_imglist(img_root,  val_list) 
    
#     assert len(img_path_list) == 1241
    test_list = [939,   74,  853, 1014,  305, 1045,  451, 1019,  466,  532,  510,
        659,  398,  441,  288,  774,  761,  578,  188,  280, 1218,  679,
         11,  643,  294,  764,  614,  702, 1005, 1107,  640,  740, 1105,
       1050,  113,  229, 1184,  307,  832,  928, 1067,  414, 1207,  473,
        936,  955,   87,  357,  985,  911,  416,  589,  731,  873,  845,
        743,  881,  858,  814, 1219,  739,  663,  837,  399,  817,  180,
       1003,  153, 1130,  906, 1092,  360, 1153,  436,  455, 1149,   63,
       1134,  945, 1022,  613, 1181,  760,  330,   65,  996,  802,  542,
        989,  813,  306,  768,  932,  688, 1136, 1164,  114,  475,   59,
        105,  189,  697,   29,  478,  751,  291, 1061,  844, 1051, 1158,
       1090,  167,  736,  538,  225,  198,  518,  965,  419,  505,  284,
       1117, 1173, 1023,  620,   47, 1035,  230,  864,   78,    1,  334,
        375,  435,  508,  651,  472,  347,  145,  450,  266,  149,  973,
        941,  883, 1055,  353,  763,  332,  666,  290,  479,   92,  859,
         58,  175,  556,  708,  699,  675,  581,  570,  710,  209,  616,
        796,  245,   81,  866,  966,  281, 1124,   67,  584,   88, 1221,
        270,  525,   37,  637,  400,  183,  977,  828, 1037,   21,  173,
        106, 1204,  922, 1145, 1030,  453,  238,  979,  257, 1095,   86,
        870,  322]
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    for count, input_image in enumerate(np.array(img_path_list)[test_list]): 
        
        canvas, all_peaks = process(model, input_image)
       
        dir_list = input_image.split('/')
        '''
        result = {}
        result['keypoint'] = all_peaks
        result['path'] = os.path.join(dir_list[-4], dir_list[-3], dir_list[-2], 'png_data_rotated.npz')
        save_path = os.path.join('/data1/yinzihao/boneage_data/cascaded_add', dir_list[-4], dir_list[-3], dir_list[-2])
        
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'keypoint_label.pickle'), 'wb') as f:
            pickle.dump(result, f)
            
       
        '''
        output_path = os.path.join(save_dir, dir_list[-4] + '.png')
        print('writing an immage at', output_path)
        cv2.imwrite(output_path, canvas) 
        
       
    
    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))


