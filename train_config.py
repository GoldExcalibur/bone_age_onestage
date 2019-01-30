#coding:utf-8 
import os
import glob
import sys
import pickle
import numpy as np
from scipy import interp
import socket
hostname = socket.gethostname()
import getpass
username = getpass.getuser()
import datetime

class Config(object):
    
    ##########################################################################################
    # debug设置为 True,则在新建目录，读写方面采取消极措施
    ##########################################################################################
    debug = True
    
    model = 'CPN101'
    num_class = 15
    vec_pair = [[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 4, 12, 13],  [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 14]] 
    num_vec = len(vec_pair[0]) 
    checkpoint_dir = '/home/yinzihao/work_deliver/keypoint_model/384.288.model/checkpoint_nok_in_350'
    
    target_length = (640.0, 860.0, 442.0)

    # 标注平台的顺序调整到关键点的顺序
    order_convert = [12, 9, 7, 4, 11, 8, 6, 3, 10, 5, 2, 1, 0]
    
    # inference tables: degree2score score2age
    tabel_root = '/data1/yinzihao/boneage_excels'
    degree2score  = pickle.load(open(os.path.join(tabel_root, 'degree2score.pickle'), 'rb'))
    score2age = pickle.load(open(os.path.join(tabel_root, 'score2age.pickle'), 'rb'))
    chn2tw3 = pickle.load(open(os.path.join(tabel_root, 'chn2tw3.pickle'), 'rb'))
        
    for area in ['rus']:
        for sex in ['male', 'female']:
            table = score2age[area]['tw3'][sex]
            x = np.sort(np.array(table.keys()))
            y = np.sort(np.array([i[0] for i in table.values()]))
            xmin, xmax = x[0], x[-1]
            xx = np.array(range(xmin, xmax+1))
            yy = interp(xx, x, y)
            new_dict = dict(zip(xx, yy))
            score2age[area]['tw3'][sex] = new_dict    
    
    ##########################################################################################
    # 数据集的标签及配置
    ##########################################################################################
    datasets_prefix = ['c1b1', 'c2b2', 'c3b1', 'c3b1old']
    datasets_setting = {'train': dict(zip(datasets_prefix, [True, False, False, False])), 
                            'val': dict(zip(datasets_prefix, [True, False, False, False]))} 
    """
    datasets_setting = {'train': dict(zip(datasets_prefix, [True, True, True, False])), 
                            'val': dict(zip(datasets_prefix, [True, True, True, False]))} 
    """
    experiment_name = 'c3b1contrast' # c3b1 train to compare with maxmig
    experiment_name = 'c1c2c3_carpal_big'
    if debug:
        checkpoint = '/data1/%s/bone_age/epiphysis_model/%s' % (username, experiment_name)
    else:
        checkpoint = '/data1/%s/bone_age/epiphysis_model/%s_%s' % (username, experiment_name, datetime.datetime.now().strftime("%m%d_%H%M"))
    
    ##########################################################################################
    # 训练的配置：学习率，批大小， loss的权重, 数据增强的参数
    ##########################################################################################
    # params augmentation
    target_size = (288, 384) #(w , h)
    
    
    # params dataloader
    train_batch_size = 4
    val_batch_size = 4
    
    # params optimizer
    base_lr = 0.001
    weight_decay = 5e-4
    
    lamda0 = 1.0
    lamda1 = 0.5
    lamda2 = 0.05
    
    settings = {'rus':{'num_branch': 13, 'classes': [12, 13, 13, 11, 12, 13, 13, 11, 12, 13, 12, 13, 15], 'start_index': 0}, 
                    'carpal': {'num_branch': 1, 'classes': [8, 9, 8, 8, 8, 9, 8], 'start_index': 13},
                   }
    
    ##########################################################################################
    # 模型的配置
    ##########################################################################################
    area = 'rus'
    
    # params of CPN
    pretrained = True
    norm_type = 'batch'
    resnet_type = 'resnet50'
    gaussain_kernel = (7, 7)
    
    stride = 4
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)
    theta = 1
    
    # params of Cls
    num_branch = 13
    num_cls_list = [12, 13, 13, 11, 12, 13, 13, 11, 12, 13, 12, 13, 15, 8, 9, 8, 8, 8, 9, 8]
    backbone_type = 'resnet34'
    
cfg = Config()
