#coding=utf-8

import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np

import os
from os.path import dirname, exists, join, realpath, isdir, isfile
import argparse
import math

import sys
BoneAgePath = dirname(dirname(dirname(realpath(__file__))))
sys.path.insert(0, BoneAgePath)
from keypoint_epiphysis.dataloader import AggDataLoader, ConcatDataset
from keypoint_epiphysis import aggregate_datasets
from keypoint_epiphysis.networks.multi_branch_cls import MultiBranchCls
from keypoint_epiphysis.networks.network import FusionNet

from utils.tensorboard_logger import tf_Logger
from keypoint_epiphysis.utils.logger import Logger
from keypoint_epiphysis.utils.evaluation import accuracy, AverageMeter, final_preds
from keypoint_epiphysis.utils.misc import save_model, adjust_learning_rate
from keypoint_epiphysis.utils.osutils import mkdir_p
from keypoint_epiphysis.utils import Mytransforms
from collections import namedtuple

from keypoint_epiphysis.train_config import cfg
import tensorflow as tf

target_size = cfg.target_size

train_transform = Mytransforms.Compose([
                Mytransforms.RandomRotate(10, prob=0.5),
                Mytransforms.RandomHorizontalFlip(prob=0.5),
                Mytransforms.RandomVerticalFlip(prob=0.5),
#                 Mytransforms.RandomDrop(prob_list = [0.4, 0.3, 0.3]),
                Mytransforms.Resized(target_size),
            ])
    
val_transform = Mytransforms.Compose([
               Mytransforms.Resized(target_size)
          ])

class mean_variance_loss(nn.Module):
    def __init__(self, lamda0, lamda1, lamda2):
        super(mean_variance_loss, self).__init__()
        self.lamda0 = lamda0
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.crossentropy = nn.CrossEntropyLoss(size_average = True)
        self.mseloss = nn.MSELoss(size_average = True)
        
    def forward(self, output, output_target):
        batch_size, classes_num = output.size()
        softmax_loss = self.crossentropy(output, output_target.type(torch.cuda.LongTensor))
        probs = F.softmax(output, dim = 1)
        classes = torch.Tensor(range(classes_num)).cuda()
        mean = torch.sum(probs * classes, dim = 1)
        variance = torch.sum(probs*((mean.unsqueeze(1) - classes)**2), dim = 1)
        mloss = self.mseloss(mean, output_target)/2.0
        vloss = torch.mean(variance)
#         print mloss, vloss, softmax_loss
        loss = self.lamda0 * softmax_loss + self.lamda1 * mloss + self.lamda2 * vloss 
        result_dict = {}
        result_dict['loss'] = loss
        result_dict['mean'] = mean
        result_dict['mloss'] = mloss
        result_dict['vloss'] = vloss
        result_dict['sfloss'] = softmax_loss
        return result_dict   
    
class Average_Meter_Dict(object):
    def __init__(self, keys):
        self.keys = keys
        self.dict = {}
        for k in keys:
            self.dict[k] = AverageMeter()
            
    def update(self, vals_dict, n = 1):
        for k, v in vals_dict.items():
            if k in self.dict.keys():
                self.dict[k].update(v.data.item(), n)
               
    def items(self):
        item_list = []
        for k, v in self.dict.items():
            item_list.append((k, v))
        return item_list
    
    def __getitem__(self, key):
        return self.dict[key]
        
            

def construct_model(config):
    restype, num_class, pretrained = config.resnet_type, config.num_class, config.pretrained
    target_size, num_branch, num_cls_list = config.target_size, config.num_branch, config.num_cls_list
    backbone = config.backbone_type
    model = FusionNet(restype,num_class, target_size, num_branch, num_cls_list, backbone, pretrained)
#     model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    return model

def main(args):
    checkpoint_name = cfg.checkpoint
    if not isdir(checkpoint_name):
        mkdir_p(checkpoint_name)
        
    area = cfg.area
    classes_list, num_branch = cfg.num_cls_list, cfg.num_branch
    
    ############################################################
    #   preparing dataset
    ############################################################
    transform_dict = {'train': train_transform, 'val': val_transform}
    

    datasets_setting = cfg.datasets_setting
    loader_dict = {}
    for mode, datasets_flag in datasets_setting.items():
        loaders = []
        for prefix, flag in datasets_flag.items():
            if flag:
                dataset = aggregate_datasets.__dict__[prefix + '_datasets']
                loader = AggDataLoader(dataset, cfg, area, mode, transform_dict[mode])
                loaders.append(loader)
        loader_dict[mode] = ConcatDataset(loaders)
        
    print 'Train Dataset Total Len: {}'.format(loader_dict['train'].__len__())
    print 'Valid Dataset Total Len: {}'.format(loader_dict['val'].__len__())
    
    train_loader = torch.utils.data.DataLoader(
                   loader_dict['train'],
                   batch_size = cfg.train_batch_size, shuffle = False, num_workers = args.workers, 
                   pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
                   loader_dict['val'],
                   batch_size = cfg.val_batch_size, shuffle = False, num_workers = args.workers, 
                   pin_memory = True)
    
    train_display_period = max(int(train_loader.__len__()//5), 1)
    val_display_period = max(int(val_loader.__len__()//5), 1)
    
    
    ###########################################################
    #  preparing model, optimizer, scheduler, loss function
    ###########################################################
    model = construct_model(cfg)
    
    optimizer = optim.Adam(model.parameters(), cfg.base_lr, weight_decay = cfg.weight_decay)
#     optimizer = optim.SGD(model.parameters(), cfg.base_lr, momentum = 0.9, weight_decay = 
#                            cfg.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.5, 
                             patience= 10, min_lr = 1e-9, verbose=True)
   
    lamda0, lamda1, lamda2 = cfg.lamda0, cfg.lamda1, cfg.lamda2
    MAELoss = nn.L1Loss(reduction = 'mean')
    MSELoss = nn.MSELoss()
    CELoss = nn.CrossEntropyLoss()
    MWLoss = mean_variance_loss(lamda0, lamda1, lamda2)
    ###########################################################
    # preparing txt logger and tensorboard logger
    ###########################################################
    logger = Logger(join(checkpoint_name, 'log.txt'))
    logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss'])
    
    tf_logger_dict = {}
    for k in ['loss', 'mloss', 'vloss', 'sfloss']:
        tf_logger_dict[k] = tf.summary.FileWriter(join(checkpoint_name, k))
    megred = tf.summary.merge_all()
 

    min_val_loss = None
    for epoch in range(args.start_epoch, args.epochs):

        present_lr = optimizer.param_groups[0]['lr']
        print 'Epoch {e:d} Learning Rate {l:.9f}'.format(e = epoch, l = present_lr)
        
        print 'Train'
        model.train()
        train_losses = AverageMeter()
        Train_Loss_Dict = Average_Meter_Dict(['loss', 'mloss', 'vloss', 'sfloss'])
        
        for iter_num, (inputs, deg_targets, kp_targets) in enumerate(train_loader):
            current_batch_size = inputs.size(0)
            inputs = inputs.cuda()
            deg_targets = deg_targets.cuda()
            kp_targets = [k.cuda() for k in kp_targets]
            
            global_outs, crop_image_batch, crop_info_batch, prob_list = model(inputs)
        
            heatmap_loss = 0.0
            for index, (global_out, kp_target) in enumerate(zip(global_outs, kp_targets)):
                heatmap_loss += MSELoss(global_out, kp_target)
            
            deg_loss = 0.0
            train_loss_recorder = {'loss': 0.0, 'mloss': 0.0, 'vloss': 0.0, 'sfloss': 0.0}
            for index, prob in enumerate(prob_list):
                deg_loss_dict = MWLoss(prob, deg_targets[:,index])
                deg_loss += deg_loss_dict['loss']
                for k, v in deg_loss_dict.items():
                    if k in train_loss_recorder.keys():
                        train_loss_recorder[k] += v
                        
            loss_sum = heatmap_loss + deg_loss
            train_losses.update(loss_sum.data.item(), current_batch_size)
            Train_Loss_Dict.update(train_loss_recorder, current_batch_size)
            
            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            assert loss_sum.data.item() == train_losses.val
            train_display_period = 1
            if (iter_num+1) % train_display_period == 0:
                print 'iter {:d} train_loss {:.3f} '.format(iter_num + 1, train_losses.val)
        
        assert 0
        
        print '-'*88
        print 'Val'
        model.eval()
        val_losses = AverageMeter()
        Val_Loss_Dict = Average_Meter_Dict(['loss', 'mloss', 'vloss', 'sfloss'])
        errors_list = [AverageMeter() for i in range(len(classes_list))]
        
        with torch.no_grad():
            for viter_num, (vepiphysis_list, vtargets) in enumerate(val_loader):
                val_data_size = vepiphysis_list[0].size(0)
                vloss_sum = 0.0
                val_loss_recorder = {'loss': 0.0, 'mloss': 0.0, 'vloss': 0.0, 'sfloss': 0.0}
                
                vbranch_id = 0
                for vepiphysis, vtarget in zip(vepiphysis_list, vtargets):
                    vinputs = vepiphysis.cuda()
                    voutput_target = vtarget.squeeze(1).cuda()
                    voutput = model(vinputs, vbranch_id)
                    vloss_dict = MWLoss(voutput, voutput_target)
                    
                    
#                     degrees errors for different epiphysis
                    with torch.no_grad():
                        error = MAELoss(vloss_dict['mean'], voutput_target)
                        errors_list[vbranch_id].update(error.data.item(), val_data_size)
                    vloss_sum += vloss_dict['loss']
                    
                    for k, v in vloss_dict.items():
                        if k in val_loss_recorder.keys():
                            val_loss_recorder[k] += v
            
                    vbranch_id += 1

               
                val_losses.update(vloss_sum.data.item(), val_data_size)
                Val_Loss_Dict.update(val_loss_recorder, val_data_size)
                
                if (viter_num + 1) % val_display_period == 0:
                    print 'iter {ii:d} val_loss {v:.3f}'.format(ii = viter_num + 1, v = val_losses.val)
        
        print 'train_loss: {g:.3f} val_loss: {t:.3f}'.format(g = train_losses.avg, t = val_losses.avg)
        logger.append([epoch + 1, present_lr, train_losses.avg, val_losses.avg])
        print 'degrees predicted errors list per epoch', [e.avg for e in errors_list]
    
#         tf_info = {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}
        tf_info = {}
        for k, v in Train_Loss_Dict.items():
            tf_info['train_' + k] = v.avg
        for k, v in Val_Loss_Dict.items():
            tf_info['val_' + k] = v.avg
        
        assert tf_info['train_loss'] == train_losses.avg
        assert tf_info['val_loss'] == val_losses.avg
        
        '''
        for tag, value in tf_info.items():
            tf_logger.scalar_summary(tag, value, epoch + 1)
        
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            tf_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            tf_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
        '''
        for k, tf_logger in tf_logger_dict.items():
            summary_train = tf.Summary(value = [tf.Summary.Value(tag='train_loss', simple_value = Train_Loss_Dict[k].avg)])
            summary_val = tf.Summary(value = [tf.Summary.Value(tag='val_loss', simple_value = Val_Loss_Dict[k].avg)])
            tf_logger.add_summary(summary_train, epoch + 1)
            tf_logger.add_summary(summary_val, epoch + 1)
         
            
    
        is_best = False
#         val_loss_epoch = val_losses.avg
        val_loss_epoch = np.mean(np.array([v.avg for v in errors_list]))
        if min_val_loss is None:
            min_val_loss = val_loss_epoch
        else:
            if min_val_loss > val_loss_epoch:
                is_best = True
                min_val_loss = val_loss_epoch
        

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best = is_best, checkpoint = checkpoint_name)
        
        scheduler.step(val_losses.avg)
            
            
        print '*'*88
        print 
    
    logger.close()
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'pytorch bone degree training')
    parser.add_argument('-w', '--workers', default = 6, type = int, metavar = 'N', help =
                        'number of data loading workers (default: 6)')
    parser.add_argument('-g', '--gpu', default = '6', type = str, metavar = 'N', help = 
                        'id of GPU to use(default: 6)')
    
    parser.add_argument('--start_epoch', default = 0, type = int, metavar = 'N', help = 'num of start epoch(default: 0)')
    parser.add_argument('--epochs', default = 500, type = int, metavar ='N', help = 'number of total epochs to run (default: 100)')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
    
