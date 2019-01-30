import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state, preds, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    scipy.io.savemat(os.path.join(checkpoint, 'preds.mat'), mdict={'preds' : preds})

    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        scipy.io.savemat(os.path.join(checkpoint, 'preds_best.mat'), mdict={'preds' : preds})

def copy_log(filepath = 'checkpoint'):
    filepath = os.path.join(filepath, 'log.txt')
    shutil.copyfile(filepath, os.path.join('log_backup.txt'))

def save_model(state, is_best, checkpoint='checkpoint', filename='latest.pth.tar'):
    file_name = filename
    if is_best:
        file_name = 'best.pth.tar'
    filepath = os.path.join(checkpoint, file_name)
    torch.save(state, filepath)

    # if snapshot and state.epoch % snapshot == 0:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(optimizer, min_lr, epoch, last_change_epoch, schedule, gamma, scheduler, last_val_loss):
    """Sets the learning rate to the initial LR decayed by schedule"""
    origin_lr = optimizer.param_groups[0]['lr']
    if last_val_loss is not None:
        scheduler.step(last_val_loss)
    present_lr = optimizer.param_groups[0]['lr']
    if present_lr < origin_lr:
        last_change_epoch = epoch
    if epoch - last_change_epoch <= 50:
        return last_change_epoch, present_lr
    else:
        if epoch in schedule:
            for param_group in optimizer.param_groups:
                origin_group_lr = param_group['lr']
                param_group['lr'] = max(gamma*origin_group_lr, min_lr)
            last_change_epoch = epoch
        
        return last_change_epoch, optimizer.state_dict()['param_groups'][0]['lr']
