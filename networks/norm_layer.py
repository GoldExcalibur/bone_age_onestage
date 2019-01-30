import torch.nn as nn
import sys
import traceback

__all__ = ['batch', 'instance']

class Norm_Layer(nn.Module):
    def __init__(self, norm_type):
        super(Norm_Layer, self).__init__()
        self.norm_type = norm_type
    def __call__(self, feature_num):
        if self.norm_type == 'batch':
            
            return nn.BatchNorm2d(feature_num)
        elif self.norm_type == 'instance':
            
            return nn.InstanceNorm2d(feature_num, affine = True)
      
s = traceback.extract_stack()
fname =  s[0][0]
ntype = None
if 'train' in fname:
    from keypoint_epiphysis.train_config import cfg as cfg_train
    ntype = cfg_train.norm_type
else: # bone_age_test, hand2crop
    sys.path.insert(0, '../..')
    from TestConfig import Config as cfg_test
    ntype = cfg_test.keypoint_norm_type

# from config import cfg
NormLayer = Norm_Layer(norm_type=ntype)
