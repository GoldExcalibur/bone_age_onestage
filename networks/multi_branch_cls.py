import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from norm_layer import NormLayer
from res_net import resnet
from dense_net import densenet
from packaging import version
if version.parse(torch.__version__) < version.parse('0.4.0'):
    from torch.autograd import Variable
    PYTORCH_VERSION_LESS_THAN_040 = True
else:
    PYTORCH_VERSION_LESS_THAN_040 = False

def _backbone_net(model_type, pretrained):
    net_str_index = model_type.find('net')
    type_str = model_type[:net_str_index]
    efficient_index = model_type.find('efficient') 
    
    efficient_flag = False
    if efficient_index != -1:
        efficient_flag = True
        model_type = model_type[:efficient_index-1]
  
    if type_str == 'res':
        print model_type, 'pretrained:', pretrained
        return resnet(model_type, pretrained)
    elif type_str == 'dense':
        print model_type, 'pretrained:', pretrained, 'efficient:', efficient_flag
        return densenet(model_type, pretrained, efficient_flag)
    else:
        raise Exception('backbone net type not supported!')
    
class MultiBranchCls(nn.Module):
    def __init__(self, target_size, num_branch, num_cls_list, backbone_type, pretrained = False):
        super(MultiBranchCls, self).__init__()
        
        self.feature_net, feature_net_last_channels = _backbone_net(backbone_type, pretrained)
        self.num_cls_list = num_cls_list
        
        assert type(target_size).__name__ == 'tuple'
        assert type(num_cls_list).__name__ == 'list'
        h, w = target_size
        fc_in_channels = (h/32 - 7 + 1) * (w/32 - 7 + 1) * feature_net_last_channels
        
        if num_branch == 1:
            self.fc_branch = nn.ModuleList([nn.Sequential(
                nn.Linear(fc_in_channels, len(num_cls_list))
                )])
        else:                                
            self.fc_branch = nn.ModuleList([nn.Sequential(
                nn.Linear(fc_in_channels, num_cls_list[i])
                ) for i in xrange(num_branch)])
        
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if PYTORCH_VERSION_LESS_THAN_040:
                        nn.init.kaiming_normal(m.weight, mode='fan_out') # pytorch 0.3.1
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                        
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n ))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
                
            
    def forward(self, x, branch_index):
        out = self.feature_net(x)
        out = out.view(out.size(0), -1)
        out = self.fc_branch[branch_index](out)
        return out.squeeze(1)
