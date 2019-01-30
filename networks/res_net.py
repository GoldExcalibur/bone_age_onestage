import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from norm_layer import NormLayer
from collections import OrderedDict
import torchvision.models as models

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        out_planes = self.expansion * planes
        self.conv2 = conv3x3(planes, out_planes)
        self.bn2 = NormLayer(planes)
        self.downsample = nn.Sequential()
        
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False),
                NormLayer(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.relu = nn.ReLU(inplace = True)
                       
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        
        out_planes  = self.expansion*planes
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = NormLayer(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(out_planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = NormLayer(64)
        self.relu = nn.ReLU(inplace = True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
      
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = block.expansion * planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x 

def resnet(model_type, pretrained):
    model = None
    settings = {
        'resnet18': {'block': BasicBlock, 'num': [2, 2, 2, 2]},
        'resnet34': {'block': BasicBlock, 'num': [3, 4, 6, 3]},
        'resnet50': {'block': BottleNeck, 'num': [3, 4, 6, 3]},
        'resnet101': {'block': BottleNeck, 'num': [3, 4, 23, 3]},
        'resnet152': {'block': BottleNeck, 'num': [3, 8, 36, 3]}
    }
    if model_type in settings.keys():
        block = settings[model_type]['block']
        num_list = settings[model_type]['num']
        model = ResNet(block, num_list)
    else:
        raise Exception('res_net type not supported!')
        
    if pretrained:
        state_dict = model.state_dict()
        '''
        pretrained_state_dict = model_zoo.load_url(model_urls[model_type])
        '''
        pretrained_state_dict = models.__dict__[model_type](pretrained = pretrained).state_dict()
        cnt  = 0
        for k, v in pretrained_state_dict.items():
            if k in state_dict.keys():
                state_dict[k] = v
                cnt += 1
        print 'loading params (%d, %d)' % (cnt, len(pretrained_state_dict))
        model.load_state_dict(state_dict)
        
    return model, block.expansion * 512