import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from .globalNet import globalNet
from .refineNet import refineNet
from .multi_branch_cls import MultiBranchCls

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet_type, num_class, pretrained=True):
        super(CPN, self).__init__()
        
        
        assert resnet_type in ['resnet'+ str(i) for i in [18, 34, 50, 101, 162]]
        resnet_depth = int(resnet_type[-2:])
        if resnet_depth in [18, 34]:
            channel_settings = [512, 256, 128, 64]
        elif resnet_depth in [50, 101, 162]:
            channel_settings = [2048, 1024, 512, 256]
       
        stride_settings = [8, 4, 2, 1]
        self.resnet = resnet.__dict__[resnet_type](pretrained)
        self.global_net = globalNet(channel_settings, stride_settings, num_class)
        self.refine_net = refineNet(channel_settings[-1], stride_settings, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)
        return global_outs, refine_out
    
class FusionNet(nn.Module):
    def __init__(self, resnet_type, num_class, target_size, num_branch, num_cls_list, backbone_type, pretrained = True):
        super(FusionNet, self).__init__()
        self.attention_net = CPN(resnet_type, num_class, pretrained)
        self.threshold = 0.1
#         self.conv1x1 = nn.Conv2d(num_class, 1, 1)
        self.classification_net = MultiBranchCls(target_size, num_branch, num_cls_list, backbone_type, pretrained)
        self.order_branch = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14]
        
    def find_suitable_patch(self, prob_map_single, x_single):
        W, H = prob_map_single.size()
        threshold_cnt = W * H * self.threshold
        min_prob, max_prob = prob_map_single.min(), prob_map_single.max()
        
        min_dis_cnt = W * H #set to some large number
        best_minw, best_minh, best_maxw, best_maxh = None, None, None, None
        while (min_prob + 1e-4 < max_prob):
            search_prob = (min_prob + max_prob)/2.0
            cnt = torch.nonzero(prob_map_single > search_prob)
            minw = cnt[:, 0].min().item()
            maxw = cnt[:, 0].max().item()
            minh = cnt[:, 1].min().item()
            maxh = cnt[:, 1].max().item()
            cnt_in_bbox = (maxw - minw) * (maxh - minh)
            
#             print search_prob, cnt_in_bbox, threshold_cnt
            
            dis_cnt = threshold_cnt - cnt_in_bbox
            if abs(dis_cnt) < min_dis_cnt and not(cnt_in_bbox == 0):
                best_minw = minw
                best_maxw = maxw
                best_minh = minh
                best_maxh = maxh
                min_dis_cnt = abs(dis_cnt)
#                 print min_dis_cnt, search_prob
                
            if dis_cnt > 0:
                max_prob = search_prob
            elif dis_cnt < 0:
                min_prob = search_prob
            else:
                break
#         print '*' * 80
        crop_image = x_single[:,best_minw:best_maxw, best_minh:best_maxh]
#         crop_info = dict(zip(['minw', 'maxw', 'minh', 'maxh'], [minw, maxw, minh, maxh]))
        crop_info = torch.Tensor([best_minw, best_maxw, best_minh, best_maxh]).cuda()
        return crop_image, crop_info
    
    def crop(self, prob_map, x):
        N, C, W, H = prob_map.size()
        crop_image_batch = torch.zeros(N,C,3,W,H).cuda()
        crop_info_batch = torch.zeros(N,C,4).cuda()
        for i in range(N):
            for j in range(C):
                crop_image, crop_info = self.find_suitable_patch(prob_map[i, j], x[i])
                crop_image_fix = F.interpolate(crop_image.unsqueeze(0).float(), size = [384, 288], mode = 'bilinear')
   
                crop_image_batch[i,j] = crop_image_fix
                crop_info_batch[i, j] = crop_info
        return crop_image_batch, crop_info_batch
            
        
    def forward(self, x):
        global_outs, refine_out = self.attention_net(x)
        refine_out_original = F.interpolate(refine_out, scale_factor = 4.0, mode = 'bilinear')
        prob_map = torch.sigmoid(refine_out_original)
        crop_image_batch, crop_info_batch = self.crop(prob_map, x)
        '''
        refine_out_aggregate = self.con1x1(refine_out)
        prob_map_aggregate = torch.sigmoid(refine_out_aggregate)
        '''
        print '#' * 80
        prob_list = []
        for index, order in enumerate(self.order_branch):
            prob = self.classification_net(crop_image_batch[:, order].float(), index)
            prob_list.append(prob)
           
        return global_outs, crop_image_batch, crop_info_batch, prob_list
        
        