import torch.nn.functional as F
import torch
from torch import nn

class MyLoss(nn.Module):
    def __init__(self, ele_range, voxel_ele_res, cla_res=1):
        super(MyLoss, self).__init__()
        self.ele_range = ele_range*100  # to cm
        if (self.ele_range*20) % (cla_res*10) != 0:
            print('The class interval is improper')
            exit()
        self.cla_res = cla_res   # in cm
        self.voxel_ele_res = voxel_ele_res*100  # in cm
        self.num_voxels_ele = int(self.ele_range*2 / self.voxel_ele_res)

        self.num_classes = int(2*self.ele_range/cla_res)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def label2class(self, ele_gt):
        # ele_gt: [N,]
        assert ele_gt.numel() > 0
        class_label = torch.floor((ele_gt + self.ele_range) / self.cla_res).type(torch.long)
        class_label = self.num_classes - class_label - 1

        return class_label

    def forward(self, ele_pred, ele_gt, ele_mask):
        # ele_pred: [B, num_classes, H, W]  without softmax
        # ele_gt:   [B, H, W]
        # ele_mask: [B, H, W]

        ele_mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)

        ele_pred = ele_pred.permute(0, 2, 3, 1)
        ele_pred = ele_pred[ele_mask, :]
        ele_gt = ele_gt[ele_mask]

        # class_voxel = self.label2class(ele_gt, 'voxel')
        # loss_voxel = self.loss_func1(voxel_prob, class_voxel)
        class_ele = self.label2class(ele_gt)
        loss_ele = self.loss_func(ele_pred, class_ele)

        return loss_ele
