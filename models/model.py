from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature

class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res):
        super(Elevation, self).__init__()
        self.stereo = stereo
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.ele_range = ele_range   # in meter

        self.cla_res = cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)

        self.feature_extraction = efficientnet_feature(self.stereo)

        if self.stereo:
            #  regressor for stereo
            self.ele_head = EleCla3D(self.feature_extraction.feat_channel, num_grids, self.num_classes)
        else:
            #  regressor for mono
            self.ele_head = EleCla2D(self.feature_extraction.feat_channel, num_grids, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, imgs_left, proj_index_left, *args):
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]

        features_left = self.feature_extraction(imgs_left)
        B, C, H, W = features_left.shape
        features_left = features_left.reshape(B, C, -1)
        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]
        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        if self.stereo:
            imgs_right, proj_index_right = args[0], args[1]
            features_right = self.feature_extraction(imgs_right)
            features_right = features_right.reshape(B, C, -1)
            linear_indices = proj_index_right[:, 1, :] * W + proj_index_right[:, 0, :]
            voxel_feat_right = features_right.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
            voxel_feat_right = voxel_feat_right.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

            voxel_feature = voxel_feat_left - voxel_feat_right
            voxel_feature = voxel_feature.permute(0, 1, 4, 2, 3)  # [B, C, Y, Z, X]
        else:
            voxel_feature = voxel_feat_left    # [B, C, Z, X, Y]

        ele_pred = self.ele_head(voxel_feature)    # [B, num_class, Z, X]   without softmax

        if not self.training:
            ele_pred = F.softmax(ele_pred, dim=1)
            ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

            # pred_class = torch.max(ele_pred.data, 1)[1]
            # ele_pred = self.ele_values[pred_class.type(torch.long)]

        return ele_pred