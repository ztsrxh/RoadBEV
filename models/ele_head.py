
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
from .efficientnet import efficientnet_cla

class EleCla2D(nn.Module):
    def __init__(self, feat_channel, num_grids, num_classes):
        super(EleCla2D, self).__init__()
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.feat_channel = feat_channel
        self.channel_reshaped = self.feat_channel * self.num_grids_y
        self.num_classes = num_classes

        self.inplanes = int(self.channel_reshaped/8)
        self.first_conv = nn.Sequential(
                        convbn(self.channel_reshaped, self.inplanes, 5, 1, 2, 1),
                        nn.ReLU(inplace=True))

        self.effnet_reg = efficientnet_cla(self.inplanes, self.num_classes)

        self.final_conv = nn.Sequential(convbn(self.num_classes, self.num_classes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1,
                                                  padding=0, bias=False))

    def forward(self, feat_voxel):
        # feat_voxel: [B, C, Z, X, Y]
        B = feat_voxel.shape[0]
        #### get the BEV feature.  shape: [B, C_, num_grids_z, num_grids_x]
        feat_bev = feat_voxel.permute(0, 4, 1, 2, 3).reshape(B, self.channel_reshaped, self.num_grids_z, self.num_grids_x)  # [B,Y*C,Z,X]
        feat_bev = self.first_conv(feat_bev)
        feat_bev = self.effnet_reg(feat_bev)
        ele_cla_prob = self.final_conv(feat_bev)  # [B, num_classes, Z, X]

        return ele_cla_prob

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.attention_block = attention_block(channels_3d=in_channels * 2, num_heads=16, block=(2,2,2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)  # 1/2
        conv2 = self.conv2(conv1)  # 1/2
        conv3 = self.attention_block(conv2)

        conv6 = F.relu(self.conv6(conv3) + self.redir1(x), inplace=True)
        return conv6

class EleCla3D(nn.Module):
    def __init__(self, feat_channel, num_grids, num_classes):
        super(EleCla3D, self).__init__()
        self.feat_channel = feat_channel
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            convbn_3d(self.feat_channel, self.feat_channel, kernel_size=(3, 5, 5), stride=1, pad=(1, 2, 2)),  # 32
            nn.ReLU(inplace=True),
            hourglass(self.feat_channel),
            convbn_3d(self.feat_channel, self.feat_channel, kernel_size=(3, 5, 5), stride=1, pad=(1, 2, 2)),
            nn.ReLU(inplace=True),
            convbn_3d(self.feat_channel, int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            hourglass(int(self.feat_channel / 2)),
            convbn_3d(int(self.feat_channel / 2), int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            hourglass(int(self.feat_channel / 2)),
            convbn_3d(int(self.feat_channel / 2), int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(self.feat_channel / 2), 1, kernel_size=3, padding=1, stride=1, bias=False)
            )

    def forward(self, feat_voxel):
        # feat_voxel: [B, C, Y, Z, X]
        prob_volume = self.layer1(feat_voxel)       # [B, 1, Y, Z, X], without softmax
        prob_volume = F.interpolate(prob_volume, [self.num_classes, prob_volume.size()[3], prob_volume.size()[4]], mode='trilinear',
                              align_corners=True)
        prob_volume = prob_volume.squeeze(1)

        return prob_volume
