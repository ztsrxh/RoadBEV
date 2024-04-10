import argparse
import shutil
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from utils.dataset import RSRD
from torch.cuda.amp import GradScaler
from models.loss import MyLoss
from torch.utils.data import DataLoader
from models.model import Elevation
import pickle
import os
from utils.metric import Metric
from utils.experiment import *
import numpy as np


@make_nograd_func
def test_sample(test_loader):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(len(test_loader))
    model.eval()
    for i, sample in enumerate(test_loader):
        if args.stereo:
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

        starter.record()
        if args.stereo:
            pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
        else:
            pred = model(imgs_left, proj_index_left)

        ender.record()
        torch.cuda.synchronize()
        times[i] = starter.elapsed_time(ender)

        metric.compute(pred, ele_gt, ele_mask)

    mean_time = times.mean().item()
    print("Inference time: {:.2f}ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))

    metric_values = metric.get_metric()
    return metric_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Elevation')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--loadckpt', default='./checkpoints/20240407064559/checkpoint_epoch50_007500.ckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=837, metavar='S', help='random seed')

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.stereo:
        args.down_scale = 2
        print('Testing RoadBEV-stereo!')
    else:
        args.down_scale = 4
        print('Testing RoadBEV-mono!')

    # dataset, dataloader
    test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale)
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    print('test set:', len(test_set))

    # model, optimizer
    ele_range = test_set.y_range
    voxel_ele_res = test_set.grid_res[1]
    num_grids = [test_set.num_grids_x, test_set.num_grids_y, test_set.num_grids_z]

    model = Elevation(args.stereo, num_grids, ele_range, args.cla_res).cuda()
    print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    metric = Metric(ele_range, args.cla_res, test_set.num_grids_z, distance_wise=True)

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict, strict=True)

    [metric_all, metric_depthwise] = test_sample(test_loader)
    info = 'test:    abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f' % (metric_all[0], metric_all[1], metric_all[2]*100)
    print(info)

    metric.plot_depthwise(metric_depthwise)
