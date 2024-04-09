import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from utils.dataset import RSRD
from torch.cuda.amp import GradScaler
from models.loss import MyLoss
from torch.utils.data import DataLoader
from models.model import Elevation
import pickle
from torch.hub import load_state_dict_from_url
import os
from utils.metric import Metric
from utils.experiment import *
import numpy as np
from datetime import datetime


def train():
    global_step = 0
    for epoch_idx in tqdm(range(args.epochs)):
        for i, sample in enumerate(train_loader):
            global_step += 1
            if args.stereo:
                (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
            else:
                (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
            imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if args.stereo:
                    ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                else:
                    ele_pred = model(imgs_left, proj_index_left)
                loss_all = loss_func(ele_pred, ele_gt, ele_mask)

            scaler.scale(loss_all).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % args.summary_freq == 0:
                loss_data = loss_all.data.item()
                if np.isnan(loss_data):
                    print('nan loss!')
                    exit()
                info = 'train--> epoch%2d, lr:%.6f, loss:%.4f' % (epoch_idx+1, optimizer.param_groups[0]['lr'], loss_data)
                log_file.write(info + '\n')
                log_file.flush()
                print(info)

            if global_step % (3*args.summary_freq) == 0:
                torch.save(model.state_dict(), "{}/checkpoint_epoch{:0>2}_{:0>6}.ckpt".format(args.logdir, epoch_idx+1, global_step))

                torch.cuda.empty_cache()
                [metric_all, _] = test_sample(test_loader)
                info = 'test:    abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f' % (metric_all[0], metric_all[1], metric_all[2]*100)
                log_file.write(info + '\n')
                log_file.flush()
                print(info)

@make_nograd_func
def test_sample(test_loader):
    model.eval()
    for i, sample in enumerate(test_loader):
        if args.stereo:
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

        if args.stereo:
            ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
        else:
            ele_pred = model(imgs_left, proj_index_left)
        metric.compute(ele_pred, ele_gt, ele_mask)
    model.train()
    metric_values = metric.get_metric()
    metric.clear()
    return metric_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--lr', type=float, default=8e-4, help='maximum learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', default='./checkpoints/', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--summary_freq', type=int, default=20, help='summary_freq')
    parser.add_argument('--seed', type=int, default=307, metavar='S', help='random seed')

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.stereo:
        args.down_scale = 2
        print('trining RoadBEV-stereo!')
    else:
        args.down_scale = 4
        print('trining RoadBEV-mono!')

    # dataset, dataloader
    train_set = RSRD(training=True, stereo=args.stereo, down_scale=args.down_scale)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale)
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    print('dataset size - train:%d, test%d' % (len(train_set), len(test_set)))

    # model, optimizer
    ele_range = train_set.y_range
    voxel_ele_res = train_set.grid_res[1]
    num_grids = [train_set.num_grids_x, train_set.num_grids_y, train_set.num_grids_z]
    model = Elevation(args.stereo, num_grids, ele_range, args.cla_res).cuda()
    print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.train()

    loss_func = MyLoss(ele_range, voxel_ele_res, args.cla_res).cuda()
    metric = Metric(ele_range, args.cla_res, train_set.num_grids_z, distance_wise=False)

    url = 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth'
    try:
        weights = load_state_dict_from_url(url, progress=True)
    except:
        print('please manually download pretrained weights at:', url)
        exit(0)

    weights_new = {}
    target_keys = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4']
    for key, value in weights.items():
        if any(k in key for k in target_keys):
            weights_new[key.replace('features.', 'l')] = value
    model.feature_extraction.load_state_dict(weights_new, strict=False)

    if args.loadckpt is not None:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict, strict=True)

    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, pct_start=0.02,
                                                    three_phase=False,
                                                    div_factor=20, anneal_strategy='linear',
                                                    steps_per_epoch=len(train_loader))

    # logging
    args.logdir = os.path.join(args.logdir, datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    print('logging dir:', args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    shutil.copy('./utils/dataset.py', os.path.join(args.logdir, 'dataset.py'))
    shutil.copy('./models/model.py', os.path.join(args.logdir, 'model.py'))
    shutil.copy('./models/efficientnet.py', os.path.join(args.logdir, 'efficientnet.py'))
    shutil.copy('./models/ele_head.py', os.path.join(args.logdir, 'ele_head.py'))
    shutil.copy('train.py', os.path.join(args.logdir, 'train.py'))
    log_file = open(os.path.join(args.logdir, 'log.txt'), 'a')

    train()


