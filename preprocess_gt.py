from tqdm import tqdm
import argparse
from models.model import Elevation
from utils.dataset import RSRD
import torch
import pickle
import numpy as np
import open3d as o3d
import os


def get_item(index):
    sample_cur = dataset.data_all[index]
    l2c_calib_cur = dataset.get_lidar2cam(sample_cur['time'])
    path_base = sample_cur['path']
    idx_str = path_base.find('/')
    path_base = path_base[idx_str + 1:]

    R_cur2enu = dataset.get_RT_lidar(sample_cur)

    ########   calculate the euler angles of the camera (relative to local ENU coord)   ########
    [pitch_cam, roll_cam, _] = dataset.matrix2euler(l2c_calib_cur['R'] @ np.linalg.inv(R_cur2enu))
    pitch_cam -= 1.5708  # pi/2
    R_X = np.array(
        [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]],
        dtype=np.float32)
    R_Z = np.array(
        [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]],
        dtype=np.float32)
    R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status

    ########  read point cloud and transform into camera's coord, then crop the ROI  #######
    path_pcd = os.path.join(dataset.data_path, path_base, 'pcd', sample_cur['time']) + '.pcd'
    cloud = o3d.io.read_point_cloud(path_pcd)
    cloud = cloud.rotate(l2c_calib_cur['R'], center=(0, 0, 0))
    cloud = cloud.translate(tuple(l2c_calib_cur['T'].reshape(-1)))  # the point cloud in the camera's coord

    cloud_camvert = cloud.rotate(R_cam2vert, center=(0, 0, 0))
    # crop the point cloud according to the given range of interest
    cloud_camvert = dataset.vol_roi.crop_point_cloud(cloud_camvert)

    ele_gt, ele_mask = dataset.get_gt_elevation(cloud_camvert)

    return ele_gt, ele_mask, sample_cur['time']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')
    parser.add_argument('--save_dir', type=str, required=True, help='save path for preprocessed GT maps')
    parser.add_argument('--dataset', type=str, required=True, choices=['train', 'test'], help='generating for train or test sets')
    args = parser.parse_args()

    training = args.dataset == 'train'
    dataset = RSRD(training=training, stereo=False, down_scale=2)
    if training:
        path = os.path.join(args.save_dir, 'train')
    else:
        path = os.path.join(args.save_dir, 'test')
    os.makedirs(path, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        ele_gt, ele_mask, stamp = get_item(i)
        with open(os.path.join(path, stamp + '.pkl'), 'wb') as f:
            pickle.dump([ele_gt, ele_mask], f)
        print(stamp)
