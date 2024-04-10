import numpy as np
import math
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image
from torchvision import transforms
import open3d as o3d

class RSRD(Dataset):
    def __init__(self, training=True, stereo=False, down_scale=2):
        super(RSRD, self).__init__()
        self.training = training
        self.stereo = stereo
        self.down_scale = down_scale

        self.calib_path = '/dataset/RSRD_calib/'  # path for calibration files
        self.data_path = '/dataset/RSRD-dense/train/'     # path for the training set of RSRD-dense
        preprocessed_path = '/dataset/RSRD-dense/ele_preprocessed/'  # path for preprocessed GT maps

        if self.training:
            self.load_dataset_names('./filenames/train/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'train')
        else:
            self.load_dataset_names('./filenames/test/')
            self.preprocessed_path = os.path.join(preprocessed_path, 'test')

        #######################
        # settings about range of interest  !! do not change !!
        #######################
        self.base_height = 1.1  # in meter, the reference height of the camera w.r.t. road surface
        self.y_range = 0.2  # in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
        self.roi_x = torch.tensor([-1, 0.92])    # in meter, the lateral range of interest (in the horizontal coordinate of camera)
        self.roi_z = torch.tensor([2.16, 7.08])    # in meter, the longitudinal range of interest
        self.grid_res = torch.tensor([0.03, 0.01, 0.03])  # in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
        #######################

        self.num_grids_x = int((self.roi_x[1] - self.roi_x[0]) / self.grid_res[0])
        self.num_grids_z = int((self.roi_z[1] - self.roi_z[0]) / self.grid_res[2])
        self.num_grids_y = int(self.y_range*2 / self.grid_res[1])

        # generate the centers of every horizontal grid
        hori_centers = torch.zeros((self.num_grids_z, self.num_grids_x, 2), dtype=torch.float32)
        hori_centers[:, :, 0] = (torch.arange(self.num_grids_x) * self.grid_res[0] + self.roi_x[0] + self.grid_res[0]/2).unsqueeze(0).repeat([self.num_grids_z, 1])
        hori_centers[:, :, 1] = (-torch.arange(self.num_grids_z) * self.grid_res[2] + self.roi_z[1] - self.grid_res[2]/2).unsqueeze(1).repeat([1, self.num_grids_x])
        self.map_centers = hori_centers.reshape(-1, 2)
        self.num_center = self.map_centers.shape[0]

        # generate the centers of every 3D voxel
        voxel_centers = torch.zeros((self.num_grids_z, self.num_grids_x, self.num_grids_y, 3), dtype=torch.float32)
        voxel_centers[:, :, :, [0, 2]] = hori_centers.unsqueeze(2).repeat([1, 1, self.num_grids_y, 1])
        voxel_centers[:, :, :, 1] = (torch.arange(self.num_grids_y) * self.grid_res[1] + self.base_height - self.y_range + self.grid_res[1]/2).unsqueeze(0).unsqueeze(0).repeat([self.num_grids_z, self.num_grids_x, 1])
        self.voxel_centers = voxel_centers.reshape(-1, 3).transpose(1, 0)

        # parameters for cropping ROI point clouds
        self.crop_bounding = np.array([[self.roi_x[0], 0, self.roi_z[0]],
                                       [self.roi_x[0], 0, self.roi_z[1]],
                                       [self.roi_x[1], 0, self.roi_z[1]],
                                       [self.roi_x[1], 0, self.roi_z[0]]]).astype("float64")
        self.vol_roi = o3d.visualization.SelectionPolygonVolume()
        self.vol_roi.orthogonal_axis = "Y"
        self.vol_roi.axis_max = 1.5
        self.vol_roi.axis_min = 0.5
        self.vol_roi.bounding_polygon = o3d.utility.Vector3dVector(self.crop_bounding)

        # pre_read the extrinsic parameters between camera and lidar
        # intrinsics (after rectification): calib_params["K"]
        # stereo baseline(in mm): calib_params["B"]
        # lidar -> left camera extrinsics: calib_params["R"], calib_params["T"]
        calib_files = ['calib_20230317_half.pkl', 'calib_20230321_half.pkl', 'calib_20230406_half.pkl', 'calib_20230408_half.pkl', 'calib_20230409_half.pkl']
        self.calib_params_all = {}
        for file in calib_files:
            with open(os.path.join(self.calib_path, file), 'rb') as f:
                calib_params = pickle.load(f)
            calib_params['K'] = calib_params['K'].astype(np.float32)
            calib_params['R'] = calib_params['R'].astype(np.float32)
            calib_params['T'] = calib_params['T'].astype(np.float32)
            calib_params['B'] = calib_params['B']/1000   # mm -> m
            calib_params['K_feat_T'] = torch.from_numpy(calib_params['K'] / self.down_scale)
            calib_params['K_feat_T'][2, 2] = 1
            calib_params['R_inv'] = np.linalg.inv(calib_params['R']).astype(np.float32)
            date = file[6:14]
            self.calib_params_all[date] = calib_params

        self.transform_jpg = transforms.Compose([
            transforms.ToTensor(),  # image --> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] --> [-1, 1]
        ])
        self.gen = torch.Generator()
        self.gen.manual_seed(1000)

    def load_dataset_names(self, sample_path):
        data_all = []
        files = sorted(os.listdir(sample_path))
        for file in files:
            with open(os.path.join(sample_path, file), 'rb') as f:
                data = pickle.load(f)
            data_all += data
        self.data_all = data_all

    def get_lidar2cam(self, date_stamp):
        # name in format: 20230408023213.400
        date = date_stamp[:8]
        return self.calib_params_all[date]

    def yaw_convert(self, yaw):
        '''
            convert the yaw data from [0, 2pi] to [-pi, pi]
        '''
        if np.pi <= yaw <= 2 * np.pi:
            yaw -= 2 * np.pi
        return yaw

    def lla_to_enu(self, C_lat, C_lon, C_alt, O_lat, O_lon, O_alt):
        '''
            Calculate the relative location with respect to the selected origin in the local ENU coordinate. unit: meter
            C_lat, C_lon, C_alt: current location
            O_lat, O_lon, O_alt: origin location
        '''
        Ea = 6378137
        Eb = 6356752.3142
        C_lat = math.radians(C_lat)
        C_lon = math.radians(C_lon)
        O_lat = math.radians(O_lat)
        O_lon = math.radians(O_lon)
        Ec = Ea * (1 - (Ea - Eb) / Ea * (math.sin(C_lat)) ** 2) + C_alt
        d_lat = C_lat - O_lat
        d_lon = C_lon - O_lon
        e = d_lon * Ec * math.cos(C_lat)
        n = d_lat * Ec
        u = C_alt - O_alt
        return np.array([e, n, u])

    def get_RT_lidar(self, loc_pose):
        #### pre-process
        # convert from angle to radius， then rectify to range -pi~pi
        rotX_cur = 0.017453 * loc_pose['pitch']
        rotY_cur = 0.017453 * loc_pose['roll']
        rotZ_cur = 0.017453 * loc_pose['yaw']
        rotZ_cur = self.yaw_convert(rotZ_cur)

        # rotation order ZXY， the derived R is rotation matrix from current pose to local ENU
        R_X1 = np.array([[1, 0, 0], [0, np.cos(rotX_cur), -np.sin(rotX_cur)], [0, np.sin(rotX_cur), np.cos(rotX_cur)]])
        R_Y1 = np.array([[np.cos(rotY_cur), 0, np.sin(rotY_cur)], [0, 1, 0], [-np.sin(rotY_cur), 0, np.cos(rotY_cur)]])
        R_Z1 = np.array([[np.cos(rotZ_cur), -np.sin(rotZ_cur), 0], [np.sin(rotZ_cur), np.cos(rotZ_cur), 0], [0, 0, 1]])
        R_cur2enu = R_Z1 @ R_X1 @ R_Y1   # the rotation from current lidar to enu

        return R_cur2enu

    def get_gt_elevation(self, cloud_camvert):
        xyz = torch.from_numpy(np.asarray(cloud_camvert.points, dtype=np.float32))
        # xyz = self.random_sample(xyz, int(len(xyz)/3))  # down sample

        N, _ = xyz.shape
        y = xyz[:, 1]*100  # m --> cm
        xz = xyz[:, [0, 2]]
        elevation_gt = torch.zeros(self.num_center, dtype=torch.float32)
        count_map = torch.zeros(self.num_center, dtype=torch.int8)

        xyz_center_b = self.map_centers
        xyz_center_b = xyz_center_b.unsqueeze(dim=1).repeat([1, N, 1])  # [num_center,2] -> [num_center, 1, 2] -> [num_center, num_points_all, 2]
        xyz_center_b = xyz_center_b - xz.unsqueeze(0).repeat([self.num_center, 1, 1])  # distance of every grid center to all points

        inner_bool = torch.logical_and(torch.abs(xyz_center_b[:, :, 0]) <= self.grid_res[0]/2, torch.abs(xyz_center_b[:, :, 1]) <= self.grid_res[2]/2)
        group_idx = torch.arange(N, dtype=torch.long).reshape(1, N).repeat([self.num_center, 1])
        inner_indexes = group_idx * inner_bool.long()  # [num_center, num_points_all] mask
        # index the inner points in grids and construct the map
        valid_indexes = inner_indexes.nonzero()
        for valid in valid_indexes:  # the loop will run N times (as every point will fall into one grid)
            elevation_gt[valid[0]] = elevation_gt[valid[0]] + y[inner_indexes[valid[0], valid[1]]]  # record the y value
            count_map[valid[0]] = count_map[valid[0]] + 1

        elevation_mask = count_map > 0
        elevation_gt[elevation_mask] = elevation_gt[elevation_mask] / count_map[elevation_mask]   # calculate the average elevation of every grid
        elevation_gt = elevation_gt.reshape(self.num_grids_z, self.num_grids_x)
        elevation_mask = elevation_mask.reshape(self.num_grids_z, self.num_grids_x)

        # finally, obtain the elevation map relative to the base height (in cm)
        elevation_gt[elevation_mask] = self.base_height*100 - elevation_gt[elevation_mask]

        return elevation_gt, elevation_mask

    def get_gt_preprocessed(self, time):
        with open(os.path.join(self.preprocessed_path, time)+'.pkl', 'rb') as f:
            [ele_gt, ele_mask] = pickle.load(f)
        return ele_gt, ele_mask

    def random_sample(self, xyz, num_center):
        N = xyz.shape[0]
        centroids = torch.multinomial(torch.ones(N), num_center, replacement=False, generator=self.gen)

        return xyz[centroids, :]

    def matrix2euler(self, m):
        # order='XYZ'
        d = np.clip
        m = m.reshape(-1)
        a, f, g, k, l, n, e = m[0], m[1], m[2], m[4], m[5], m[7], m[8]
        y = np.arcsin(d(g, -1, 1))
        if 0.99999 > np.abs(g):
            x = np.arctan2(- l, e)
            z = np.arctan2(- f, a)
        else:
            x = np.arctan2(n, k)
            z = 0
        return np.array([x, y, z], dtype=np.float32)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        sample_cur = self.data_all[index]
        l2c_calib_cur = self.get_lidar2cam(sample_cur['time'])
        path_base = sample_cur['path']
        idx_str = path_base.find('/')
        path_base = path_base[idx_str + 1:]

        ########   calculate the euler angles of the camera (relative to local ENU coord)   ########
        R_cur2enu = self.get_RT_lidar(sample_cur)
        [pitch_cam, roll_cam, _] = self.matrix2euler(l2c_calib_cur['R'] @ np.linalg.inv(R_cur2enu))
        pitch_cam -= 1.5708  # pi/2
        R_X = np.array(
            [[1, 0, 0], [0, np.cos(pitch_cam), np.sin(pitch_cam)], [0, -np.sin(pitch_cam), np.cos(pitch_cam)]], dtype=np.float32)
        R_Z = np.array(
            [[np.cos(roll_cam), np.sin(roll_cam), 0], [-np.sin(roll_cam), np.cos(roll_cam), 0], [0, 0, 1]], dtype=np.float32)
        R_cam2vert = R_X @ R_Z  # the rotation matrix from the current camera coord to the vertical status
        R_vert2cam = torch.from_numpy(np.linalg.inv(R_cam2vert))

        ######   create the GT elevation map  ########
        ele_gt, ele_mask = self.get_gt_preprocessed(sample_cur['time'])

        ##########  read the RGB images   ############
        path_img = os.path.join(self.data_path, path_base, 'left_half', sample_cur['time']) + '.jpg'
        img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
        imgs_left = self.transform_jpg(img)

        voxel_cam_left = R_vert2cam @ self.voxel_centers
        if self.stereo:
            #########   calculate the index relationship between 3D voxels and 2D pixels   ##############
            voxel_cam_right = voxel_cam_left
            voxel_cam_right[0, :] = voxel_cam_right[0, :] - l2c_calib_cur['B']
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            uvz_right = l2c_calib_cur['K_feat_T'] @ voxel_cam_right  # projection index on right image plane
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)
            voxel_uv_right = torch.floor(uvz_right[:2, :] / uvz_right[2:, :]).type(torch.long)

            path_img = os.path.join(self.data_path, path_base, 'right_half', sample_cur['time']) + '.jpg'
            img = PIL.Image.open(path_img).crop((0, 0, 960, 528))
            imgs_right = self.transform_jpg(img)

            return imgs_left, imgs_right, ele_gt, ele_mask, voxel_uv_left, voxel_uv_right, sample_cur['time']
        else:
            uvz_left = l2c_calib_cur['K_feat_T'] @ voxel_cam_left
            voxel_uv_left = torch.floor(uvz_left[:2, :] / uvz_left[2:, :]).type(torch.long)
            return imgs_left, ele_gt, ele_mask, voxel_uv_left, sample_cur['time']

if __name__ == '__main__':
    dataset = RSRD(down_scale=2, training=False, stereo=False)
    for i in range(len(dataset)):
        dataset.__getitem__(i)

