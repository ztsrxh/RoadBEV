import numpy as np
from utils.experiment import make_nograd_func
import torch
import matplotlib.pyplot as plt

class Metric():
    def __init__(self, ele_range, cla_res, num_grids_z, distance_wise=False):
        self.ele_range = ele_range*100
        self.res = cla_res  # in cm
        self.num_classes = int(2 * self.ele_range / self.res)

        self.metric_all = np.zeros(3,)
        self.count_all = 0

        # if compute the distance-wise metric in the ROI grid
        self.distance_wise = distance_wise
        self.intervals = 11  # number of grids for every segment
        self.num_intervals = int(num_grids_z/self.intervals)+1
        self.metric_wise = np.zeros((self.num_intervals, 3))
        self.count_wise = np.zeros(self.num_intervals)

    @make_nograd_func
    def clear(self):
        self.count_all = 0
        self.metric_all *= 0
        self.count_wise *= 0
        self.metric_wise *= 0

    @make_nograd_func
    def plot_depthwise(self, metric_depthwise):
        plt.figure()
        plt.subplot(121)
        plt.plot(np.flip(metric_depthwise[:, 0]), marker='*')
        plt.title('Abs_err')
        plt.subplot(122)
        plt.plot(np.flip(metric_depthwise[:, 1]), marker='*')
        plt.title('RMSE')
        plt.show()

    @make_nograd_func
    def get_metric(self):
        metric_all = self.metric_all / self.count_all
        if self.distance_wise:
            metric_wise = self.metric_wise / self.count_wise.reshape(-1, 1)
            return [metric_all, metric_wise]
        else:
            return [metric_all, None]

    @make_nograd_func
    def compute_values(self, ele_gt, ele_pred):
        abs_err = torch.abs(ele_gt - ele_pred)
        rmse = (ele_gt - ele_pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        err_mask = abs_err > 0.5
        ratio_thresh = torch.mean(err_mask.float())

        return np.array(torch.tensor([torch.mean(abs_err), rmse, ratio_thresh], device='cpu'))

    @make_nograd_func
    def compute(self, ele_pred, ele_gt, mask):
        # ele_pred: [B, H, W]
        mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(mask_roi, mask)

        self.count_all += 1
        self.metric_all += self.compute_values(ele_gt[ele_mask], ele_pred[ele_mask])

        if self.distance_wise:
            for i in range(self.num_intervals):
                try:
                    ele_gt_ = ele_gt[:, i * self.intervals:(i+1)*self.intervals, :]
                    ele_pred_ = ele_pred[:, i*self.intervals:(i+1)*self.intervals, :]
                    ele_mask_ = ele_mask[:, i*self.intervals:(i+1)*self.intervals, :]
                except:
                    ele_gt_ = ele_gt[:, i*self.intervals:, :]
                    ele_pred_ = ele_pred[:, i*self.intervals:, :]
                    ele_mask_ = ele_mask[:, i*self.intervals:, :]
                gt_valid = ele_gt_[ele_mask_]
                if len(gt_valid) > 0:
                    pred_valid = ele_pred_[ele_mask_]
                    values = self.compute_values(gt_valid, pred_valid)
                    self.metric_wise[i, :] += values
                    self.count_wise[i] += 1
