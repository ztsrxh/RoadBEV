import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.dataset import RSRD
from matplotlib.ticker import LinearLocator
from matplotlib import cm

def plot_bev_mesh():
    x = -torch.arange(rsrd.num_grids_x) * rsrd.grid_res[0] + rsrd.roi_x[1] - rsrd.grid_res[0]/2
    z = torch.arange(rsrd.num_grids_z) * rsrd.grid_res[2] + rsrd.roi_z[0] + rsrd.grid_res[2]/2
    Z, X = np.meshgrid(np.array(z), np.array(x))

    with open(path_pred, 'rb') as f:
        ele_pred = pickle.load(f)
    ele_pred = np.array(ele_pred)
    ele_pred = np.transpose(np.flip(ele_pred, axis=0))
    ele_max = np.max(ele_pred)
    ele_min = np.min(ele_pred)

    fig = plt.figure(figsize=(10, 5), dpi=250)
    ax1 = fig.add_subplot(111)
    im1 = ax1.pcolormesh(Z, X, ele_pred, vmax=ele_max, vmin=ele_min, cmap='plasma')
    plt.colorbar(im1, ax=ax1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_3d_surface():
    x = torch.arange(rsrd.num_grids_x) * rsrd.grid_res[0] + rsrd.roi_x[0] - rsrd.grid_res[0]/2
    z = -torch.arange(rsrd.num_grids_z) * rsrd.grid_res[2] + rsrd.roi_z[1] + rsrd.grid_res[2]/2
    X, Z = np.meshgrid(np.array(x), np.array(z))

    with open(path_pred, 'rb') as f:
        ele_pred = pickle.load(f)
    ele_pred = np.array(ele_pred)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Z, ele_pred, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=1, antialiased=False)
    # ax.plot_wireframe(X, Z, ele_pred, rstride=1, cstride=1,  linewidth=1)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_box_aspect([rsrd.num_grids_x, rsrd.num_grids_z, 10])
    ax.set_zlim(-20, 20)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    rsrd = RSRD()
    path_pred = './bev_pred/20230408023811.200.pkl'

    plot_bev_mesh()
    plot_3d_surface()
