# -*-coding:utf-8-*-

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def vis_keypoints_3D(vals, front):

    ntu60_connections = [[0, 1], [1, 20], [20, 2], [2, 3],
                         [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],
                         [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],
                         [0, 16], [16, 17], [17, 18], [18, 19],
                         [0, 12], [12, 13], [13, 14], [14, 15]]

    ntu60_LR = np.array([0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0,
                         1, 1, 1, 1], dtype=bool)

    lcolor = '#AA0000'
    rcolor = '#00008B'

    for ind, (i, j) in enumerate(ntu60_connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        if (ind is 0) or (ind is 1) or (ind is 2) or (ind is 3):
            front.plot(x, y, z, alpha=0.8, lw=3, c='#DAA520', marker='o', markersize=5, markerfacecolor='#DAA520')

        else:
            front.plot(x, y, z, alpha=0.6, lw=3, c=lcolor if ntu60_LR[ind] else rcolor, marker='o', markersize=5,
                    markerfacecolor=lcolor if ntu60_LR[ind] else rcolor)

    # ---------------------------------------------------------------------

    front.set_title('Front View', fontsize=12)

    front.patch.set_facecolor('#FFFFFF')
    front.patch.set_alpha(0.1)

    plt.rcParams["axes.edgecolor"] = "#FFFFFF"
    plt.rc('grid', alpha=0.2, lw=1, linestyle="-", c='#00FFFF')
    front.grid(True)

    front.set_xlim3d([-1000, 1000])
    front.set_ylim3d([-1200, 1200])
    front.set_zlim3d([-1200, 1200])

    front.set_xticks(range(-1000, 1000, 500))
    front.set_yticks(range(-1200, 1200, 600))
    front.set_zticks(range(-1200, 1200, 600))

    plt.setp(front.get_xticklabels(), visible=False)
    plt.setp(front.get_yticklabels(), visible=False)
    plt.setp(front.get_zticklabels(), visible=False)
    front.tick_params(axis='both', which='major', length=0)

    front.set_xlabel('X Label', fontsize=10)
    front.set_ylabel('Y Label', fontsize=10)
    front.set_zlabel('Z Label', fontsize=10)

    front.xaxis._axinfo['juggled'] = (2, 0, 1)
    front.yaxis._axinfo['juggled'] = (2, 1, 1)
    front.zaxis._axinfo['juggled'] = (2, 2, 2)

    front.view_init(elev=90, azim=-90)

    return 0


def vis_keypoints_3D_VA(VA_vals, VA_front):

    ntu60_connections = [[0, 1], [1, 20], [20, 2], [2, 3],
                         [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],
                         [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],
                         [0, 16], [16, 17], [17, 18], [18, 19],
                         [0, 12], [12, 13], [13, 14], [14, 15]]

    ntu60_LR = np.array([0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0,
                         1, 1, 1, 1], dtype=bool)

    lcolor = '#AA0000'
    rcolor = '#00008B'

    for ind, (i, j) in enumerate(ntu60_connections):
        x, y, z = [np.array([VA_vals[i, c], VA_vals[j, c]]) for c in range(3)]
        if (ind is 0) or (ind is 1) or (ind is 2) or (ind is 3):
            VA_front.plot(x, y, z, alpha=0.8, lw=3, c='#DAA520', marker='o', markersize=5, markerfacecolor='#DAA520')
        else:
            VA_front.plot(x, y, z, alpha=0.6, lw=3, c=lcolor if ntu60_LR[ind] else rcolor, marker='o', markersize=5,
                    markerfacecolor=lcolor if ntu60_LR[ind] else rcolor)

    # ---------------------------------------------------------------------

    VA_front.set_title('VA Front View', fontsize=12)

    VA_front.patch.set_facecolor('#FFFFFF')
    VA_front.patch.set_alpha(0.1)

    plt.rcParams["axes.edgecolor"] = "#FFFFFF"
    plt.rc('grid', alpha=0.2, lw=1, linestyle="-", c='#00FFFF')
    VA_front.grid(True)

    VA_front.set_xlim3d([-1000, 1000])
    VA_front.set_ylim3d([-1200, 1200])
    VA_front.set_zlim3d([-1200, 1200])

    VA_front.set_xticks(range(-1000, 1000, 500))
    VA_front.set_yticks(range(-1200, 1200, 600))
    VA_front.set_zticks(range(-1200, 1200, 600))

    # 顯示刻度
    plt.setp(VA_front.get_xticklabels(), visible=False)
    plt.setp(VA_front.get_yticklabels(), visible=False)
    plt.setp(VA_front.get_zticklabels(), visible=False)

    VA_front.tick_params(axis='both', which='major', length=0)

    # 顯示 x,y,z Label
    VA_front.set_xlabel('X Label', fontsize=10)
    VA_front.set_ylabel('Y Label', fontsize=10)
    VA_front.set_zlabel('Z Label', fontsize=10)

    # 顯示 x,y,z 軸位置
    VA_front.xaxis._axinfo['juggled'] = (2, 0, 1)
    VA_front.yaxis._axinfo['juggled'] = (2, 1, 1)
    VA_front.zaxis._axinfo['juggled'] = (2, 2, 2)

    VA_front.view_init(elev=90, azim=-90)

    return 0


if __name__ == '__main__':

    fig = plt.figure('3D Skeleton', figsize=(9, 6), dpi=100)
    vis = True
    if vis:
        for i in range(300):
            plt.ion()

            ax2 = fig.add_subplot(1, 2, 1, projection='3d')
            ax3 = fig.add_subplot(1, 2, 2, projection='3d')

            output = np.load('/home/antony/2s-AGCN/VA_vis/action1_drink_water.npy')
            output2 = np.load('/home/antony/2s-AGCN/VA_vis/action1_drink_water_va.npy')
            output = output * 1000
            output2 = output2 * 1000

            output = torch.Tensor(output).permute(1, 2, 0)  # (1, 25, 3)
            output2 = torch.Tensor(output2).permute(1, 2, 0)  # (1, 25, 3)

            vis_keypoints_3D(output[i, :, :], front=ax2)
            vis_keypoints_3D_VA(output2[i, :, :], VA_front=ax3)
            plt.savefig('/home/antony/2s-AGCN/VA_vis/action1_drink_water/'
                        'vis_action0_{:03d}.png'.format(i))  # 儲存圖片
            # plt.draw()
            # plt.pause(0)
            plt.clf()
