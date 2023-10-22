import numpy as np

import torch.nn as nn


#################### * RICH 5 * ####################


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def spatial_cross(x, fidx1, fidx2):
    x_spc1 = np.transpose(x[:, :, np.array(fidx1) - 1] - x, (1, 2, 0))
    x_spc2 = np.transpose(x[:, :, np.array(fidx2) - 1] - x, (1, 2, 0))
    x_spcc = 100 * np.cross(x_spc1, x_spc2)

    return x_spcc


# S2
def edge(x):
    x = np.transpose(x, [0, 4, 1, 2, 3])
    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    ori_list = [
        (1, 2),
        (2, 21),
        (3, 21),
        (4, 3),
        (5, 21),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 21),
        (10, 9),
        (11, 10),
        (12, 11),
        (13, 1),
        (14, 13),
        (15, 14),
        (16, 15),
        (17, 1),
        (18, 17),
        (19, 18),
        (20, 19),
        (21, 21),
        (22, 23),
        (23, 8),
        (24, 25),
        (25, 12),
    ]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []

            for r in range(len(ori_list)):
                x_m = x[n, m, :, :, :]
                x_spc1 = np.transpose(
                    x_m[:, :, np.array(ori_list[r][0]) - 1]
                    - x_m[:, :, np.array(ori_list[r][1] - 1)],
                    (1, 0),
                )
                x_spcc = np.copy(x_spc1)
                x_spc_r.append(x_spcc)
            x_spc_m.append(x_spc_r)
        x_spc_n.append(x_spc_m)

    x_spc_n_to_array = np.array(x_spc_n)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 3, 2, 1)

    return x_spc_n_to_array


# S3
def surface(x):
    x = np.transpose(x, [0, 4, 1, 2, 3])

    # Spatial cross
    fidx1 = [
        17,
        21,
        4,
        21,
        6,
        5,
        6,
        22,
        21,
        11,
        12,
        24,
        1,
        13,
        16,
        14,
        18,
        17,
        18,
        19,
        5,
        8,
        8,
        12,
        12,
    ]
    fidx2 = [
        13,
        1,
        21,
        3,
        21,
        7,
        8,
        23,
        10,
        9,
        10,
        25,
        14,
        15,
        14,
        15,
        1,
        19,
        20,
        18,
        9,
        23,
        22,
        25,
        24,
    ]

    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_list = []
        for m in range(M):
            x_m = x[n, m, :, :, :]
            x_spc_list.append(spatial_cross(x_m, fidx1, fidx2))
        x_spc_n.append(x_spc_list)

    x_spc_n_to_array = np.array(x_spc_n)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)

    return x_spc_n_to_array


# T2
def motion(x):
    x = np.transpose(x, [0, 4, 2, 1, 3])
    N = x.shape[0]
    M = x.shape[1]
    T = x.shape[2]
    C = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []
            for r in range(T - 1):
                x_m = x[n, m, :, :, :]
                x_spc1 = x_m[r + 1, :, :] - x_m[r, :, :]
                x_spcc = np.copy(x_spc1)
                x_spc_r.append(x_spcc)
            x_last = x_m[0, :, :] - x_m[0, :, :]
            x_spc_r.append(x_last)
            x_spc_m.append(x_spc_r)
        x_spc_n.append(x_spc_m)

    x_spc_n_to_array = np.array(x_spc_n)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 3, 2, 4, 1)

    return x_spc_n_to_array


# T3
def velocity(x):
    x = np.transpose(x, [0, 4, 1, 2, 3])

    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_list = []
        for m in range(M):
            x_m = x[n, m, :, :, :]
            x_spt_list = []
            x_spt_list.append(np.transpose(x_m[:, 0, :] - x_m[:, 0, :], (1, 0)))
            for t in range(T - 2):
                x_spc1 = np.transpose(x_m[:, t, :] - x_m[:, t + 1, :], (1, 0))
                x_spc2 = np.transpose(x_m[:, t + 2, :] - x_m[:, t + 1, :], (1, 0))
                x_spcc = x_spc1 + x_spc2
                x_spt_list.append(x_spcc)

            x_spt_list.append(np.transpose(x_m[:, 0, :] - x_m[:, 0, :], (1, 0)))
            x_spc_list.append(x_spt_list)
        x_spc_n.append(x_spc_list)
    x_spc_n_to_array = np.array(x_spc_n)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)

    return x_spc_n_to_array
