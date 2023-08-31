import numpy as np
from random import sample


class Data_transform():
    def __init__(self, data_transform):
        self.data_transform = data_transform
        # Spatial diff
        self.fidx = [2, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

        # Spatial cross
        self.fidx1 = [17, 21, 4, 21, 6, 5, 6, 22, 21, 11, 12, 24, 1, 13, 16, 14, 18, 17, 18, 19, 5, 8, 8, 12, 12]
        self.fidx2 = [13, 1, 21, 3, 21, 7, 8, 23, 10, 9, 10, 25, 14, 15, 14, 15, 1, 19, 20, 18, 9, 23, 22, 25, 24]

    def __call__(self, data):

        x_spd_list, x_spc_list, x_tpcross_list = [], [], []
        # x, location = data
        
        # if self.data_transform:
        #     # print('x.shape',x.shape)
        #     C, T, V, M = x.shape
        #     x_new = np.zeros((C * 5, T, V, M))
        #     # x_new = np.zeros((C, T, V, M))
        #     x_new[:C, :, :, :] = x
        
        #     # ----- spatial relative -----#
        #     for i in range(V):
        #         x_new[:C, :, i, :] = x[:, :, i, :] - x[:, :, 20, :]
        
        #     # ----- spatial edge & surface -----#
        #     for m in range(M):
        #         x_m = x[:, :, :, m]
        #         # spatial diff
        #         x_spd_list.append(self.spatial_diff(x_m))
        #         # spatial cross
        #         x_spc_list.append(self.spatial_cross(x_m))
        #     x_new[C:(2 * C)] = np.transpose(np.array(x_spd_list), [3, 1, 2, 0])
        #     x_new[(2 * C):(3 * C)] = np.transpose(np.array(x_spc_list), [3, 1, 2, 0])
        
        #     # ----- temporal_diff ----- #
        #     for i in range(T - 1):
        #         x_new[(3 * C):(4 * C), i, :, :] = x[:, i + 1, :, :] - x[:, i, :, :]
        #         print('x_new[(3 * C):(4 * C), i, :, :]\n',x_new[(3 * C):(4 * C), i, :, :] )
        #     # --- velocity ---#
        #     for i in range(T - 2):
        #         x_new[(4 * C):, i, :, :] = (x[:, i, :, :] - x[:, i + 1, :, :]) + (x[:, i + 2, :, :] - x[:, i + 1, :, :])
        
        #     return (x_new, location)


        # x, location = data
        # if self.data_transform:
        #     C, T, V, M = x.shape
        #     x_new = np.zeros((C*3, T, V, M))
        #     x_new[:C, :, :, :] = x
        #     for i in range(T-1):
        #         for j in range(V):
        #             x_new[C:(2 * C), i, :, :] = x[:, i + 1, :, :] - x[:, i, :, :]
        #     for i in range(V):
        #         x_new[(2 * C):, :, i, :] = x[:, :, i, :] - x[:, :, 1, :]
        #     return (x_new, location)
        # else:
        #     return (x, location)

        x, location = data
        if self.data_transform:
            C, T, V, M = x.shape
            x_new = np.zeros((C, T, V, M))
            for i in range(V):
                x_new[:C, :, i, :] = x[:, :, i, :] - x[:, :, 20, :]
            # print(x_new[0, 0, :, 0])
            return (x_new, location)
        else:
            return (x, location)

        # x, location = data
        # # print(x[0, 0, :, 0])
        # if self.data_transform:
        #     C, T, V, M = x.shape
        #     return (x, location)
        # else:
        #     return (x, location)

    def spatial_diff(self, x):
        return np.transpose(x[:, :, np.array(self.fidx) - 1] - x, (1, 2, 0))

    def spatial_cross(self, x):
        x_spc1 = np.transpose(x[:, :, np.array(self.fidx1) - 1] - x, (1, 2, 0))
        x_spc2 = np.transpose(x[:, :, np.array(self.fidx2) - 1] - x, (1, 2, 0))
        x_spcc = 100 * np.cross(x_spc1, x_spc2)
        # print(x_spcc.shape)# (300, 25, 3)
        return x_spcc


class Occlusion_part():
    def __init__(self, occlusion_part):
        self.occlusion_part = occlusion_part

        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1                      # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1                   # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1                          # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1          # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1                          # trunk
        self.parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1      # left body
        self.parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1   # right body

    def __call__(self, data):
        # print('data',data)
        # x, location = data
        x=data #change
        
        # print(location.shape)
        for part in self.occlusion_part:
            x[:, :, self.parts[part], :] = 0
            
        # return (x, location)
        return x #change


class Occlusion_rand():
    def __init__(self, occlusion_rand, data_shape):
        C, T, V, M = data_shape
        index_list = []
        self.mask = np.random.rand(T, V, M)
        # self.mask[self.mask > occlusion_rand] = 1
        # self.mask[self.mask <= occlusion_rand] = 0

        for i in range(T):
            np.random.seed()

            index = np.random.choice(a=25, size=13, replace=False,
                                     p=[0.3 / 10.8, 0.2 / 10.8, 0.2 / 10.8, 0.3 / 10.8, 0.2 / 10.8,
                                        0.3 / 10.8, 0.4 / 10.8, 0.5 / 10.8, 0.2 / 10.8, 0.3 / 10.8,
                                        0.4 / 10.8, 0.5 / 10.8, 0.4 / 10.8, 0.5 / 10.8, 0.6 / 10.8,
                                        0.7 / 10.8, 0.4 / 10.8, 0.5 / 10.8, 0.6 / 10.8, 0.7 / 10.8,
                                        0.0 / 10.8, 0.7 / 10.8, 0.6 / 10.8, 0.7 / 10.8, 0.6 / 10.8])

            # index_list.append(index)
            # np.save('test_occluded_13', index_list)
            index = np.load('test_occluded_13.npy')

            # print(index)
            # print('Ok')
            self.mask[i, index[i], :] = 0
            self.mask[self.mask > 0] = 1

    def __call__(self, data):
        x, location = data
        x = x * self.mask[np.newaxis, :, :, :]
        return (x, location)


class Jittering_joint():
    def __init__(self, jittering_joint, data_shape, sigma=1):
        C, T, V, M = data_shape
        noise = sigma * np.random.randn(T, V, M)
        self.mask = np.random.rand(T, V, M)
        self.mask[self.mask > jittering_joint] = 1
        self.mask[self.mask <= jittering_joint] = 0
        self.mask = 1 - self.mask
        self.mask *= noise

    def __call__(self, data):
        x, location = data
        x = x + self.mask[np.newaxis, :, :, :]
        return (x, location)


class Jittering_frame():
    def __init__(self, jittering_frame, data_shape, sigma=1):
        C, T, V, M = data_shape
        noise = sigma * np.random.randn(T)
        self.mask = np.random.rand(T)
        self.mask[self.mask > jittering_frame] = 1
        self.mask[self.mask <= jittering_frame] = 0
        self.mask = 1 - self.mask
        self.mask *= noise

    def __call__(self, data):
        x, location = data
        x = x + self.mask[np.newaxis, :, np.newaxis, np.newaxis]
        return (x, location)



class Occlusion_time():
    def __init__(self, occlusion_time):
        self.occlusion_time = int(occlusion_time // 2)

    def __call__(self, data):
        x, location = data
        if not self.occlusion_time == 0:
            x[:, (50-self.occlusion_time):(50+self.occlusion_time), :, :] = 0
        return (x, location)


class Occlusion_block():
    def __init__(self, threshold):
        if threshold == 0:
            self.threshold = 0
        else:
            self.threshold = 50 * (threshold + 2)

    def __call__(self, data):
        x, location = data
        if self.threshold:
            y_max = np.max(location[1,:,:,:])
            mask = location[1] > (y_max - self.threshold)
            for i in range(x.shape[0]):
                x[i][mask] = 0
        return (x, location)
