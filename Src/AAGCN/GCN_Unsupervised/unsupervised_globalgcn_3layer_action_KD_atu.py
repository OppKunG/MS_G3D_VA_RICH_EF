import torch
import torch.nn as nn
from torch.autograd import Variable

import math
import numpy as np
import random

from Src.AAGCN.GCN_Unsupervised.GCNEncoder.AGCNEncoder_3layer_action_atu import AGCNEncoder,AAGCN_VA_RICH
from Src.AAGCN.GCN_Unsupervised.rotation import *
from Src.AAGCN.GCN_Unsupervised.GCNEncoder.graph import Graph
from Model.Extension.va import ViewAdaptive


random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.l3 = nn.Linear(64,3) # 3layer

    def forward(self,x):
        N, M,C,T,V = x.size()
        x = x.reshape(N*M,C,T,V)
        x = x.view(N,M,64,T,V).permute(0,1,3,4,2) # 3layer
        x = self.l3(x)
        x = x.permute(0,4,2,3,1)
        return x

class Model_teacher(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=300,predict_seg=1):
        super(Model_teacher, self).__init__()

        self.full_body = AGCNEncoder(num_class, num_point, num_person, graph, graph_args,in_channels, adaptive=True, attention=True,ATU_layer=ATU_layer) #change

        self.FULL_BODY = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])

        if GCNEncoder == "AGCN":
            self.Encoder = AGCNEncoder(num_class, num_point, num_person, graph, graph_args,64,ATU_layer=ATU_layer) #change 3layer

        self.Decoder = Decoder()
        
        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate

        self.predict_seg = predict_seg

        self.Classifier = AGCNClassifier()

        self.AAGCN = AAGCN_VA_RICH()

        # Skeleton parts
        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1                      # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1                   # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1                          # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1          # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1                          # trunk
        self.parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1      # left body
        self.parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1   # right body
        self.parts[8] = np.array([], dtype=np.int32)  # no occlusion
        self.parts[9] = np.array([], dtype=np.int32)
        self.parts[10] = np.array([], dtype=np.int32)
        self.parts[11] = np.array([], dtype=np.int32)
        self.parts[12] = np.array([], dtype=np.int32)
        self.parts[13] = np.array([], dtype=np.int32)
        self.parts[14] = np.array([], dtype=np.int32)
        self.parts[15] = np.array([], dtype=np.int32)
        self.parts[16] = np.array([], dtype=np.int32)
        self.parts[17] = np.array([], dtype=np.int32)
        self.parts[18] = np.array([], dtype=np.int32)
        self.parts[19] = np.array([], dtype=np.int32)
        self.parts[20] = np.array([], dtype=np.int32)
        self.parts[21] = np.array([], dtype=np.int32)
        self.parts[22] = np.array([], dtype=np.int32)
        self.parts[23] = np.array([], dtype=np.int32)
        self.parts[24] = np.array([], dtype=np.int32)

        miss_rate = 0.6
        self.joint_list=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])-1
        self.joint_occlusion_num=int(len(self.joint_list) *miss_rate)


    def forward(self,x,is_test=False):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        joint_occlusion = np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)
        x_mask = np.ones((x.shape[0],3, 300, 25,2))
        num_parts = 13 #len(parts)
        part_indices = np.random.choice(num_parts, size=N, replace=True)

        x[:,:,:, self.parts[7], :] = 0
        x_mask[:, :,:, self.parts[7], :] = 0 #change occlusion 

        # Block data
        for i_b in range(x.shape[0]):
            # Randomly choose parts to be blocked
            part_index = part_indices[i_b]
            part = self.parts[part_index+1] # Add 1 because the index of the location starts from 1
            if part_index<=8:
                # Selected parts
                x[i_b, :, :, part, :] = 0
                x_mask[i_b, :, :, part, :] = 0
            else:
                miss_rate = np.random.choice([0.2,0.3,0.4,0.5,0.6])#0,0.2,0.3,0.4,0.5,0.6
                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint_list) * miss_rate)
                    joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b,:,t, joint_occlusion, :] = 0

        # The dimension of defining GCN_feature is (n, m, 64, t, v)
        GCN_feature = torch.empty(N,M,64,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #3layer
        # GCN_feature = torch.empty(N,M,256,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #10layer

        # Assign the characteristics of each part to gcn_feature, respectively
        GCN_feature[:,:,:,:,self.FULL_BODY-1] = self.full_body(x[:,:,:,self.FULL_BODY-1,:]) #torch.Size([16, 2, 64, 300, 25])

        # Decoder, the output dimension is (N, C, T, V, M)
        output = self.Decoder(GCN_feature)

        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(device)
        output_mask=output*(1-x_mask)+ori_data*(x_mask)

        output_action = self.AAGCN(output_mask)
        return output,GCN_feature,output_action,x_mask,None

class Model_student(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=300,predict_seg=1):
        super(Model_student, self).__init__()
        self.full_body = AGCNEncoder(num_class, 25, num_person, graph, graph_args,in_channels, adaptive=True, attention=True,ATU_layer=ATU_layer) #change
        self.FULL_BODY = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
        if GCNEncoder == "AGCN":
            self.Encoder = AGCNEncoder(num_class, num_point, num_person, graph, graph_args,64,ATU_layer=ATU_layer) # 3 layer

        self.Decoder = Decoder()

        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        self.predict_seg = predict_seg
        self.Classifier = AGCNClassifier()

        self.AAGCN = AAGCN_VA_RICH()

        self.parts = dict()
        self.parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1                      # left arm
        self.parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1                   # right arm
        self.parts[3] = np.array([22, 23, 24, 25]) - 1                          # two hands
        self.parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1          # two legs
        self.parts[5] = np.array([1, 2, 3, 4, 21]) - 1                          # trunk
        self.parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1      # left body
        self.parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1   # right body
        self.parts[8] = np.array([], dtype=np.int32)  # no occlusion
        self.parts[9] = np.array([], dtype=np.int32)
        self.parts[10] = np.array([], dtype=np.int32)
        self.parts[11] = np.array([], dtype=np.int32)
        self.parts[12] = np.array([], dtype=np.int32)
        self.parts[13] = np.array([], dtype=np.int32)
        self.parts[14] = np.array([], dtype=np.int32)
        self.parts[15] = np.array([], dtype=np.int32)
        self.parts[16] = np.array([], dtype=np.int32)
        self.parts[17] = np.array([], dtype=np.int32)
        self.parts[18] = np.array([], dtype=np.int32)
        self.parts[19] = np.array([], dtype=np.int32)
        self.parts[20] = np.array([], dtype=np.int32)
        self.parts[21] = np.array([], dtype=np.int32)
        self.parts[22] = np.array([], dtype=np.int32)
        self.parts[23] = np.array([], dtype=np.int32)
        self.parts[24] = np.array([], dtype=np.int32)


        miss_rate = 0.2
        self.joint_list=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])-1
        self.joint_occlusion_num=int(len(self.joint_list) *miss_rate)


    def forward(self,x,is_test=False):
        N, C, T, V, M = x.size()
        ori_data = x.clone()
        joint_occlusion = np.random.choice(self.joint_list, self.joint_occlusion_num, replace=False)
        x_mask = np.ones((x.shape[0],3, 300, 25,2))
        num_parts = 13 # len(parts)
        part_indices = np.random.choice(num_parts, size=N, replace=True)

        # x[:,:,:, self.parts[1], :] = 0
        # x_mask[:, :,:, self.parts[1], :] = 0  #change occlusion 

        # for i_b in range(x.shape[0]):
        #     miss_rate = np.random.choice([0.2])#0,0.2,0.3,0.4,0.5,0.6
        #     for t in range(x.shape[2]):
        #         joint_occlusion_num = int(len(self.joint_list) * miss_rate)
        #         joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
        #         x[i_b, :, t, joint_occlusion, :] = 0
        #         x_mask[i_b,:,t, joint_occlusion, :] = 0

        # # 遮擋數據
        # for i_b in range(x.shape[0]):
        #     # skeletons = x[i_b]
        #     # 隨機選擇要遮擋的部位
        #     part_index = part_indices[i_b]
        #     part = self.parts[part_index+1] # 加1是因為部位索引從1開始
        #     # 遮擋所選部位
        #     x[i_b, :, :, part, :] = 0
        #     x_mask[i_b, :, :, part, :] = 0

        # Block data
        for i_b in range(x.shape[0]):
            # Randomly choose parts to be blocked
            part_index = part_indices[i_b]
            part = self.parts[part_index+1] # Add 1 because the index of the location starts from 1
            if part_index<=8:
                # Selected parts
                x[i_b, :, :, part, :] = 0
                x_mask[i_b, :, :, part, :] = 0
            else:
                miss_rate = np.random.choice([0.2,0.3,0.4,0.5,0.6])#0,0.2,0.3,0.4,0.5,0.6
                for t in range(x.shape[2]):
                    joint_occlusion_num = int(len(self.joint_list) * miss_rate)
                    joint_occlusion = np.random.choice(self.joint_list, joint_occlusion_num, replace=False)
                    x[i_b, :, t, joint_occlusion, :] = 0
                    x_mask[i_b,:,t, joint_occlusion, :] = 0



        # The dimension of defining GCN_feature is (n, m, 64, t, v)
        GCN_feature = torch.empty(N,M,64,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #3layer
        # GCN_feature = torch.empty(N,M,256,T,V).cuda().to(torch.float32) # N,M,C,T,V  [16, 2, 64, 300, 25] #10layer

        # Assign the characteristics of each part to gcn_feature, respectively
        GCN_feature[:,:,:,:,self.FULL_BODY-1] = self.full_body(x[:,:,:,self.FULL_BODY-1,:]) #torch.Size([16, 2, 64, 300, 25])

        # GCN_feature = GCN_feature.permute(0,2,3,4,1) #N,C,T,V,M #torch.Size([16, 64, 300, 25, 2])
        # GCN_feature = self.Encoder(GCN_feature)

        # Decoder, the output dimension is (n, c, t, v, m)
        output = self.Decoder(GCN_feature)

        x_mask = torch.tensor(x_mask, dtype=torch.float32, requires_grad=False).to(device)
        output_mask=output*(1-x_mask)+ori_data*(x_mask)


        output_action = self.AAGCN(output_mask)
        return output,GCN_feature,output_action,x_mask,None

class AGCNClassifier(nn.Module):
    def __init__(self):
        super(AGCNClassifier, self).__init__()
         # N*M,C,T,V
        num_class = 60
        self.fc = nn.Linear(64, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. /num_class))


    def forward(self,x):
        # print('x.shape: ',x.shape) #torch.Size([16, 64, 300, 25])
        N, M,C,T,V = x.size()
        x = x.reshape(N, M, C, T*V)
        x = x.mean(3).mean(1)

        return self.fc(x)


################################################### MS-AAGCN ################################################################################
def import_class(name):
    print(name)
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def spatial_cross(x, fidx1, fidx2):
    x_spc1 = np.transpose(x[:, :, np.array(fidx1) - 1] - x, (1, 2, 0))  # (300, 25, 3)
    x_spc2 = np.transpose(x[:, :, np.array(fidx2) - 1] - x, (1, 2, 0))
    x_spcc = 100 * np.cross(x_spc1, x_spc2)
    # print(x_spcc.shape)  # (300, 25, 3)
    return x_spcc


def edge(x):
    x = np.transpose(x, [0, 4, 1, 2, 3])
    N = x.shape[0]
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    ori_list = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8),
                (24, 25), (25, 12)]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []

            for r in range(len(ori_list)):
                x_m = x[n, m, :, :, :]
                # print('x_m :', x_m.shape)
                x_spc1 = np.transpose(
                    x_m[:, :, np.array(ori_list[r][0]) - 1] - x_m[:, :, np.array(ori_list[r][1] - 1)], (1, 0))
                x_spcc = np.copy(x_spc1)
                x_spc_r.append(x_spcc)
            x_spc_m.append(x_spc_r)
        x_spc_n.append(x_spc_m)

    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (16, 2, 300, 300, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 3, 2, 1)
    # print('real', x_spc_n_to_array.shape)  # (16, 3, 300, 300, 2)
    # x_spc_n_to_array = torch.tensor(x_spc_n_to_array).cuda()

    # print(x_spc_n_to_array.shape)  #(16, 3, 300, 25, 2)

    return x_spc_n_to_array

def surface(x):

    x = np.transpose(x, [0, 4, 1, 2, 3])
    # print(data.shape) #(40091, 2, 3, 300, 25)

    # Spatial cross
    # fidx0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    fidx1 = [17, 21, 4, 21, 6, 5, 6, 22, 21, 11, 12, 24, 1, 13, 16, 14, 18, 17, 18, 19, 5, 8, 8, 12, 12]
    fidx2 = [13, 1, 21, 3, 21, 7, 8, 23, 10, 9, 10, 25, 14, 15, 14, 15, 1, 19, 20, 18, 9, 23, 22, 25, 24]

    N = x.shape[0]  # (40091, 2, 3, 300, 25)
    M = x.shape[1]
    C = x.shape[2]
    T = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_list = []
        for m in range(M):
            x_m = x[n, m, :, :, :]
            # print('xm',x_m.shape) #(3, 300, 25)
            x_spc_list.append(spatial_cross(x_m, fidx1, fidx2))
            x_spc_list_to_array = np.array(x_spc_list)
            # print(x_spc_list_to_array.shape)  # (40091, 2, 300, 25, 3)
        x_spc_n.append(x_spc_list)

    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (40091, 2, 300, 25, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)
    # print('x_spc_n_to_array', x_spc_n_to_array)
    # print('real', x_spc_n_to_array.shape)  # (40091, 3, 300, 25, 2)

    return x_spc_n_to_array

def motion(x):
    x = np.transpose(x, [0, 4, 2, 1, 3])
    N = x.shape[0]  # (16, 2, 3, 300, 25)
    M = x.shape[1]
    T = x.shape[2]
    C = x.shape[3]
    V = x.shape[4]

    x_spc_n = []
    for n in range(N):
        x_spc_m = []
        for m in range(M):
            x_spc_r = []
            for r in range(T-1):
                x_m = x[n, m, :, :, :]  # (40091, 2, 300, 3, 25)
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

def velocity(x):

    x = np.transpose(x, [0, 4, 1, 2, 3])
    # print(x.shape) #(40091, 2, 3, 300, 25)
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
                # print('xm',x_m.shape) #(3, 300, 25)
                x_spc1 = np.transpose(x_m[:, t, :] - x_m[:, t + 1, :], (1, 0))  # (25, 3)
                x_spc2 = np.transpose(x_m[:, t + 2, :] - x_m[:, t + 1, :], (1, 0))  # (25, 3)
                x_spcc = x_spc1 + x_spc2
                x_spt_list.append(x_spcc)
                # print(np.array(x_spt_list).shape)
            x_spt_list.append(np.transpose(x_m[:, 0, :] - x_m[:, 0, :], (1, 0)))
            x_spc_list.append(x_spt_list)
        x_spc_n.append(x_spc_list)
    x_spc_n_to_array = np.array(x_spc_n)
    # print(x_spc_n_to_array.shape)  # (40091, 2, 300, 25, 3)
    x_spc_n_to_array = x_spc_n_to_array.transpose(0, 4, 2, 3, 1)
    # print('x_spc_n_to_array', x_spc_n_to_array)
    # print('real', x_spc_n_to_array.shape)  # (40091, 3, 300, 25, 2)

    return x_spc_n_to_array


class Unit_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Unit_TCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))

        return x

class Unit_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(Unit_GCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  #Bk(3, 25, 25)
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:

            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            # print(ker_jpt)
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

            # self.bn = nn.BatchNorm2d(out_channels)
            # bn_init(self.bn, 1)
        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            Bk = self.PA  # Bk(3, 25, 25)
            # print('Bk', Bk[0, :, 0])
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)  # [32, 25, 4800]
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)  # [32, 4800, 25]
                Ck = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                # print('Ck', Ck[0, :, 0])
                A1 = Bk[i] + Ck * self.alpha
                # print('Bk[0]', Bk[0])
                # print('Bk[1]', Bk[1])
                # print('Bk[2]', Bk[2])
                # print('alpha', self.alpha)
                # print('A1', A1[0, :, 0])
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        else:
            A = self.A.cuda(x.get_device()) * self.mask
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:

            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

        return y

class TCN_GCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_Unit, self).__init__()
        self.gcn1 = Unit_GCN(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.tcn1 = Unit_TCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = Unit_TCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.attention:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y   

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=15,
                 drop_out=0, adaptive=True, attention=True,ATU_layer=2,T=300):
        super(Model, self).__init__()

        self.graph = Graph()
        A = self.graph.A

        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_Unit(15, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_Unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_Unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_Unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_Unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_Unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_Unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_Unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_Unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_Unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.soft = nn.Softmax(-2)
        self.fc = nn.Linear(256, num_class)
        self.pool = nn.AvgPool2d(kernel_size=(1, 1)) # downsample / only first set
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # VA function
        self.va = ViewAdaptive()

    def forward(self, x):
        N, C, T, V, M = x.size()

        ''''''
        # # ------------------------------------S1+VA+Preprocess(S2_S3_T2_T3)--------------------------------------#
        ''''''
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.permute(0, 2, 1)  # (16, 300, 150) (N, M * V * C, T)
        x = self.va(x)  # (16, 300, 150) (N, M * V * C, T)
        x = x.permute(0, 2, 1)  # (16, 150, 300)
        x = x.view(N, M, V, C, T).permute(0, 1, 2, 3, 4).contiguous().view(N, M, V, C, T)  # (16, 2, 25, 3, 300)
        x = x.permute(0, 3, 4, 2, 1)  # (16, 3, 300, 25, 2)
        x = x.cpu().detach().numpy()
        
        S2 = edge(x)  # (16, 3, 300, 25, 2) E
        S3 = surface(x)  # (16, 3, 300, 25, 2) S
        T2 = motion(x)  # (16, 3, 300, 25, 2) D
        T3 = velocity(x)

        # Early Fusion
        x = np.concatenate((x, S2, S3, T2, T3), axis=1)

        N = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]
        V = x.shape[3]
        M = x.shape[4]
        
        x = torch.Tensor(x)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # (16, 150, 300)
        x = self.data_bn(x.cuda())
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # torch.Size([16, 150, 300])


        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

