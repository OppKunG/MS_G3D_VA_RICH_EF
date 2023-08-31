  
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange, repeat
from torch import einsum
import copy
from GCN_Unsupervised.GCNEncoder.graph import Graph
import sys
import random

random.seed(0)
np.set_printoptions(threshold=sys.maxsize,precision=3,suppress=True,linewidth=800)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


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


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8 , dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # 每個head的維度，即H*D
        self.heads = heads # head的數量
        self.scale = dim_head ** -0.5 # 用來縮放dot-product的值，保持它們的範圍在合理的大小

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        N, T, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h=h), qkv)

        dots = einsum('n h i d, n h j d -> n h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('n h i j, n h j d -> n h i d', attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out


class ATULayer(nn.Module):
    """
    This is the TAU in paper
    """
    def __init__(self,T):
        super(ATULayer,self).__init__()
        self.attn = Residual(PreNorm(1600, Attention(1600, heads=8,dim_head=128, dropout=0.))) #change #3layer
        # self.attn = Residual(PreNorm(6400, Attention(6400, heads=8,dim_head=128, dropout=0.))) #change #3layer
        self.linear = nn.Linear(T,T)
        self.Tanh =  nn.Tanh()

    def forward(self,x):
        # N,M,C,V,T = x.shape
        # x = x.permute(0,1,4,2,3).reshape(N*M,T,C*V)
        # x = self.attn(x)
        # x = x.reshape(N,M,T,C,V).permute(0,1,3,4,2)

        N,C,V,T = x.shape
        x = x.permute(0,3,1,2).reshape(N,T,C*V)
        x = self.attn(x)
        x = x.reshape(N,T,C,V).permute(0,2,3,1)

        x = self.linear(x) # N,C,V,T
        x = self.Tanh(x)
        return x


class ATNet(nn.Module):
    def __init__(self,layers,T):
        super(ATNet,self).__init__()
        self.layer = layers
        ATUlayer = ATULayer(T)
        self.ATUNet = nn.ModuleList([copy.deepcopy(ATUlayer) for _ in range(self.layer)])

    def forward(self,x):
        # N,M,C,T,V = x.shape #torch.Size([128, 2, 64, 300, 25])
        # x = x.permute(0,1,2,4,3)
        # for ATU in self.ATUNet:
        #     x = ATU(x)
        # x = x.permute(0,1,2,4,3)

        N,C,T,V = x.shape #torch.Size([128, 2, 64, 300, 25])
        x = x.permute(0,1,3,2)
        for ATU in self.ATUNet:
            x = ATU(x)
        x = x.permute(0,1,3,2)
        return x

class unit_gcn(nn.Module): #no aa
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3,ATU_layer=2,T=300, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        num_jpts = A.shape[-1] #aa

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1)) #in_channels=3,inter_channels=16
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive: #aa
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  #Bk(3, 25, 25)
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attentionautoEncoder
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
        self.attention = attention #aa 

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
        # print('x.size:',x.size())#torch.Size([8, 3, 300, 25])
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        # print('self.conv_a[i](x)1:',self.conv_a)
        # print('x.size:',x.size())#torch.Size([32, 3, 300, 25])
        # print('N,V,self.inter_c,T',N,V,self.inter_c,T) #32,25,16,300
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        # print('y.size:',y.size())#torch.Size([12, 64, 300, 25])(N, C, T, V)
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


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention) #change 
        # self.gcn1 = unit_gcn(64, out_channels, A) #change in_channels to 64
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class unit_gcn_atu(nn.Module): #no aa
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3,ATU_layer=2,T=300, adaptive=True, attention=True):
        super(unit_gcn_atu, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset
        num_jpts = A.shape[-1] #aa

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1)) #in_channels=3,inter_channels=16
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive: #aa
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  #Bk(3, 25, 25)
            self.alpha = nn.Parameter(torch.zeros(1))
            # self.beta = nn.Parameter(torch.ones(1))
            # nn.init.constant_(self.PA, 1e-6)
            # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
            # self.A = self.PA
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # self.beta = nn.Parameter(torch.zeros(1))
            # self.gamma = nn.Parameter(torch.zeros(1))
            # unified attention
            # self.Attention = nn.Parameter(torch.ones(num_jpts))

            # temporal attentionautoEncoder
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
        self.attention = attention #aa 

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.AttentionTemporalNet = ATNet(ATU_layer,T)

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
        # print('x.size:',x.size())#torch.Size([8, 3, 300, 25])
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        # print('y.size:',y.size())#torch.Size([12, 64, 300, 25])(N, C, T, V)
        if self.attention:
            y = self.AttentionTemporalNet(y)

        return y


class TCN_GCN_unit_atu(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True,ATU_layer=2):
        super(TCN_GCN_unit_atu, self).__init__()
        self.gcn1 = unit_gcn_atu(in_channels, out_channels, A, adaptive=adaptive, attention=attention,ATU_layer=ATU_layer) #change 
        # self.gcn1 = unit_gcn(64, out_channels, A) #change in_channels to 64
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AGCNEncoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, adaptive=True, attention=True,ATU_layer=2): #change 3 to 64
        super(AGCNEncoder, self).__init__()

        self.graph = Graph()
        A = self.graph.A
        # print('in_channels: ',in_channels)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point) #2*3*25=150

        self.num_class = num_class

        # self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l1 = TCN_GCN_unit_atu(in_channels, 64, A, residual=False, adaptive=adaptive, attention=attention,ATU_layer=ATU_layer)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        # self.l2 = TCN_GCN_unit_atu(64, 64, A, adaptive=adaptive, attention=attention,ATU_layer=ATU_layer)
        # self.l3 = TCN_GCN_unit_atu(64, 64, A, adaptive=adaptive, attention=attention,ATU_layer=ATU_layer)

        # self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        # self.l5 = TCN_GCN_unit(64, 128, A, adaptive=adaptive, attention=attention)
        # self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        # self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        # self.l8 = TCN_GCN_unit(128, 256, A, adaptive=adaptive, attention=attention)
        # self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        # self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)


        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size() #[16, 64, 300, 25, 2]

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) #[16, 3200, 300] 3200=64*25*2  #150=25*2*3
        # print('x123: ',x.shape) #torch.Size([16, 150, 300])
        # print('self.data_bn',self.data_bn)
        x = self.data_bn(x) 
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # print('x: ',x.size()) #torch.Size([32, 256, 75, 25])
        # N*M,C,T,V
        c_new = x.size(1)
        # print('c_new: ',N, M, c_new, T, V)
        x = x.view(N, M, c_new, T, V)
        return x

class AAGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, attention=True):
        super(AAGCN, self).__init__()

        self.graph = Graph()
        A = self.graph.A
        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        # self.soft = nn.Softmax(-2)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

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


class MSAAGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=15,
                 drop_out=0, adaptive=True, attention=True,ATU_layer=2,T=300):
        super(MSAAGCN, self).__init__()

        self.graph = Graph()
        A = self.graph.A

        self.num_class = num_class

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(15, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.soft = nn.Softmax(-2)
        self.fc = nn.Linear(256, num_class)
        self.pool = nn.AvgPool2d(kernel_size=(1, 1)) # downsaple / only first set
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        from model.va import ViewAdaptive
        self.va = ViewAdaptive()

    def forward(self, x):
        N, C, T, V, M = x.size()

        ''''''
        # # ------------------------------------S1+va+preprocess(S2_S3_T2_T3)--------------------------------------#
        ''''''
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = x.permute(0, 2, 1)  # (16, 300, 150) (N, M * V * C, T)
        x = self.va(x)  # (16, 300, 150) (N, M * V * C, T)
        x = x.permute(0, 2, 1)  # (16, 150, 300)
        x = x.view(N, M, V, C, T).permute(0, 1, 2, 3, 4).contiguous().view(N, M, V, C, T)  # (16, 2, 25, 3, 300)
        x = x.permute(0, 3, 4, 2, 1)  # (16, 3, 300, 25, 2)
        x = x.cpu().detach().numpy()  # torch to numpy

        S2 = edge(x)  # (16, 3, 300, 25, 2) E
        S3 = surface(x)  # (16, 3, 300, 25, 2) S
        T2 = motion(x)  # (16, 3, 300, 25, 2) D
        T3 = velocity(x)

        x = np.concatenate((x, S2, S3, T2, T3), axis=1)

        N = x.shape[0]
        C = x.shape[1]
        T = x.shape[2]
        V = x.shape[3]
        M = x.shape[4]
        x = torch.Tensor(x)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # (16, 150, 300)
        x = self.data_bn(x.cuda())
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # (32, 3, 300, 25)

        ''''''

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

        # c_new = x10.size(1)
        # x11 = x10.view(N, M, c_new, -1)
        # x11 = x11.mean(3).mean(1)
        # x11 = self.drop_out(x11)

        return self.fc(x)
        # return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], self.fc(x11)
        # return self.fc(x11)  