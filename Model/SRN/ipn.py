import sys

import numpy as np
import random

import torch.nn as nn

# AAGCN Model
from Model.AAGCN.Utils.graph import Graph
from Model.AAGCN.aagcn import AAGCN
from Model.AAGCN.msaagn import MSAAGCN

random.seed(0)
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True, linewidth=800)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class IPN(nn.Module):
    def __init__(
        self,
        num_point,
        num_person,
        in_channels,
        ATU_layer,
        adaptive,
        attention,
    ):
        super(IPN, self).__init__()

        self.graph = Graph()
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(
            num_person * in_channels * num_point
        )  # 2*3*25=150

        # Model
        self.l1 = MSAAGCN(
            in_channels,
            64,
            A,
            adaptive,
            attention,
            ATU_layer,
        )
        self.l2 = AAGCN(64, 64, A, adaptive, attention)
        self.l3 = AAGCN(64, 64, A, adaptive, attention)

        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()  # [16, 64, 300, 25, 2]
        x = (
            x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        )  # [16, 3200, 300] 3200=64*25*2  # 150=25*2*3
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, T, V)
        return x
