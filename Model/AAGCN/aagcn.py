import torch.nn as nn
from Model.AAGCN.Components.gcn import GCN
from Model.AAGCN.Components.tcn import TCN


class AAGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        adaptive=True,
        attention=True,
    ):
        super(AAGCN, self).__init__()
        self.gcn1 = GCN(
            in_channels, out_channels, A, adaptive=adaptive, attention=attention
        )
        self.tcn1 = TCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
