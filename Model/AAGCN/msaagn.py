import torch.nn as nn

from Model.AAGCN.Components.gcn_msa import GCN_MSA
from Model.AAGCN.Components.tcn import TCN

class MSAAGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        residual=True,
        stride=1,
        adaptive=True,
        attention=True,
        ATU_layer=2,
    ):
        super(MSAAGCN, self).__init__()
        self.gcn1 = GCN_MSA(
            in_channels,
            out_channels,
            A,
            adaptive=adaptive,
            attention=attention,
            ATU_layer=ATU_layer,
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
