import torch.nn as nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.l3 = nn.Linear(64, 3)

    def forward(self, x):
        N, M, C, T, V = x.size()
        x = x.reshape(N * M, C, T, V)
        x = x.view(N, M, 64, T, V).permute(0, 1, 3, 4, 2)
        x = self.l3(x)
        x = x.permute(0, 4, 2, 3, 1)
        return x
