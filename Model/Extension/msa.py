import copy
from einops import rearrange

import torch.nn as nn
from torch import einsum


#################### * MSA * ####################


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        N, T, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "n t (h d) -> n h t d", h=h), qkv)

        dots = einsum("n h i d, n h j d -> n h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("n h i j, n h j d -> n h i d", attn, v)
        out = rearrange(out, "n h t d -> n t (h d)")
        out = self.to_out(out)
        return out


class ATULayer(nn.Module):
    def __init__(self, T):
        super(ATULayer, self).__init__()
        # self.attn = Residual(
        #     PreNorm(1600, Attention(1600, heads=8, dim_head=128, dropout=0.0))
        # )
        self.attn = Residual(
            PreNorm(2400, Attention(2400, heads=8, dim_head=128, dropout=0.0))
        )
        self.linear = nn.Linear(T, T)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        N, C, V, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(N, T, C * V)
        # x = x.reshape(-1, 2400)
        x = self.attn(x)
        x = x.reshape(N, T, C, V).permute(0, 2, 3, 1)

        x = self.linear(x)  # N,C,V,T
        x = self.Tanh(x)
        return x


class ATNet(nn.Module):
    def __init__(self, layers, T):
        super(ATNet, self).__init__()
        self.layer = layers
        ATUlayer = ATULayer(T)
        self.ATUNet = nn.ModuleList(
            [copy.deepcopy(ATUlayer) for _ in range(self.layer)]
        )

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2)
        for ATU in self.ATUNet:
            x = ATU(x)
        x = x.permute(0, 1, 3, 2)
        return x
