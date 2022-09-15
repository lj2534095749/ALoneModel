import torch
from torch import nn

from cbl import CBL as BaseConv


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP
    1 CBL(i=c,o=hid,k=1,s=1,p=same) ->
    3 max_pool(k=5,s=1,p=same)
      max_pool(k=9,s=1,p=same)
      max_pool(k=13,s=1,p=same) ->
    cat(1 CBL,3 max_pool) ->
    1 CBL(i=4c,o=c,k=1,s=1,p=same)
    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        # CBL(i=c,o=hid,k=1,s=1,p=same), (b,hid,h,w)
        x = self.conv1(x)
        # MaxPool2d(k, s=1, p=same), (b,hid,h,w)
        # cat, (b,conv2_channels,h,w)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        # CBL(i=conv2_channels,o=ok=1,s=1,p=same), (b,o,h,w)
        x = self.conv2(x)
        return x

