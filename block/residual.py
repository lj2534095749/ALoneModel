from torch import nn

from cbl import CBL as BaseConv
from block.dwcbl import DWConv


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Bottleneck(nn.Module):
    # Standard bottleneck, Res Unit
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        # CBS(i=c,o=hid,k=1,s=1,p=same) -> CBS(i=c,o=o,k=3,s=1,p=same)
        # (b,c,w,h) -> (b,hid,w,h) -> (b,o,w,h)
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
