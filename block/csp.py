import torch
import torch.nn as nn

from block.cbl import CBL as BaseConv
from block.residual import Bottleneck


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)  # CBS(i=c,o=hid,k=1,s=1,p=same), (b,hid,w,h)
        x_2 = self.conv2(x)  # CBS(i=c,o=hid,k=1,s=1,p=same), (b,hid,w,h)
        x_1 = self.m(x_1)  # n * Res(i=hid,o=hid), (b,hid,w,h)
        x = torch.cat((x_1, x_2), dim=1)  # (b,hid+hid,w,h)
        return self.conv3(x)  # CBS(i=2*hid,o=o,k=1,s=1,p=same), (b,o,w,h)

