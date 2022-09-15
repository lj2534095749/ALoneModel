import torch
import torch.nn as nn

from block.cbl import CBL as BaseConv


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        # (b,4c,w/2,h/2) -> (b,out_channels,w/2,h/2)
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # ---------------------------------------------------------#
        #  [m::n] 从a[m]开始，每跳n个取一个，m为空则默认m=0
        #  shape of x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # ---------------------------------------------------------#
        patch_top_left = x[..., ::2, ::2]  # (b, c, w/2, h/2), patch_top_left(..., 0 2 4..., 0 2 4...)
        patch_top_right = x[..., ::2, 1::2]  # (b, c, w/2, h/2), patch_top_right(..., 0 2 4..., 1 3 5...)
        patch_bot_left = x[..., 1::2, ::2]  # (b, c, w/2, h/2), patch_bot_left(..., 1 3 5..., 0 2 4...)
        patch_bot_right = x[..., 1::2, 1::2]  # (b, c, w/2, h/2), patch_bot_right(..., 1 3 5..., 1 3 5...)
        # (b,4c,w/2,h/2)
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
