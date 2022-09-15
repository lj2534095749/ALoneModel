import torch.nn as nn

from block import *


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        # ------------------------------------------------- #
        # 输入图片是640, 640, 3
        # 初始的基本通道是64
        # ------------------------------------------------- #
        base_channels = int(wid_mul * 64)  # 64

        # round
        # ------------------------------------------------- #
        # 将数字四舍五入到小数位数的给定精度。
        # 如果省略了ndigits或无，则返回值为整数。
        # ------------------------------------------------- #
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # ------------------------------------------------- #
        # Focus(i=3,o=c,k=3)
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # ------------------------------------------------- #
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        # ------------------------------------------------- #
        # CBS(i=c,o=2c,k=3,s=2,p=same)
        #   320, 320, 64 -> 160, 160, 128
        # CSP1_1(i=2c,o=2c,n=3)
        #   160, 160, 128 -> 160, 160, 128
        # ------------------------------------------------- #
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        # ------------------------------------------------- #
        # CBS(i=2c,o=4c,k=3,s=2,p=same)
        #   160, 160, 128 -> 80, 80, 256
        # CSP1_3(i=4c,o=4c,n=3)
        #   80, 80, 256 -> 80, 80, 256
        # ------------------------------------------------- #
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        # ------------------------------------------------- #
        # CBS(i=4c,o=8c,k=3,s=2,p=same)
        #   80, 80, 256 -> 40, 40, 512
        # CSP1_3(i=8c,o=8c,n=3)
        #   40, 40, 512 -> 40, 40, 512
        # ------------------------------------------------- #
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        # ------------------------------------------------- #
        # CBS(i=8c,o=16c,k=3,s=2,p=same)
        #   40, 40, 512 -> 20, 20, 1024
        # SPP(i=16c,o=16c)
        #   20, 20, 1024 -> 20, 20, 1024
        # CSP2_1(i=16c,o=16c,n=1), no residual
        #   20, 20, 1024 -> 20, 20, 1024
        # ------------------------------------------------- #
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
