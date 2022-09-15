import torch
import torch.nn as nn

from backbone.cspdarknet import CSPDarknet
from block.cbl import CBL as BaseConv
from block.csp import CSPLayer
from block.dwcbl import DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()

        # backbone
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        # backbone
        out_features = self.backbone(input)

        # ------------------------------------------------- #
        # [
        #   x2: "dark3": (80, 80, 256),
        #   x1: "dark4": (40, 40, 512),
        #   x0: "dark5": (20, 20, 1024),
        # ]
        # ------------------------------------------------- #
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # ------------------------------------------------- #
        # CBS(i=16c,o=8c,k=1,s=1,p=same)
        #   20, 20, 1024 -> 20, 20, 512
        # ------------------------------------------------- #
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32

        # ------------------------------------------------- #
        # upsample(scale_factor=2, mode="nearest")
        #   20, 20, 512 -> 40, 40, 512
        # ------------------------------------------------- #
        f_out0 = self.upsample(fpn_out0)  # 512/16

        # ------------------------------------------------- #
        # cat
        #   f_out0(40, 40, 512), x1(40, 40, 512) -> 40, 40, 1024
        # ------------------------------------------------- #
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16

        # ------------------------------------------------- #
        # CSP2_3(i=16c,o=8c,n=3)
        #   40, 40, 1024 -> 40, 40, 512
        # ------------------------------------------------- #
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        # ------------------------------------------------- #
        # CBS(i=8c,o=4c,k=1,s=1,p=same)
        #   40, 40, 512 -> 40, 40, 256
        # ------------------------------------------------- #
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16

        # ------------------------------------------------- #
        # upsample(scale_factor=2, mode="nearest")
        #   40, 40, 256 -> 80, 80, 256
        # ------------------------------------------------- #
        f_out1 = self.upsample(fpn_out1)  # 256/8

        # ------------------------------------------------- #
        # cat
        #   f_out1(80, 80, 256), x2(80, 80, 256) -> 80, 80, 512
        # ------------------------------------------------- #
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8

        # ------------------------------------------------- #
        # CSP2_3(i=8c,o=4c,n=3)
        #   80, 80, 512 -> 80, 80, 256
        # ------------------------------------------------- #
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        # ------------------------------------------------- #
        # CBS(i=4c,o=4c,k=3,s=2,p=same)
        #   80, 80, 256 -> 40, 40, 256
        # ------------------------------------------------- #
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16

        # ------------------------------------------------- #
        # cat
        #   p_out1(40, 40, 256), x2(40, 40, 256) -> 40, 40, 512
        # ------------------------------------------------- #
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16

        # ------------------------------------------------- #
        # CSP2_3(i=8c,o=8c,n=3)
        #   40, 40, 512 -> 40, 40, 512
        # ------------------------------------------------- #
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        # ------------------------------------------------- #
        # CBS(i=8c,o=8c,k=3,s=2,p=same)
        #   40, 40, 512 -> 20, 20, 512
        # ------------------------------------------------- #
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32

        # ------------------------------------------------- #
        # cat
        #   p_out0(20, 20, 512), x2(20, 20, 512) -> 20, 20, 1024
        # ------------------------------------------------- #
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32

        # ------------------------------------------------- #
        # CSP2_3(i=16c,o=16c,n=3)
        #   20, 20, 1024 -> 20, 20, 1024
        # ------------------------------------------------- #
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs