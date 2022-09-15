from torch import nn

from block.cbl import CBL as BaseConv
from block.spp import SPPBottleneck
from block.residual import ResLayer


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    # Darknet21 Darknet53
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        # 1 CBL(i=3,o=32,k=3,s=1,p=same) -> make_group_layer(num_blocks=1)
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )

        # after stem, 1C -> 2C
        in_channels = stem_out_channels * 2  # 64

        # darknet21 or darknet53
        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        """starts with conv layer then has `num_blocks` `ResLayer`
        1 CBL -> num_blocks RES
            1 CBL : (i=c,o=2c,k=3,s=2,p=same).
            1 RES : (i=2c,o=c,k=1,s=1,p=same) -> (i=c,o=2c,k=3,s=1,p=same)
        C,W,H -> CBL(2C,W/2,H/2) -> RES(C,W/2,H/2 -> 2C,W/2,H/2)
        """
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        """spp block
        1 CBL(i=2c,o=c,k=1,s=1,p=same) ->
        1 CBL(i=c,o=2c,k=3,s=1,p=same) ->
        1 SPP(i=2c,o=c,k=3,s=1,p=same) ->
        1 CBL(i=c,o=2c,k=3,s=1,p=same) ->
        1 CBL(i=2c,o=c,k=1,s=1,p=same)
        """
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):

        outputs = {}

        # 608, 608, 3 -> 608, 608, 32
        # ------------------------------------------------- #
        # CBL(i=3,o=32,k=3,s=1,p=same)
        #   608, 608, 3 -> 608, 608, 32
        # make_group_layer(num_blocks=1)
        #   CBL(i=c,o=2c,k=3,s=2,p=same)
        #       608, 608, 32 -> 304, 304, 64
        #   RES(i=2c,o=c,k=1,s=1,p=same) -> (i=c,o=2c,k=3,s=1,p=same)
        #       304, 304, 64 -> 304, 304, 32 -> 304, 304, 64
        # ------------------------------------------------- #
        x = self.stem(x)
        outputs["stem"] = x

        # 304, 304, 64  -> 152, 152, 128
        # ------------------------------------------------- #
        # CBL(i=2c,o=4c,k=3,s=2,p=same)
        #   304, 304, 64  -> 152, 152, 128
        # 2 * RES[CBL(i=4c,o=2c,k=1,s=1,p=same) -> CBL(i=2c,o=4c,k=3,s=1,p=same)]
        #   2 * (152, 152, 128 -> 152, 152, 64 -> 152, 152, 128)
        # ------------------------------------------------- #
        x = self.dark2(x)
        outputs["dark2"] = x

        # 152, 152, 128  -> 76, 76, 256
        # ------------------------------------------------- #
        # CBL(i=4c,o=8c,k=3,s=2,p=same)
        #   152, 152, 128  -> 76, 76, 256
        # 8 * RES[CBL(i=8c,o=4c,k=1,s=1,p=same) -> CBL(i=4c,o=8c,k=3,s=1,p=same)]
        #   8 * (76, 76, 256 -> 76, 76, 128 -> 76, 76, 256)
        # ------------------------------------------------- #
        x = self.dark3(x)
        outputs["dark3"] = x

        # 76, 76, 256 -> 38, 38, 512
        # ------------------------------------------------- #
        # CBL(i=8c,o=16c,k=3,s=2,p=same)
        #   76, 76, 256  -> 38, 38, 512
        # 8 * RES[CBL(i=16c,o=8c,k=1,s=1,p=same) -> CBL(i=8c,o=16c,k=3,s=1,p=same)]
        #   8 * (38, 38, 512 -> 38, 38, 256 -> 38, 38, 512)
        # ------------------------------------------------- #
        x = self.dark4(x)
        outputs["dark4"] = x

        # 38, 38, 512 -> 19, 19, 512
        # ------------------------------------------------- #
        # CBL(i=16c,o=32c,k=3,s=2,p=same)
        #   38, 38, 512  -> 19, 19, 1024
        # 8 * RES[CBL(i=32c,o=16c,k=1,s=1,p=same) -> CBL(i=32c,o=16c,k=3,s=1,p=same)]
        #   8 * (19, 19, 1024 -> 19, 19, 512 -> 19, 19, 1024)
        # SPP[
        #   CBL(i=32c,o=16c,k=1,s=1,p=same)
        #       19, 19, 1024 -> 19, 19, 512
        #   CBL(i=16c,o=32c,k=3,s=1,p=same)
        #       19, 19, 512 -> 19, 19, 1024
        #   SPP(i=32c,o=16c,k=3,s=1,p=same)
        #       19, 19, 1024 -> 19, 19, 512
        #   CBL(i=16c,o=32c,k=3,s=1,p=same)
        #       19, 19, 512 -> 19, 19, 1024
        #   CBL(i=32c,o=16c,k=1,s=1,p=same)
        #       19, 19, 1024 -> 19, 19, 512
        # ]
        # ------------------------------------------------- #
        x = self.dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}
