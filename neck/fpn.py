import torch
from torch import nn

from backbone.darknet_spp import Darknet
from block.cbl import CBL as BaseConv


class YOLOFPN(nn.Module):
    """Neck
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
        in_channels=512,
    ):
        super().__init__()

        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(in_channels, in_channels // 2, 1)
        self.out1 = self._make_embedding([in_channels // 2, in_channels], in_channels + in_channels // 2)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        """
        1 CBL(i=_in,o=_out,k=ks,s=1,p=same)
        """
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        """
        1 CBL(i=c/2+c,o=c/2,k=1,s=1,p=same) ->
        1 CBL(i=c/2,o=c,k=3,s=1,p=same) ->
        1 CBL(i=c,o=c/2,k=1,s=1,p=same) ->
        1 CBL(i=c/2,o=c,k=3,s=1,p=same) ->
        1 CBL(i=2,o=c/2,k=1,s=1,p=same)
        """
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        #  c = 512, w = 13, h = 13
        #  x0(b, c, w, h)
        #  x1(b, c, 2w, 2h)
        #  x1(b, c/2, 4w, 4h)
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        #  yolo branch 1
        #  CBL(i=c,o=c/2,k=1,s=1,p=same)
        #  b,c,w,h -> b,c/2,w,h
        x1_in = self.out1_cbl(x0)
        #  upsample(scale_factor=2, mode="nearest")
        #  b,c/2,w,h -> b,c/2,2w,2h
        x1_in = self.upsample(x1_in)
        #  b,c/2,2w,2h -> b,c/2+c,2w,2h
        x1_in = torch.cat([x1_in, x1], 1)
        #  b,c/2+c,2w,2h -> b,c/2,2w,2h
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        #  CBL(i=c/2,o=c/4,k=1,s=1,p=same)
        #  b,c/2,2w,2h -> b,c/4,2w,2h
        x2_in = self.out2_cbl(out_dark4)
        #  upsample(scale_factor=2, mode="nearest")
        #  b,c/4,2w,2h -> b,c/4,4w,4h
        x2_in = self.upsample(x2_in)
        #  b,c/4,4w,4h -> b,c/4+c/2,4w,4h
        x2_in = torch.cat([x2_in, x2], 1)
        #  b,c/4+c/2,4w,4h -> b,c/4,4w,4h
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)
        return outputs

