import torch.nn as nn

from block.activation import get_activation


class CBL(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block
    # 输入图像宽高为偶数，且 stride = 1 or 2
    # 输出结果通道数 = out_channels
    # 输出结果宽高 = w/stride, h/stride
    """

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        """CBL
        padding = same padding
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核尺寸
        :param stride: 卷积步长
        :param groups: 分组卷积的组数，关于分组卷积可以看这个：https://blog.csdn.net/qq_34243930/article/details/107231539
        :param bias: 卷积层是否加上偏移量
        :param act: 激活函数名
        """
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))